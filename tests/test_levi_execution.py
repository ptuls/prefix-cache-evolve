"""Regression tests for the Levi workflow adapter."""

import asyncio
import json
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prefix_cache_evolve.evaluator_entry import EvaluatorResult
from prefix_cache_evolve.workflow import execution
from prefix_cache_evolve.workflow.configuration import ConfigLoader
from prefix_cache_evolve.workflow.execution import (
    LeviRunner,
    LeviScoreFunction,
    _configured_levi_failure_memory,
    _configured_reasoning_max_tokens,
    _enable_levi_code_feedback_support,
    _enable_levi_degenerate_centroid_fallback,
    _enable_levi_failure_memory_support,
    _enable_levi_reasoning_completion_support,
    _enable_levi_reproducibility_support,
    _module_name_from_package_path,
    _persist_levi_paradigm_candidate_capture,
)


def test_workflow_config_loader_validates_top_level_and_nested_settings() -> None:
    loader = ConfigLoader()

    with pytest.raises(ValueError, match=r"(?s)unexpected.*Extra inputs are not permitted"):
        loader.from_dict({"unexpected": {}})

    with pytest.raises(ValueError, match=r"(?s)parallel_evaluations.*greater than 0"):
        loader.from_dict({"evaluator": {"parallel_evaluations": 0}})

    with pytest.raises(ValueError, match=r"(?s)prompt.*Input should be a valid dictionary"):
        loader.from_dict({"prompt": []})

    with pytest.raises(ValueError, match=r"(?s)failure_memory.mode.*run_only.*global"):
        loader.from_dict({"failure_memory": {"mode": "unknown"}})


def test_workflow_config_resolves_provider_and_search_seed() -> None:
    config = ConfigLoader().from_dict(
        {
            "max_iterations": 3,
            "llm": {
                "default_provider": "anthropic",
                "primary_model": "mutation-model",
                "secondary_model": "gemini/paradigm-model",
                "api_base": "http://localhost:8000/v1",
                "api_key_env": "LOCAL_MODEL_API_KEY",
            },
            "search": {"seed": 17},
        }
    )

    assert config.mutation_model == "anthropic/mutation-model"
    assert config.paradigm_model == "gemini/paradigm-model"
    assert config.search_seed == 17
    assert config.api_base == "http://localhost:8000/v1"
    assert config.api_key_env == "LOCAL_MODEL_API_KEY"
    assert config.failure_memory.mode == "run_only"


def test_failure_memory_defaults_to_independent_runs_and_allows_global_opt_in(tmp_path) -> None:
    evaluator_path = tmp_path / "evaluator.py"
    evaluator_path.write_text("", encoding="utf-8")
    run_only = ConfigLoader().from_dict({"max_iterations": 1})
    global_path = tmp_path / "global.jsonl"
    global_config = ConfigLoader().from_dict(
        {
            "max_iterations": 1,
            "failure_memory": {
                "mode": "global",
                "global_path": str(global_path),
                "max_global_events": 25,
            },
        }
    )

    assert _configured_levi_failure_memory(
        run_only,
        evaluator_path,
        "def build_candidate():",
    ) == ("run_only", None, None, 1_000)

    mode, configured_path, scope, max_events = _configured_levi_failure_memory(
        global_config,
        evaluator_path,
        "def build_candidate():",
    )
    assert mode == "global"
    assert configured_path == global_path
    assert scope is not None
    assert max_events == 25


def test_levi_score_function_exposes_combined_score() -> None:
    def evaluate_factory(factory):
        return EvaluatorResult(
            metrics={
                "combined_score": factory(),
                "ignored_inf": float("inf"),
                "ignored_text": "n/a",
            },
            artifacts={},
        )

    score_fn = LeviScoreFunction(evaluate_factory)

    assert score_fn(lambda: 2.5) == {"score": 2.5, "combined_score": 2.5}


def test_levi_score_function_prefers_source_aware_evaluator() -> None:
    source = "def build_candidate():\n    return 1\n"

    namespace = {"__source_code__": source}
    exec(source, namespace)

    def evaluate_factory(_factory):
        return EvaluatorResult(metrics={"combined_score": 1.0}, artifacts={})

    def evaluate_source(candidate_source: str):
        assert candidate_source == source
        return EvaluatorResult(
            metrics={"combined_score": 2.0, "scoring_fn_complexity": 7},
            artifacts={},
        )

    score_fn = LeviScoreFunction(evaluate_factory, evaluate_source)

    assert score_fn(namespace["build_candidate"]) == {
        "score": 2.0,
        "combined_score": 2.0,
        "scoring_fn_complexity": 7.0,
    }


def test_levi_score_function_fails_closed_when_source_is_required() -> None:
    def evaluate_factory(_factory):
        return EvaluatorResult(metrics={"combined_score": 1.0}, artifacts={})

    def evaluate_source(_candidate_source: str):
        return EvaluatorResult(metrics={"combined_score": 2.0}, artifacts={})

    score_fn = LeviScoreFunction(evaluate_factory, evaluate_source)

    assert score_fn(lambda: None) == {
        "error": "candidate source unavailable; source-aware evaluation is required"
    }


def test_levi_score_function_clamps_invalid_score() -> None:
    def evaluate_factory(_factory):
        return EvaluatorResult(metrics={"combined_score": float("nan")}, artifacts={})

    score_fn = LeviScoreFunction(evaluate_factory)

    assert score_fn(lambda: None) == {"score": 0.0}


def test_levi_score_function_forwards_failure_feedback() -> None:
    def evaluate_factory(_factory):
        return EvaluatorResult(
            metrics={
                "combined_score": 2.0,
                "success": True,
                "per_example_scores": [0.2, 0.8],
                "feedback_per_example": ["weak first workload", "weak second workload"],
            },
            artifacts={},
        )

    score_fn = LeviScoreFunction(evaluate_factory)

    assert score_fn(lambda: None) == {
        "score": 2.0,
        "combined_score": 2.0,
        "success": 1.0,
        "per_example_scores": [0.2, 0.8],
        "feedback_per_example": ["weak first workload", "weak second workload"],
    }


def test_levi_score_function_rejects_unsuccessful_results() -> None:
    def evaluate_factory(_factory):
        return EvaluatorResult(
            metrics={
                "combined_score": -2001.0,
                "success": False,
                "error": "candidate rejected by static policy checks",
            },
            artifacts={},
        )

    score_fn = LeviScoreFunction(evaluate_factory)

    assert score_fn(lambda: None) == {"error": "candidate rejected by static policy checks"}


def test_levi_score_function_forwards_structured_repair_feedback() -> None:
    def evaluate_factory(_factory):
        return EvaluatorResult(
            metrics={
                "combined_score": -2001.0,
                "success": False,
                "error": "candidate rejected by static policy checks",
            },
            artifacts={
                "repair_feedback": [
                    "Delete or simplify at least 20 effective AST nodes.",
                ],
            },
        )

    score_fn = LeviScoreFunction(evaluate_factory)

    assert score_fn(lambda: None) == {
        "error": (
            "Repair before retry: Delete or simplify at least 20 effective AST nodes. "
            "Failure: candidate rejected by static policy checks."
        )
    }


def test_levi_code_adapter_accepts_failure_feedback() -> None:
    from levi.artifacts.code import CodeAdapter
    from levi.core import Program
    from levi.prompts import ProgramWithScore

    config = SimpleNamespace(
        function_signature="def build_candidate():",
        problem_description="test problem",
        prompt_overrides={},
    )
    adapter = CodeAdapter(config)
    parents = [ProgramWithScore(Program(content="def build_candidate():\n    pass\n"))]

    _enable_levi_code_feedback_support()
    prompt = adapter.build_mutation_prompt(
        parents,
        feedback=["weak validation workload validation/agentic_replan"],
    )

    assert "## Evaluator Feedback" in prompt
    assert "weak validation workload validation/agentic_replan" in prompt
    assert "These diagnostics describe the selected parent" in prompt
    assert "## Preflight Repair Checks" in prompt
    assert "Never reference request_type or prompt_tokens" in prompt
    assert "dedicated simplification mutation" in prompt
    assert prompt.index("## Evaluator Feedback") < prompt.index("## Output")


def test_levi_repeated_failure_memory_reaches_later_prompts(tmp_path, monkeypatch) -> None:
    from levi.artifacts.code import CodeAdapter
    from levi.core import Program
    from levi.pipeline.state import PipelineState
    from levi.prompts import ProgramWithScore

    feedback_path = tmp_path / "failure_feedback.jsonl"
    monkeypatch.setattr(execution, "_levi_failure_feedback_path", feedback_path)
    feedback_path.write_text(
        "\n".join(
            [
                json.dumps({"key": "stale", "guidance": "stale prior-run failure"}),
                json.dumps({"key": "stale", "guidance": "stale prior-run failure"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _enable_levi_failure_memory_support(feedback_path)
    state = object.__new__(PipelineState)
    state.budget_tracker = SimpleNamespace(eval_count=0)
    state.error_count = 0
    state.period_errors = 0
    state.period_error_messages = set()
    state.all_error_counts = {}

    error = "Static policy violations: sanitized request field request_type is not a policy signal"
    state.record_error(error)
    state.record_error(error)
    with feedback_path.open("a", encoding="utf-8") as stream:
        stream.write("[]\n{partial-json\n")

    config = SimpleNamespace(
        function_signature="def build_candidate():",
        problem_description="test problem",
        prompt_overrides={},
        init=SimpleNamespace(diversity_prompt=None),
    )
    adapter = CodeAdapter(config)
    parents = [ProgramWithScore(Program(content="def build_candidate():\n    pass\n"))]
    _enable_levi_code_feedback_support()

    mutation_prompt = adapter.build_mutation_prompt(parents)
    diversity_prompt = adapter.build_diversity_prompt([])

    assert feedback_path.is_file()
    assert "## Failure Memory" in mutation_prompt
    assert "from this Levi run" in mutation_prompt
    assert "(2x) Remove request_type entirely" in mutation_prompt
    assert "stale prior-run failure" not in mutation_prompt
    assert "## Failure Memory" in diversity_prompt
    assert "(2x) Remove request_type entirely" in diversity_prompt


def test_levi_global_failure_memory_reaches_compatible_later_runs(tmp_path) -> None:
    from levi.pipeline.state import PipelineState

    global_path = tmp_path / "global_failure_feedback.jsonl"
    first_run_path = tmp_path / "run-1" / "failure_feedback.jsonl"
    _enable_levi_failure_memory_support(
        first_run_path,
        global_path=global_path,
        scope="prefix-cache-contract",
    )
    state = object.__new__(PipelineState)
    state.budget_tracker = SimpleNamespace(eval_count=0)
    state.error_count = 0
    state.period_errors = 0
    state.period_error_messages = set()
    state.all_error_counts = {}
    error = "Static policy violations: sanitized request field request_type is not a policy signal"
    state.record_error(error)
    state.record_error(error)
    state.record_error("Timeout")

    second_run_path = tmp_path / "run-2" / "failure_feedback.jsonl"
    _enable_levi_failure_memory_support(
        second_run_path,
        global_path=global_path,
        scope="prefix-cache-contract",
    )
    compatible_feedback = execution._recurring_levi_failure_feedback()

    _enable_levi_failure_memory_support(
        second_run_path,
        global_path=global_path,
        scope="different-contract",
    )
    incompatible_feedback = execution._recurring_levi_failure_feedback()

    _enable_levi_failure_memory_support(second_run_path)
    isolated_feedback = execution._recurring_levi_failure_feedback()

    assert compatible_feedback == [
        (2, "Remove request_type entirely; it is not available to candidate policies.")
    ]
    assert incompatible_feedback == []
    assert isolated_feedback == []
    assert '"scope": "prefix-cache-contract"' in global_path.read_text(encoding="utf-8")
    assert "timeout" not in global_path.read_text(encoding="utf-8").lower()


def test_levi_duplicate_init_behaviors_fall_back_to_distinct_uniform_centroids(monkeypatch) -> None:
    from levi.behavior import BehaviorExtractor
    from levi.pool.cvt_map_elites import CVTMAPElitesPool

    extractor = BehaviorExtractor(ast_features=["cyclomatic_complexity"])
    pool = CVTMAPElitesPool(extractor, n_centroids=4, data_driven_centroids=True)
    monkeypatch.setattr(
        pool,
        "_init_cvt_centroids",
        lambda: np.array([[0.1], [0.3], [0.7], [0.9]]),
    )
    duplicate_behaviors = [np.array([0.5])] * 4

    _enable_levi_degenerate_centroid_fallback()
    n_centroids, labels = pool.set_centroids_from_data(duplicate_behaviors, n_centroids=4)

    assert n_centroids == 4
    assert len(np.unique(pool._centroids, axis=0)) == 4
    assert len(np.unique(labels)) == 1
    assert np.array_equal(pool._mins, np.zeros(1))
    assert np.array_equal(pool._maxs, np.ones(1))


def test_compact_paradigm_shift_override_reaches_levi_prompt() -> None:
    from levi.artifacts.code import CodeAdapter
    from levi.config import BudgetConfig, LeviConfig
    from levi.core import EvaluationResult, Program

    run_config = ConfigLoader().load(Path("configs/prefix_kv_cache.yaml"))
    levi_config = LeviConfig(
        problem_description="test problem",
        function_signature="def build_candidate():",
        seed_program="def build_candidate():\n    pass\n",
        score_fn=lambda _factory: {"score": 1.0},
        budget=BudgetConfig(evaluations=2),
        prompt_overrides=run_config.evolve_kwargs()["prompt_overrides"],
    )
    adapter = CodeAdapter(levi_config)
    representative = SimpleNamespace(
        program=Program(content=levi_config.seed_program),
        result=EvaluationResult(scores={"score": 1.0}),
    )

    prompt = adapter.build_paradigm_shift_prompt([(0, representative)], n_evaluations=1)

    assert "Target at most 550 effective AST nodes" in prompt
    assert "Candidates above 650 nodes are exploration-only" in prompt
    assert "COMPLETELY DIFFERENT strategy" not in prompt


def test_configured_reasoning_max_tokens_prefers_paradigm_budget() -> None:
    config = SimpleNamespace(
        punctuated_equilibrium={"max_tokens": 12_000},
        pipeline={"max_tokens": 6_000},
    )

    assert _configured_reasoning_max_tokens(config) == 12_000


def test_levi_reasoning_calls_use_configured_token_budget(monkeypatch) -> None:
    from levi.pipeline.state import PipelineState

    calls = []

    async def fake_acompletion(
        _self,
        client_spec,
        *,
        prompt,
        temperature=None,
        max_tokens=None,
        timeout=None,
        **extras,
    ):
        calls.append(
            {
                "client_spec": client_spec,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
                **extras,
            }
        )
        return "response"

    monkeypatch.setattr(PipelineState, "acompletion", fake_acompletion)
    _enable_levi_reasoning_completion_support(12_000)
    state = object.__new__(PipelineState)

    result = asyncio.run(
        state.acompletion(
            "openai/gpt-5.5",
            prompt=[{"role": "user", "content": "write code"}],
            max_tokens=4_096,
            reasoning_effort="medium",
        )
    )

    assert result == "response"
    assert calls[0]["max_tokens"] == 12_000
    assert calls[0]["reasoning_effort"] == "medium"


def test_levi_non_reasoning_calls_keep_requested_token_budget(monkeypatch) -> None:
    from levi.pipeline.state import PipelineState

    calls = []

    async def fake_acompletion(_self, _client_spec, *, max_tokens=None, **_kwargs):
        calls.append(max_tokens)
        return "response"

    monkeypatch.setattr(PipelineState, "acompletion", fake_acompletion)
    _enable_levi_reasoning_completion_support(12_000)
    state = object.__new__(PipelineState)

    asyncio.run(state.acompletion("openai/gpt-5.4-mini", prompt="mutate", max_tokens=4_096))

    assert calls == [4_096]


def test_levi_reproducibility_seeds_selection_and_model_requests(monkeypatch) -> None:
    from levi.pipeline.state import PipelineState

    calls = []

    async def fake_acompletion(_self, _client_spec, *, seed=None, **kwargs):
        calls.append({"seed": seed, **kwargs})
        return "response"

    monkeypatch.setattr(PipelineState, "acompletion", fake_acompletion)
    _enable_levi_reproducibility_support(41)
    state = object.__new__(PipelineState)

    first_python = random.random()
    first_numpy = float(np.random.random())
    asyncio.run(state.acompletion("openai/model", prompt="first"))
    asyncio.run(state.acompletion("openai/model", prompt="second"))

    _enable_levi_reproducibility_support(41)
    assert random.random() == first_python
    assert float(np.random.random()) == first_numpy
    assert [call["seed"] for call in calls] == [41, 42]
    assert all(call["drop_params"] is True for call in calls)


def test_levi_request_defaults_resolve_api_key_without_recording_it(monkeypatch) -> None:
    from levi.pipeline.state import PipelineState

    calls = []

    async def fake_acompletion(_self, _client_spec, **kwargs):
        calls.append(kwargs)
        return "response"

    monkeypatch.setattr(PipelineState, "acompletion", fake_acompletion)
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "secret-value")
    _enable_levi_reproducibility_support(
        7,
        api_base="http://localhost:8000/v1",
        api_key_env="LOCAL_MODEL_API_KEY",
    )
    state = object.__new__(PipelineState)

    asyncio.run(state.acompletion("openai/local-model", prompt="mutate"))

    assert calls[0]["api_base"] == "http://localhost:8000/v1"
    assert calls[0]["api_key"] == "secret-value"
    assert calls[0]["seed"] == 7


def test_persist_levi_paradigm_candidate_capture_keeps_rejected_code(
    tmp_path,
    monkeypatch,
) -> None:
    import prefix_cache_evolve.workflow.execution as execution

    monkeypatch.setattr(execution, "_levi_paradigm_candidate_output_dir", tmp_path)
    _persist_levi_paradigm_candidate_capture(
        {
            "trigger_evaluation": 20,
            "budget_progress": 0.2,
            "candidates": [
                {
                    "code": "def build_candidate():\n    return None\n",
                    "result": {"score": 72.5, "combined_score": 72.5},
                },
                {
                    "code": "def build_candidate():\n    return 1\n",
                    "result": {"error": "candidate rejected"},
                },
            ],
        },
        {
            "paradigm_accepted": False,
            "evaluations": [{"source": "paradigm_shift", "accepted": False}],
        },
    )

    event_dir = tmp_path / "eval_0020"
    assert (event_dir / "00_paradigm_shift.py").is_file()
    assert (event_dir / "01_variant.py").is_file()
    manifest = json.loads((event_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidates"][0]["score"] == 72.5
    assert manifest["candidates"][1]["error"] == "candidate rejected"
    assert manifest["stats"]["paradigm_accepted"] is False


def test_levi_runner_loads_package_evaluator_as_picklable_function() -> None:
    evaluator_path = (
        Path(__file__).resolve().parents[1]
        / "src/prefix_cache_evolve/problems/prefix_kv_cache/evaluator.py"
    )

    assert (
        _module_name_from_package_path(evaluator_path)
        == "prefix_cache_evolve.problems.prefix_kv_cache.evaluator"
    )

    runner = LeviRunner(
        evolve_code=lambda *_args, **_kwargs: None,
        evaluator_path=evaluator_path,
        problem_description="test",
        function_signature=(
            "def build_candidate(capacity_blocks: int, block_size_tokens: int, "
            "seed: int | None = None):"
        ),
    )
    score_fn = LeviScoreFunction(runner._evaluate_factory)

    pickle.loads(pickle.dumps(score_fn))


def test_levi_runner_loads_file_evaluator_with_future_dataclass(tmp_path) -> None:
    evaluator_path = tmp_path / "evaluator.py"
    evaluator_path.write_text(
        """
from __future__ import annotations

from dataclasses import dataclass

from prefix_cache_evolve.evaluator_entry import EvaluatorResult


@dataclass
class Payload:
    child: Payload | None = None


def evaluate_factory(factory):
    return EvaluatorResult(metrics={"combined_score": 1.0}, artifacts={"payload": Payload()})


def evaluate_source(source):
    return EvaluatorResult(
        metrics={"combined_score": 2.0, "source_length": len(source)},
        artifacts={"payload": Payload()},
    )
""",
        encoding="utf-8",
    )

    runner = LeviRunner(
        evolve_code=lambda *_args, **_kwargs: None,
        evaluator_path=evaluator_path,
        problem_description="test",
        function_signature="def candidate_factory():",
    )

    assert runner._evaluate_factory(lambda: None).metrics["combined_score"] == 1.0
    assert (
        runner._evaluate_best_program("def build_candidate():\n    return None\n").metrics[
            "combined_score"
        ]
        == 2.0
    )


def test_levi_runner_records_generated_snapshot_path(tmp_path, monkeypatch) -> None:
    captured = {}
    program_path = tmp_path / "program.py"
    program_path.write_text("def build_candidate():\n    return None\n", encoding="utf-8")

    def evolve_code(_description, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            best_program=program_path.read_text(encoding="utf-8"),
            best_score=2.0,
            total_evaluations=3,
            total_cost=0.1,
            archive_size=2,
            runtime_seconds=1.0,
        )

    runner = LeviRunner(
        evolve_code=evolve_code,
        evaluator_path=(
            Path(__file__).resolve().parents[1]
            / "src/prefix_cache_evolve/problems/prefix_kv_cache/evaluator.py"
        ),
        problem_description="test",
        function_signature="def build_candidate():",
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_best_program",
        lambda _source: EvaluatorResult(
            metrics={"combined_score": 2.0},
            artifacts={},
        ),
    )
    config = SimpleNamespace(
        evolve_kwargs=lambda: {},
        search_seed=17,
        model="anthropic/example-model",
        paradigm_model=None,
        mutation_model=None,
        api_base="http://localhost:8000/v1",
        api_key_env="LOCAL_MODEL_API_KEY",
        pipeline={"n_llm_workers": 1},
    )

    result = runner.run(program_path, config)

    output_dir = Path(captured["output_dir"])
    assert result.metadata["levi_output_dir"] == str(output_dir)
    assert result.metadata["levi_snapshot_path"] == str(output_dir / "snapshot.json")
    assert result.metadata["levi_paradigm_candidates_dir"] == str(
        output_dir / "paradigm_candidates"
    )
    assert result.metadata["levi_failure_feedback_path"] == str(
        output_dir / "failure_feedback.jsonl"
    )
    assert result.metadata["levi_failure_memory_mode"] == "run_only"
    assert result.metadata["levi_global_failure_feedback_path"] is None
    assert result.metadata["levi_failure_feedback_scope"] is None
    assert result.metadata["levi_global_failure_feedback_max_events"] == 1_000
    assert result.metadata["search_seed"] == 17
    assert result.metadata["model"] == "anthropic/example-model"
    assert result.metadata["api_base"] == "http://localhost:8000/v1"
    assert result.metadata["api_key_env"] == "LOCAL_MODEL_API_KEY"
    assert result.metadata["pipeline"] == {"n_llm_workers": 1}
    assert result.metadata["runtime"]["python"]
    assert result.metadata["runtime"]["packages"]["numpy"]


def test_levi_runner_prefers_configured_function_signature(tmp_path, monkeypatch) -> None:
    captured = {}
    program_path = tmp_path / "program.py"
    program_path.write_text(
        "def score_eviction(block, now, frequency, priority):\n    return 0.0\n",
        encoding="utf-8",
    )

    def evolve_code(_description, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            best_program=program_path.read_text(encoding="utf-8"),
            best_score=1.0,
        )

    runner = LeviRunner(
        evolve_code=evolve_code,
        evaluator_path=(
            Path(__file__).resolve().parents[1]
            / "src/prefix_cache_evolve/problems/prefix_kv_cache/evaluator.py"
        ),
        problem_description="test",
        function_signature="def build_candidate():",
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_best_program",
        lambda _source: EvaluatorResult(metrics={"combined_score": 1.0}, artifacts={}),
    )
    config = SimpleNamespace(
        function_signature="def score_eviction(block, now, frequency, priority):",
        evolve_kwargs=lambda: {},
    )

    runner.run(program_path, config)

    assert captured["function_signature"] == (
        "def score_eviction(block, now, frequency, priority):"
    )
