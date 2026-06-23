"""Regression tests for the Levi workflow adapter."""

import asyncio
import importlib.util
import json
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prefix_cache_evolve.evaluator_entry import EvaluatorResult
from prefix_cache_evolve.workflow.config import ConfigLoader, yaml_documents_equal
from prefix_cache_evolve.workflow.execution import (
    LeviRunner,
    LeviScoreFunction,
    _configured_paradigm_max_tokens,
    _configured_paradigm_model_names,
    _enable_levi_code_feedback_support,
    _enable_levi_degenerate_centroid_fallback,
    _enable_levi_paradigm_completion_support,
    _enable_levi_reproducibility_support,
    _module_name_from_package_path,
    _persist_levi_paradigm_candidate_capture,
)
from tests.support import score_identity

requires_levi = pytest.mark.skipif(
    importlib.util.find_spec("levi") is None,
    reason="Levi is installed only with the evolution extra",
)


def test_workflow_config_loader_validates_top_level_and_nested_settings() -> None:
    loader = ConfigLoader()

    with pytest.raises(ValueError, match=r"(?s)unexpected.*Extra inputs are not permitted"):
        loader.from_dict({"unexpected": {}})

    with pytest.raises(ValueError, match=r"(?s)parallel_evaluations.*greater than 0"):
        loader.from_dict({"evaluator": {"parallel_evaluations": 0}})

    with pytest.raises(ValueError, match=r"(?s)prompt.*Input should be a valid dictionary"):
        loader.from_dict({"prompt": []})


def test_yaml_document_comparison_ignores_formatting_but_detects_value_changes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("search:\n  seed: 17\nllm:\n  temperature: 0.3\n", encoding="utf-8")
    second.write_text(
        "# Same parsed document with different key order and flow style.\n"
        "llm: {temperature: 0.3}\nsearch: {seed: 17}\n",
        encoding="utf-8",
    )

    assert yaml_documents_equal(first, second)

    second.write_text("llm: {temperature: 0.3}\nsearch: {seed: 18}\n", encoding="utf-8")

    assert not yaml_documents_equal(first, second)

    second.write_text("search: [\n", encoding="utf-8")

    assert not yaml_documents_equal(first, second)

    first.write_text("false\n", encoding="utf-8")
    second.write_text("{}\n", encoding="utf-8")

    assert not yaml_documents_equal(first, second)


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


def test_levi_score_function_preserves_score_identity() -> None:
    def evaluate_factory(_factory):
        return EvaluatorResult(
            metrics={
                **score_identity(),
                "combined_score": 2.5,
            },
            artifacts={},
        )

    score_fn = LeviScoreFunction(evaluate_factory)

    assert score_fn(lambda: None) == {
        "score": 2.5,
        **score_identity(),
        "combined_score": 2.5,
    }


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


@requires_levi
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
    assert "### Failure Cases With Measurements" in prompt
    assert "Each item is an aggregate from one evaluated workload" in prompt
    assert "weak validation workload validation/agentic_replan" in prompt
    assert "## Preflight Repair Checks" in prompt
    assert "dedicated simplification mutation" in prompt
    assert prompt.index("## Evaluator Feedback") < prompt.index("## Output")


@requires_levi
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


@requires_levi
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


def test_configured_paradigm_max_tokens_prefers_paradigm_budget() -> None:
    config = SimpleNamespace(
        punctuated_equilibrium={"max_tokens": 12_000},
        pipeline={"max_tokens": 6_000},
    )

    assert _configured_paradigm_max_tokens(config) == 12_000


@requires_levi
def test_production_config_resolves_reasoning_paradigm_budget_and_model() -> None:
    config = ConfigLoader().load(Path("configs/prefix_kv_cache.yaml"))

    assert _configured_paradigm_max_tokens(config) == 12_000
    assert _configured_paradigm_model_names(config) == frozenset({"openai/gpt-5.5"})


@requires_levi
def test_configured_paradigm_models_prefer_explicit_heavy_models() -> None:
    config = SimpleNamespace(
        model=None,
        paradigm_model="openai/default-paradigm",
        punctuated_equilibrium={
            "heavy_models": [
                "openai/reasoning-paradigm",
                "anthropic/alternate-paradigm",
            ]
        },
    )

    assert _configured_paradigm_model_names(config) == frozenset(
        {
            "openai/reasoning-paradigm",
            "anthropic/alternate-paradigm",
        }
    )


@requires_levi
def test_configured_paradigm_models_mirror_levi_empty_heavy_model_fallback() -> None:
    config = SimpleNamespace(
        model=None,
        mutation_model="openai/mutation-model",
        paradigm_model="openai/default-paradigm",
        punctuated_equilibrium={"heavy_models": []},
    )

    assert _configured_paradigm_model_names(config) == frozenset({"openai/default-paradigm"})


@requires_levi
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
    _enable_levi_paradigm_completion_support(
        12_000,
        paradigm_model_names=frozenset({"openai/gpt-5.5"}),
    )
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


@requires_levi
def test_levi_paradigm_model_without_reasoning_effort_uses_configured_budget(
    monkeypatch,
) -> None:
    from levi.pipeline.state import PipelineState

    calls = []

    async def fake_acompletion(_self, _client_spec, *, max_tokens=None, **_kwargs):
        calls.append(max_tokens)
        return "response"

    monkeypatch.setattr(PipelineState, "acompletion", fake_acompletion)
    _enable_levi_paradigm_completion_support(
        12_000,
        paradigm_model_names=frozenset({"openai/gpt-5.5"}),
    )
    state = object.__new__(PipelineState)

    asyncio.run(state.acompletion("openai/gpt-5.5", prompt="shift", max_tokens=4_096))

    assert calls == [12_000]


@requires_levi
def test_levi_mutation_calls_keep_requested_token_budget(monkeypatch) -> None:
    from levi.pipeline.state import PipelineState

    calls = []

    async def fake_acompletion(_self, _client_spec, *, max_tokens=None, **_kwargs):
        calls.append(max_tokens)
        return "response"

    monkeypatch.setattr(PipelineState, "acompletion", fake_acompletion)
    _enable_levi_paradigm_completion_support(
        12_000,
        paradigm_model_names=frozenset({"openai/gpt-5.5"}),
    )
    state = object.__new__(PipelineState)

    asyncio.run(
        state.acompletion(
            "openai/gpt-5.4-mini",
            prompt="mutate",
            max_tokens=4_096,
            reasoning_effort="medium",
        )
    )

    assert calls == [4_096]


@requires_levi
def test_levi_native_or_nonlegacy_paradigm_budget_is_preserved(monkeypatch) -> None:
    from levi.pipeline.state import PipelineState

    calls = []

    async def fake_acompletion(_self, _client_spec, *, max_tokens=None, **_kwargs):
        calls.append(max_tokens)
        return "response"

    monkeypatch.setattr(PipelineState, "acompletion", fake_acompletion)
    _enable_levi_paradigm_completion_support(
        12_000,
        paradigm_model_names=frozenset({"openai/gpt-5.5"}),
    )
    state = object.__new__(PipelineState)

    asyncio.run(state.acompletion("openai/gpt-5.5", prompt="shift", max_tokens=16_000))

    assert calls == [16_000]


@requires_levi
def test_levi_disabled_reasoning_still_uses_configured_output_budget(monkeypatch) -> None:
    from levi.pipeline.state import PipelineState

    calls = []

    async def fake_acompletion(_self, _client_spec, *, max_tokens=None, **_kwargs):
        calls.append(max_tokens)
        return "response"

    monkeypatch.setattr(PipelineState, "acompletion", fake_acompletion)
    _enable_levi_paradigm_completion_support(
        12_000,
        paradigm_model_names=frozenset({"openai/gpt-5.5"}),
    )
    state = object.__new__(PipelineState)

    asyncio.run(
        state.acompletion(
            "openai/gpt-5.5",
            prompt="shift",
            max_tokens=4_096,
            reasoning_effort="disabled",
        )
    )

    assert calls == [12_000]


@requires_levi
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


@requires_levi
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
                    "result": {
                        **score_identity(),
                        "score": 72.5,
                        "combined_score": 72.5,
                    },
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
    candidate = manifest["candidates"][0]
    assert candidate["score"] == 72.5
    for key, value in score_identity().items():
        assert candidate[key] == value
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


@requires_levi
def test_levi_runner_records_generated_snapshot_path(tmp_path, monkeypatch) -> None:
    captured = {}
    program_path = tmp_path / "program.py"
    program_path.write_text("def build_candidate():\n    return None\n", encoding="utf-8")

    def evolve_code(_description, **kwargs):
        captured.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "snapshot.json").write_text(
            json.dumps(
                {
                    "metadata": {"best_score": 2.0},
                    "run_state": {"best_score": 2.0},
                    "elites": [{"primary_score": 2.0}],
                    "score_history": [{"score": 2.0, "best_score": 2.0}],
                }
            ),
            encoding="utf-8",
        )
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
            metrics={
                **score_identity(),
                "combined_score": 2.0,
            },
            artifacts=score_identity(),
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
    assert result.metadata["search_seed"] == 17
    assert result.metadata["model"] == "anthropic/example-model"
    assert result.metadata["api_base"] == "http://localhost:8000/v1"
    assert result.metadata["api_key_env"] == "LOCAL_MODEL_API_KEY"
    assert result.metadata["pipeline"] == {"n_llm_workers": 1}
    assert result.metadata["runtime"]["python"]
    assert result.metadata["runtime"]["packages"]["numpy"]
    snapshot = json.loads(output_dir.joinpath("snapshot.json").read_text(encoding="utf-8"))
    score_records = (
        snapshot,
        snapshot["metadata"],
        snapshot["run_state"],
        snapshot["elites"][0],
        snapshot["score_history"][0],
    )
    for record in score_records:
        for key, value in score_identity().items():
            assert record[key] == value


@requires_levi
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
