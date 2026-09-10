"""Behavior-preservation and accounting tests for opt-in cost profiling."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from prefix_cache_evolve.evaluators.baselines import BASELINE_REGISTRY
from prefix_cache_evolve.evaluators.complexity import scoring_fn_complexity
from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.evaluators.policy_costs import (
    Distribution,
    baseline_source,
    equivalent_behavior,
    profile_policy,
    retained_size,
)
from prefix_cache_evolve.evaluators.prefix_kv_cache import PrefixKVCacheEvaluator
from prefix_cache_evolve.problems.prefix_kv_cache import runner
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.evaluator import _candidate_source_violations
from prefix_cache_evolve.problems.prefix_kv_cache.runner import (
    _persist_behavior_size_frontier,
    _resolve_search_seed,
)
from prefix_cache_evolve.problems.prefix_kv_cache.sandbox import _fresh_policy
from prefix_cache_evolve.problems.prefix_kv_cache.sandbox import main as sandbox_main
from prefix_cache_evolve.tools import analyze_policy_costs
from prefix_cache_evolve.tools.analyze_policy_costs import assess_budgets, pareto_names

_DYNAMIC_CALLBACK_SOURCE = """
class Policy:
    def on_request_start(self, request, now):
        self.score_admission = self._reject

    def score_admission(self, block, now):
        return 1.0

    def _reject(self, block, now):
        return -1.0

    def score_eviction(self, block, now):
        return float(now - block.last_accessed_at)

    def on_cache_hit(self, block, request, now):
        pass

    on_cache_miss = on_cache_hit

def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return Policy()
"""

_CLOSURE_STATE_SOURCE = """
class Policy:
    def __init__(self):
        history = {}
        def observe(request, now):
            history[request.request_id] = [now] * 32
        self.on_request_start = observe

    def score_admission(self, block, now):
        return 1.0

    def score_eviction(self, block, now):
        return float(now - block.last_accessed_at)

    def on_cache_hit(self, block, request, now):
        pass

    on_cache_miss = on_cache_hit

def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return Policy()
"""

_SIZE_HOOK_SOURCE = _DYNAMIC_CALLBACK_SOURCE.replace(
    "        self.score_admission = self._reject", "        pass"
).replace(
    "    def _reject(self, block, now):",
    "    def __sizeof__(self):\n"
    "        self.score_admission = self._reject\n"
    "        return 0\n\n"
    "    def _reject(self, block, now):",
)

_BUILTIN_CALLBACK_SOURCE = _CLOSURE_STATE_SOURCE.replace(
    "        history = {}\n"
    "        def observe(request, now):\n"
    "            history[request.request_id] = [now] * 32\n"
    "        self.on_request_start = observe",
    "        self.on_request_start = {}.setdefault",
)

_CONTAINER_POLICY_SOURCE = _DYNAMIC_CALLBACK_SOURCE.replace(
    "class Policy:",
    "class Policy(dict):\n    def __init__(self):\n        self.history = []\n",
).replace("        self.score_admission = self._reject", "        self.history.append([now] * 32)")


def small_config():
    return EvaluatorConfig(
        request_count=12,
        seeds=(3,),
        train_families=(),
        validation_families=("hotset_cold_scan", "session_continuation_growth"),
        capacity_sweep_blocks=(4, 8),
    )


@pytest.mark.parametrize("name", ("lru", "tinylfu_lru"))
def test_profiling_preserves_baseline_trials_and_accounts_for_callbacks_and_state(name):
    factory = BASELINE_REGISTRY.factories()[name]
    config = small_config().with_updates(request_count=16)
    regular = PrefixKVCacheEvaluator(config, splits=("validation",))(factory)
    measured = profile_policy(factory, config, complexity=0, splits=("validation",))
    assert equivalent_behavior(regular, measured)
    assert regular.combined_score == measured.combined_score
    profile = json.loads(measured.candidate_metadata["policy_cost_profile"])
    assert len(profile["trials"]) == 4
    for trial in profile["trials"]:
        assert trial["callbacks"]["on_request_start"]["wall_ns"]["count"] == 16
        assert (
            trial["eviction_scans"]["candidates"]["total"]
            == (trial["callbacks"]["score_eviction"]["wall_ns"]["count"])
        )
        assert [row["requests"] for row in trial["state_samples"]] == [0, 1, 2, 4, 8, 16]
        first, last = trial["state_samples"][0], trial["state_samples"][-1]
        assert last["unique_prefixes"] > 0
        if name == "tinylfu_lru":
            assert last["retained_bytes"] > first["retained_bytes"]
            assert last["resident_blocks"] <= trial["capacity_blocks"] < last["unique_prefixes"]


@pytest.mark.parametrize(
    "source, outcome",
    [
        pytest.param(_DYNAMIC_CALLBACK_SOURCE, "reject", id="replaced-callback"),
        pytest.param(_CLOSURE_STATE_SOURCE, "state", id="closure-state"),
        pytest.param(_SIZE_HOOK_SOURCE, "accept", id="size-hook-not-invoked"),
        pytest.param(_BUILTIN_CALLBACK_SOURCE, "state", id="builtin-owner"),
        pytest.param(_CONTAINER_POLICY_SOURCE, "state", id="container-attributes"),
    ],
)
def test_profiling_preserves_callback_behavior_and_counts_owned_state(source, outcome):
    config = small_config().with_updates(reject_unsupported_source_patterns=True)
    complexity = scoring_fn_complexity(source)
    assert not _candidate_source_violations(source, complexity, config)

    def factory(capacity, block_size, seed):
        return _fresh_policy(source, ("build_candidate",), capacity, block_size, seed)

    regular = PrefixKVCacheEvaluator(config, splits=("validation",))(
        factory, scoring_fn_complexity=complexity
    )
    measured = profile_policy(factory, config, complexity=complexity, splits=("validation",))
    assert equivalent_behavior(regular, measured)
    if outcome == "reject":
        assert all(trial.token_hit_rate == 0 for trial in measured.trials)
    elif outcome == "accept":
        assert any(trial.token_hit_rate > 0 for trial in measured.trials)
    else:
        profiles = [json.loads(measured.candidate_metadata["policy_cost_profile"])]
        for trial in profiles[0]["trials"]:
            first, last = trial["state_samples"][0], trial["state_samples"][-1]
            assert last["requests"] == config.request_count
            assert last["retained_bytes"] > first["retained_bytes"] + 2000
        assert (
            assess_budgets(
                profiles, callback_p99_us=None, scan_p99_us=None, state_budget_bytes=2000
            )["status"]
            == "exceeded"
        )


def test_callback_defaults_and_closures_share_one_state_graph_without_globals():
    state = []

    def callback(positional=state, *, keyword=state):
        return positional, keyword, state

    assert retained_size(callback) > retained_size(state)
    initial = retained_size(callback)
    state.extend(range(100))
    growth = retained_size(callback) - initial
    assert 3000 < growth < 6000  # Three references must not triple-count the state.
    assert retained_size(callback) < 10000  # Module globals must stay outside the graph.


def test_bound_callback_state_includes_its_owner_and_handles_cycles():
    class Owner:
        def __init__(self):
            self.state = []
            self.callback = self.observe

        def observe(self):
            return self.state

    owner = Owner()
    initial = retained_size(owner.callback)
    owner.state.extend(range(100))
    assert retained_size(owner.callback) > initial + 3000


def test_native_state_inspection_avoids_hooks_and_includes_nested_container_attributes():
    class Meta(type):
        def __getattribute__(cls, name):
            raise AssertionError("inspection invoked a metaclass hook")

    class State(dict, metaclass=Meta):
        __slots__ = ("__history",)

        def __init__(self, history):
            self.__history = history

        def __getattribute__(self, name):
            raise AssertionError("inspection invoked an attribute hook")

        def __sizeof__(self):
            raise AssertionError("inspection invoked a size hook")

        def __iter__(self):
            raise AssertionError("inspection invoked an iterator hook")

        def keys(self):
            raise AssertionError("inspection invoked a mapping hook")

    history = []
    state = [State(history)]
    initial = retained_size(state)
    history.extend(range(100))
    assert retained_size(state) > initial + 3000


def test_source_convention_includes_bases_and_referenced_module_helpers():
    tiny = baseline_source("tinylfu_lru")
    lfu = baseline_source("lfu")
    assert "class _BasePolicy" in tiny
    assert "class _TinyLFULRUPolicy" in tiny
    assert "class _LRUPolicy" not in tiny
    assert "def baseline_tinylfu" not in tiny
    assert "def _recency_tiebreak" in lfu
    assert scoring_fn_complexity(tiny) == 227


def test_histogram_quantiles_bound_exact_percentiles_and_storage_is_bounded():
    distribution = Distribution()
    for value in range(10000):
        distribution.add(value)
    summary = distribution.summary()
    assert summary["count"] == 10000
    assert summary["total"] == sum(range(10000))
    assert summary["max"] == 9999
    for percentile in (50, 95, 99):
        exact = percentile * 100 - 1
        assert exact <= summary[f"p{percentile}_upper"] <= (exact + 1) * 1.01
    assert len(distribution.buckets) < 1000
    assert Distribution().summary()["p99_upper"] == 0


def test_simplification_rejects_behavior_identity_and_validity_changes():
    result = PrefixKVCacheEvaluator(small_config(), splits=("validation",))()
    simpler = replace(
        result, trials=tuple(replace(trial, scoring_fn_complexity=12) for trial in result.trials)
    )
    assert equivalent_behavior(result, simpler)
    changed = replace(
        simpler, trials=(replace(simpler.trials[0], token_hit_rate=0.987), *simpler.trials[1:])
    )
    assert not equivalent_behavior(result, changed)
    assert not equivalent_behavior(result, replace(simpler, panel_sha256="different"))
    assert not equivalent_behavior(result, replace(simpler, success=False))
    assert not equivalent_behavior(replace(result, trials=()), replace(simpler, trials=()))


def test_pareto_retains_smaller_alternatives_and_equal_pairs():
    rows = {
        "large": {"raw_before_complexity": 20, "implementation_ast_nodes": 500},
        "small": {"raw_before_complexity": 19, "implementation_ast_nodes": 100},
        "same": {"raw_before_complexity": 19, "implementation_ast_nodes": 100},
        "dominated": {"raw_before_complexity": 18, "implementation_ast_nodes": 300},
    }
    assert pareto_names(rows) == ["large", "same", "small"]


def test_budget_assessment_uses_worst_trial_and_leaves_unspecified_costs_unassessed():
    result = profile_policy(
        BASELINE_REGISTRY.factories()["lru"], small_config(), complexity=0, splits=("validation",)
    )
    profiles = [json.loads(result.candidate_metadata["policy_cost_profile"])]
    assert (
        assess_budgets(profiles, callback_p99_us=None, scan_p99_us=None, state_budget_bytes=None)[
            "status"
        ]
        == "unassessed"
    )
    assessed = assess_budgets(
        profiles, callback_p99_us=None, scan_p99_us=None, state_budget_bytes=1
    )
    assert assessed["status"] == "exceeded"
    assert assessed["failed"] == ["sampled_retained_bytes"]
    assert assessed["deployment_approved"] is False


def test_exploration_config_uses_new_seed_and_keeps_size_limits():
    path = Path("configs/prefix_kv_cache_replay_exploration.yaml")
    config = load_evaluator_config(path)
    assert config.search_score_mode == "raw_before_complexity"
    assert config.max_candidate_complexity == 750
    assert config.promotion_max_candidate_complexity == 650
    assert _resolve_search_seed(str(path)).name == ("joint_four_domain_simplified_20260908.py")
    assert _resolve_search_seed(str(path), Path("override.py")) == Path("override.py")
    historical = load_evaluator_config(Path("configs/prefix_kv_cache_replay_audited.yaml"))
    assert historical.search_score_mode == "combined"
    assert config == historical.with_updates(search_score_mode="raw_before_complexity")


@pytest.mark.parametrize("override", (False, True))
def test_synthetic_snapshot_resolves_archived_seed_after_relocation(
    tmp_path, monkeypatch, override
):
    config = small_config()
    source = _SIZE_HOOK_SOURCE if override else _DYNAMIC_CALLBACK_SOURCE
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    config_path = inputs / "search.yaml"
    original_document = {
        "search": {"seed_program": "initial.py"},
        "problem": {"settings": config.model_dump(mode="json")},
    }
    config_path.write_text(json.dumps(original_document))
    initial = inputs / "initial.py"
    initial.write_text(_DYNAMIC_CALLBACK_SOURCE)
    assert _resolve_search_seed(str(config_path)).read_text() == _DYNAMIC_CALLBACK_SOURCE

    def factory(capacity, block_size, seed):
        return _fresh_policy(source, ("build_candidate",), capacity, block_size, seed)

    evaluation = PrefixKVCacheEvaluator(config, splits=("train", "validation"))(factory)
    identity = {
        key: getattr(evaluation, key)
        for key in ("verifier_version", "evaluation_context_sha256", "panel_sha256")
    }
    monkeypatch.setattr(runner, "_evaluate_baselines", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_evaluate_candidate_program", lambda *args: evaluation)
    monkeypatch.setattr(runner, "_repository_state", lambda: {})
    saved = runner.save_run_artifacts(
        SimpleNamespace(best_program=source, metrics=identity, artifacts=identity),
        tmp_path / "runs",
        iterations=0,
        config_label="synthetic",
        seed_source=source,
        report_config=config,
        report_config_file=str(config_path),
        config_snapshot=config_path,
    )
    config_path.unlink()
    initial.unlink()
    relocated = tmp_path / "relocated"
    saved.rename(relocated)
    snapshot = relocated / "config_snapshot.yaml"
    assert _resolve_search_seed(str(snapshot)) == relocated / "seed_program.py"
    assert _resolve_search_seed(str(snapshot)).read_text() == source
    assert load_evaluator_config(snapshot) == config
    assert json.loads((relocated / "source_config.yaml").read_text()) == original_document


def test_cost_cli_writes_profiles_and_consistent_size_reports(tmp_path, monkeypatch):
    config = small_config().with_updates(sandbox_image="sha256:test")
    monkeypatch.setattr(analyze_policy_costs, "load_evaluator_config", lambda path: config)

    def evaluate(source, settings, *, splits, profile, baseline):
        assert profile and splits == ("train", "validation")
        assert source == ""
        result = profile_policy(
            BASELINE_REGISTRY.factories()[baseline], settings, complexity=0, splits=splits
        )
        result.candidate_metadata["profile_source_sha256"] = hashlib.sha256(
            baseline_source(baseline).encode()
        ).hexdigest()
        return result

    monkeypatch.setattr(analyze_policy_costs, "evaluate_in_docker", evaluate)
    output = tmp_path / "profiles"
    args = [
        "--config",
        "configs/prefix_kv_cache.yaml",
        "--output-dir",
        str(output),
        "--baseline",
        "tinylfu_lru",
        "--repeats",
        "2",
    ]
    result = CliRunner().invoke(analyze_policy_costs.main, args)
    assert result.exit_code == 0, result.output
    report = json.loads((output / "report.json").read_text())
    assert report["policies"]["tinylfu_lru"]["implementation_ast_nodes"] == 227
    assert report["policies"]["tinylfu_lru"]["budgets"]["status"] == "unassessed"
    assert "Callback p99 upper" in (output / "report.md").read_text()
    assert CliRunner().invoke(analyze_policy_costs.main, args).exit_code != 0


def test_profile_worker_protocol_supports_baselines_and_rejects_invalid_source(tmp_path):
    payload = {
        "source": "",
        "baseline": "tinylfu_lru",
        "profile": True,
        "config": small_config().model_dump(mode="json"),
        "splits": ["validation"],
    }
    request = tmp_path / "request.json"
    request.write_text(json.dumps(payload))
    result = CliRunner().invoke(sandbox_main, [str(request)])
    assert result.exit_code == 0, result.output
    measured = json.loads(result.output)
    assert (
        measured["candidate_metadata"]["profile_source_sha256"]
        == hashlib.sha256(baseline_source("tinylfu_lru").encode()).hexdigest()
    )
    payload.update(baseline=None, source="import os\nos._exit(0)")
    request.write_text(json.dumps(payload))
    result = CliRunner().invoke(sandbox_main, [str(request)])
    assert result.exit_code != 0
    assert "Static policy violations" in result.output


def test_newer_candidate_profile_matches_regular_isolated_factory_behavior():
    path = Path(
        "src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/"
        "joint_mooncake_synthetic_wildchat_lmcache_20260907.py"
    )
    source = path.read_text()
    config = small_config()

    def factory(capacity, block_size, seed):
        return _fresh_policy(source, ("build_candidate",), capacity, block_size, seed)

    complexity = scoring_fn_complexity(source, form_aware=True)
    regular = PrefixKVCacheEvaluator(config, splits=("validation",))(
        factory, scoring_fn_complexity=complexity
    )
    measured = profile_policy(factory, config, complexity=complexity, splits=("validation",))
    assert equivalent_behavior(regular, measured)


def test_archive_retains_smaller_sources_and_fails_closed_on_mixed_identities(tmp_path):
    config = small_config()
    result = PrefixKVCacheEvaluator(config, splits=("validation",))()
    identity = {
        key: getattr(result, key)
        for key in ("verifier_version", "evaluation_context_sha256", "panel_sha256")
    }

    def elite(source, raw):
        return {
            **identity,
            "code": source,
            "scores": {
                "success": 1.0,
                "invalid_fraction": 0.0,
                "selection_raw_score_before_complexity": raw,
            },
        }

    small = "class Small:\n    pass\n"
    large = "class Large:\n    def method(self):\n        return 1.0\n"
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"elites": [elite(small, 10.0), elite(large, 11.0)]}))
    _persist_behavior_size_frontier(
        tmp_path, metadata={"levi_snapshot_path": str(snapshot)}, config=config
    )
    directory = tmp_path / "behavior_size_frontier"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest["candidates"]) == 2
    assert len(list(directory.glob("*.py"))) == 2
    changed = elite(large, 11.0)
    changed["panel_sha256"] = "a" * 64
    snapshot.write_text(json.dumps({"elites": [elite(small, 10.0), changed]}))
    _persist_behavior_size_frontier(
        tmp_path, metadata={"levi_snapshot_path": str(snapshot)}, config=config
    )
    assert "mixed panels" in (tmp_path / "behavior_size_frontier_error.json").read_text()
