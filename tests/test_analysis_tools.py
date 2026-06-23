"""Tests for analysis tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from prefix_cache_evolve.problems.prefix_kv_cache.configuration import (
    DEFAULT_CONFIG_PATH,
    load_evaluator_config,
)
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents.registry import current_incumbent
from prefix_cache_evolve.tools import analyze_rediscovery
from prefix_cache_evolve.tools.analyze_eviction import (
    VARIANT_SOURCES,
    CounterfactualTotals,
)
from prefix_cache_evolve.tools.analyze_reasoning_kv import (
    _write_markdown as _write_reasoning_kv_markdown,
)
from prefix_cache_evolve.tools.analyze_reasoning_kv import (
    run_analysis as run_reasoning_kv_analysis,
)
from prefix_cache_evolve.tools.analyze_rediscovery import (
    _experiment_summary,
    _gap_recovery,
)
from prefix_cache_evolve.tools.analyze_regret import (
    AdmissionPolicySpec,
    EvictionPolicySpec,
    _summarize_groups,
    _write_admission_eviction_matrix_markdown,
    _write_admission_policy_markdown,
    _write_causal_component_markdown,
    _write_shadow_price_markdown,
    run_admission_eviction_matrix,
    run_admission_policy_sweep,
    run_causal_component_factorial,
    run_shadow_price_analysis,
)
from prefix_cache_evolve.tools.analyze_regret import (
    _write_markdown as _write_regret_markdown,
)
from prefix_cache_evolve.tools.analyze_regret import (
    run_analysis as run_regret_analysis,
)
from prefix_cache_evolve.workflow.config import ConfigLoader

_POLICY_ROOT = Path("src/prefix_cache_evolve/problems/prefix_kv_cache")
_WEAK_SEED_PATH = _POLICY_ROOT / "seeds/weak_initial.py"


def _group(admission: float, eviction: float) -> dict[str, object]:
    total = admission + eviction
    return {
        "invalid": False,
        "block_hit_rate": 0.5,
        "token_hit_rate": 0.5,
        "prefill_tokens_saved": 10.0,
        "p95_latency_proxy": 2.0,
        "admission_score_count": 2,
        "eviction_count": 1,
        "total_regret_tokens": total,
        "admission_side_regret_tokens": admission,
        "eviction_side_regret_tokens": eviction,
        "admission_minus_eviction_regret_tokens": admission - eviction,
        "admission_dominates": admission > eviction,
        "decision_normalized_admission_dominates": admission / 2 > eviction,
    }


def _agentic_metrics(*, wasted_admission_token_rate: float = 0.1) -> dict[str, float]:
    return {
        "token_hit_rate": 0.4,
        "request_token_hit_rate_p10": 0.2,
        "worst_quarter_token_hit_rate": 0.3,
        "wasted_admission_token_rate": wasted_admission_token_rate,
        "policy_underfill_rate": 0.05,
        "short_reuse_after_eviction_missed_token_rate": 0.02,
        "cache_churn_per_1k": 100.0,
    }


class _AdmitAllPolicy:
    def on_request_start(self, request, now):
        return None

    def on_cache_hit(self, block, request, now):
        return None

    def on_cache_miss(self, block, request, now):
        return None

    def score_admission(self, block, now):
        return 1.0

    def score_eviction(self, block, now):
        return 0.0


class _RejectAllPolicy(_AdmitAllPolicy):
    def score_admission(self, block, now):
        return -1.0


def test_counterfactual_totals_distinguish_corrected_and_introduced_regret() -> None:
    totals = CounterfactualTotals()

    totals.record(
        legal_victims=3,
        incumbent_distance=2.0,
        alternative_distance=10.0,
        furthest_distance=10.0,
        changed=True,
    )
    totals.record(
        legal_victims=2,
        incumbent_distance=10.0,
        alternative_distance=2.0,
        furthest_distance=10.0,
        changed=True,
    )

    summary = totals.summary()
    assert summary["multiple_legal_victim_rate"] == 1.0
    assert summary["changed_decision_rate"] == 1.0
    assert summary["better_next_reuse_rate_on_changed"] == 0.5
    assert summary["worse_next_reuse_rate_on_changed"] == 0.5
    assert summary["corrected_avoidable_decisions"] == 1
    assert summary["introduced_avoidable_decisions"] == 1
    assert summary["corrected_short_reuse_decisions"] == 1
    assert summary["introduced_short_reuse_decisions"] == 1


def test_eviction_analysis_variants_are_function_only_sources() -> None:
    for source in VARIANT_SOURCES.values():
        namespace = {}
        exec(source, namespace)

        assert callable(namespace["score_eviction"])


def test_reasoning_kv_analysis_compares_capacity_modes(tmp_path: Path) -> None:
    payload = run_reasoning_kv_analysis(DEFAULT_CONFIG_PATH, request_count=4, seeds=(3,))
    markdown_path = tmp_path / "reasoning.md"
    _write_reasoning_kv_markdown(markdown_path, payload)

    assert payload["schema"] == "prefix-kv-cache-reasoning-kv-analysis-v1"
    assert set(payload["modes"]) == {"prefix_only", "shared"}
    prefix_only = payload["modes"]["prefix_only"]["incumbent"]
    shared = payload["modes"]["shared"]["incumbent"]
    assert prefix_only["decode_kv_blocks_requested"] == 0
    assert shared["decode_kv_blocks_requested"] > 0
    assert shared["decode_kv_allocation_failure_rate"] > 0
    assert "Decode allocation failure is reported but is not yet a score term" in (
        markdown_path.read_text(encoding="utf-8")
    )


def test_regret_summary_falsifies_universal_claim_on_one_counterexample() -> None:
    summary = _summarize_groups((_group(8.0, 2.0), _group(1.0, 3.0), _group(0.0, 0.0)))

    assert summary["verdict"] == "falsified"
    assert summary["regretful_group_count"] == 2
    assert summary["admission_dominant_group_count"] == 1
    assert summary["admission_dominance_rate"] == 0.5
    assert summary["aggregate_admission_side_regret_tokens"] == 9.0
    assert summary["aggregate_eviction_side_regret_tokens"] == 5.0
    assert summary["aggregate_admission_regret_tokens_per_decision"] == 1.5
    assert summary["aggregate_eviction_regret_tokens_per_decision"] == 5 / 3
    assert summary["decision_normalized_admission_dominant_group_count"] == 1
    assert summary["decision_normalized_admission_dominance_rate"] == 0.5
    assert summary["mean_token_hit_rate"] == 0.5
    assert summary["total_prefill_tokens_saved"] == 30.0


def test_regret_analysis_emits_grouped_json_and_markdown(tmp_path: Path) -> None:
    payload = run_regret_analysis(
        DEFAULT_CONFIG_PATH,
        request_count=4,
        seeds=(3,),
        splits=("validation",),
        workloads=("shared_system_prompt",),
    )
    markdown_path = tmp_path / "regret.md"
    _write_regret_markdown(markdown_path, payload)

    assert payload["schema"] == "prefix-kv-cache-admission-eviction-regret-audit-v1"
    assert len(payload["groups"]) == 2
    assert {group["capacity_blocks"] for group in payload["groups"]} == {24, 48}
    assert "universal claim passes only" in markdown_path.read_text(encoding="utf-8")


def test_shadow_price_analysis_emits_calibrated_decision_trajectories(
    tmp_path: Path,
) -> None:
    payload = run_shadow_price_analysis(
        DEFAULT_CONFIG_PATH,
        request_count=24,
        seeds=(3,),
        splits=("validation",),
        workloads=("priority_burst_recovery",),
    )
    markdown_path = tmp_path / "shadow-price.md"
    _write_shadow_price_markdown(markdown_path, payload)

    assert payload["schema"] == "prefix-kv-cache-shadow-price-tracking-v1"
    assert len(payload["groups"]) == 2
    assert all(group["trajectory"] for group in payload["groups"])
    decision = payload["groups"][0]["trajectory"][0]
    assert "oracle_shadow_price" in decision
    assert "policy_implied_shadow_price" in decision
    assert "tracking_error" in decision
    assert "fast-change admission-regret share" in markdown_path.read_text(encoding="utf-8").lower()


def test_causal_component_factorial_emits_four_paired_cells(
    tmp_path: Path,
) -> None:
    payload = run_causal_component_factorial(
        DEFAULT_CONFIG_PATH,
        request_count=8,
        seeds=(3,),
        splits=("validation",),
        workloads=("priority_burst_recovery",),
        capacity_blocks=(8,),
    )
    markdown_path = tmp_path / "causal-components.md"
    _write_causal_component_markdown(markdown_path, payload)

    assert payload["schema"] == "prefix-kv-cache-causal-component-factorial-v1"
    assert payload["summary"]["group_count"] == 1
    assert "future-aware avoidable-eviction" in payload["group_score_definition"]
    assert payload["summary"]["residual_eviction_value_group_count"] in {0, 1}
    assert (
        payload["summary"]["oracle_admission_dissolved_eviction_value_group_count"]
        + payload["summary"]["residual_eviction_value_group_count"]
        == 1
    )
    group = payload["groups"][0]
    assert set(group["cells"]) == {"II", "OI", "IO", "OO"}
    assert set(group["effects"]["group_score"]) == {
        "tau_A",
        "tau_E",
        "tau_AE",
        "eviction_effect_after_oracle_admission",
        "admission_effect_after_oracle_eviction",
    }
    report = markdown_path.read_text(encoding="utf-8")
    assert "Causal Admission-Eviction Component Factorial" in report
    assert "Eviction effect after oracle admission" in report
    assert "Groups with residual eviction value" in report


def test_admission_policy_sweep_holds_eviction_fixed_to_lru(tmp_path: Path) -> None:
    policy_specs = (
        AdmissionPolicySpec("reject_all", lambda *_: _RejectAllPolicy()),
        AdmissionPolicySpec("admit_all", lambda *_: _AdmitAllPolicy()),
    )
    payload = run_admission_policy_sweep(
        DEFAULT_CONFIG_PATH,
        request_count=4,
        seeds=(3,),
        splits=("validation",),
        workloads=("shared_system_prompt",),
        policy_specs=policy_specs,
    )
    markdown_path = tmp_path / "admission-policy-sweep.md"
    _write_admission_policy_markdown(markdown_path, payload)

    assert payload["schema"] == "prefix-kv-cache-admission-policy-regret-sweep-v1"
    assert payload["eviction_policy"] == "lru"
    assert set(payload["policies"]) == {"reject_all", "admit_all"}
    assert all(policy["policy"].endswith("+fixed_lru") for policy in payload["policies"].values())
    assert "Eviction is fixed to legal-leaf LRU" in markdown_path.read_text(encoding="utf-8")


def test_admission_eviction_matrix_crosses_all_supplied_policies(tmp_path: Path) -> None:
    admission_specs = (
        AdmissionPolicySpec("reject_all", lambda *_: _RejectAllPolicy()),
        AdmissionPolicySpec("admit_all", lambda *_: _AdmitAllPolicy()),
    )
    eviction_specs = (
        EvictionPolicySpec("lru", lambda *_: _AdmitAllPolicy()),
        EvictionPolicySpec("oracle_next_use", lambda *_: _AdmitAllPolicy(), True),
    )
    payload = run_admission_eviction_matrix(
        DEFAULT_CONFIG_PATH,
        request_count=4,
        seeds=(3,),
        splits=("validation",),
        workloads=("shared_system_prompt",),
        admission_specs=admission_specs,
        eviction_specs=eviction_specs,
    )
    markdown_path = tmp_path / "matrix.md"
    _write_admission_eviction_matrix_markdown(markdown_path, payload)

    assert payload["schema"] == "prefix-kv-cache-admission-eviction-policy-matrix-v1"
    assert payload["combination_count"] == 4
    assert set(payload["combinations"]) == {
        "reject_all+lru",
        "reject_all+oracle_next_use",
        "admit_all+lru",
        "admit_all+oracle_next_use",
    }
    assert "Best Eviction Per Admission" in markdown_path.read_text(encoding="utf-8")


def test_rediscovery_config_matches_evaluator_without_incumbent_prompt_leakage() -> None:
    operative = load_evaluator_config(Path("configs/prefix_kv_cache.yaml"))
    rediscovery = load_evaluator_config(Path("configs/prefix_kv_cache_rediscovery.yaml"))
    workflow = ConfigLoader().load(Path("configs/prefix_kv_cache_rediscovery.yaml"))
    prompt = workflow.problem_description

    assert (
        rediscovery.with_updates(
            search_score_mode="combined",
            search_guidance_families=(),
        )
        == operative
    )
    assert rediscovery.search_score_mode == "robust_min"
    assert rediscovery.search_guidance_families == ("agentic_tool_workflows",)
    assert "65.649" not in prompt
    assert "TinyLFU" not in prompt
    assert "pressure-aware" not in prompt
    assert "preserve the supplied parent" not in prompt
    assert "MultiTimescaleDecay maintains a bounded per-key vector" in prompt
    assert "available search-space tools, not required architecture" in prompt
    assert "Preserve or simplify canonical" not in prompt
    assert "deliberately weak" in prompt
    assert "Never use request_type or prompt_tokens" in prompt
    assert "never access a block or request field that is not enumerated" in prompt
    assert workflow.pipeline["n_inspirations"] == 2
    assert workflow.cvt["n_centroids"] == 16
    assert workflow.init["diversity_model"] == "openai/gpt-5.4-mini"
    assert "validation_shadow_price_tracking_rmse" in workflow.behavior["score_keys"]


@pytest.mark.parametrize(
    ("candidate", "seed", "incumbent", "expected"),
    (
        (6.0, 2.0, 10.0, 0.5),
        (12.0, 2.0, 10.0, 1.25),
        (8.0, 10.0, 9.0, 0.0),
        (10.0, 10.0, 9.0, 1.0),
    ),
)
def test_gap_recovery_is_normalized_to_seed_and_incumbent(
    candidate: float,
    seed: float,
    incumbent: float,
    expected: float,
) -> None:
    assert _gap_recovery(candidate, seed, incumbent) == pytest.approx(expected)


def test_experiment_requires_repeated_weak_seed_behavioral_rediscovery() -> None:
    runs = [
        {
            "seed_tier": "weak_initial",
            "search_seed": seed,
            "checks": {"behaviorally_close": True},
        }
        for seed in (11, 23)
    ]

    summary = _experiment_summary(runs)

    assert summary["verdict"] == "supported"
    assert summary["weak_initial_behaviorally_close_count"] == 2
    assert summary["distinct_weak_initial_close_search_seed_count"] == 2


def test_run_analysis_prefers_generated_mutation_and_reports_preliminary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/prefix_kv_cache_rediscovery.yaml")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    initial_source = _WEAK_SEED_PATH.read_text(encoding="utf-8")
    incumbent_source = current_incumbent("production").source_path.read_text(encoding="utf-8")
    (run_dir / "seed_program.py").write_text(initial_source, encoding="utf-8")
    (run_dir / "best_program.py").write_text(initial_source, encoding="utf-8")
    (run_dir / "best_generated_mutation.py").write_text(incumbent_source, encoding="utf-8")
    (run_dir / "run_summary.json").write_text(
        '{"iterations": 100, "total_evaluations": 98, "total_cost": 2.5}\n',
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text('{"search_seed": 17}\n', encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text(
        "# Formatting and comments do not change YAML identity.\n"
        + config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    def fake_decomposition(_config, candidate_path):
        source = candidate_path.read_text(encoding="utf-8")
        score = 10.0 if source == incumbent_source else 0.0
        panel = {
            "combined_score": score,
            "success": True,
            "score_breakdown": {"complexity_cost": 1.0},
        }
        return {
            "effective_complexity": 100,
            "selection": {
                **panel,
                "workload_metrics": {
                    "train/agentic_tool_workflows": _agentic_metrics(),
                },
            },
            "probe": {
                **panel,
                "workload_metrics": {
                    "probe/agent_trace_branching": _agentic_metrics(),
                },
            },
            "hidden": {**panel, "workload_metrics": {}},
        }

    monkeypatch.setattr(
        analyze_rediscovery,
        "_candidate_panel_decomposition",
        fake_decomposition,
    )

    payload = analyze_rediscovery.run_analysis(config_path, run_dirs=(run_dir,))

    assert payload["schema"] == "prefix-kv-cache-behavioral-rediscovery-v2"
    assert payload["summary"]["verdict"] == "preliminary"
    assert payload["runs"][0]["candidate_kind"] == "best_generated_mutation"
    assert payload["runs"][0]["seed_tier"] == "weak_initial"
    assert payload["runs"][0]["search_seed"] == 17
    assert payload["runs"][0]["search_winner_source_differs_from_seed"] is False
    assert payload["runs"][0]["checks"]["behaviorally_close"]


def test_run_analysis_rejects_close_result_when_agentic_gate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/prefix_kv_cache_rediscovery.yaml")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    initial_source = _WEAK_SEED_PATH.read_text(encoding="utf-8")
    incumbent_source = current_incumbent("production").source_path.read_text(encoding="utf-8")
    (run_dir / "seed_program.py").write_text(initial_source, encoding="utf-8")
    (run_dir / "best_generated_mutation.py").write_text(incumbent_source, encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    def fake_decomposition(_config, candidate_path):
        source = candidate_path.read_text(encoding="utf-8")
        score = 10.0 if source == incumbent_source else 0.0
        panel = {
            "combined_score": score,
            "success": True,
            "score_breakdown": {"complexity_cost": 0.0},
        }
        return {
            "effective_complexity": 100,
            "selection": {
                **panel,
                "workload_metrics": {
                    "train/agentic_tool_workflows": _agentic_metrics(
                        wasted_admission_token_rate=0.5
                    ),
                },
            },
            "probe": {
                **panel,
                "workload_metrics": {
                    "probe/agent_trace_branching": _agentic_metrics(),
                },
            },
            "hidden": {**panel, "workload_metrics": {}},
        }

    monkeypatch.setattr(
        analyze_rediscovery,
        "_candidate_panel_decomposition",
        fake_decomposition,
    )

    run = analyze_rediscovery.run_analysis(config_path, run_dirs=(run_dir,))["runs"][0]

    assert run["checks"]["charged_gap_recovery_at_least_threshold_all_panels"]
    assert not run["checks"]["agentic_surrogate_probe_gate"]
    assert not run["checks"]["behaviorally_close"]
