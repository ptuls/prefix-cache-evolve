"""Tests for the admission-vs-eviction regret audit."""

from pathlib import Path

from prefix_cache_evolve.problems.prefix_kv_cache.configuration import DEFAULT_CONFIG_PATH
from prefix_cache_evolve.tools.analyze_regret import (
    AdmissionPolicySpec,
    EvictionPolicySpec,
    _summarize_groups,
    _write_admission_eviction_matrix_markdown,
    _write_admission_policy_markdown,
    _write_markdown,
    run_admission_eviction_matrix,
    run_admission_policy_sweep,
    run_analysis,
)


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
    payload = run_analysis(
        DEFAULT_CONFIG_PATH,
        request_count=4,
        seeds=(3,),
        splits=("validation",),
        workloads=("shared_system_prompt",),
    )
    markdown_path = tmp_path / "regret.md"
    _write_markdown(markdown_path, payload)

    assert payload["schema"] == "prefix-kv-cache-admission-eviction-regret-audit-v1"
    assert len(payload["groups"]) == 2
    assert {group["capacity_blocks"] for group in payload["groups"]} == {24, 48}
    assert "universal claim passes only" in markdown_path.read_text(encoding="utf-8")


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
