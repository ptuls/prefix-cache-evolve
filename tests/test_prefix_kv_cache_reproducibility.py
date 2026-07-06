"""Tests for reproducible synthetic workload manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prefix_cache_evolve.evaluators.complexity import scoring_fn_complexity
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluatorConfig,
    PrefixKVCacheEvaluator,
    baseline_lru_blocks,
)
from prefix_cache_evolve.evaluators.verifier import VERIFIER_VERSION
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents import (
    build_discovery_incumbent as build_candidate,
)
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents.registry import (
    current_incumbent,
    incumbent_record,
    incumbent_records,
    validate_incumbent_registry,
)
from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import (
    build_workload_manifest,
    file_sha256,
    stable_workload_manifest_payload,
)
from tests.support import score_identity

_DISCOVERY_INCUMBENT = current_incumbent("discovery")
_DISCOVERY_INCUMBENT_PATH = _DISCOVERY_INCUMBENT.source_path
_PRODUCTION_INCUMBENT = current_incumbent("production")
_PRODUCTION_INCUMBENT_PATH = _PRODUCTION_INCUMBENT.source_path
_INCUMBENT_IDS = tuple(record.incumbent_id for record in incumbent_records())
_ONE_SEED_DISCOVERY_TOKEN_HIT_RATES = {
    "train/shared_system_prompt": 0.834446919079436,
    "train/rag_template_reuse": 0.820967146548542,
    "train/long_context_mixed": 0.844537815126050,
    "train/session_continuation_growth": 0.640120967741935,
    "train/agentic_tool_workflows": 0.479706785964534,
    "validation/phase_shift_prompts": 0.794450529390288,
    "validation/multi_tenant_skew": 0.802102891475779,
    "validation/hotset_cold_scan": 0.638077634011091,
    "validation/concurrent_long_generation": 0.890390390390390,
    "validation/stochastic_serving_mix": 0.420017873100983,
    "validation/rolling_template_versions": 0.846597462514418,
    "validation/heavy_tailed_prefix_lengths": 0.634691195795007,
    "validation/priority_burst_recovery": 0.505823186871361,
    "validation/priority_one_off_noise": 0.596301188903567,
    "validation/tenant_phase_shift_cycles": 0.518910030537937,
    "probe/agent_trace_branching": 0.375389888603256,
    "probe/cyclic_working_set_pressure": 0.881192396313364,
}


def _single_workload_config(**updates: object) -> EvaluatorConfig:
    """Return a compact deterministic config for identity tests."""
    values = {
        "capacity_blocks": 4,
        "request_count": 8,
        "seeds": (11,),
        "train_families": ("shared_system_prompt",),
        "validation_families": (),
        "probe_families": (),
        "hidden_families": (),
    }
    values.update(updates)
    return EvaluatorConfig(**values)


def _manifest_record(
    *,
    verifier_version: str = VERIFIER_VERSION,
    evaluation_verifier_version: str | None = None,
) -> dict[str, object]:
    """Return a minimal internally versioned workload manifest."""
    return {
        "schema": "v1",
        **score_identity(verifier_version=verifier_version),
        "evaluation": {
            **score_identity(verifier_version=evaluation_verifier_version or verifier_version),
            "stream_count": 1,
        },
        "streams": [{"request_stream_sha256": "stream"}],
    }


def test_workload_manifest_is_deterministic_and_records_actual_seed() -> None:
    config = _single_workload_config(
        capacity_sweep_blocks=(4, 8),
    )

    first = build_workload_manifest(config, splits=("train",))
    second = build_workload_manifest(config, splits=("train",))

    assert first == second
    assert first["verifier_version"] == VERIFIER_VERSION
    assert first["evaluation"]["verifier_version"] == VERIFIER_VERSION
    assert first["evaluation"]["stream_count"] == 1
    assert first["streams"][0]["actual_seed"] == 1011
    assert len(first["streams"][0]["request_stream_sha256"]) == 64
    assert len(first["panel_sha256"]) == 64


def test_deterministic_policy_evaluation_repeats_exactly() -> None:
    config = EvaluatorConfig(
        capacity_blocks=4,
        capacity_sweep_blocks=(4, 8),
        request_count=12,
        seeds=(11, 23),
        train_families=(),
        validation_families=("stochastic_serving_mix",),
        probe_families=(),
        hidden_families=(),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))

    first = evaluator(baseline_lru_blocks)
    second = evaluator(baseline_lru_blocks)

    assert first == second


def test_evaluator_identity_matches_workload_manifest() -> None:
    config = _single_workload_config(
        capacity_sweep_blocks=(4, 8),
    )

    result = PrefixKVCacheEvaluator(config, splits=("train",))(baseline_lru_blocks)
    manifest = build_workload_manifest(config, splits=("train",))

    assert result.panel_sha256 == manifest["panel_sha256"]
    assert result.evaluation_context_sha256 == manifest["evaluation_context_sha256"]


def test_score_config_change_preserves_panel_but_changes_context() -> None:
    config = _single_workload_config()
    changed = config.with_updates(churn_weight=config.churn_weight + 0.1)

    first = build_workload_manifest(config, splits=("train",))
    second = build_workload_manifest(changed, splits=("train",))

    assert first["panel_sha256"] == second["panel_sha256"]
    assert first["evaluation_context_sha256"] != second["evaluation_context_sha256"]


def test_rescore_preserves_panel_but_changes_context() -> None:
    config = _single_workload_config()
    result = PrefixKVCacheEvaluator(config, splits=("train",))(baseline_lru_blocks)
    changed = config.with_updates(churn_weight=config.churn_weight + 0.1)

    rescored = PrefixKVCacheEvaluator(changed, splits=("train",)).rescore_trials(result.trials)

    assert rescored.panel_sha256 == result.panel_sha256
    assert rescored.evaluation_context_sha256 != result.evaluation_context_sha256


def test_discovery_panel_matches_committed_workload_fingerprint() -> None:
    config = load_evaluator_config(Path("configs/prefix_kv_cache_discovery.yaml"))
    committed = json.loads(
        Path("docs/results/discovery_workload_manifest.json").read_text(encoding="utf-8")
    )

    generated = build_workload_manifest(config)

    assert generated["panel_sha256"] == committed["panel_sha256"]
    assert generated["evaluation_context_sha256"] == committed["evaluation_context_sha256"]
    assert generated["panel_sha256"] == (
        "4607782d231560f5d51c5f0347a789b7b82a7e8ff4d78ec5f1adb576c68d2c8f"
    )
    assert generated["evaluation_context_sha256"] == (
        "b4d8e05f8eecb686ee399e7145458efcf8c6b81955a8ed4adc4ed850df2fb99d"
    )


def test_stable_manifest_payload_ignores_environment_metadata() -> None:
    manifest = _manifest_record()
    manifest["generator"] = {"python_version": "3.11.9", "source_file": "old.py"}
    regenerated = {
        **manifest,
        "generator": {"python_version": "3.12.3", "source_file": "workloads.py"},
    }

    assert stable_workload_manifest_payload(manifest) == stable_workload_manifest_payload(
        regenerated
    )


def test_stable_manifest_payload_rejects_verifier_version_drift() -> None:
    current = _manifest_record()
    prior = _manifest_record(verifier_version="0.9.0")

    assert stable_workload_manifest_payload(current) != stable_workload_manifest_payload(prior)


def test_stable_manifest_payload_refuses_mixed_verifier_versions() -> None:
    manifest = _manifest_record(evaluation_verifier_version="0.9.0")

    with pytest.raises(ValueError, match="refuses mixed verifier versions"):
        stable_workload_manifest_payload(manifest)


def test_pressure_aware_incumbent_matches_one_seed_discovery_scores() -> None:
    config = load_evaluator_config(Path("configs/prefix_kv_cache_discovery.yaml")).with_updates(
        seeds=(11,)
    )
    source = _DISCOVERY_INCUMBENT_PATH.read_text(encoding="utf-8")
    complexity = scoring_fn_complexity(source, form_aware=config.form_aware_complexity)

    result = PrefixKVCacheEvaluator(
        config,
        splits=("train", "validation", "probe"),
    )(build_candidate, scoring_fn_complexity=complexity)

    assert complexity == 648
    assert result.combined_score == pytest.approx(75.46120113909609, rel=0.0, abs=1e-12)
    assert set(result.workload_metrics) == set(_ONE_SEED_DISCOVERY_TOKEN_HIT_RATES)
    for workload, expected in _ONE_SEED_DISCOVERY_TOKEN_HIT_RATES.items():
        assert result.workload_metrics[workload]["token_hit_rate"] == pytest.approx(
            expected,
            rel=0.0,
            abs=1e-12,
        )


def test_incumbent_registry_preserves_exact_sources_and_metadata() -> None:
    records = validate_incumbent_registry()
    source = _PRODUCTION_INCUMBENT_PATH.read_text(encoding="utf-8")

    assert {record.role for record in records} == {"discovery", "historical", "production"}
    assert file_sha256(_PRODUCTION_INCUMBENT_PATH) == _PRODUCTION_INCUMBENT.source_sha256
    assert scoring_fn_complexity(source, form_aware=True) == 372
    assert _PRODUCTION_INCUMBENT.provenance["source_artifact_sha256"] == (
        _PRODUCTION_INCUMBENT.source_sha256
    )


@pytest.mark.parametrize("incumbent_id", _INCUMBENT_IDS)
def test_registered_incumbent_matches_pinned_benchmark_identity(incumbent_id: str) -> None:
    incumbent = incumbent_record(incumbent_id)
    benchmark = incumbent.benchmark
    config = load_evaluator_config(Path(str(benchmark["config_path"])))
    source = incumbent.source_path.read_text(encoding="utf-8")
    complexity = scoring_fn_complexity(source, form_aware=config.form_aware_complexity)

    result = PrefixKVCacheEvaluator(config, splits=tuple(benchmark["splits"]))(
        incumbent.load_factory(),
        scoring_fn_complexity=complexity,
    )

    assert result.verifier_version == benchmark["verifier_version"]
    assert result.panel_sha256 == benchmark["panel_sha256"]
    assert result.evaluation_context_sha256 == benchmark["evaluation_context_sha256"]
    assert result.combined_score == benchmark["selection_combined_score"]
    assert (
        result.split_metrics["validation"]["token_hit_rate"]
        == (benchmark["validation_token_hit_rate"])
    )
    assert result.split_metrics["probe"]["token_hit_rate"] == benchmark["probe_token_hit_rate"]
