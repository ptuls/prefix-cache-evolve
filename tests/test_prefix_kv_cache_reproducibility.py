"""Tests for reproducible synthetic workload manifests."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from prefix_cache_evolve.evaluators.complexity import scoring_fn_complexity
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluatorConfig,
    PrefixKVCacheEvaluator,
    baseline_lru_blocks,
)
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.pressure_aware_incumbent import (
    build_candidate,
)
from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import (
    build_workload_manifest,
    stable_workload_manifest_payload,
)

_DISCOVERY_INCUMBENT_PATH = Path(
    "src/prefix_cache_evolve/problems/prefix_kv_cache/pressure_aware_incumbent.py"
)
_PRODUCTION_INCUMBENT_PATH = Path(
    "src/prefix_cache_evolve/problems/prefix_kv_cache/production_incumbent.py"
)
_PROMOTED_PRODUCTION_ARTIFACT_PATH = Path(
    "artifacts/prefix_kv_cache_simplification_runs/20260609T051303Z/elite_65_649.py"
)
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


def test_workload_manifest_is_deterministic_and_records_actual_seed() -> None:
    config = EvaluatorConfig(
        capacity_blocks=4,
        capacity_sweep_blocks=(4, 8),
        request_count=8,
        seeds=(11,),
        train_families=("shared_system_prompt",),
        validation_families=(),
        probe_families=(),
        hidden_families=(),
    )

    first = build_workload_manifest(config, splits=("train",))
    second = build_workload_manifest(config, splits=("train",))

    assert first == second
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


def test_discovery_panel_matches_committed_workload_fingerprint() -> None:
    config = load_evaluator_config(Path("configs/prefix_kv_cache_discovery.yaml"))
    committed = json.loads(
        Path("docs/results/discovery_workload_manifest.json").read_text(encoding="utf-8")
    )

    generated = build_workload_manifest(config)

    assert generated["panel_sha256"] == committed["panel_sha256"]
    assert generated["panel_sha256"] == (
        "4607782d231560f5d51c5f0347a789b7b82a7e8ff4d78ec5f1adb576c68d2c8f"
    )


def test_stable_manifest_payload_ignores_environment_metadata() -> None:
    manifest = {
        "schema": "v1",
        "panel_sha256": "panel",
        "evaluation": {"stream_count": 1},
        "streams": [{"request_stream_sha256": "stream"}],
        "generator": {"python_version": "3.11.9", "source_file": "old.py"},
    }
    regenerated = {
        **manifest,
        "generator": {"python_version": "3.12.3", "source_file": "workloads.py"},
    }

    assert stable_workload_manifest_payload(manifest) == stable_workload_manifest_payload(
        regenerated
    )


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


def test_production_incumbent_matches_promoted_artifact() -> None:
    source = _PRODUCTION_INCUMBENT_PATH.read_text(encoding="utf-8")
    artifact_source = _PROMOTED_PRODUCTION_ARTIFACT_PATH.read_text(encoding="utf-8")

    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(artifact_source))
    assert scoring_fn_complexity(source, form_aware=True) == 572
