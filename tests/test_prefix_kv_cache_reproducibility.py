"""Tests for reproducible synthetic workload manifests."""

from __future__ import annotations

import json
from pathlib import Path

from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluatorConfig,
    PrefixKVCacheEvaluator,
    baseline_lru_blocks,
)
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import (
    build_workload_manifest,
)


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
