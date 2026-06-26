"""Tests for evaluator result models and their aggregation schema."""

from prefix_cache_evolve.evaluators.results import (
    MAX_AGGREGATION_FIELDS,
    MEAN_AGGREGATION_FIELDS,
    SERIALIZED_TRIAL_FIELDS,
    TrialMetrics,
)
from prefix_cache_evolve.evaluators.scoring import aggregate_trials


def test_prefix_kv_cache_facade_preserves_result_exports() -> None:
    from prefix_cache_evolve.evaluators import prefix_kv_cache

    assert prefix_kv_cache.TrialMetrics is TrialMetrics


def test_trial_metric_schema_drives_serialization_and_aggregation() -> None:
    first = TrialMetrics(
        split="validation",
        workload="unit",
        seed=1,
        token_hit_rate=0.25,
        memory_occupancy_peak=3,
        structural_metrics={"mean_depth": 2.0},
    )
    second = TrialMetrics(
        split="validation",
        workload="unit",
        seed=2,
        token_hit_rate=0.75,
        memory_occupancy_peak=5,
        structural_metrics={"mean_depth": 4.0},
    )

    serialized = first.as_dict()
    aggregate = aggregate_trials([first, second])

    assert set(SERIALIZED_TRIAL_FIELDS) <= set(serialized)
    assert set(MEAN_AGGREGATION_FIELDS) <= set(serialized)
    assert set(MAX_AGGREGATION_FIELDS) <= set(serialized)
    assert serialized["mean_depth"] == 2.0
    assert aggregate["token_hit_rate"] == 0.5
    assert aggregate["memory_occupancy_peak"] == 5
    assert aggregate["mean_depth"] == 3.0
