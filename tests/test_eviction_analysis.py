"""Tests for eviction-decision analysis helpers."""

from __future__ import annotations

from prefix_cache_evolve.tools.analyze_eviction import (
    VARIANT_SOURCES,
    CounterfactualTotals,
)


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
