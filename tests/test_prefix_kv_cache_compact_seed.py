"""Tests for the compact prefix KV-cache seed policy."""

from types import SimpleNamespace

from prefix_cache_evolve.problems.prefix_kv_cache.incumbents import incumbent_record
from prefix_cache_evolve.problems.prefix_kv_cache.seeds.structured_recurrence import (
    StructuredRecurrencePolicy,
)
from prefix_cache_evolve.tools.ablate_structured import AblationStructuredPolicy
from prefix_cache_evolve.tools.tune_compact import (
    DEFAULT_PARAMETERS,
    TunableCompactPolicy,
)

CompactReusePolicy = incumbent_record("historical_compact_20260607").load_symbol(
    "CompactReusePolicy"
)


def _block() -> SimpleNamespace:
    return SimpleNamespace(
        prefix_hash=7,
        descendant_count=0,
        depth=2,
        estimated_recompute_cost=0.0,
        last_accessed_at=0,
        hit_count=0,
        active_ref_count=0,
        subtree_active_ref_count=0,
        subtree_hit_rate=0.0,
        access_gap_mean=None,
        access_gap_var=None,
        token_count=4,
    )


def _request(priority: int) -> SimpleNamespace:
    return SimpleNamespace(
        priority=priority,
        recent_admission_pressure=0.0,
        recent_miss_rate=0.0,
    )


def test_tunable_default_matches_compact_seed() -> None:
    compact = CompactReusePolicy(1, 4)
    tunable = TunableCompactPolicy(DEFAULT_PARAMETERS)
    block = _block()

    for now, priority, callback in (
        (0, 4, "on_cache_miss"),
        (8, 0, "on_cache_hit"),
        (20, 2, "on_cache_miss"),
    ):
        request = _request(priority)
        compact.on_request_start(request, now)
        tunable.on_request_start(request, now)
        getattr(compact, callback)(block, request, now)
        getattr(tunable, callback)(block, request, now)

        assert compact.score_admission(block, now) == tunable.score_admission(block, now)
        assert compact.score_eviction(block, now) == tunable.score_eviction(block, now)


def test_structured_ablation_default_matches_structured_seed() -> None:
    structured = StructuredRecurrencePolicy(8, 4)
    ablation = AblationStructuredPolicy(8, 4)
    block = _block()

    for now, priority, callback in (
        (0, 4, "on_cache_miss"),
        (8, 0, "on_cache_hit"),
        (20, 2, "on_cache_miss"),
    ):
        request = _request(priority)
        structured.on_request_start(request, now)
        ablation.on_request_start(request, now)
        getattr(structured, callback)(block, request, now)
        getattr(ablation, callback)(block, request, now)

        assert structured.score_admission(block, now) == ablation.score_admission(block, now)
        assert structured.score_eviction(block, now) == ablation.score_eviction(block, now)
