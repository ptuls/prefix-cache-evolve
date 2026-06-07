"""Tests for the compact prefix KV-cache seed policy."""

from types import SimpleNamespace

from prefix_cache_evolve.problems.prefix_kv_cache.compact_seed import CompactReusePolicy
from prefix_cache_evolve.problems.prefix_kv_cache.seeds.structured_recurrence import (
    StructuredRecurrencePolicy,
)
from prefix_cache_evolve.problems.prefix_kv_cache.seeds.structured_seed import (
    StructuredSeedPolicy,
)
from prefix_cache_evolve.tools.ablate_structured import AblationStructuredPolicy
from prefix_cache_evolve.tools.tune_compact import (
    DEFAULT_PARAMETERS,
    TunableCompactPolicy,
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


def test_compact_frequency_state_decays_by_half_life() -> None:
    policy = CompactReusePolicy(
        1,
        4,
        frequency_half_life=8.0,
        priority_half_life=None,
    )
    block = _block()

    policy.on_request_start(_request(priority=0), now=0)
    policy.on_cache_miss(block, _request(priority=0), now=0)

    assert policy._values(block.prefix_hash, 0)[0] == 1.0
    assert policy._values(block.prefix_hash, 8)[0] == 0.5


def test_compact_priority_maximum_decays_and_can_be_refreshed() -> None:
    policy = CompactReusePolicy(
        1,
        4,
        frequency_half_life=None,
        priority_half_life=8.0,
    )
    block = _block()

    policy.on_request_start(_request(priority=4), now=0)
    policy.on_cache_miss(block, _request(priority=4), now=0)
    assert policy._values(block.prefix_hash, 8)[1] == 2.0

    policy.on_request_start(_request(priority=3), now=8)
    policy.on_cache_hit(block, _request(priority=3), now=8)
    assert policy._values(block.prefix_hash, 8)[1] == 3.0


def test_compact_decay_terms_can_be_disabled_independently() -> None:
    policy = CompactReusePolicy(
        1,
        4,
        frequency_half_life=None,
        priority_half_life=None,
    )
    block = _block()

    policy.on_request_start(_request(priority=4), now=0)
    policy.on_cache_miss(block, _request(priority=4), now=0)

    assert policy._values(block.prefix_hash, 100) == (1.0, 4.0)


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

        assert compact.score_admission(block, now) == tunable.score_admission(
            block, now
        )
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

        assert structured.score_admission(block, now) == ablation.score_admission(
            block, now
        )
        assert structured.score_eviction(block, now) == ablation.score_eviction(
            block, now
        )


def test_selected_structured_seed_uses_bounded_canonical_state() -> None:
    policy = StructuredSeedPolicy(2, 4)
    request = _request(priority=0)

    for key in range(100):
        block = _block()
        block.prefix_hash = key
        policy.on_request_start(request, key)
        policy.on_cache_miss(block, request, key)

    assert policy._state.state_size == 64
