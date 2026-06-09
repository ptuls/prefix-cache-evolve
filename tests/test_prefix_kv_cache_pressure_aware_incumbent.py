"""Tests for the promoted pressure-aware prefix KV-cache incumbent."""

from types import SimpleNamespace

import pytest

from prefix_cache_evolve.problems.prefix_kv_cache.pressure_aware_incumbent import (
    CompactReusePolicy,
)


def _block() -> SimpleNamespace:
    return SimpleNamespace(
        prefix_hash=7,
        descendant_count=0,
        depth=2,
        estimated_recompute_cost=0.0,
        last_accessed_at=0,
        last_access_gap=None,
        hit_count=0,
    )


def _request(
    *,
    priority: int = 0,
    recent_admission_pressure: float = 0.0,
    recent_miss_rate: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        priority=priority,
        recent_admission_pressure=recent_admission_pressure,
        recent_miss_rate=recent_miss_rate,
    )


def test_admission_pressure_accumulates_with_decay_and_is_bounded() -> None:
    policy = CompactReusePolicy(1, 4)
    request = _request(recent_admission_pressure=1.0, recent_miss_rate=0.5)

    policy.on_request_start(request, now=0)
    first_pressure = policy._admission_pressure
    policy.on_request_start(request, now=1)

    expected = first_pressure * 2.0 ** (-1.0 / 4.0) + first_pressure
    assert policy._admission_pressure == pytest.approx(min(1.5, expected))

    for now in range(2, 20):
        policy.on_request_start(request, now=now)

    assert policy._admission_pressure == 1.5


def test_pressure_penalizes_low_evidence_deep_admission_more() -> None:
    policy = CompactReusePolicy(1, 4)
    request = _request(recent_admission_pressure=1.0, recent_miss_rate=1.0)
    policy.on_request_start(request, now=0)

    shallow = _block()
    shallow.depth = 2
    deep = _block()
    deep.prefix_hash = 8
    deep.depth = 7

    assert policy.score_admission(deep, now=0) < policy.score_admission(shallow, now=0)


def test_cache_miss_does_not_treat_request_priority_as_reuse_evidence() -> None:
    policy = CompactReusePolicy(1, 4)
    request = _request(priority=4)
    block = _block()

    policy.on_request_start(request, now=0)
    policy.on_cache_miss(block, request, now=0)

    assert policy._state.values(block.prefix_hash, now=0)[1] == 0.0


def test_observed_reuse_state_is_bounded() -> None:
    policy = CompactReusePolicy(1, 4)
    request = _request()

    for key in range(policy._state.max_keys + 20):
        block = _block()
        block.prefix_hash = key
        policy.on_cache_miss(block, request, now=key)

    assert policy._state.state_size == policy._state.max_keys


def test_repeated_deep_misses_relax_depth_penalty_without_relaxing_first_miss() -> None:
    policy = CompactReusePolicy(1, 4)
    request = _request()
    block = _block()
    block.depth = 10

    policy.on_cache_miss(block, request, now=0)
    first_score = policy.score_admission(block, now=0)
    for _ in range(7):
        policy.on_cache_miss(block, request, now=0)
    repeated_score = policy.score_admission(block, now=0)

    assert first_score < 0.0
    assert repeated_score > 0.0


def test_persistent_pressure_strengthens_throttle_above_threshold() -> None:
    policy = CompactReusePolicy(1, 4)
    block = _block()

    policy._admission_pressure = 0.6
    low_pressure_score = policy.score_admission(block, now=0)
    policy._admission_pressure = 0.8
    threshold_score = policy.score_admission(block, now=0)
    policy._admission_pressure = 1.0
    persistent_pressure_score = policy.score_admission(block, now=0)

    assert threshold_score - persistent_pressure_score > low_pressure_score - threshold_score


def test_moderate_pressure_disproportionately_throttles_low_priority_admissions() -> None:
    policy = CompactReusePolicy(1, 4)
    low_priority = _block()
    high_priority = _block()
    high_priority.prefix_hash = 8
    policy.on_cache_hit(high_priority, _request(priority=2), now=0)

    policy._admission_pressure = 0.25
    low_at_threshold = policy.score_admission(low_priority, now=0)
    high_at_threshold = policy.score_admission(high_priority, now=0)
    policy._admission_pressure = 0.75
    low_under_pressure = policy.score_admission(low_priority, now=0)
    high_under_pressure = policy.score_admission(high_priority, now=0)

    assert low_at_threshold - low_under_pressure > high_at_threshold - high_under_pressure
