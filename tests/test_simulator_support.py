"""Tests for extracted simulator support responsibilities."""

from __future__ import annotations

import math

import pytest

from prefix_cache_evolve.evaluators import prefix_kv_cache, simulator_support
from prefix_cache_evolve.evaluators.contracts import RequestInfo
from prefix_cache_evolve.evaluators.simulator_support import (
    _AdmissionAccounting,
    _BlockState,
    _CapacityOutcome,
    _FutureReuseTracker,
    _request_arrival_steps,
)
from prefix_cache_evolve.evaluators.utilities import request_prefix_hashes
from prefix_cache_evolve.evaluators.workloads import WorkloadRequest


def _request(
    request_id: int,
    tokens: tuple[int, ...],
    *,
    arrival_step: int | None = None,
) -> WorkloadRequest:
    return WorkloadRequest(
        info=RequestInfo(
            request_id=request_id,
            tenant_id=0,
            session_id=0,
            prompt_length=len(tokens),
            priority=0,
            request_type="test",
            prompt_tokens=(),
        ),
        true_output_length=0,
        prompt_tokens=tokens,
        arrival_step=arrival_step,
    )


def _block(**overrides: object) -> _BlockState:
    values = {
        "prefix_hash": 1,
        "parent_hash": None,
        "depth": 1,
        "start_token": 0,
        "end_token": 4,
        "token_count": 4,
        "prefix_role": "system",
        "tenant_id": 0,
        "created_at": 0,
        "last_accessed_at": 0,
    }
    values.update(overrides)
    return _BlockState(**values)


@pytest.mark.parametrize(
    "name",
    (
        "_ActiveDecode",
        "_AdmissionAccounting",
        "_AdmissionAudit",
        "_AdmissionOutcome",
        "_BlockState",
        "_CapacityOutcome",
        "_FutureReuseTracker",
        "_correlation",
        "_request_arrival_steps",
        "_window_mean",
    ),
)
def test_prefix_kv_cache_preserves_simulator_support_exports(name: str) -> None:
    assert getattr(prefix_kv_cache, name) is getattr(simulator_support, name)


def test_request_arrival_steps_preserve_defaults_and_validate_order() -> None:
    assert _request_arrival_steps(
        (
            _request(0, (1,), arrival_step=2),
            _request(1, (2,), arrival_step=2),
            _request(2, (3,), arrival_step=7),
        )
    ) == (2, 2, 7)
    assert _request_arrival_steps((_request(0, (1,)), _request(1, (2,)))) == (0, 1)

    with pytest.raises(ValueError, match="arrival steps must be monotonic"):
        _request_arrival_steps(
            (
                _request(0, (1,), arrival_step=3),
                _request(1, (2,), arrival_step=2),
            )
        )


def test_future_reuse_tracker_advances_counts_and_distances() -> None:
    requests = (
        _request(0, (1, 2), arrival_step=3),
        _request(1, (1, 2), arrival_step=8),
    )
    prefix_hash = request_prefix_hashes(requests[0], block_size_tokens=2)[0]
    tracker = _FutureReuseTracker(requests, block_size_tokens=2, enabled=True)

    assert tracker.remaining_count(prefix_hash) == 2.0
    assert tracker.next_distance(prefix_hash, now=3) == 0.0

    tracker.advance([_block(prefix_hash=prefix_hash)], now=3)

    assert tracker.remaining_count(prefix_hash) == 1.0
    assert tracker.next_distance(prefix_hash, now=3) == 5.0

    tracker.advance([_block(prefix_hash=prefix_hash)], now=8)

    assert tracker.remaining_count(prefix_hash) == 0.0
    assert math.isinf(tracker.next_distance(prefix_hash, now=8))


def test_admission_accounting_records_useful_and_wasted_intervals() -> None:
    accounting = _AdmissionAccounting()

    accounting.record(
        _block(admission_tracked=True, resident_hit_count=2),
        evicted=False,
    )
    accounting.record(
        _block(prefix_hash=2, admission_tracked=True),
        evicted=True,
    )
    accounting.record(
        _block(prefix_hash=3, admission_tracked=False),
        evicted=True,
    )

    assert accounting.useful_count == 1
    assert accounting.wasted_count == 1
    assert accounting.admitted_tokens == 8
    assert accounting.useful_tokens == 4
    assert accounting.wasted_tokens == 4
    assert accounting.saved_tokens == 8
    assert accounting.evicted_without_hit_count == 1


def test_capacity_outcome_merge_accumulates_all_effects() -> None:
    outcome = _CapacityOutcome(
        evictions=1,
        avoidable_evictions=1,
        decode_blocks_requested=2,
        decode_blocks_allocated=1,
    )

    outcome.merge(
        _CapacityOutcome(
            evictions=2,
            high_descendant_evictions=1,
            avoidable_short_reuse_evictions=1,
            value_weighted_avoidable_evictions=1,
            value_weighted_avoidable_eviction_regret_tokens=4.0,
            decode_blocks_requested=3,
            decode_blocks_allocated=2,
            decode_allocation_failure_blocks=1,
            decode_pressure_evictions=2,
        )
    )

    assert outcome == _CapacityOutcome(
        evictions=3,
        high_descendant_evictions=1,
        avoidable_evictions=1,
        avoidable_short_reuse_evictions=1,
        value_weighted_avoidable_evictions=1,
        value_weighted_avoidable_eviction_regret_tokens=4.0,
        decode_blocks_requested=5,
        decode_blocks_allocated=3,
        decode_allocation_failure_blocks=1,
        decode_pressure_evictions=2,
    )
