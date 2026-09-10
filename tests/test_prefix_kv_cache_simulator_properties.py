"""Property-based invariants for the deterministic prefix-cache simulator."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st

from prefix_cache_evolve.evaluators.baselines import (
    baseline_lru_blocks,
    baseline_no_cache,
)
from prefix_cache_evolve.evaluators.contracts import PolicyFactory
from prefix_cache_evolve.evaluators.prefix_kv_cache import PrefixKVCacheSimulator
from prefix_cache_evolve.evaluators.results import TrialMetrics
from prefix_cache_evolve.evaluators.telemetry import (
    EvictionDecisionSnapshot,
    RequestSnapshot,
)
from prefix_cache_evolve.evaluators.workloads import WorkloadRequest, build_workload

_PROPERTY_SETTINGS = settings(max_examples=50, derandomize=True, deadline=None)
_WORKLOAD_FAMILIES = (
    "shared_system_prompt",
    "session_continuation_growth",
    "multi_tenant_skew",
    "concurrent_long_generation",
    "priority_burst_recovery",
)
_POLICY_FACTORIES: tuple[tuple[str, PolicyFactory], ...] = (
    ("lru", baseline_lru_blocks),
    ("no_cache", baseline_no_cache),
)
_UNIT_INTERVAL_METRICS = (
    "block_hit_rate",
    "token_hit_rate",
    "priority_weighted_token_hit_rate",
    "high_priority_token_hit_rate",
    "low_priority_token_hit_rate",
    "priority_request_fraction",
    "request_token_hit_rate_p10",
    "request_token_hit_rate_p50",
    "high_priority_request_token_hit_rate_p10",
    "worst_quarter_token_hit_rate",
    "final_quarter_token_hit_rate",
    "admission_rate",
    "policy_bypass_token_rate",
    "policy_underfill_rate",
    "forced_bypass_token_rate",
    "short_reuse_after_eviction_missed_token_rate",
    "avoidable_eviction_rate",
    "value_weighted_avoidable_eviction_rate",
    "decode_kv_allocation_failure_rate",
    "decode_pressure_eviction_rate",
    "recovery_token_hit_rate",
)
_NONNEGATIVE_METRICS = (
    "prefill_tokens_saved",
    "recompute_tokens",
    "recompute_cost",
    "lookup_block_count",
    "eviction_count",
    "admission_count",
    "admission_score_count",
    "admission_rejection_count",
    "policy_bypass_tokens",
    "forced_bypass_count",
    "forced_bypass_tokens",
    "memory_occupancy_mean",
    "memory_occupancy_peak",
    "prefix_kv_occupancy_mean",
    "prefix_kv_occupancy_peak",
    "decode_kv_occupancy_mean",
    "decode_kv_occupancy_peak",
)


@dataclasses.dataclass
class _TraceCollector:
    requests: list[RequestSnapshot] = dataclasses.field(default_factory=list)
    evictions: list[EvictionDecisionSnapshot] = dataclasses.field(default_factory=list)

    def on_request_complete(self, snapshot: RequestSnapshot) -> None:
        self.requests.append(snapshot)

    def on_eviction_decision(self, snapshot: EvictionDecisionSnapshot) -> None:
        self.evictions.append(snapshot)


@st.composite
def _request_streams(draw):
    family = draw(st.sampled_from(_WORKLOAD_FAMILIES))
    request_count = draw(st.integers(min_value=2, max_value=10))
    block_size = draw(st.sampled_from((2, 4, 8)))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    requests = build_workload(
        family,
        request_count=request_count,
        block_size_tokens=block_size,
        seed=seed,
    )

    arrival_deltas = draw(
        st.lists(
            st.integers(min_value=0, max_value=3), min_size=request_count, max_size=request_count
        )
    )
    session_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=3), min_size=request_count, max_size=request_count
        )
    )
    priorities = draw(
        st.lists(
            st.integers(min_value=0, max_value=3), min_size=request_count, max_size=request_count
        )
    )
    arrival_step = 0
    varied_requests = []
    for request, delta, session_id, priority in zip(
        requests,
        arrival_deltas,
        session_ids,
        priorities,
        strict=True,
    ):
        arrival_step += delta
        varied_requests.append(
            replace(
                request,
                info=replace(request.info, session_id=session_id, priority=priority),
                arrival_step=arrival_step,
            )
        )

    return (
        tuple(varied_requests),
        family,
        block_size,
        seed,
        draw(st.integers(min_value=1, max_value=12)),
        draw(st.sampled_from(("prefix_only", "shared"))),
        draw(st.sampled_from((1, 4, 16, 64))),
    )


def _run_simulation(
    requests: tuple[WorkloadRequest, ...],
    *,
    family: str,
    block_size: int,
    seed: int,
    capacity: int,
    capacity_mode: str,
    active_tokens_per_step: int,
    factory: PolicyFactory,
) -> tuple[TrialMetrics, _TraceCollector]:
    trace = _TraceCollector()
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=capacity,
        block_size_tokens=block_size,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.1,
        eviction_cost_per_block=0.2,
        active_tokens_per_step=active_tokens_per_step,
        kv_capacity_mode=capacity_mode,
        observer=trace,
        eviction_decision_observer=trace,
    )
    metrics = simulator.run(
        factory(capacity, block_size, seed),
        requests,
        split="property",
        workload=family,
        seed=seed,
    )
    return metrics, trace


def _serialized_run(metrics: TrialMetrics, trace: _TraceCollector) -> bytes:
    payload = {
        "metrics": dataclasses.asdict(metrics),
        "requests": [dataclasses.asdict(snapshot) for snapshot in trace.requests],
        "evictions": [dataclasses.asdict(snapshot) for snapshot in trace.evictions],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _assert_request_invariants(
    metrics: TrialMetrics,
    trace: _TraceCollector,
    requests: tuple[WorkloadRequest, ...],
    capacity: int,
) -> None:
    assert not metrics.invalid
    assert len(trace.requests) == len(requests)
    assert metrics.matched_lengths == tuple(snapshot.matched_blocks for snapshot in trace.requests)

    total_prompt_tokens = sum(request.info.prompt_length for request in requests)
    total_prompt_blocks = sum(snapshot.prompt_blocks for snapshot in trace.requests)
    total_hit_tokens = sum(snapshot.hit_tokens for snapshot in trace.requests)
    total_hit_blocks = sum(snapshot.matched_blocks for snapshot in trace.requests)
    assert sum(snapshot.prompt_tokens for snapshot in trace.requests) == total_prompt_tokens
    assert metrics.prefill_tokens_saved == total_hit_tokens
    assert metrics.prefill_tokens_saved + metrics.recompute_tokens == total_prompt_tokens
    assert metrics.token_hit_rate == total_hit_tokens / total_prompt_tokens
    assert metrics.block_hit_rate == total_hit_blocks / total_prompt_blocks

    for snapshot in trace.requests:
        assert 0 <= snapshot.matched_blocks <= snapshot.prompt_blocks
        assert 0 <= snapshot.hit_tokens <= snapshot.prompt_tokens
        assert 0 <= snapshot.resident_blocks <= snapshot.capacity_blocks == capacity
        assert len(snapshot.cache) == snapshot.resident_blocks
        assert 0.0 <= snapshot.cumulative_token_hit_rate <= 1.0

        block_ids = {block.block_id for block in snapshot.cache}
        assert len(block_ids) == len(snapshot.cache)
        assert all(
            block.parent_id is None or block.parent_id in block_ids for block in snapshot.cache
        )

        hit_blocks = [block for block in snapshot.cache if block.hit_this_request]
        assert len(hit_blocks) == snapshot.matched_blocks
        assert sum(block.token_count for block in hit_blocks) == snapshot.hit_tokens
        assert {block.depth for block in hit_blocks} == set(range(1, snapshot.matched_blocks + 1))
        assert all(block.in_request and block.active_ref_count > 0 for block in hit_blocks)


def _assert_eviction_invariants(trace: _TraceCollector) -> None:
    for decision in trace.evictions:
        assert decision.candidates
        candidate_hashes = {candidate.block.prefix_hash for candidate in decision.candidates}
        assert decision.victim_prefix_hash in candidate_hashes
        assert all(candidate.block.active_ref_count == 0 for candidate in decision.candidates)


def _assert_metric_invariants(metrics: TrialMetrics, capacity: int) -> None:
    for name in _UNIT_INTERVAL_METRICS:
        value = getattr(metrics, name)
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0
    for name in _NONNEGATIVE_METRICS:
        value = getattr(metrics, name)
        assert math.isfinite(value)
        assert value >= 0

    assert metrics.memory_occupancy_peak <= capacity
    assert metrics.prefix_kv_occupancy_peak <= capacity
    assert metrics.decode_kv_occupancy_peak <= capacity
    assert metrics.admission_score_count == (
        metrics.admission_count + metrics.admission_rejection_count + metrics.forced_bypass_count
    )


@_PROPERTY_SETTINGS
@given(case=_request_streams())
def test_simulator_invariants_hold_for_generated_request_streams(case) -> None:
    requests, family, block_size, seed, capacity, capacity_mode, active_tokens_per_step = case
    run_arguments = {
        "family": family,
        "block_size": block_size,
        "seed": seed,
        "capacity": capacity,
        "capacity_mode": capacity_mode,
        "active_tokens_per_step": active_tokens_per_step,
    }

    for _, factory in _POLICY_FACTORIES:
        metrics, trace = _run_simulation(requests, factory=factory, **run_arguments)
        replay_metrics, replay_trace = _run_simulation(requests, factory=factory, **run_arguments)

        _assert_request_invariants(metrics, trace, requests, capacity)
        _assert_eviction_invariants(trace)
        _assert_metric_invariants(metrics, capacity)
        assert _serialized_run(metrics, trace) == _serialized_run(replay_metrics, replay_trace)
