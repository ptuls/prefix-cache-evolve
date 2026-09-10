"""State, accounting, and tracking support for the KV-cache simulator."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Iterable

from prefix_cache_evolve.evaluators.results import AdmissionDecisionDiagnostic
from prefix_cache_evolve.evaluators.utilities import percentile as _percentile
from prefix_cache_evolve.evaluators.utilities import (
    request_prefix_hashes as _request_prefix_hashes,
)
from prefix_cache_evolve.evaluators.workloads import WorkloadRequest


def _request_arrival_steps(requests: tuple[WorkloadRequest, ...]) -> tuple[int, ...]:
    """Returns monotonic logical arrival times, preserving sequential defaults."""
    arrival_steps = []
    previous_step = -1
    for request_index, request in enumerate(requests):
        arrival_step = request_index if request.arrival_step is None else request.arrival_step
        if arrival_step < previous_step:
            raise ValueError("workload arrival steps must be monotonic")
        arrival_steps.append(arrival_step)
        previous_step = arrival_step
    return tuple(arrival_steps)


def _correlation(left: list[float], right: list[float]) -> float:
    """Return a finite Pearson correlation, or zero for a constant series."""
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value**2 for value in left_centered) * sum(value**2 for value in right_centered)
    )
    if denominator == 0.0:
        return 0.0
    return (
        sum(
            left_value * right_value
            for left_value, right_value in zip(left_centered, right_centered, strict=True)
        )
        / denominator
    )


def _window_mean(values: Iterable[float]) -> float:
    """Return the mean of a bounded online window, or zero before observations."""
    values = tuple(values)
    return sum(values) / len(values) if values else 0.0


@dataclass
class _BlockState:
    prefix_hash: int
    parent_hash: int | None
    depth: int
    start_token: int
    end_token: int
    token_count: int
    prefix_role: str
    tenant_id: int
    created_at: int
    last_accessed_at: int
    prev_last_accessed_at: int | None = None
    last_access_gap: int | None = None
    observed_accessed_at: int | None = None
    access_gap_mean: float | None = None
    access_gap_mean_square: float | None = None
    access_gap_sample_count: int = 0
    hit_count: int = 0
    active_ref_count: int = 0
    resident: bool = False
    admission_tracked: bool = False
    resident_hit_count: int = 0
    resident_children: set[int] = field(default_factory=set)
    known_children: set[int] = field(default_factory=set)

    @property
    def block_id(self) -> int:
        return self.prefix_hash


@dataclass
class _AdmissionAccounting:
    """Finalized utility for successful admission residency intervals."""

    useful_count: int = 0
    wasted_count: int = 0
    admitted_tokens: int = 0
    useful_tokens: int = 0
    wasted_tokens: int = 0
    saved_tokens: int = 0
    evicted_without_hit_count: int = 0

    def record(self, block: _BlockState, *, evicted: bool) -> None:
        """Record one tracked interval before its state is reset."""
        if not block.admission_tracked:
            return
        self.admitted_tokens += block.token_count
        self.saved_tokens += block.resident_hit_count * block.token_count
        if block.resident_hit_count > 0:
            self.useful_count += 1
            self.useful_tokens += block.token_count
            return
        self.wasted_count += 1
        self.wasted_tokens += block.token_count
        if evicted:
            self.evicted_without_hit_count += 1


@dataclass
class _AdmissionAudit:
    """Same-state admission regret measured in future reusable tokens."""

    accepted_count: int = 0
    rejected_count: int = 0
    avoidable_admission_count: int = 0
    avoidable_admission_regret_tokens: float = 0.0
    avoidable_rejection_count: int = 0
    avoidable_rejection_regret_tokens: float = 0.0
    record_decisions: bool = False
    decisions: list[AdmissionDecisionDiagnostic] = field(default_factory=list)
    shadow_samples: list[tuple[float, float, float, float]] = field(default_factory=list)

    def record(
        self,
        *,
        now: int,
        request_index: int,
        block: _BlockState,
        score: float,
        accepted: bool,
        feasible: bool,
        incoming_value_tokens: float,
        displaced_value_tokens: float,
        capacity_weight_tokens: int,
    ) -> None:
        """Record one explicit policy admission score against a local oracle."""
        regret = 0.0
        if accepted:
            self.accepted_count += 1
            if feasible:
                regret = max(0.0, displaced_value_tokens - incoming_value_tokens)
            if regret:
                self.avoidable_admission_count += 1
                self.avoidable_admission_regret_tokens += regret
        else:
            self.rejected_count += 1
            if feasible:
                regret = max(0.0, incoming_value_tokens - displaced_value_tokens)
            if regret:
                self.avoidable_rejection_count += 1
                self.avoidable_rejection_regret_tokens += regret

        weight = max(1, capacity_weight_tokens)
        incoming_density = incoming_value_tokens / weight
        oracle_shadow_price = displaced_value_tokens / weight
        oracle_surplus_density = incoming_density - oracle_shadow_price
        if feasible:
            self.shadow_samples.append(
                (
                    score,
                    oracle_surplus_density,
                    oracle_shadow_price,
                    regret / weight,
                )
            )
        if self.record_decisions:
            self.decisions.append(
                AdmissionDecisionDiagnostic(
                    now=now,
                    request_index=request_index,
                    prefix_hash=block.prefix_hash,
                    depth=block.depth,
                    token_count=block.token_count,
                    capacity_weight_tokens=weight,
                    score=score,
                    accepted=accepted,
                    feasible=feasible,
                    incoming_value_tokens=incoming_value_tokens,
                    displaced_value_tokens=displaced_value_tokens,
                    incoming_value_density=incoming_density,
                    oracle_shadow_price=oracle_shadow_price,
                    oracle_surplus_density=oracle_surplus_density,
                    regret_tokens=regret,
                )
            )

    def shadow_price_metrics(self) -> dict[str, float]:
        """Return calibrated policy-versus-oracle water-level diagnostics."""
        if not self.shadow_samples:
            return {}
        score_square = sum(score**2 for score, _, _, _ in self.shadow_samples)
        score_surplus = sum(score * surplus for score, surplus, _, _ in self.shadow_samples)
        scale = max(0.0, score_surplus / score_square) if score_square else 0.0
        tracking_errors = [surplus - scale * score for score, surplus, _, _ in self.shadow_samples]
        shadow_prices = [shadow_price for _, _, shadow_price, _ in self.shadow_samples]
        changes = [
            0.0,
            *(
                abs(current - previous)
                for previous, current in zip(shadow_prices, shadow_prices[1:])
            ),
        ]
        regret_densities = [regret for _, _, _, regret in self.shadow_samples]
        positive_changes = [change for change in changes if change > 0.0]
        fast_threshold = _percentile(positive_changes, 75) if positive_changes else math.inf
        fast_indexes = [index for index, change in enumerate(changes) if change >= fast_threshold]
        fast_regret = sum(regret_densities[index] for index in fast_indexes)
        total_regret = sum(regret_densities)
        fast_mean = (
            sum(regret_densities[index] for index in fast_indexes) / len(fast_indexes)
            if fast_indexes
            else 0.0
        )
        overall_mean = total_regret / len(self.shadow_samples)
        return {
            "oracle_shadow_price_mean": mean(shadow_prices),
            "oracle_shadow_price_stddev": (
                pstdev(shadow_prices) if len(shadow_prices) > 1 else 0.0
            ),
            "oracle_shadow_price_change_mean": mean(changes),
            "oracle_shadow_price_change_p95": _percentile(changes, 95),
            "shadow_price_score_scale": scale,
            "shadow_price_tracking_rmse": math.sqrt(mean(error**2 for error in tracking_errors)),
            "shadow_price_tracking_mae": mean(abs(error) for error in tracking_errors),
            "shadow_price_tracking_bias": mean(tracking_errors),
            "fast_shadow_price_decision_fraction": (len(fast_indexes) / len(self.shadow_samples)),
            "fast_shadow_price_regret_share": (fast_regret / total_regret if total_regret else 0.0),
            "fast_shadow_price_regret_lift": (
                fast_mean / overall_mean if overall_mean > 0.0 else 1.0
            ),
            "shadow_price_change_regret_correlation": _correlation(
                changes,
                regret_densities,
            ),
        }


@dataclass
class _CapacityOutcome:
    """Capacity effects produced outside one prompt admission."""

    evictions: int = 0
    high_descendant_evictions: int = 0
    avoidable_evictions: int = 0
    avoidable_short_reuse_evictions: int = 0
    value_weighted_avoidable_evictions: int = 0
    value_weighted_avoidable_eviction_regret_tokens: float = 0.0
    decode_blocks_requested: int = 0
    decode_blocks_allocated: int = 0
    decode_allocation_failure_blocks: int = 0
    decode_pressure_evictions: int = 0

    def merge(self, other: _CapacityOutcome) -> None:
        """Accumulate another capacity outcome."""
        self.evictions += other.evictions
        self.high_descendant_evictions += other.high_descendant_evictions
        self.avoidable_evictions += other.avoidable_evictions
        self.avoidable_short_reuse_evictions += other.avoidable_short_reuse_evictions
        self.value_weighted_avoidable_evictions += other.value_weighted_avoidable_evictions
        self.value_weighted_avoidable_eviction_regret_tokens += (
            other.value_weighted_avoidable_eviction_regret_tokens
        )
        self.decode_blocks_requested += other.decode_blocks_requested
        self.decode_blocks_allocated += other.decode_blocks_allocated
        self.decode_allocation_failure_blocks += other.decode_allocation_failure_blocks
        self.decode_pressure_evictions += other.decode_pressure_evictions


@dataclass
class _AdmissionOutcome:
    """Result of one accepted admission attempt."""

    admitted: bool = False
    capacity: _CapacityOutcome = field(default_factory=_CapacityOutcome)


@dataclass
class _ActiveDecode:
    """Decode KV allocation state for one in-flight request."""

    started_at: int
    release_at: int
    total_tokens: int
    attempted_blocks: int = 0
    allocated_blocks: int = 0


class _FutureReuseTracker:
    """Live future-reuse metadata for oracle-style reporting baselines."""

    def __init__(
        self,
        requests: tuple[WorkloadRequest, ...],
        *,
        block_size_tokens: int,
        enabled: bool,
    ) -> None:
        self.enabled = enabled
        self._remaining_counts: dict[int, int] = {}
        self._future_positions: dict[int, deque[int]] = {}
        if not enabled:
            return

        arrival_steps = _request_arrival_steps(requests)
        for now, request in zip(arrival_steps, requests, strict=True):
            for prefix_hash in _request_prefix_hashes(request, block_size_tokens):
                self._remaining_counts[prefix_hash] = self._remaining_counts.get(prefix_hash, 0) + 1
                self._future_positions.setdefault(prefix_hash, deque()).append(now)

    def advance(self, blocks: list[_BlockState], now: int) -> None:
        if not self.enabled:
            return
        for block in blocks:
            prefix_hash = block.prefix_hash
            self._remaining_counts[prefix_hash] = max(
                0,
                self._remaining_counts.get(prefix_hash, 0) - 1,
            )
            positions = self._future_positions.get(prefix_hash)
            if positions:
                positions.popleft()

    def remaining_count(self, prefix_hash: int) -> float | None:
        if not self.enabled:
            return None
        return float(self._remaining_counts.get(prefix_hash, 0))

    def next_distance(self, prefix_hash: int, now: int) -> float | None:
        if not self.enabled:
            return None
        positions = self._future_positions.get(prefix_hash) or ()
        if not positions:
            return math.inf
        return float(max(0, positions[0] - now))
