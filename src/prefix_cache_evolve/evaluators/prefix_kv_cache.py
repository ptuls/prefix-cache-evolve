"""Deterministic prefix KV-cache evaluator for scoring eviction heuristics."""

from __future__ import annotations

import inspect
import math
import tracemalloc
from collections import deque
from dataclasses import dataclass, field, replace
from statistics import mean, pstdev
from typing import Callable, Iterable, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from prefix_cache_evolve.evaluators import scoring as _scoring
from prefix_cache_evolve.evaluators.baselines import (
    BASELINE_REGISTRY as BASELINE_REGISTRY,
)
from prefix_cache_evolve.evaluators.baselines import (
    BASELINES as BASELINES,
)
from prefix_cache_evolve.evaluators.baselines import (
    REPORTING_BASELINES as REPORTING_BASELINES,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_cost_aware_lru as baseline_cost_aware_lru,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_depth_prefer_shallow as baseline_depth_prefer_shallow,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_future_reuse_heuristic as baseline_future_reuse_heuristic,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_lfu_blocks as baseline_lfu_blocks,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_lru_blocks,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_no_cache as baseline_no_cache,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_oracle_future_reuse as baseline_oracle_future_reuse,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_prefix_anchor as baseline_prefix_anchor,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_prefix_fanout as baseline_prefix_fanout,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_recompute_cost_greedy as baseline_recompute_cost_greedy,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_sglang_radix_attention as baseline_sglang_radix_attention,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_tenant_fair_lru as baseline_tenant_fair_lru,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_tinylfu_lru as baseline_tinylfu_lru,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_vllm_apc as baseline_vllm_apc,
)
from prefix_cache_evolve.evaluators.complexity import (
    scoring_fn_complexity as scoring_fn_complexity,
)
from prefix_cache_evolve.evaluators.contracts import (
    PolicyFactory as PolicyFactory,
)
from prefix_cache_evolve.evaluators.contracts import (
    PrefixBlockInfo as PrefixBlockInfo,
)
from prefix_cache_evolve.evaluators.contracts import (
    PrefixKVPolicy as PrefixKVPolicy,
)
from prefix_cache_evolve.evaluators.contracts import (
    RequestInfo as RequestInfo,
)
from prefix_cache_evolve.evaluators.fingerprints import (
    evaluation_context_sha256,
    panel_sha256,
    request_stream_fingerprint_record,
)
from prefix_cache_evolve.evaluators.scoring import (
    aggregate_by as _aggregate_by,
)
from prefix_cache_evolve.evaluators.scoring import (
    workload_base_score as _workload_base_score,
)
from prefix_cache_evolve.evaluators.telemetry import (
    CacheBlockSnapshot,
    EvictionCandidateSnapshot,
    EvictionDecisionObserver,
    EvictionDecisionSnapshot,
    RequestSnapshot,
    SimulatorObserver,
)
from prefix_cache_evolve.evaluators.utilities import (
    depth_band as _depth_band,
)
from prefix_cache_evolve.evaluators.utilities import (
    jain_fairness as _jain_fairness,
)
from prefix_cache_evolve.evaluators.utilities import (
    percentile as _percentile,
)
from prefix_cache_evolve.evaluators.utilities import (
    prefix_role as _prefix_role,
)
from prefix_cache_evolve.evaluators.utilities import (
    request_prefix_hashes as _request_prefix_hashes,
)
from prefix_cache_evolve.evaluators.utilities import (
    stable_hash as _stable_hash,
)
from prefix_cache_evolve.evaluators.utilities import (
    structural_metrics as _structural_metrics,
)
from prefix_cache_evolve.evaluators.utilities import (
    window_token_hit_rates as _window_token_hit_rates,
)
from prefix_cache_evolve.evaluators.verifier import (
    VERIFIER_VERSION,
    VERIFIER_VERSION_PATTERN,
)
from prefix_cache_evolve.evaluators.workloads import WorkloadRequest, build_workload

_HIGH_DESCENDANT_MIN_COUNT = 2
_COLD_DEEP_MIN_DEPTH = 5
_SHORT_REUSE_DISTANCE_STEPS = 8
_TEMPORAL_WINDOWS = 4
_ACCESS_GAP_EW_ALPHA = 0.25
_REGIME_WINDOW_REQUESTS = 32
_KV_CAPACITY_MODES = ("prefix_only", "shared")
_aggregate_trials = _scoring.aggregate_trials


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


def _flatten_scoring_settings(value: object) -> object:
    """Flatten a nested scoring mapping into evaluator settings."""
    if not isinstance(value, Mapping):
        return value
    values = dict(value)
    scoring = values.pop("scoring", None)
    if scoring is None:
        return values
    if not isinstance(scoring, Mapping):
        raise ValueError("scoring must be a mapping")
    duplicates = sorted(set(values).intersection(scoring))
    if duplicates:
        raise ValueError(
            "scoring fields must not also appear at the settings root: " + ", ".join(duplicates)
        )
    values.update(scoring)
    return values


@dataclass
class WorkloadConfig:
    """Configures one workload family inside one split."""

    family: str
    split: str
    request_count: int = 96
    seed_offset: int = 0


class EvaluatorConfig(BaseModel):
    """Configuration for prefix KV-cache evaluation and scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    verifier_version: str = Field(default=VERIFIER_VERSION, pattern=VERIFIER_VERSION_PATTERN)
    capacity_blocks: PositiveInt = 24
    capacity_sweep_blocks: tuple[PositiveInt, ...] = ()
    block_size_tokens: PositiveInt = 16
    workload_token_granularity: PositiveInt = 8
    seeds: tuple[int, ...] = (11, 23, 37)
    policy_seed: int = 0
    train_families: tuple[str, ...] = (
        "shared_system_prompt",
        "rag_template_reuse",
        "long_context_mixed",
        "session_continuation_growth",
        "agentic_tool_workflows",
    )
    validation_families: tuple[str, ...] = (
        "phase_shift_prompts",
        "multi_tenant_skew",
        "hotset_cold_scan",
        "concurrent_long_generation",
        "stochastic_serving_mix",
        "rolling_template_versions",
        "heavy_tailed_prefix_lengths",
        "priority_burst_recovery",
        "priority_one_off_noise",
        "tenant_phase_shift_cycles",
    )
    probe_families: tuple[str, ...] = (
        "agent_trace_branching",
        "cyclic_working_set_pressure",
    )
    hidden_families: tuple[str, ...] = (
        "adversarial_unique_prompts",
        "cross_family_mixture",
        "tenant_session_reentry",
        "stochastic_serving_mix_shifted",
        "rolling_template_versions_shifted",
        "heavy_tailed_prefix_lengths_shifted",
        "priority_burst_recovery_shifted",
        "cyclic_working_set_pressure_shifted",
        "priority_one_off_noise_shifted",
        "tenant_phase_shift_cycles_shifted",
    )
    request_count: PositiveInt = 96
    family_request_multipliers: dict[str, PositiveInt] = Field(
        default_factory=lambda: {
            "tenant_phase_shift_cycles": 3,
            "tenant_phase_shift_cycles_shifted": 4,
        }
    )
    prefill_cost_per_token: NonNegativeFloat = 1.0
    lookup_cost_per_block: NonNegativeFloat = 0.035
    eviction_cost_per_block: NonNegativeFloat = 0.2
    active_tokens_per_step: PositiveInt = 64
    kv_capacity_mode: Literal["prefix_only", "shared"] = "prefix_only"
    w_avg_tok: NonNegativeFloat = 80.0
    w_avg_blk: NonNegativeFloat = 60.0
    min_workload_weight: NonNegativeFloat = 0.5
    min_seed_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    request_tail_weight: NonNegativeFloat = 12.0
    worst_window_weight: NonNegativeFloat = 12.0
    priority_hit_weight: NonNegativeFloat = 8.0
    wasted_admission_weight: NonNegativeFloat = 6.0
    admission_utility_weight: NonNegativeFloat = 1.0
    avoidable_eviction_weight: NonNegativeFloat = 8.0
    latency_norm: NonNegativeFloat = 0.0
    latency_weight: NonNegativeFloat = 35.0
    latency_cap: NonNegativeFloat = 40.0
    churn_weight: NonNegativeFloat = 0.015
    churn_cap: NonNegativeFloat = 25.0
    underfill_weight: NonNegativeFloat = 12.0
    underfill_cap: NonNegativeFloat = 15.0
    fairness_weight: NonNegativeFloat = 80.0
    fairness_cap: NonNegativeFloat = 30.0
    k_complex: NonNegativeFloat = 0.065
    complexity_exponent: PositiveFloat = 0.75
    v_min: float = -1_000.0
    invalid_surcharge: NonNegativeFloat = 1_000.0
    timeout_s: PositiveFloat = 30.0
    max_memory_bytes: PositiveInt = 64 * 1024 * 1024
    form_aware_complexity: bool = False
    max_candidate_complexity: PositiveInt | None = None
    promotion_max_candidate_complexity: PositiveInt | None = None
    surrogate_probe_tripwire_thresholds: dict[str, NonNegativeFloat] = Field(
        default_factory=lambda: {
            "agentic_branching": 0.12,
            "cyclic_working_set": 0.25,
        }
    )
    fixed_admission_policy: str | None = None
    candidate_policy_surface: Literal["full", "eviction_only"] = "full"
    search_score_mode: Literal["combined", "raw_before_complexity", "robust_min"] = "combined"
    search_guidance_families: tuple[str, ...] = ()
    reject_unsupported_source_patterns: bool = False

    @field_validator("verifier_version")
    @classmethod
    def _require_implemented_verifier_version(cls, value: str) -> str:
        if value != VERIFIER_VERSION:
            raise ValueError(f"this checkout implements verifier {VERIFIER_VERSION}, not {value}")
        return value

    @model_validator(mode="before")
    @classmethod
    def _flatten_scoring_settings(cls, value: object) -> object:
        """Accept the YAML scoring subsection while retaining a flat runtime API."""
        return _flatten_scoring_settings(value)

    @model_validator(mode="after")
    def _validate_tripwire_channels(self) -> EvaluatorConfig:
        """Require an explicit threshold for every supported tripwire channel."""
        expected = {"agentic_branching", "cyclic_working_set"}
        configured = set(self.surrogate_probe_tripwire_thresholds)
        if configured != expected:
            missing = sorted(expected - configured)
            unknown = sorted(configured - expected)
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            raise ValueError(
                "surrogate_probe_tripwire_thresholds must configure exactly "
                f"{sorted(expected)} ({'; '.join(details)})"
            )
        return self

    @model_validator(mode="after")
    def _validate_search_guidance(self) -> EvaluatorConfig:
        """Require robust search guidance to use non-quarantined train families."""
        guidance = set(self.search_guidance_families)
        if self.search_score_mode != "robust_min":
            if guidance:
                raise ValueError("search_guidance_families requires search_score_mode='robust_min'")
            return self
        if not guidance:
            raise ValueError("robust_min search requires at least one search guidance family")
        unknown = guidance - set(self.train_families)
        if unknown:
            raise ValueError(
                "search guidance families must be configured train families: "
                + ", ".join(sorted(unknown))
            )
        quarantined = guidance & (set(self.probe_families) | set(self.hidden_families))
        if quarantined:
            raise ValueError(
                "search guidance families must not be probe or hidden families: "
                + ", ".join(sorted(quarantined))
            )
        return self

    def with_updates(self, **updates: object) -> EvaluatorConfig:
        """Return a validated copy with the supplied settings overlaid."""
        normalized = _flatten_scoring_settings(updates)
        if not isinstance(normalized, Mapping):
            raise TypeError("evaluator config updates must be a mapping")
        return type(self).model_validate({**self.model_dump(), **dict(normalized)})

    def effective_capacity_blocks(self) -> tuple[int, ...]:
        """Returns the capacities evaluated for each workload and seed."""
        values = self.capacity_sweep_blocks or (self.capacity_blocks,)
        capacities: list[int] = []
        for value in values:
            capacity = int(value)
            if capacity <= 0:
                raise ValueError("capacity blocks must be positive")
            if capacity not in capacities:
                capacities.append(capacity)
        return tuple(capacities)

    def effective_capacity_tokens(self) -> tuple[int, ...]:
        """Returns evaluated cache capacities expressed in tokens."""
        if self.block_size_tokens <= 0:
            raise ValueError("block size tokens must be positive")
        return tuple(
            capacity * self.block_size_tokens for capacity in self.effective_capacity_blocks()
        )

    def effective_workload_token_granularity(self) -> int:
        """Returns the canonical token granularity used to build synthetic traffic."""
        granularity = int(self.workload_token_granularity)
        if granularity <= 0:
            raise ValueError("workload token granularity must be positive")
        return granularity

    def workload_configs(self, splits: Iterable[str]) -> tuple[WorkloadConfig, ...]:
        """Return expanded workload configurations for the requested splits."""
        configs: list[WorkloadConfig] = []
        for split in splits:
            families = {
                "train": self.train_families,
                "validation": self.validation_families,
                "probe": self.probe_families,
                "hidden": self.hidden_families,
            }[split]
            for index, family in enumerate(families):
                multiplier = int(self.family_request_multipliers.get(family, 1))
                if multiplier <= 0:
                    raise ValueError("family request multipliers must be positive")
                configs.append(
                    WorkloadConfig(
                        family=family,
                        split=split,
                        request_count=self.request_count * multiplier,
                        seed_offset=1000 * (index + 1),
                    )
                )
        return tuple(configs)


@dataclass
class TrialMetrics:
    """Metrics for one workload family and random seed."""

    split: str
    workload: str
    seed: int
    capacity_blocks: int = 0
    panel_sha256: str = ""
    block_hit_rate: float = 0.0
    token_hit_rate: float = 0.0
    priority_weighted_token_hit_rate: float = 0.0
    high_priority_token_hit_rate: float = 0.0
    low_priority_token_hit_rate: float = 0.0
    priority_request_fraction: float = 0.0
    request_token_hit_rate_p10: float = 0.0
    request_token_hit_rate_p50: float = 0.0
    high_priority_request_token_hit_rate_p10: float = 0.0
    worst_quarter_token_hit_rate: float = 0.0
    final_quarter_token_hit_rate: float = 0.0
    quarter_token_hit_rate_stddev: float = 0.0
    prefill_tokens_saved: float = 0.0
    recompute_tokens: float = 0.0
    recompute_cost: float = 0.0
    lookup_block_count: int = 0
    lookup_blocks_per_request: float = 0.0
    eviction_count: int = 0
    admission_count: int = 0
    admission_score_count: int = 0
    admission_rejection_count: int = 0
    admission_rate: float = 0.0
    avoidable_admission_count: int = 0
    avoidable_admission_rate: float = 0.0
    avoidable_admission_regret_tokens: float = 0.0
    avoidable_admission_regret_token_rate: float = 0.0
    avoidable_rejection_count: int = 0
    avoidable_rejection_rate: float = 0.0
    avoidable_rejection_regret_tokens: float = 0.0
    avoidable_rejection_regret_token_rate: float = 0.0
    oracle_shadow_price_mean: float = 0.0
    oracle_shadow_price_stddev: float = 0.0
    oracle_shadow_price_change_mean: float = 0.0
    oracle_shadow_price_change_p95: float = 0.0
    shadow_price_score_scale: float = 0.0
    shadow_price_tracking_rmse: float = 0.0
    shadow_price_tracking_mae: float = 0.0
    shadow_price_tracking_bias: float = 0.0
    fast_shadow_price_decision_fraction: float = 0.0
    fast_shadow_price_regret_share: float = 0.0
    fast_shadow_price_regret_lift: float = 0.0
    shadow_price_change_regret_correlation: float = 0.0
    useful_admission_count: int = 0
    useful_admission_rate: float = 0.0
    wasted_admission_count: int = 0
    wasted_admission_rate: float = 0.0
    admitted_token_count: int = 0
    useful_admission_token_count: int = 0
    useful_admission_token_rate: float = 0.0
    wasted_admission_token_count: int = 0
    wasted_admission_token_rate: float = 0.0
    admission_saved_tokens: int = 0
    admission_saved_tokens_per_admission: float = 0.0
    admission_token_utility: float = 0.0
    evicted_without_hit_count: int = 0
    evicted_without_hit_rate: float = 0.0
    policy_bypass_tokens: int = 0
    policy_bypass_token_rate: float = 0.0
    policy_underfill_rate: float = 0.0
    cache_churn_per_1k: float = 0.0
    forced_bypass_count: int = 0
    forced_bypass_tokens: int = 0
    forced_bypass_token_rate: float = 0.0
    short_reuse_after_eviction_missed_tokens: int = 0
    short_reuse_after_eviction_missed_token_rate: float = 0.0
    eviction_reuse_distance_p50: float = 0.0
    eviction_reuse_distance_p95: float = 0.0
    avoidable_eviction_count: int = 0
    avoidable_eviction_rate: float = 0.0
    avoidable_short_reuse_eviction_count: int = 0
    avoidable_short_reuse_eviction_rate: float = 0.0
    value_weighted_avoidable_eviction_count: int = 0
    value_weighted_avoidable_eviction_rate: float = 0.0
    value_weighted_avoidable_eviction_regret_tokens: float = 0.0
    value_weighted_avoidable_eviction_regret_token_rate: float = 0.0
    tenant_count: int = 0
    tenant_fairness_penalty: float = 0.0
    tenant_token_hit_rate_p10: float = 0.0
    tenant_jain_fairness: float = 1.0
    p50_latency_proxy: float = 0.0
    p95_latency_proxy: float = 0.0
    p99_latency_proxy: float = 0.0
    high_priority_p95_latency_proxy: float = 0.0
    high_priority_p99_latency_proxy: float = 0.0
    p95_recompute_cost: float = 0.0
    recovery_request_count: int = 0
    recovery_token_hit_rate: float = 0.0
    recovery_p95_latency_proxy: float = 0.0
    recovery_phase_count: int = 0
    worst_recovery_phase_token_hit_rate: float = 0.0
    final_recovery_phase_token_hit_rate: float = 0.0
    worst_recovery_phase_p95_latency_proxy: float = 0.0
    memory_occupancy_mean: float = 0.0
    memory_occupancy_peak: int = 0
    prefix_kv_occupancy_mean: float = 0.0
    prefix_kv_occupancy_peak: int = 0
    decode_kv_occupancy_mean: float = 0.0
    decode_kv_occupancy_peak: int = 0
    decode_kv_blocks_requested: int = 0
    decode_kv_blocks_allocated: int = 0
    decode_kv_allocation_failure_blocks: int = 0
    decode_kv_allocation_failure_rate: float = 0.0
    decode_pressure_eviction_count: int = 0
    decode_pressure_eviction_rate: float = 0.0
    arrival_span_steps: int = 0
    active_request_count_peak: int = 0
    max_prefill_cost: float = 0.0
    scoring_fn_complexity: int = 0
    invalid: bool = False
    invalid_reason: str = ""
    matched_lengths: tuple[int, ...] = ()
    structural_metrics: dict[str, float] = field(default_factory=dict)
    admission_decisions: tuple[AdmissionDecisionDiagnostic, ...] = ()

    def as_dict(self) -> dict[str, float | int | bool | str]:
        """Return scalar trial metrics as a serializable mapping."""
        metrics: dict[str, float | int | bool | str] = {
            "block_hit_rate": self.block_hit_rate,
            "capacity_blocks": self.capacity_blocks,
            "token_hit_rate": self.token_hit_rate,
            "priority_weighted_token_hit_rate": self.priority_weighted_token_hit_rate,
            "high_priority_token_hit_rate": self.high_priority_token_hit_rate,
            "low_priority_token_hit_rate": self.low_priority_token_hit_rate,
            "priority_request_fraction": self.priority_request_fraction,
            "request_token_hit_rate_p10": self.request_token_hit_rate_p10,
            "request_token_hit_rate_p50": self.request_token_hit_rate_p50,
            "high_priority_request_token_hit_rate_p10": (
                self.high_priority_request_token_hit_rate_p10
            ),
            "worst_quarter_token_hit_rate": self.worst_quarter_token_hit_rate,
            "final_quarter_token_hit_rate": self.final_quarter_token_hit_rate,
            "quarter_token_hit_rate_stddev": self.quarter_token_hit_rate_stddev,
            "prefill_tokens_saved": self.prefill_tokens_saved,
            "recompute_tokens": self.recompute_tokens,
            "recompute_cost": self.recompute_cost,
            "lookup_block_count": self.lookup_block_count,
            "lookup_blocks_per_request": self.lookup_blocks_per_request,
            "eviction_count": self.eviction_count,
            "admission_count": self.admission_count,
            "admission_score_count": self.admission_score_count,
            "admission_rejection_count": self.admission_rejection_count,
            "admission_rate": self.admission_rate,
            "avoidable_admission_count": self.avoidable_admission_count,
            "avoidable_admission_rate": self.avoidable_admission_rate,
            "avoidable_admission_regret_tokens": self.avoidable_admission_regret_tokens,
            "avoidable_admission_regret_token_rate": (self.avoidable_admission_regret_token_rate),
            "avoidable_rejection_count": self.avoidable_rejection_count,
            "avoidable_rejection_rate": self.avoidable_rejection_rate,
            "avoidable_rejection_regret_tokens": self.avoidable_rejection_regret_tokens,
            "avoidable_rejection_regret_token_rate": (self.avoidable_rejection_regret_token_rate),
            "oracle_shadow_price_mean": self.oracle_shadow_price_mean,
            "oracle_shadow_price_stddev": self.oracle_shadow_price_stddev,
            "oracle_shadow_price_change_mean": self.oracle_shadow_price_change_mean,
            "oracle_shadow_price_change_p95": self.oracle_shadow_price_change_p95,
            "shadow_price_score_scale": self.shadow_price_score_scale,
            "shadow_price_tracking_rmse": self.shadow_price_tracking_rmse,
            "shadow_price_tracking_mae": self.shadow_price_tracking_mae,
            "shadow_price_tracking_bias": self.shadow_price_tracking_bias,
            "fast_shadow_price_decision_fraction": self.fast_shadow_price_decision_fraction,
            "fast_shadow_price_regret_share": self.fast_shadow_price_regret_share,
            "fast_shadow_price_regret_lift": self.fast_shadow_price_regret_lift,
            "shadow_price_change_regret_correlation": (self.shadow_price_change_regret_correlation),
            "useful_admission_count": self.useful_admission_count,
            "useful_admission_rate": self.useful_admission_rate,
            "wasted_admission_count": self.wasted_admission_count,
            "wasted_admission_rate": self.wasted_admission_rate,
            "admitted_token_count": self.admitted_token_count,
            "useful_admission_token_count": self.useful_admission_token_count,
            "useful_admission_token_rate": self.useful_admission_token_rate,
            "wasted_admission_token_count": self.wasted_admission_token_count,
            "wasted_admission_token_rate": self.wasted_admission_token_rate,
            "admission_saved_tokens": self.admission_saved_tokens,
            "admission_saved_tokens_per_admission": (self.admission_saved_tokens_per_admission),
            "admission_token_utility": self.admission_token_utility,
            "evicted_without_hit_count": self.evicted_without_hit_count,
            "evicted_without_hit_rate": self.evicted_without_hit_rate,
            "policy_bypass_tokens": self.policy_bypass_tokens,
            "policy_bypass_token_rate": self.policy_bypass_token_rate,
            "policy_underfill_rate": self.policy_underfill_rate,
            "cache_churn_per_1k": self.cache_churn_per_1k,
            "forced_bypass_count": self.forced_bypass_count,
            "forced_bypass_tokens": self.forced_bypass_tokens,
            "forced_bypass_token_rate": self.forced_bypass_token_rate,
            "short_reuse_after_eviction_missed_tokens": (
                self.short_reuse_after_eviction_missed_tokens
            ),
            "short_reuse_after_eviction_missed_token_rate": (
                self.short_reuse_after_eviction_missed_token_rate
            ),
            "eviction_reuse_distance_p50": self.eviction_reuse_distance_p50,
            "eviction_reuse_distance_p95": self.eviction_reuse_distance_p95,
            "avoidable_eviction_count": self.avoidable_eviction_count,
            "avoidable_eviction_rate": self.avoidable_eviction_rate,
            "avoidable_short_reuse_eviction_count": (self.avoidable_short_reuse_eviction_count),
            "avoidable_short_reuse_eviction_rate": (self.avoidable_short_reuse_eviction_rate),
            "value_weighted_avoidable_eviction_count": (
                self.value_weighted_avoidable_eviction_count
            ),
            "value_weighted_avoidable_eviction_rate": (self.value_weighted_avoidable_eviction_rate),
            "value_weighted_avoidable_eviction_regret_tokens": (
                self.value_weighted_avoidable_eviction_regret_tokens
            ),
            "value_weighted_avoidable_eviction_regret_token_rate": (
                self.value_weighted_avoidable_eviction_regret_token_rate
            ),
            "tenant_count": self.tenant_count,
            "tenant_fairness_penalty": self.tenant_fairness_penalty,
            "tenant_token_hit_rate_p10": self.tenant_token_hit_rate_p10,
            "tenant_jain_fairness": self.tenant_jain_fairness,
            "p50_latency_proxy": self.p50_latency_proxy,
            "p95_latency_proxy": self.p95_latency_proxy,
            "p99_latency_proxy": self.p99_latency_proxy,
            "high_priority_p95_latency_proxy": self.high_priority_p95_latency_proxy,
            "high_priority_p99_latency_proxy": self.high_priority_p99_latency_proxy,
            "p95_recompute_cost": self.p95_recompute_cost,
            "recovery_request_count": self.recovery_request_count,
            "recovery_token_hit_rate": self.recovery_token_hit_rate,
            "recovery_p95_latency_proxy": self.recovery_p95_latency_proxy,
            "recovery_phase_count": self.recovery_phase_count,
            "worst_recovery_phase_token_hit_rate": (self.worst_recovery_phase_token_hit_rate),
            "final_recovery_phase_token_hit_rate": (self.final_recovery_phase_token_hit_rate),
            "worst_recovery_phase_p95_latency_proxy": (self.worst_recovery_phase_p95_latency_proxy),
            "memory_occupancy_mean": self.memory_occupancy_mean,
            "memory_occupancy_peak": self.memory_occupancy_peak,
            "prefix_kv_occupancy_mean": self.prefix_kv_occupancy_mean,
            "prefix_kv_occupancy_peak": self.prefix_kv_occupancy_peak,
            "decode_kv_occupancy_mean": self.decode_kv_occupancy_mean,
            "decode_kv_occupancy_peak": self.decode_kv_occupancy_peak,
            "decode_kv_blocks_requested": self.decode_kv_blocks_requested,
            "decode_kv_blocks_allocated": self.decode_kv_blocks_allocated,
            "decode_kv_allocation_failure_blocks": self.decode_kv_allocation_failure_blocks,
            "decode_kv_allocation_failure_rate": self.decode_kv_allocation_failure_rate,
            "decode_pressure_eviction_count": self.decode_pressure_eviction_count,
            "decode_pressure_eviction_rate": self.decode_pressure_eviction_rate,
            "arrival_span_steps": self.arrival_span_steps,
            "active_request_count_peak": self.active_request_count_peak,
            "max_prefill_cost": self.max_prefill_cost,
            "scoring_fn_complexity": self.scoring_fn_complexity,
            "invalid": self.invalid,
            "invalid_reason": self.invalid_reason,
        }
        metrics.update(self.structural_metrics)
        return metrics


@dataclass
class EvaluationResult:
    """Aggregated evaluator result."""

    verifier_version: str
    evaluation_context_sha256: str
    panel_sha256: str
    combined_score: float
    success: bool
    invalid_fraction: float
    split_metrics: dict[str, dict[str, float | int | bool | str]]
    workload_metrics: dict[str, dict[str, float | int | bool | str]]
    capacity_metrics: dict[str, dict[str, float | int | bool | str]]
    candidate_metadata: dict[str, float | int | bool | str]
    score_breakdown: dict[str, float]
    trials: tuple[TrialMetrics, ...] = ()


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


@dataclass(frozen=True, slots=True)
class AdmissionDecisionDiagnostic:
    """One scored admission decision with quarantined oracle quantities."""

    now: int
    request_index: int
    prefix_hash: int
    depth: int
    token_count: int
    capacity_weight_tokens: int
    score: float
    accepted: bool
    feasible: bool
    incoming_value_tokens: float
    displaced_value_tokens: float
    incoming_value_density: float
    oracle_shadow_price: float
    oracle_surplus_density: float
    regret_tokens: float


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


class InvalidCandidateError(RuntimeError):
    """Raised when a candidate returns an invalid score or crashes."""


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


class PrefixKVCacheSimulator:
    """Owns cache state and applies scoring-only candidate policies."""

    def __init__(
        self,
        *,
        capacity_blocks: int,
        block_size_tokens: int,
        prefill_cost_per_token: float,
        lookup_cost_per_block: float,
        eviction_cost_per_block: float,
        active_tokens_per_step: int = 64,
        kv_capacity_mode: str = "prefix_only",
        expose_future_reuse: bool = False,
        max_memory_bytes: int | None = None,
        observer: SimulatorObserver | None = None,
        eviction_decision_observer: EvictionDecisionObserver | None = None,
        record_admission_diagnostics: bool = False,
        oracle_admission: bool = False,
        oracle_eviction: bool = False,
    ) -> None:
        self.capacity_blocks = capacity_blocks
        self.block_size_tokens = block_size_tokens
        self.prefill_cost_per_token = prefill_cost_per_token
        self.lookup_cost_per_block = lookup_cost_per_block
        self.eviction_cost_per_block = eviction_cost_per_block
        self.active_tokens_per_step = active_tokens_per_step
        if kv_capacity_mode not in _KV_CAPACITY_MODES:
            raise ValueError(f"kv capacity mode must be one of {', '.join(_KV_CAPACITY_MODES)}")
        self.kv_capacity_mode = kv_capacity_mode
        self.expose_future_reuse = expose_future_reuse
        self.max_memory_bytes = max_memory_bytes
        self.observer = observer
        self.eviction_decision_observer = eviction_decision_observer
        self.record_admission_diagnostics = record_admission_diagnostics
        self.oracle_admission = oracle_admission
        self.oracle_eviction = oracle_eviction
        self.blocks: dict[int, _BlockState] = {}
        self._release_events: dict[int, list[int]] = {}
        self._resident_hashes: set[int] = set()
        self._leaf_hashes: set[int] = set()
        self._descendant_counts: dict[int, int] = {}
        self._subtree_access_counts: dict[int, int] = {}
        self._subtree_hit_counts: dict[int, int] = {}
        self._subtree_active_ref_counts: dict[int, int] = {}
        self._evicted_hashes: set[int] = set()
        self._last_evicted_at: dict[int, int] = {}
        self._active_decodes: list[_ActiveDecode] = []
        self._decode_resident_blocks = 0
        self._recent_admission_pressure: deque[float] = deque(maxlen=_REGIME_WINDOW_REQUESTS)
        self._recent_miss_rates: deque[float] = deque(maxlen=_REGIME_WINDOW_REQUESTS)

    def run(
        self,
        policy: PrefixKVPolicy,
        requests: tuple[WorkloadRequest, ...],
        *,
        split: str,
        workload: str,
        seed: int,
        scoring_fn_complexity: int = 0,
    ) -> TrialMetrics:
        """Simulate a policy over a fixed request sequence."""
        total_blocks = 0
        total_tokens = 0
        hit_blocks = 0
        hit_tokens = 0
        priority_weighted_total_tokens = 0
        priority_weighted_hit_tokens = 0
        high_priority_tokens = 0
        high_priority_hit_tokens = 0
        high_priority_request_count = 0
        low_priority_tokens = 0
        low_priority_hit_tokens = 0
        recompute_tokens = 0
        recompute_cost = 0.0
        lookup_block_count = 0
        admission_count = 0
        admission_score_count = 0
        admission_rejection_count = 0
        policy_bypass_tokens = 0
        eviction_count = 0
        forced_bypass_count = 0
        forced_bypass_tokens = 0
        latencies: list[float] = []
        high_priority_latencies: list[float] = []
        high_priority_request_token_hit_rates: list[float] = []
        recompute_costs: list[float] = []
        request_token_hit_rates: list[float] = []
        request_hit_records: list[tuple[int, int]] = []
        recovery_hit_tokens = 0
        recovery_tokens = 0
        recovery_latencies: list[float] = []
        recovery_request_count = 0
        recovery_phase_records: list[list[tuple[int, int, float]]] = []
        previous_was_recovery = False
        occupancies: list[int] = []
        prefix_occupancies: list[int] = []
        decode_occupancies: list[int] = []
        active_request_releases: dict[int, int] = {}
        active_request_count = 0
        active_request_count_peak = 0
        max_prefill_cost = 0.0
        matched_lengths: list[int] = []
        tenant_hits: dict[int, int] = {}
        tenant_tokens: dict[int, int] = {}
        depth_total_blocks: dict[str, int] = {}
        depth_hit_blocks: dict[str, int] = {}
        depth_total_tokens: dict[str, int] = {}
        depth_hit_tokens: dict[str, int] = {}
        prefix_role_hit_tokens: dict[str, int] = {
            "system": 0,
            "developer": 0,
            "user": 0,
        }
        high_descendant_evictions = 0
        cold_deep_admission_opportunities = 0
        cold_deep_admissions = 0
        reuse_after_eviction_missed_blocks = 0
        reuse_after_eviction_missed_tokens = 0
        short_reuse_after_eviction_missed_tokens = 0
        eviction_reuse_distances: list[float] = []
        admission_accounting = _AdmissionAccounting()
        admission_audit = _AdmissionAudit(
            record_decisions=self.record_admission_diagnostics,
        )
        avoidable_eviction_count = 0
        avoidable_short_reuse_eviction_count = 0
        value_weighted_avoidable_eviction_count = 0
        value_weighted_avoidable_eviction_regret_tokens = 0.0
        decode_capacity_outcome = _CapacityOutcome()

        try:
            self._validate_policy(policy)
            self._check_memory_limit()
            future_reuse = _FutureReuseTracker(
                requests,
                block_size_tokens=self.block_size_tokens,
                enabled=self.expose_future_reuse,
            )
            audit_future_reuse = _FutureReuseTracker(
                requests,
                block_size_tokens=self.block_size_tokens,
                enabled=True,
            )
            arrival_steps = _request_arrival_steps(requests)
            for request_index, (now, request) in enumerate(
                zip(arrival_steps, requests, strict=True)
            ):
                decode_capacity_outcome.merge(
                    self._advance_decodes(
                        policy,
                        now,
                        future_reuse,
                        audit_future_reuse,
                        admission_accounting,
                    )
                )
                self._release_expired(now)
                for release_at in sorted(step for step in active_request_releases if step <= now):
                    active_request_count -= active_request_releases.pop(release_at)
                request_blocks = self._materialize_chain(request, now)
                future_reuse.advance(request_blocks, now)
                audit_future_reuse.advance(request_blocks, now)
                max_prefill_cost = max(
                    max_prefill_cost,
                    sum(self._estimated_recompute_cost(block) for block in request_blocks),
                )
                total_blocks += len(request_blocks)
                total_tokens += request.info.prompt_length
                for block in request_blocks:
                    band = _depth_band(block.depth)
                    depth_total_blocks[band] = depth_total_blocks.get(band, 0) + 1
                    depth_total_tokens[band] = depth_total_tokens.get(band, 0) + block.token_count
                tenant_tokens[request.info.tenant_id] = (
                    tenant_tokens.get(request.info.tenant_id, 0) + request.info.prompt_length
                )

                visible_request = replace(
                    request.info,
                    request_id=_stable_hash(("candidate-request", seed, request.info.request_id)),
                    request_type="request",
                    prompt_tokens=(),
                    recent_admission_pressure=_window_mean(self._recent_admission_pressure),
                    recent_miss_rate=_window_mean(self._recent_miss_rates),
                )
                self._call_hook(policy.on_request_start, visible_request, now)
                matched_len = self.match_resident_prefix(request_blocks)
                lookup_blocks = matched_len + int(matched_len < len(request_blocks))
                lookup_block_count += lookup_blocks
                matched_lengths.append(matched_len)
                per_request_evictions = 0
                admission_count_before = admission_count
                policy_bypass_tokens_before = policy_bypass_tokens
                forced_bypass_tokens_before = forced_bypass_tokens
                request_hit_capacity = self.total_resident_count >= self.capacity_blocks
                hit_blocks += matched_len
                tokens_hit = sum(block.token_count for block in request_blocks[:matched_len])
                hit_tokens += tokens_hit
                priority_weight = 1 + max(0, request.info.priority)
                priority_weighted_total_tokens += request.info.prompt_length * priority_weight
                priority_weighted_hit_tokens += tokens_hit * priority_weight
                if request.info.priority > 0:
                    high_priority_tokens += request.info.prompt_length
                    high_priority_hit_tokens += tokens_hit
                else:
                    low_priority_tokens += request.info.prompt_length
                    low_priority_hit_tokens += tokens_hit
                request_token_hit_rates.append(tokens_hit / max(1, request.info.prompt_length))
                request_hit_records.append((tokens_hit, request.info.prompt_length))
                is_recovery_request = "recovery" in request.info.request_type
                if is_recovery_request:
                    if not previous_was_recovery:
                        recovery_phase_records.append([])
                    recovery_request_count += 1
                    recovery_hit_tokens += tokens_hit
                    recovery_tokens += request.info.prompt_length
                if request.info.priority > 0:
                    high_priority_request_count += 1
                    high_priority_request_token_hit_rates.append(
                        tokens_hit / max(1, request.info.prompt_length)
                    )
                tenant_hits[request.info.tenant_id] = (
                    tenant_hits.get(request.info.tenant_id, 0) + tokens_hit
                )

                duration = max(
                    1,
                    math.ceil(request.true_output_length / self.active_tokens_per_step),
                )
                active_request_count += 1
                active_request_releases[now + duration] = (
                    active_request_releases.get(now + duration, 0) + 1
                )
                active_request_count_peak = max(active_request_count_peak, active_request_count)
                for block in request_blocks[:matched_len]:
                    band = _depth_band(block.depth)
                    depth_hit_blocks[band] = depth_hit_blocks.get(band, 0) + 1
                    depth_hit_tokens[band] = depth_hit_tokens.get(band, 0) + block.token_count
                    if block.prefix_role in prefix_role_hit_tokens:
                        prefix_role_hit_tokens[block.prefix_role] += block.token_count
                    block.last_accessed_at = now
                    block.hit_count += 1
                    self._record_hit(block)
                    block.resident_hit_count += 1
                    self._pin(block, now + duration)
                    self._call_hook(
                        policy.on_cache_hit,
                        self._info(block, now, future_reuse),
                        visible_request,
                        now,
                    )

                admission_blocked = False
                forced_bypass_active = False
                for block in request_blocks[matched_len:]:
                    recompute_tokens += block.token_count
                    recompute_cost += self._estimated_recompute_cost(block)
                    if block.prefix_hash in self._evicted_hashes:
                        reuse_after_eviction_missed_blocks += 1
                        reuse_after_eviction_missed_tokens += block.token_count
                        reuse_distance = max(
                            0,
                            now - self._last_evicted_at.get(block.prefix_hash, now),
                        )
                        eviction_reuse_distances.append(float(reuse_distance))
                        if reuse_distance <= _SHORT_REUSE_DISTANCE_STEPS:
                            short_reuse_after_eviction_missed_tokens += block.token_count
                    self._call_hook(
                        policy.on_cache_miss,
                        self._info(block, now, future_reuse),
                        visible_request,
                        now,
                    )
                    is_cold_deep = block.depth >= _COLD_DEEP_MIN_DEPTH and block.hit_count == 0
                    if is_cold_deep:
                        cold_deep_admission_opportunities += 1
                    if admission_blocked:
                        if forced_bypass_active:
                            forced_bypass_tokens += block.token_count
                        else:
                            policy_bypass_tokens += block.token_count
                        continue
                    admission_score_count += 1
                    score = self._score(
                        policy.score_admission,
                        self._info(block, now, future_reuse),
                        now,
                    )
                    (
                        incoming_value_tokens,
                        displaced_value_tokens,
                        admission_feasible,
                    ) = self._admission_decision_values(block, audit_future_reuse)
                    if self.oracle_admission:
                        score = (
                            incoming_value_tokens - displaced_value_tokens
                            if admission_feasible
                            else -1.0
                        )
                    admission_audit.record(
                        now=now,
                        request_index=request_index,
                        block=block,
                        score=score,
                        accepted=score > 0.0,
                        feasible=admission_feasible,
                        incoming_value_tokens=incoming_value_tokens,
                        displaced_value_tokens=displaced_value_tokens,
                        capacity_weight_tokens=self.block_size_tokens,
                    )
                    if score <= 0.0:
                        admission_rejection_count += 1
                        policy_bypass_tokens += block.token_count
                        admission_blocked = True
                        continue
                    admission_outcome = self._admit_block(
                        policy,
                        block,
                        now,
                        duration,
                        future_reuse,
                        audit_future_reuse,
                        admission_accounting,
                    )
                    capacity_outcome = admission_outcome.capacity
                    per_request_evictions += capacity_outcome.evictions
                    request_hit_capacity = (
                        request_hit_capacity
                        or capacity_outcome.evictions > 0
                        or self.total_resident_count >= self.capacity_blocks
                    )
                    eviction_count += capacity_outcome.evictions
                    high_descendant_evictions += capacity_outcome.high_descendant_evictions
                    avoidable_eviction_count += capacity_outcome.avoidable_evictions
                    avoidable_short_reuse_eviction_count += (
                        capacity_outcome.avoidable_short_reuse_evictions
                    )
                    value_weighted_avoidable_eviction_count += (
                        capacity_outcome.value_weighted_avoidable_evictions
                    )
                    value_weighted_avoidable_eviction_regret_tokens += (
                        capacity_outcome.value_weighted_avoidable_eviction_regret_tokens
                    )
                    if admission_outcome.admitted:
                        admission_count += 1
                        if is_cold_deep:
                            cold_deep_admissions += 1
                    else:
                        forced_bypass_count += 1
                        forced_bypass_tokens += block.token_count
                        forced_bypass_active = True
                        admission_blocked = True

                uncached_cost = sum(
                    self._estimated_recompute_cost(block) for block in request_blocks[matched_len:]
                )
                latency = (
                    uncached_cost
                    + lookup_blocks * self.lookup_cost_per_block
                    + per_request_evictions * self.eviction_cost_per_block
                )
                latencies.append(latency)
                recompute_costs.append(uncached_cost)
                if request.info.priority > 0:
                    high_priority_latencies.append(latency)
                if is_recovery_request:
                    recovery_latencies.append(latency)
                    recovery_phase_records[-1].append(
                        (tokens_hit, request.info.prompt_length, latency)
                    )
                previous_was_recovery = is_recovery_request
                decode_start_outcome = self._start_decode(
                    policy,
                    now,
                    request.true_output_length,
                    duration,
                    future_reuse,
                    audit_future_reuse,
                    admission_accounting,
                )
                decode_capacity_outcome.merge(decode_start_outcome)
                request_hit_capacity = (
                    request_hit_capacity
                    or decode_start_outcome.evictions > 0
                    or self.total_resident_count >= self.capacity_blocks
                )
                occupancies.append(self.total_resident_count)
                prefix_occupancies.append(self.resident_count)
                decode_occupancies.append(self.decode_resident_count)
                self._recent_admission_pressure.append(float(request_hit_capacity))
                self._recent_miss_rates.append(
                    1.0 - tokens_hit / max(1, request.info.prompt_length)
                )
                self._emit_request_snapshot(
                    index=request_index,
                    now=now,
                    request=request,
                    request_blocks=request_blocks,
                    matched_len=matched_len,
                    hit_tokens=tokens_hit,
                    admissions=admission_count - admission_count_before,
                    evictions=per_request_evictions + decode_start_outcome.evictions,
                    bypassed_tokens=(
                        policy_bypass_tokens
                        - policy_bypass_tokens_before
                        + forced_bypass_tokens
                        - forced_bypass_tokens_before
                    ),
                    latency=latency,
                    cumulative_hit_tokens=hit_tokens,
                    cumulative_total_tokens=total_tokens,
                    cumulative_evictions=eviction_count + decode_capacity_outcome.evictions,
                )
        except InvalidCandidateError as exc:
            return TrialMetrics(
                split=split,
                workload=workload,
                seed=seed,
                capacity_blocks=self.capacity_blocks,
                scoring_fn_complexity=scoring_fn_complexity,
                invalid=True,
                invalid_reason=str(exc),
                matched_lengths=tuple(matched_lengths),
            )

        request_count = max(len(requests), 1)
        tenant_rates = self._tenant_hit_rates(tenant_hits, tenant_tokens)
        fairness_penalty = max(tenant_rates) - min(tenant_rates) if tenant_rates else 0.0
        resident_admissions = [
            block for block in self.blocks.values() if block.resident and block.admission_tracked
        ]
        for block in resident_admissions:
            admission_accounting.record(block, evicted=False)
        quarter_hit_rates = _window_token_hit_rates(
            request_hit_records,
            window_count=_TEMPORAL_WINDOWS,
        )
        recovery_phase_hit_rates = [
            sum(hit_tokens for hit_tokens, _, _ in records)
            / max(1, sum(tokens for _, tokens, _ in records))
            for records in recovery_phase_records
        ]
        recovery_phase_p95_latencies = [
            _percentile([latency for _, _, latency in records], 95)
            for records in recovery_phase_records
        ]
        total_eviction_count = eviction_count + decode_capacity_outcome.evictions
        total_high_descendant_evictions = (
            high_descendant_evictions + decode_capacity_outcome.high_descendant_evictions
        )
        total_avoidable_eviction_count = (
            avoidable_eviction_count + decode_capacity_outcome.avoidable_evictions
        )
        total_avoidable_short_reuse_eviction_count = (
            avoidable_short_reuse_eviction_count
            + decode_capacity_outcome.avoidable_short_reuse_evictions
        )
        total_value_weighted_avoidable_eviction_count = (
            value_weighted_avoidable_eviction_count
            + decode_capacity_outcome.value_weighted_avoidable_evictions
        )
        total_value_weighted_avoidable_eviction_regret_tokens = (
            value_weighted_avoidable_eviction_regret_tokens
            + decode_capacity_outcome.value_weighted_avoidable_eviction_regret_tokens
        )
        memory_occupancy_mean = mean(occupancies) if occupancies else 0.0
        prefix_kv_occupancy_mean = mean(prefix_occupancies) if prefix_occupancies else 0.0
        decode_kv_occupancy_mean = mean(decode_occupancies) if decode_occupancies else 0.0
        policy_bypass_token_rate = policy_bypass_tokens / max(1, total_tokens)
        policy_underfill_rate = policy_bypass_token_rate * max(
            0.0,
            1.0 - prefix_kv_occupancy_mean / self.capacity_blocks,
        )
        structural_metrics = _structural_metrics(
            depth_total_blocks=depth_total_blocks,
            depth_hit_blocks=depth_hit_blocks,
            depth_total_tokens=depth_total_tokens,
            depth_hit_tokens=depth_hit_tokens,
            prefix_role_hit_tokens=prefix_role_hit_tokens,
            total_hit_tokens=hit_tokens,
            high_descendant_evictions=total_high_descendant_evictions,
            eviction_count=total_eviction_count,
            cold_deep_admission_opportunities=cold_deep_admission_opportunities,
            cold_deep_admissions=cold_deep_admissions,
            reuse_after_eviction_missed_blocks=reuse_after_eviction_missed_blocks,
            reuse_after_eviction_missed_tokens=reuse_after_eviction_missed_tokens,
            recompute_tokens=recompute_tokens,
        )
        shadow_price_metrics = admission_audit.shadow_price_metrics()
        return TrialMetrics(
            split=split,
            workload=workload,
            seed=seed,
            capacity_blocks=self.capacity_blocks,
            block_hit_rate=hit_blocks / total_blocks if total_blocks else 0.0,
            token_hit_rate=hit_tokens / total_tokens if total_tokens else 0.0,
            priority_weighted_token_hit_rate=(
                priority_weighted_hit_tokens / priority_weighted_total_tokens
                if priority_weighted_total_tokens
                else 0.0
            ),
            high_priority_token_hit_rate=(
                high_priority_hit_tokens / high_priority_tokens if high_priority_tokens else 0.0
            ),
            low_priority_token_hit_rate=(
                low_priority_hit_tokens / low_priority_tokens if low_priority_tokens else 0.0
            ),
            priority_request_fraction=high_priority_request_count / request_count,
            request_token_hit_rate_p10=_percentile(request_token_hit_rates, 10),
            request_token_hit_rate_p50=_percentile(request_token_hit_rates, 50),
            high_priority_request_token_hit_rate_p10=_percentile(
                high_priority_request_token_hit_rates,
                10,
            ),
            worst_quarter_token_hit_rate=min(quarter_hit_rates, default=0.0),
            final_quarter_token_hit_rate=quarter_hit_rates[-1] if quarter_hit_rates else 0.0,
            quarter_token_hit_rate_stddev=(
                pstdev(quarter_hit_rates) if len(quarter_hit_rates) > 1 else 0.0
            ),
            prefill_tokens_saved=hit_tokens,
            recompute_tokens=recompute_tokens,
            recompute_cost=recompute_cost,
            lookup_block_count=lookup_block_count,
            lookup_blocks_per_request=lookup_block_count / request_count,
            eviction_count=total_eviction_count,
            admission_count=admission_count,
            admission_score_count=admission_score_count,
            admission_rejection_count=admission_rejection_count,
            admission_rate=admission_count / max(1, admission_score_count),
            avoidable_admission_count=admission_audit.avoidable_admission_count,
            avoidable_admission_rate=(
                admission_audit.avoidable_admission_count / max(1, admission_audit.accepted_count)
            ),
            avoidable_admission_regret_tokens=(admission_audit.avoidable_admission_regret_tokens),
            avoidable_admission_regret_token_rate=(
                admission_audit.avoidable_admission_regret_tokens / max(1, total_tokens)
            ),
            avoidable_rejection_count=admission_audit.avoidable_rejection_count,
            avoidable_rejection_rate=(
                admission_audit.avoidable_rejection_count / max(1, admission_audit.rejected_count)
            ),
            avoidable_rejection_regret_tokens=(admission_audit.avoidable_rejection_regret_tokens),
            avoidable_rejection_regret_token_rate=(
                admission_audit.avoidable_rejection_regret_tokens / max(1, total_tokens)
            ),
            oracle_shadow_price_mean=shadow_price_metrics.get(
                "oracle_shadow_price_mean",
                0.0,
            ),
            oracle_shadow_price_stddev=shadow_price_metrics.get(
                "oracle_shadow_price_stddev",
                0.0,
            ),
            oracle_shadow_price_change_mean=shadow_price_metrics.get(
                "oracle_shadow_price_change_mean",
                0.0,
            ),
            oracle_shadow_price_change_p95=shadow_price_metrics.get(
                "oracle_shadow_price_change_p95",
                0.0,
            ),
            shadow_price_score_scale=shadow_price_metrics.get(
                "shadow_price_score_scale",
                0.0,
            ),
            shadow_price_tracking_rmse=shadow_price_metrics.get(
                "shadow_price_tracking_rmse",
                0.0,
            ),
            shadow_price_tracking_mae=shadow_price_metrics.get(
                "shadow_price_tracking_mae",
                0.0,
            ),
            shadow_price_tracking_bias=shadow_price_metrics.get(
                "shadow_price_tracking_bias",
                0.0,
            ),
            fast_shadow_price_decision_fraction=shadow_price_metrics.get(
                "fast_shadow_price_decision_fraction",
                0.0,
            ),
            fast_shadow_price_regret_share=shadow_price_metrics.get(
                "fast_shadow_price_regret_share",
                0.0,
            ),
            fast_shadow_price_regret_lift=shadow_price_metrics.get(
                "fast_shadow_price_regret_lift",
                0.0,
            ),
            shadow_price_change_regret_correlation=shadow_price_metrics.get(
                "shadow_price_change_regret_correlation",
                0.0,
            ),
            useful_admission_count=admission_accounting.useful_count,
            useful_admission_rate=(admission_accounting.useful_count / max(1, admission_count)),
            wasted_admission_count=admission_accounting.wasted_count,
            wasted_admission_rate=(admission_accounting.wasted_count / max(1, admission_count)),
            admitted_token_count=admission_accounting.admitted_tokens,
            useful_admission_token_count=admission_accounting.useful_tokens,
            useful_admission_token_rate=(
                admission_accounting.useful_tokens / max(1, admission_accounting.admitted_tokens)
            ),
            wasted_admission_token_count=admission_accounting.wasted_tokens,
            wasted_admission_token_rate=(
                admission_accounting.wasted_tokens / max(1, admission_accounting.admitted_tokens)
            ),
            admission_saved_tokens=admission_accounting.saved_tokens,
            admission_saved_tokens_per_admission=(
                admission_accounting.saved_tokens / max(1, admission_count)
            ),
            admission_token_utility=(
                admission_accounting.saved_tokens / max(1, admission_count * self.block_size_tokens)
            ),
            evicted_without_hit_count=admission_accounting.evicted_without_hit_count,
            evicted_without_hit_rate=(
                admission_accounting.evicted_without_hit_count / max(1, total_eviction_count)
            ),
            policy_bypass_tokens=policy_bypass_tokens,
            policy_bypass_token_rate=policy_bypass_token_rate,
            policy_underfill_rate=policy_underfill_rate,
            cache_churn_per_1k=total_eviction_count * 1000.0 / request_count,
            forced_bypass_count=forced_bypass_count,
            forced_bypass_tokens=forced_bypass_tokens,
            forced_bypass_token_rate=forced_bypass_tokens / max(1, total_tokens),
            short_reuse_after_eviction_missed_tokens=(short_reuse_after_eviction_missed_tokens),
            short_reuse_after_eviction_missed_token_rate=(
                short_reuse_after_eviction_missed_tokens / max(1, recompute_tokens)
            ),
            eviction_reuse_distance_p50=_percentile(eviction_reuse_distances, 50),
            eviction_reuse_distance_p95=_percentile(eviction_reuse_distances, 95),
            avoidable_eviction_count=total_avoidable_eviction_count,
            avoidable_eviction_rate=total_avoidable_eviction_count / max(1, total_eviction_count),
            avoidable_short_reuse_eviction_count=(total_avoidable_short_reuse_eviction_count),
            avoidable_short_reuse_eviction_rate=(
                total_avoidable_short_reuse_eviction_count / max(1, total_eviction_count)
            ),
            value_weighted_avoidable_eviction_count=(total_value_weighted_avoidable_eviction_count),
            value_weighted_avoidable_eviction_rate=(
                total_value_weighted_avoidable_eviction_count / max(1, total_eviction_count)
            ),
            value_weighted_avoidable_eviction_regret_tokens=(
                total_value_weighted_avoidable_eviction_regret_tokens
            ),
            value_weighted_avoidable_eviction_regret_token_rate=(
                total_value_weighted_avoidable_eviction_regret_tokens / max(1, total_tokens)
            ),
            tenant_count=len(tenant_rates),
            tenant_fairness_penalty=fairness_penalty,
            tenant_token_hit_rate_p10=_percentile(tenant_rates, 10),
            tenant_jain_fairness=_jain_fairness(tenant_rates),
            p50_latency_proxy=_percentile(latencies, 50),
            p95_latency_proxy=_percentile(latencies, 95),
            p99_latency_proxy=_percentile(latencies, 99),
            high_priority_p95_latency_proxy=_percentile(high_priority_latencies, 95),
            high_priority_p99_latency_proxy=_percentile(high_priority_latencies, 99),
            p95_recompute_cost=_percentile(recompute_costs, 95),
            recovery_request_count=recovery_request_count,
            recovery_token_hit_rate=recovery_hit_tokens / max(1, recovery_tokens),
            recovery_p95_latency_proxy=_percentile(recovery_latencies, 95),
            recovery_phase_count=len(recovery_phase_records),
            worst_recovery_phase_token_hit_rate=min(
                recovery_phase_hit_rates,
                default=0.0,
            ),
            final_recovery_phase_token_hit_rate=(
                recovery_phase_hit_rates[-1] if recovery_phase_hit_rates else 0.0
            ),
            worst_recovery_phase_p95_latency_proxy=max(
                recovery_phase_p95_latencies,
                default=0.0,
            ),
            memory_occupancy_mean=memory_occupancy_mean,
            memory_occupancy_peak=max(occupancies) if occupancies else 0,
            prefix_kv_occupancy_mean=prefix_kv_occupancy_mean,
            prefix_kv_occupancy_peak=max(prefix_occupancies) if prefix_occupancies else 0,
            decode_kv_occupancy_mean=decode_kv_occupancy_mean,
            decode_kv_occupancy_peak=max(decode_occupancies) if decode_occupancies else 0,
            decode_kv_blocks_requested=decode_capacity_outcome.decode_blocks_requested,
            decode_kv_blocks_allocated=decode_capacity_outcome.decode_blocks_allocated,
            decode_kv_allocation_failure_blocks=(
                decode_capacity_outcome.decode_allocation_failure_blocks
            ),
            decode_kv_allocation_failure_rate=(
                decode_capacity_outcome.decode_allocation_failure_blocks
                / max(1, decode_capacity_outcome.decode_blocks_requested)
            ),
            decode_pressure_eviction_count=decode_capacity_outcome.decode_pressure_evictions,
            decode_pressure_eviction_rate=(
                decode_capacity_outcome.decode_pressure_evictions / max(1, total_eviction_count)
            ),
            arrival_span_steps=(arrival_steps[-1] - arrival_steps[0] + 1 if arrival_steps else 0),
            active_request_count_peak=active_request_count_peak,
            max_prefill_cost=max_prefill_cost,
            scoring_fn_complexity=scoring_fn_complexity,
            matched_lengths=tuple(matched_lengths),
            structural_metrics=structural_metrics,
            admission_decisions=(
                tuple(admission_audit.decisions) if self.record_admission_diagnostics else ()
            ),
        )

    @property
    def resident_count(self) -> int:
        """Return the number of resident prefix blocks."""
        return len(self._resident_hashes)

    @property
    def decode_resident_count(self) -> int:
        """Return the number of resident decode blocks."""
        return self._decode_resident_blocks

    @property
    def total_resident_count(self) -> int:
        """Return total resident prefix and decode blocks."""
        return self.resident_count + self.decode_resident_count

    def match_resident_prefix(self, blocks: list[_BlockState]) -> int:
        """Return the largest root-contiguous resident prefix length."""
        matched = 0
        for block in blocks:
            if not block.resident:
                break
            matched += 1
        return matched

    def _emit_request_snapshot(
        self,
        *,
        index: int,
        now: int,
        request: WorkloadRequest,
        request_blocks: list[_BlockState],
        matched_len: int,
        hit_tokens: int,
        admissions: int,
        evictions: int,
        bypassed_tokens: int,
        latency: float,
        cumulative_hit_tokens: int,
        cumulative_total_tokens: int,
        cumulative_evictions: int,
    ) -> None:
        """Emit a request-complete snapshot when observability is enabled."""
        if self.observer is None:
            return
        request_hashes = {block.prefix_hash for block in request_blocks}
        hit_hashes = {block.prefix_hash for block in request_blocks[:matched_len]}
        cache = tuple(
            CacheBlockSnapshot(
                block_id=f"{block.prefix_hash:016x}",
                parent_id=(f"{block.parent_hash:016x}" if block.parent_hash is not None else None),
                depth=block.depth,
                token_count=block.token_count,
                prefix_role=block.prefix_role,
                tenant_id=block.tenant_id,
                hit_count=block.hit_count,
                active_ref_count=block.active_ref_count,
                descendant_count=self._descendant_counts.get(block.prefix_hash, 0),
                last_accessed_at=block.last_accessed_at,
                is_leaf=block.prefix_hash in self._leaf_hashes,
                in_request=block.prefix_hash in request_hashes,
                hit_this_request=block.prefix_hash in hit_hashes,
            )
            for block in sorted(
                (self.blocks[prefix_hash] for prefix_hash in self._resident_hashes),
                key=lambda value: (value.depth, value.prefix_hash),
            )
        )
        self.observer.on_request_complete(
            RequestSnapshot(
                index=index,
                now=now,
                request_id=request.info.request_id,
                tenant_id=request.info.tenant_id,
                priority=request.info.priority,
                request_type=request.info.request_type,
                prompt_blocks=len(request_blocks),
                prompt_tokens=request.info.prompt_length,
                matched_blocks=matched_len,
                hit_tokens=hit_tokens,
                admissions=admissions,
                evictions=evictions,
                bypassed_tokens=bypassed_tokens,
                resident_blocks=self.resident_count,
                capacity_blocks=self.capacity_blocks,
                latency_proxy=latency,
                cumulative_token_hit_rate=(cumulative_hit_tokens / max(1, cumulative_total_tokens)),
                cumulative_evictions=cumulative_evictions,
                cache=cache,
            )
        )

    def evict_block(self, prefix_hash: int) -> None:
        """Evict a resident leaf block; useful for direct simulator tests."""
        block = self.blocks[prefix_hash]
        if block.active_ref_count or block.resident_children:
            raise ValueError("only inactive resident leaves can be evicted")
        self._remove_resident(block)

    def _admit_block(
        self,
        policy: PrefixKVPolicy,
        block: _BlockState,
        now: int,
        duration: int,
        future_reuse: _FutureReuseTracker,
        audit_future_reuse: _FutureReuseTracker,
        admission_accounting: _AdmissionAccounting,
    ) -> _AdmissionOutcome:
        if block.resident:
            self._pin(block, now + duration)
            return _AdmissionOutcome(admitted=True)

        if block.parent_hash is not None:
            parent = self.blocks.get(block.parent_hash)
            if parent is None or not parent.resident:
                return _AdmissionOutcome()

        self._make_resident(block)
        block.last_accessed_at = now
        release_at = now + duration
        self._pin(block, release_at)
        capacity_outcome = _CapacityOutcome()
        while self.total_resident_count > self.capacity_blocks:
            outcome = self._evict_one(
                policy,
                now,
                future_reuse,
                audit_future_reuse,
                admission_accounting,
            )
            if outcome is None:
                self._unpin(block)
                self._cancel_release(block, release_at)
                self._remove_resident(block)
                return _AdmissionOutcome(capacity=capacity_outcome)
            capacity_outcome.merge(outcome)
        block.admission_tracked = True
        return _AdmissionOutcome(admitted=True, capacity=capacity_outcome)

    def _evict_one(
        self,
        policy: PrefixKVPolicy,
        now: int,
        future_reuse: _FutureReuseTracker,
        audit_future_reuse: _FutureReuseTracker,
        admission_accounting: _AdmissionAccounting,
    ) -> _CapacityOutcome | None:
        """Evict one legal prefix leaf using the candidate policy."""
        evictable = self._evictable_blocks()
        if not evictable:
            return None
        scored = []
        for candidate in evictable:
            info = self._info(candidate, now, future_reuse)
            scored.append(
                (
                    self._score(policy.score_eviction, info, now),
                    candidate.prefix_hash,
                    candidate,
                    info,
                )
            )
        if self.oracle_eviction:
            _, _, victim, _ = min(
                scored,
                key=lambda item: self._oracle_eviction_key(
                    item[2],
                    now,
                    audit_future_reuse,
                ),
            )
        else:
            _, _, victim, _ = max(scored, key=lambda item: (item[0], item[1]))
        victim_next_reuse = audit_future_reuse.next_distance(victim.prefix_hash, now)
        future_values = {
            candidate.prefix_hash: self._future_value_tokens(candidate, audit_future_reuse)
            for candidate in evictable
        }
        if self.eviction_decision_observer is not None:
            self.eviction_decision_observer.on_eviction_decision(
                EvictionDecisionSnapshot(
                    now=now,
                    victim_prefix_hash=victim.prefix_hash,
                    candidates=tuple(
                        EvictionCandidateSnapshot(
                            block=info,
                            score=score,
                            next_reuse_distance=audit_future_reuse.next_distance(
                                candidate.prefix_hash,
                                now,
                            ),
                        )
                        for score, _, candidate, info in scored
                    ),
                )
            )
        alternative_next_reuse = [
            audit_future_reuse.next_distance(candidate.prefix_hash, now)
            for candidate in evictable
            if candidate.prefix_hash != victim.prefix_hash
        ]
        furthest_alternative_reuse = max(
            (distance for distance in alternative_next_reuse if distance is not None),
            default=None,
        )
        outcome = _CapacityOutcome(evictions=1)
        minimum_future_value = min(future_values.values())
        value_regret = future_values[victim.prefix_hash] - minimum_future_value
        if value_regret > 0.0:
            outcome.value_weighted_avoidable_evictions = 1
            outcome.value_weighted_avoidable_eviction_regret_tokens = value_regret
        if (
            victim_next_reuse is not None
            and furthest_alternative_reuse is not None
            and victim_next_reuse < furthest_alternative_reuse
        ):
            outcome.avoidable_evictions = 1
            if (
                victim_next_reuse <= _SHORT_REUSE_DISTANCE_STEPS
                and furthest_alternative_reuse > _SHORT_REUSE_DISTANCE_STEPS
            ):
                outcome.avoidable_short_reuse_evictions = 1
        if self._descendant_counts.get(victim.prefix_hash, 0) >= _HIGH_DESCENDANT_MIN_COUNT:
            outcome.high_descendant_evictions = 1
        admission_accounting.record(victim, evicted=True)
        self._evicted_hashes.add(victim.prefix_hash)
        self._last_evicted_at[victim.prefix_hash] = now
        self._remove_resident(victim)
        return outcome

    def _admission_decision_values(
        self,
        block: _BlockState,
        audit_future_reuse: _FutureReuseTracker,
    ) -> tuple[float, float, bool]:
        """Return incoming value, cheapest displacement, and admission feasibility."""
        incoming_value = self._future_value_tokens(block, audit_future_reuse)
        required_evictions = max(0, self.total_resident_count + 1 - self.capacity_blocks)
        if required_evictions == 0:
            return incoming_value, 0.0, True
        candidate_values = sorted(
            self._future_value_tokens(candidate, audit_future_reuse)
            for candidate in self._admission_victim_candidates(block)
        )
        if len(candidate_values) < required_evictions:
            return incoming_value, 0.0, False
        return incoming_value, sum(candidate_values[:required_evictions]), True

    def _admission_victim_candidates(self, incoming: _BlockState) -> list[_BlockState]:
        """Return blocks that would remain legal victims after admitting `incoming`."""
        return [
            candidate
            for candidate in self._evictable_blocks()
            if candidate.prefix_hash != incoming.parent_hash
        ]

    @staticmethod
    def _future_value_tokens(
        block: _BlockState,
        audit_future_reuse: _FutureReuseTracker,
    ) -> float:
        """Return the oracle upper bound on future token hits for one block."""
        remaining_reuse = audit_future_reuse.remaining_count(block.prefix_hash)
        return block.token_count * max(0.0, remaining_reuse or 0.0)

    def _oracle_eviction_key(
        self,
        block: _BlockState,
        now: int,
        audit_future_reuse: _FutureReuseTracker,
    ) -> tuple[float, float, int]:
        """Rank legal victims by future value, then furthest next use."""
        next_distance = audit_future_reuse.next_distance(block.prefix_hash, now)
        tie_distance = math.inf if next_distance is None else next_distance
        return (
            self._future_value_tokens(block, audit_future_reuse),
            -tie_distance,
            block.prefix_hash,
        )

    def _advance_decodes(
        self,
        policy: PrefixKVPolicy,
        now: int,
        future_reuse: _FutureReuseTracker,
        audit_future_reuse: _FutureReuseTracker,
        admission_accounting: _AdmissionAccounting,
    ) -> _CapacityOutcome:
        """Advance in-flight decode KV through the current logical step."""
        outcome = _CapacityOutcome()
        if self.kv_capacity_mode != "shared":
            return outcome
        previous_step = getattr(self, "_decode_step", None)
        first_step = now if previous_step is None else previous_step + 1
        for step in range(first_step, now + 1):
            self._release_expired(step)
            retained_decodes = []
            for decode in self._active_decodes:
                if decode.release_at <= step:
                    self._decode_resident_blocks -= decode.allocated_blocks
                else:
                    retained_decodes.append(decode)
            self._active_decodes = retained_decodes
            outcome.merge(
                self._grow_decodes(
                    policy,
                    step,
                    future_reuse,
                    audit_future_reuse,
                    admission_accounting,
                )
            )
        self._decode_step = now
        return outcome

    def _start_decode(
        self,
        policy: PrefixKVPolicy,
        now: int,
        true_output_length: int,
        duration: int,
        future_reuse: _FutureReuseTracker,
        audit_future_reuse: _FutureReuseTracker,
        admission_accounting: _AdmissionAccounting,
    ) -> _CapacityOutcome:
        """Start one decode and allocate its first generated KV blocks."""
        if self.kv_capacity_mode != "shared":
            return _CapacityOutcome()
        self._active_decodes.append(
            _ActiveDecode(
                started_at=now,
                release_at=now + duration,
                total_tokens=max(0, true_output_length),
            )
        )
        return self._grow_decodes(
            policy,
            now,
            future_reuse,
            audit_future_reuse,
            admission_accounting,
        )

    def _grow_decodes(
        self,
        policy: PrefixKVPolicy,
        now: int,
        future_reuse: _FutureReuseTracker,
        audit_future_reuse: _FutureReuseTracker,
        admission_accounting: _AdmissionAccounting,
    ) -> _CapacityOutcome:
        """Allocate newly generated decode blocks at one logical step."""
        outcome = _CapacityOutcome()
        for decode in self._active_decodes:
            generated_tokens = min(
                decode.total_tokens,
                max(0, now - decode.started_at + 1) * self.active_tokens_per_step,
            )
            target_blocks = math.ceil(generated_tokens / self.block_size_tokens)
            requested_blocks = max(0, target_blocks - decode.attempted_blocks)
            decode.attempted_blocks = target_blocks
            allocation = self._allocate_decode_blocks(
                policy,
                now,
                requested_blocks,
                future_reuse,
                audit_future_reuse,
                admission_accounting,
            )
            decode.allocated_blocks += allocation.decode_blocks_allocated
            outcome.merge(allocation)
        return outcome

    def _allocate_decode_blocks(
        self,
        policy: PrefixKVPolicy,
        now: int,
        requested_blocks: int,
        future_reuse: _FutureReuseTracker,
        audit_future_reuse: _FutureReuseTracker,
        admission_accounting: _AdmissionAccounting,
    ) -> _CapacityOutcome:
        """Allocate non-evictable decode blocks, evicting prefix leaves as needed."""
        outcome = _CapacityOutcome(decode_blocks_requested=requested_blocks)
        if requested_blocks <= 0:
            return outcome
        self._decode_resident_blocks += requested_blocks
        while self.total_resident_count > self.capacity_blocks:
            eviction = self._evict_one(
                policy,
                now,
                future_reuse,
                audit_future_reuse,
                admission_accounting,
            )
            if eviction is None:
                break
            outcome.merge(eviction)
            outcome.decode_pressure_evictions += eviction.evictions
        overflow = max(0, self.total_resident_count - self.capacity_blocks)
        if overflow:
            self._decode_resident_blocks -= overflow
        outcome.decode_allocation_failure_blocks = overflow
        outcome.decode_blocks_allocated = requested_blocks - overflow
        return outcome

    def _evictable_blocks(self) -> list[_BlockState]:
        return [
            self.blocks[prefix_hash]
            for prefix_hash in self._leaf_hashes
            if self.blocks[prefix_hash].active_ref_count == 0
        ]

    def _materialize_chain(
        self,
        request: WorkloadRequest,
        now: int,
    ) -> list[_BlockState]:
        blocks: list[_BlockState] = []
        prefix_tokens: list[int] = []
        tokens = request.prompt_tokens or request.info.prompt_tokens
        for depth, start in enumerate(range(0, len(tokens), self.block_size_tokens), start=1):
            chunk = tokens[start : start + self.block_size_tokens]
            prefix_tokens.extend(chunk)
            prefix_hash = _stable_hash((request.info.tenant_id, tuple(prefix_tokens)))
            parent_hash = blocks[-1].prefix_hash if blocks else None
            if prefix_hash not in self.blocks:
                self.blocks[prefix_hash] = _BlockState(
                    prefix_hash=prefix_hash,
                    parent_hash=parent_hash,
                    depth=depth,
                    start_token=start,
                    end_token=start + len(chunk),
                    token_count=len(chunk),
                    prefix_role=_prefix_role(chunk),
                    tenant_id=request.info.tenant_id,
                    created_at=now,
                    last_accessed_at=now,
                )
                if parent_hash is not None:
                    self.blocks[parent_hash].known_children.add(prefix_hash)
                    ancestor_hash: int | None = parent_hash
                    while ancestor_hash is not None:
                        self._descendant_counts[ancestor_hash] = (
                            self._descendant_counts.get(ancestor_hash, 0) + 1
                        )
                        ancestor_hash = self.blocks[ancestor_hash].parent_hash
            block = self.blocks[prefix_hash]
            self._record_access(block, now)
            blocks.append(block)
        return blocks

    def _record_access(self, block: _BlockState, now: int) -> None:
        """Record online recurrence timing before candidate callbacks fire."""
        self._adjust_subtree_counter(self._subtree_access_counts, block, 1)
        previous = block.observed_accessed_at
        block.prev_last_accessed_at = previous
        block.last_access_gap = None if previous is None else max(0, now - previous)
        block.observed_accessed_at = now
        if block.last_access_gap is None:
            return
        gap = float(block.last_access_gap)
        if block.access_gap_mean is None or block.access_gap_mean_square is None:
            block.access_gap_mean = gap
            block.access_gap_mean_square = gap * gap
        else:
            alpha = _ACCESS_GAP_EW_ALPHA
            block.access_gap_mean += alpha * (gap - block.access_gap_mean)
            block.access_gap_mean_square += alpha * (gap * gap - block.access_gap_mean_square)
        block.access_gap_sample_count = min(2, block.access_gap_sample_count + 1)

    def _record_hit(self, block: _BlockState) -> None:
        """Record one hit in the online subtree aggregate."""
        self._adjust_subtree_counter(self._subtree_hit_counts, block, 1)

    def _adjust_subtree_counter(
        self,
        counts: dict[int, int],
        block: _BlockState,
        delta: int,
    ) -> None:
        """Apply a block contribution to itself and each known ancestor."""
        prefix_hash: int | None = block.prefix_hash
        while prefix_hash is not None:
            counts[prefix_hash] = max(0, counts.get(prefix_hash, 0) + delta)
            prefix_hash = self.blocks[prefix_hash].parent_hash

    def _make_resident(self, block: _BlockState) -> None:
        if block.resident:
            return
        block.resident = True
        block.admission_tracked = False
        block.resident_hit_count = 0
        self._resident_hashes.add(block.prefix_hash)
        self._leaf_hashes.add(block.prefix_hash)
        if block.parent_hash is not None and block.parent_hash in self.blocks:
            parent = self.blocks[block.parent_hash]
            if parent.resident:
                parent.resident_children.add(block.prefix_hash)
                self._leaf_hashes.discard(parent.prefix_hash)

    def _remove_resident(self, block: _BlockState) -> None:
        if not block.resident:
            return
        if block.parent_hash is not None and block.parent_hash in self.blocks:
            parent = self.blocks[block.parent_hash]
            parent.resident_children.discard(block.prefix_hash)
            if parent.resident and not parent.resident_children:
                self._leaf_hashes.add(parent.prefix_hash)
        block.resident = False
        block.admission_tracked = False
        block.resident_hit_count = 0
        self._resident_hashes.discard(block.prefix_hash)
        self._leaf_hashes.discard(block.prefix_hash)
        block.resident_children.clear()

    def _pin(self, block: _BlockState, release_at: int) -> None:
        block.active_ref_count += 1
        self._adjust_subtree_counter(self._subtree_active_ref_counts, block, 1)
        self._release_events.setdefault(release_at, []).append(block.prefix_hash)

    def _unpin(self, block: _BlockState) -> None:
        if block.active_ref_count <= 0:
            return
        block.active_ref_count -= 1
        self._adjust_subtree_counter(self._subtree_active_ref_counts, block, -1)

    def _cancel_release(self, block: _BlockState, release_at: int) -> None:
        events = self._release_events.get(release_at)
        if not events:
            return
        try:
            events.remove(block.prefix_hash)
        except ValueError:
            return
        if not events:
            self._release_events.pop(release_at, None)

    def _release_expired(self, now: int) -> None:
        for release_at in sorted([key for key in self._release_events if key <= now]):
            for prefix_hash in self._release_events.pop(release_at):
                block = self.blocks.get(prefix_hash)
                if block is not None:
                    self._unpin(block)

    def _info(
        self,
        block: _BlockState,
        now: int,
        future_reuse: _FutureReuseTracker,
    ) -> PrefixBlockInfo:
        return PrefixBlockInfo(
            block_id=block.block_id,
            prefix_hash=block.prefix_hash,
            parent_hash=block.parent_hash,
            depth=block.depth,
            start_token=block.start_token,
            end_token=block.end_token,
            token_count=block.token_count,
            tenant_id=block.tenant_id,
            created_at=block.created_at,
            last_accessed_at=block.last_accessed_at,
            hit_count=block.hit_count,
            descendant_count=self._descendant_counts.get(block.prefix_hash, 0),
            active_ref_count=block.active_ref_count,
            estimated_recompute_cost=self._estimated_recompute_cost(block),
            prev_last_accessed_at=block.prev_last_accessed_at,
            last_access_gap=block.last_access_gap,
            access_gap_mean=(block.access_gap_mean if block.access_gap_sample_count >= 2 else None),
            access_gap_var=(
                max(
                    0.0,
                    block.access_gap_mean_square - block.access_gap_mean**2,
                )
                if block.access_gap_sample_count >= 2
                and block.access_gap_mean is not None
                and block.access_gap_mean_square is not None
                else None
            ),
            subtree_hit_rate=(
                self._subtree_hit_counts.get(block.prefix_hash, 0)
                / max(1, self._subtree_access_counts.get(block.prefix_hash, 0))
            ),
            subtree_active_ref_count=self._subtree_active_ref_counts.get(
                block.prefix_hash,
                0,
            ),
            estimated_future_reuse=future_reuse.remaining_count(block.prefix_hash),
            estimated_next_reuse_distance=future_reuse.next_distance(block.prefix_hash, now),
        )

    def _estimated_recompute_cost(self, block: _BlockState) -> float:
        return block.end_token * self.prefill_cost_per_token

    def _score(self, func: Callable[[PrefixBlockInfo, int], float], *args) -> float:
        try:
            score = func(*args)
        except Exception as exc:  # pragma: no cover - exercised by tests
            raise InvalidCandidateError(f"{func.__name__} raised {type(exc).__name__}") from exc
        if isinstance(score, bool) or not isinstance(score, (float, int)):
            raise InvalidCandidateError(f"{func.__name__} returned non-numeric score")
        score = float(score)
        if not math.isfinite(score):
            raise InvalidCandidateError(f"{func.__name__} returned non-finite score")
        self._check_memory_limit()
        return score

    def _call_hook(self, func: Callable, *args) -> None:
        try:
            func(*args)
        except Exception as exc:  # pragma: no cover - defensive
            raise InvalidCandidateError(f"{func.__name__} raised {type(exc).__name__}") from exc
        self._check_memory_limit()

    def _check_memory_limit(self) -> None:
        if not self.max_memory_bytes or not tracemalloc.is_tracing():
            return
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        if peak_memory_bytes > self.max_memory_bytes:
            raise InvalidCandidateError(
                f"candidate used {peak_memory_bytes} bytes (> {self.max_memory_bytes})"
            )

    @staticmethod
    def _validate_policy(policy: PrefixKVPolicy) -> None:
        """Ensures missing hooks become structured invalid-candidate results."""
        required_methods = (
            "on_request_start",
            "score_admission",
            "score_eviction",
            "on_cache_hit",
            "on_cache_miss",
        )
        for method_name in required_methods:
            if not callable(getattr(policy, method_name, None)):
                raise InvalidCandidateError(f"policy must implement {method_name}()")

    @staticmethod
    def _tenant_hit_rates(
        tenant_hits: dict[int, int],
        tenant_tokens: dict[int, int],
    ) -> list[float]:
        return [
            tenant_hits.get(tenant, 0) / tokens
            for tenant, tokens in tenant_tokens.items()
            if tokens > 0
        ]


class PrefixKVCacheEvaluator:
    """Callable evaluator compatible with Levi-style candidate factories."""

    def __init__(
        self,
        config: EvaluatorConfig | None = None,
        *,
        splits: tuple[str, ...] = ("train", "validation"),
        expose_future_reuse: bool = False,
        simulator_factory: Callable[..., PrefixKVCacheSimulator] = (PrefixKVCacheSimulator),
        workload_builder: Callable[..., tuple[WorkloadRequest, ...]] | None = None,
        fixed_admission_factory: Callable[..., PrefixKVPolicy] | None = None,
        record_admission_diagnostics: bool = False,
        oracle_admission: bool = False,
        oracle_eviction: bool = False,
    ) -> None:
        self.config = config or EvaluatorConfig()
        self.splits = splits
        self.expose_future_reuse = expose_future_reuse
        self._simulator_factory = simulator_factory
        self._workload_builder = workload_builder or build_workload
        self._fixed_admission_factory = fixed_admission_factory
        self._record_admission_diagnostics = record_admission_diagnostics
        self._oracle_admission = oracle_admission
        self._oracle_eviction = oracle_eviction

    def __call__(
        self,
        factory: Callable[..., PrefixKVPolicy] | None = None,
        *,
        scoring_fn_complexity: int = 0,
    ) -> EvaluationResult:
        """Evaluate a policy factory across configured workloads."""
        factory = factory or baseline_lru_blocks
        trials: list[TrialMetrics] = []
        capacity_blocks_values = self.config.effective_capacity_blocks()
        prepared_streams = {}
        stream_records = []
        for workload in self.config.workload_configs(self.splits):
            for seed in self.config.seeds:
                actual_seed = seed + workload.seed_offset
                requests = self._workload_builder(
                    workload.family,
                    request_count=workload.request_count,
                    block_size_tokens=self.config.effective_workload_token_granularity(),
                    seed=actual_seed,
                )
                prepared_streams[(workload.split, workload.family, seed)] = (
                    actual_seed,
                    requests,
                )
                stream_records.append(
                    request_stream_fingerprint_record(
                        requests,
                        split=workload.split,
                        family=workload.family,
                        base_seed=seed,
                        seed_offset=workload.seed_offset,
                        actual_seed=actual_seed,
                    )
                )
        panel_sha = self._synthetic_panel_sha(
            capacity_blocks_values,
            stream_records,
        )
        for workload in self.config.workload_configs(self.splits):
            for capacity_blocks in capacity_blocks_values:
                for seed in self.config.seeds:
                    actual_seed, requests = prepared_streams[
                        (workload.split, workload.family, seed)
                    ]
                    trials.append(
                        self._run_trial(
                            factory,
                            requests,
                            split=workload.split,
                            workload=workload.family,
                            seed=actual_seed,
                            capacity_blocks=capacity_blocks,
                            scoring_fn_complexity=scoring_fn_complexity,
                        )
                    )

        return self._result_from_trials(
            trials,
            scoring_fn_complexity,
            panel_sha=panel_sha,
        )

    def evaluate_requests(
        self,
        factory: Callable[..., PrefixKVPolicy] | None,
        requests: Iterable[WorkloadRequest],
        *,
        workload: str = "trace_replay",
        split: str = "validation",
        seed: int = 0,
        scoring_fn_complexity: int = 0,
    ) -> EvaluationResult:
        """Evaluate a fixed metadata-derived request sequence."""
        factory = factory or baseline_lru_blocks
        request_tuple = tuple(requests)
        stream = request_stream_fingerprint_record(
            request_tuple,
            split=split,
            family=workload,
            base_seed=seed,
            seed_offset=0,
            actual_seed=seed,
        )
        panel_sha = panel_sha256(
            evaluation=self._panel_evaluation_metadata(
                self.config.effective_capacity_blocks(),
                splits=(split,),
                request_count=len(request_tuple),
                base_seeds=(seed,),
                family_request_multipliers={},
                stream_count=1,
            ),
            streams=(stream,),
        )
        trials = [
            self._run_trial(
                factory,
                request_tuple,
                split=split,
                workload=workload,
                seed=seed,
                capacity_blocks=capacity_blocks,
                scoring_fn_complexity=scoring_fn_complexity,
            )
            for capacity_blocks in self.config.effective_capacity_blocks()
        ]
        return self._result_from_trials(
            trials,
            scoring_fn_complexity,
            panel_sha=panel_sha,
        )

    def rescore_trials(
        self,
        trials: Iterable[TrialMetrics],
        *,
        scoring_fn_complexity: int = 0,
    ) -> EvaluationResult:
        """Reaggregate existing trials under this evaluator's score weights."""
        trial_list = list(trials)
        panel_shas = {trial.panel_sha256 for trial in trial_list}
        if "" in panel_shas or len(panel_shas) != 1:
            raise ValueError("rescoring requires trials from exactly one fingerprinted panel")
        return self._result_from_trials(
            trial_list,
            scoring_fn_complexity,
            panel_sha=next(iter(panel_shas)),
        )

    def _synthetic_panel_sha(
        self,
        capacity_blocks: tuple[int, ...],
        streams: list[dict[str, object]],
    ) -> str:
        """Return the manifest-compatible hash for this synthetic panel."""
        return panel_sha256(
            evaluation=self._panel_evaluation_metadata(
                capacity_blocks,
                splits=self.splits,
                request_count=self.config.request_count,
                base_seeds=self.config.seeds,
                family_request_multipliers=dict(
                    sorted(self.config.family_request_multipliers.items())
                ),
                stream_count=len(streams),
            ),
            streams=streams,
        )

    def _panel_evaluation_metadata(
        self,
        capacity_blocks: tuple[int, ...],
        *,
        splits: tuple[str, ...],
        request_count: int,
        base_seeds: tuple[int, ...],
        family_request_multipliers: dict[str, int],
        stream_count: int,
    ) -> dict[str, object]:
        """Return the canonical panel configuration used by workload manifests."""
        return {
            "splits": list(splits),
            "capacity_blocks": list(capacity_blocks),
            "capacity_tokens": [
                capacity * self.config.block_size_tokens for capacity in capacity_blocks
            ],
            "physical_block_size_tokens": self.config.block_size_tokens,
            "workload_token_granularity": (self.config.effective_workload_token_granularity()),
            "request_count": request_count,
            "base_seeds": list(base_seeds),
            "family_request_multipliers": family_request_multipliers,
            "stream_count": stream_count,
        }

    def _run_trial(
        self,
        factory: Callable[..., PrefixKVPolicy],
        requests: tuple[WorkloadRequest, ...],
        *,
        split: str,
        workload: str,
        seed: int,
        capacity_blocks: int,
        scoring_fn_complexity: int,
    ) -> TrialMetrics:
        simulator = self._simulator_factory(
            capacity_blocks=capacity_blocks,
            block_size_tokens=self.config.block_size_tokens,
            prefill_cost_per_token=self.config.prefill_cost_per_token,
            lookup_cost_per_block=self.config.lookup_cost_per_block,
            eviction_cost_per_block=self.config.eviction_cost_per_block,
            active_tokens_per_step=self.config.active_tokens_per_step,
            kv_capacity_mode=self.config.kv_capacity_mode,
            expose_future_reuse=self.expose_future_reuse,
            max_memory_bytes=self.config.max_memory_bytes,
            record_admission_diagnostics=self._record_admission_diagnostics,
            oracle_admission=self._oracle_admission,
            oracle_eviction=self._oracle_eviction,
        )
        tracing_already_started = tracemalloc.is_tracing()
        if tracing_already_started:
            tracemalloc.reset_peak()
        else:
            tracemalloc.start()
        try:
            try:
                policy = _build_policy(
                    factory,
                    capacity_blocks,
                    self.config.block_size_tokens,
                    self.config.policy_seed,
                )
                if self._fixed_admission_factory is not None:
                    admission_policy = _build_policy(
                        self._fixed_admission_factory,
                        capacity_blocks,
                        self.config.block_size_tokens,
                        self.config.policy_seed,
                    )
                    policy = _FixedAdmissionPolicy(
                        admission_policy=admission_policy,
                        eviction_policy=policy,
                    )
            except Exception as exc:
                return TrialMetrics(
                    split=split,
                    workload=workload,
                    seed=seed,
                    capacity_blocks=capacity_blocks,
                    scoring_fn_complexity=scoring_fn_complexity,
                    invalid=True,
                    invalid_reason=f"factory raised {type(exc).__name__}",
                )
            trial = simulator.run(
                policy,
                requests,
                split=split,
                workload=workload,
                seed=seed,
                scoring_fn_complexity=scoring_fn_complexity,
            )
            _, peak_memory_bytes = tracemalloc.get_traced_memory()
            if self.config.max_memory_bytes and peak_memory_bytes > self.config.max_memory_bytes:
                return TrialMetrics(
                    split=split,
                    workload=workload,
                    seed=seed,
                    capacity_blocks=capacity_blocks,
                    scoring_fn_complexity=scoring_fn_complexity,
                    invalid=True,
                    invalid_reason=(
                        f"candidate used {peak_memory_bytes} bytes "
                        f"(> {self.config.max_memory_bytes})"
                    ),
                )
            return trial
        finally:
            if not tracing_already_started:
                tracemalloc.stop()

    def _result_from_trials(
        self,
        trials: list[TrialMetrics],
        scoring_fn_complexity: int,
        *,
        panel_sha: str,
    ) -> EvaluationResult:
        for trial in trials:
            trial.panel_sha256 = panel_sha
        context_sha = evaluation_context_sha256(
            verifier_version=self.config.verifier_version,
            evaluator_config=self.config.model_dump(mode="json"),
            panel_sha=panel_sha,
        )
        invalid_fraction = (
            sum(1 for trial in trials if trial.invalid) / len(trials) if trials else 1.0
        )
        selection_trials = [trial for trial in trials if trial.split != "probe"]
        if not selection_trials:
            selection_trials = trials
        selection_invalid_fraction = (
            sum(1 for trial in selection_trials if trial.invalid) / len(selection_trials)
            if selection_trials
            else 1.0
        )
        split_metrics = _aggregate_by((trial.split for trial in trials), trials)
        workload_metrics = _aggregate_by(
            (f"{trial.split}/{trial.workload}" for trial in trials), trials
        )
        capacity_metrics = _aggregate_by(
            (f"capacity_{trial.capacity_blocks}" for trial in trials), trials
        )
        score_breakdown = self._score_breakdown(
            trials,
            selection_invalid_fraction,
            scoring_fn_complexity,
        )
        return EvaluationResult(
            verifier_version=self.config.verifier_version,
            evaluation_context_sha256=context_sha,
            panel_sha256=panel_sha,
            combined_score=score_breakdown["combined_score"],
            success=selection_invalid_fraction == 0.0,
            invalid_fraction=selection_invalid_fraction,
            split_metrics=split_metrics,
            workload_metrics=workload_metrics,
            capacity_metrics=capacity_metrics,
            candidate_metadata={
                "verifier_version": self.config.verifier_version,
                "evaluation_context_sha256": context_sha,
                "panel_sha256": panel_sha,
                "capacity_blocks": self.config.capacity_blocks,
                "capacity_sweep_blocks": ",".join(
                    str(value) for value in self.config.effective_capacity_blocks()
                ),
                "capacity_sweep_tokens": ",".join(
                    str(value) for value in self.config.effective_capacity_tokens()
                ),
                "block_size_tokens": self.config.block_size_tokens,
                "kv_capacity_mode": self.config.kv_capacity_mode,
                "workload_token_granularity": self.config.effective_workload_token_granularity(),
                "policy_seed": self.config.policy_seed,
                "scoring_fn_complexity": scoring_fn_complexity,
                "churn_weight": self.config.churn_weight,
                "underfill_weight": self.config.underfill_weight,
                "fairness_weight": self.config.fairness_weight,
                "complexity_weight": self.config.k_complex,
                "complexity_exponent": self.config.complexity_exponent,
                "complexity_mode": (
                    "form_aware" if self.config.form_aware_complexity else "legacy_ast_nodes"
                ),
                "min_workload_weight": self.config.min_workload_weight,
                "min_seed_weight": self.config.min_seed_weight,
                "request_tail_weight": self.config.request_tail_weight,
                "worst_window_weight": self.config.worst_window_weight,
                "priority_hit_weight": self.config.priority_hit_weight,
                "wasted_admission_weight": self.config.wasted_admission_weight,
                "wasted_admission_metric": "wasted_admission_token_rate",
                "admission_utility_weight": self.config.admission_utility_weight,
                "avoidable_eviction_weight": self.config.avoidable_eviction_weight,
                "latency_norm": self.config.latency_norm,
                "latency_norm_scope": (
                    "configured" if self.config.latency_norm > 0.0 else "workload_capacity"
                ),
                "expose_future_reuse": self.expose_future_reuse,
                "oracle_admission": self._oracle_admission,
                "oracle_eviction": self._oracle_eviction,
                "fixed_admission_policy": self.config.fixed_admission_policy or "",
                "reporting_invalid_fraction": invalid_fraction,
                "selection_invalid_fraction": selection_invalid_fraction,
            },
            score_breakdown=score_breakdown,
            trials=tuple(trials),
        )

    def _score_trials(
        self,
        trials: list[TrialMetrics],
        invalid_fraction: float,
        complexity: int,
    ) -> float:
        return self._score_breakdown(
            trials,
            invalid_fraction,
            complexity,
        )["combined_score"]

    def _score_breakdown(
        self,
        trials: list[TrialMetrics],
        invalid_fraction: float,
        complexity: int,
    ) -> dict[str, float]:
        """Returns the combined score and its top-level weighted components."""
        if invalid_fraction > 0.0:
            combined_score = (
                self.config.v_min - 1.0 - self.config.invalid_surcharge * invalid_fraction
            )
            return {
                "combined_score": combined_score,
                "invalid_fraction": invalid_fraction,
                "invalid_surcharge_cost": (self.config.invalid_surcharge * invalid_fraction),
            }
        validation = [trial for trial in trials if trial.split == "validation"]
        if not validation:
            validation = [trial for trial in trials if trial.split not in {"hidden", "probe"}]
        if not validation:
            validation = [trial for trial in trials if trial.split == "hidden"]
        if not validation:
            validation = [trial for trial in trials if trial.split == "probe"]
        by_workload_capacity: dict[tuple[str, int], list[TrialMetrics]] = {}
        for trial in validation:
            by_workload_capacity.setdefault((trial.workload, trial.capacity_blocks), []).append(
                trial
            )
        workload_scores = []
        min_seed_weight = min(1.0, max(0.0, self.config.min_seed_weight))
        for workload_trials in by_workload_capacity.values():
            average_score = _workload_base_score(
                workload_trials,
                token_weight=self.config.w_avg_tok,
                block_weight=self.config.w_avg_blk,
                request_tail_weight=self.config.request_tail_weight,
                worst_window_weight=self.config.worst_window_weight,
                priority_hit_weight=self.config.priority_hit_weight,
                wasted_admission_weight=self.config.wasted_admission_weight,
                admission_utility_weight=self.config.admission_utility_weight,
                avoidable_eviction_weight=self.config.avoidable_eviction_weight,
                latency_weight=self.config.latency_weight,
                latency_cap=self.config.latency_cap,
                latency_norm=self.config.latency_norm,
            )
            seed_floor = min(
                _workload_base_score(
                    [trial],
                    token_weight=self.config.w_avg_tok,
                    block_weight=self.config.w_avg_blk,
                    request_tail_weight=self.config.request_tail_weight,
                    worst_window_weight=self.config.worst_window_weight,
                    priority_hit_weight=self.config.priority_hit_weight,
                    wasted_admission_weight=self.config.wasted_admission_weight,
                    admission_utility_weight=self.config.admission_utility_weight,
                    avoidable_eviction_weight=self.config.avoidable_eviction_weight,
                    latency_weight=self.config.latency_weight,
                    latency_cap=self.config.latency_cap,
                    latency_norm=self.config.latency_norm,
                )
                for trial in workload_trials
            )
            workload_scores.append(
                (1.0 - min_seed_weight) * average_score + min_seed_weight * seed_floor
            )
        mean_score = mean(workload_scores) if workload_scores else 0.0
        min_workload_score = min(workload_scores) if workload_scores else 0.0
        churn = mean(trial.cache_churn_per_1k for trial in validation) if validation else 0.0
        underfill = mean(trial.policy_underfill_rate for trial in validation) if validation else 0.0
        fairness = (
            mean(
                trial.tenant_fairness_penalty
                for trial in validation
                if trial.workload == "multi_tenant_skew"
            )
            if any(trial.workload == "multi_tenant_skew" for trial in validation)
            else 0.0
        )
        churn_cost = min(self.config.churn_cap, self.config.churn_weight * churn)
        underfill_cost = min(
            self.config.underfill_cap,
            self.config.underfill_weight * underfill,
        )
        fairness_cost = min(
            self.config.fairness_cap,
            self.config.fairness_weight * fairness,
        )
        complexity_cost = self.config.k_complex * complexity**self.config.complexity_exponent
        min_workload_contribution = self.config.min_workload_weight * min_workload_score
        combined_score = (
            mean_score
            + min_workload_contribution
            - churn_cost
            - underfill_cost
            - fairness_cost
            - complexity_cost
        )
        return {
            "combined_score": combined_score,
            "mean_workload_score": mean_score,
            "min_workload_score": min_workload_score,
            "min_workload_contribution": min_workload_contribution,
            "churn_cost": churn_cost,
            "policy_underfill_rate": underfill,
            "underfill_cost": underfill_cost,
            "fairness_cost": fairness_cost,
            "complexity_cost": complexity_cost,
            "validation_trial_count": float(len(validation)),
        }


def _build_policy(
    factory: Callable[..., PrefixKVPolicy],
    capacity_blocks: int,
    block_size_tokens: int,
    seed: int,
) -> PrefixKVPolicy:
    argument_options = (
        (capacity_blocks, block_size_tokens, seed),
        (capacity_blocks, block_size_tokens),
        (),
    )
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(*argument_options[0])

    for args in argument_options:
        try:
            signature.bind(*args)
        except TypeError:
            continue
        return factory(*args)
    raise TypeError(
        "candidate factory must accept (capacity_blocks, block_size_tokens, seed), "
        "(capacity_blocks, block_size_tokens), or no arguments"
    )


class _FixedAdmissionPolicy:
    """Uses one policy for admission and another policy for eviction."""

    _REQUIRED_METHODS = (
        "on_request_start",
        "score_admission",
        "score_eviction",
        "on_cache_hit",
        "on_cache_miss",
    )

    def __init__(
        self,
        *,
        admission_policy: PrefixKVPolicy,
        eviction_policy: PrefixKVPolicy,
    ) -> None:
        for policy in (admission_policy, eviction_policy):
            for method_name in self._REQUIRED_METHODS:
                if not callable(getattr(policy, method_name, None)):
                    raise TypeError(f"policy must implement {method_name}()")
        self._admission_policy = admission_policy
        self._eviction_policy = eviction_policy

    def on_request_start(self, request: RequestInfo, now: int) -> None:
        self._admission_policy.on_request_start(request, now)
        self._eviction_policy.on_request_start(request, now)

    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return self._admission_policy.score_admission(block, now)

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return self._eviction_policy.score_eviction(block, now)

    def on_cache_hit(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        self._admission_policy.on_cache_hit(block, request, now)
        self._eviction_policy.on_cache_hit(block, request, now)

    def on_cache_miss(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        self._admission_policy.on_cache_miss(block, request, now)
        self._eviction_policy.on_cache_miss(block, request, now)
