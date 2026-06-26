"""Typed result models and metric aggregation schema."""

from __future__ import annotations

from dataclasses import dataclass, field

MEAN_AGGREGATION_FIELDS = (
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
    "quarter_token_hit_rate_stddev",
    "prefill_tokens_saved",
    "recompute_tokens",
    "recompute_cost",
    "lookup_block_count",
    "lookup_blocks_per_request",
    "eviction_count",
    "admission_count",
    "admission_score_count",
    "admission_rejection_count",
    "admission_rate",
    "avoidable_admission_count",
    "avoidable_admission_rate",
    "avoidable_admission_regret_tokens",
    "avoidable_admission_regret_token_rate",
    "avoidable_rejection_count",
    "avoidable_rejection_rate",
    "avoidable_rejection_regret_tokens",
    "avoidable_rejection_regret_token_rate",
    "oracle_shadow_price_mean",
    "oracle_shadow_price_stddev",
    "oracle_shadow_price_change_mean",
    "oracle_shadow_price_change_p95",
    "shadow_price_score_scale",
    "shadow_price_tracking_rmse",
    "shadow_price_tracking_mae",
    "shadow_price_tracking_bias",
    "fast_shadow_price_decision_fraction",
    "fast_shadow_price_regret_share",
    "fast_shadow_price_regret_lift",
    "shadow_price_change_regret_correlation",
    "useful_admission_count",
    "useful_admission_rate",
    "wasted_admission_count",
    "wasted_admission_rate",
    "admitted_token_count",
    "useful_admission_token_count",
    "useful_admission_token_rate",
    "wasted_admission_token_count",
    "wasted_admission_token_rate",
    "admission_saved_tokens",
    "admission_saved_tokens_per_admission",
    "admission_token_utility",
    "evicted_without_hit_count",
    "evicted_without_hit_rate",
    "policy_bypass_tokens",
    "policy_bypass_token_rate",
    "policy_underfill_rate",
    "cache_churn_per_1k",
    "forced_bypass_count",
    "forced_bypass_tokens",
    "forced_bypass_token_rate",
    "short_reuse_after_eviction_missed_tokens",
    "short_reuse_after_eviction_missed_token_rate",
    "eviction_reuse_distance_p50",
    "eviction_reuse_distance_p95",
    "avoidable_eviction_count",
    "avoidable_eviction_rate",
    "avoidable_short_reuse_eviction_count",
    "avoidable_short_reuse_eviction_rate",
    "value_weighted_avoidable_eviction_count",
    "value_weighted_avoidable_eviction_rate",
    "value_weighted_avoidable_eviction_regret_tokens",
    "value_weighted_avoidable_eviction_regret_token_rate",
    "tenant_count",
    "tenant_fairness_penalty",
    "tenant_token_hit_rate_p10",
    "tenant_jain_fairness",
    "p50_latency_proxy",
    "p95_latency_proxy",
    "p99_latency_proxy",
    "high_priority_p95_latency_proxy",
    "high_priority_p99_latency_proxy",
    "p95_recompute_cost",
    "recovery_request_count",
    "recovery_token_hit_rate",
    "recovery_p95_latency_proxy",
    "recovery_phase_count",
    "worst_recovery_phase_token_hit_rate",
    "final_recovery_phase_token_hit_rate",
    "worst_recovery_phase_p95_latency_proxy",
    "memory_occupancy_mean",
    "prefix_kv_occupancy_mean",
    "decode_kv_occupancy_mean",
    "decode_kv_blocks_requested",
    "decode_kv_blocks_allocated",
    "decode_kv_allocation_failure_blocks",
    "decode_kv_allocation_failure_rate",
    "decode_pressure_eviction_count",
    "decode_pressure_eviction_rate",
    "arrival_span_steps",
    "max_prefill_cost",
    "scoring_fn_complexity",
)

MAX_AGGREGATION_FIELDS = (
    "memory_occupancy_peak",
    "prefix_kv_occupancy_peak",
    "decode_kv_occupancy_peak",
    "active_request_count_peak",
)

SERIALIZED_TRIAL_FIELDS = (
    "capacity_blocks",
    *MEAN_AGGREGATION_FIELDS,
    *MAX_AGGREGATION_FIELDS,
    "invalid",
    "invalid_reason",
)


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
        metrics = {name: getattr(self, name) for name in SERIALIZED_TRIAL_FIELDS}
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
