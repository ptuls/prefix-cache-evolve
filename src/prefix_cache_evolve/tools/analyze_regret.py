"""Test whether admission-side regret dominates eviction-side regret."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable

import click

from prefix_cache_evolve.evaluator_entry import load_candidate_factory
from prefix_cache_evolve.evaluators.baselines import (
    baseline_cost_aware_lru,
    baseline_lfu_blocks,
    baseline_lru_blocks,
    baseline_no_cache,
    baseline_oracle_future_reuse,
    baseline_tinylfu_lru,
    baseline_vllm_apc,
)
from prefix_cache_evolve.evaluators.contracts import PrefixKVPolicy
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    AdmissionDecisionDiagnostic,
    EvaluatorConfig,
    PrefixKVCacheEvaluator,
    TrialMetrics,
)
from prefix_cache_evolve.evaluators.scoring import workload_base_score
from prefix_cache_evolve.evaluators.utilities import percentile
from prefix_cache_evolve.evaluators.verifier import (
    require_single_score_identity,
)
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents import (
    build_current_incumbent as build_production_incumbent,
)
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents import (
    build_discovery_incumbent as build_incumbent,
)
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents import (
    incumbent_record,
)
from prefix_cache_evolve.problems.prefix_kv_cache.seeds.structured_recurrence import (
    build_candidate as build_structured_recurrence,
)
from prefix_cache_evolve.problems.prefix_kv_cache.seeds.structured_seed import (
    build_candidate as build_structured_seed,
)

_DEFAULT_SPLITS = ("train", "validation", "probe", "hidden")
build_compact_seed = incumbent_record("historical_compact_20260607").load_factory()


@dataclass(frozen=True, slots=True)
class AdmissionPolicySpec:
    """One distinct admission rule used in controlled policy crosses."""

    name: str
    factory: Callable[..., PrefixKVPolicy]
    aliases: tuple[str, ...] = ()
    reporting_only: bool = False


@dataclass(frozen=True, slots=True)
class EvictionPolicySpec:
    """One eviction rule crossed with every distinct admission rule."""

    name: str
    factory: Callable[..., PrefixKVPolicy]
    reporting_only: bool = False


ADMISSION_POLICY_SPECS = (
    AdmissionPolicySpec("reject_all", baseline_no_cache, aliases=("no_cache",)),
    AdmissionPolicySpec(
        "admit_all",
        baseline_lru_blocks,
        aliases=(
            "lru",
            "sglang_radix_attention",
            "lfu",
            "depth_prefer_shallow",
            "recompute_greedy",
            "cost_aware_lru",
            "prefix_fanout",
            "prefix_anchor",
            "tenant_fair_lru",
            "future_reuse_heuristic",
            "initial_hybrid",
        ),
    ),
    AdmissionPolicySpec("full_blocks_only", baseline_vllm_apc, aliases=("vllm_apc",)),
    AdmissionPolicySpec("tinylfu", baseline_tinylfu_lru, aliases=("tinylfu_lru",)),
    AdmissionPolicySpec("compact_seed", build_compact_seed),
    AdmissionPolicySpec("structured_seed", build_structured_seed),
    AdmissionPolicySpec("structured_recurrence", build_structured_recurrence),
    AdmissionPolicySpec("pressure_aware_incumbent", build_incumbent),
    AdmissionPolicySpec(
        "oracle_future_reuse",
        baseline_oracle_future_reuse,
        reporting_only=True,
    ),
)

EVICTION_POLICY_SPECS = (
    EvictionPolicySpec("lru", baseline_lru_blocks),
    EvictionPolicySpec("lfu", baseline_lfu_blocks),
    EvictionPolicySpec("cost_aware_lru", baseline_cost_aware_lru),
    EvictionPolicySpec("incumbent_value_aware", build_incumbent),
    EvictionPolicySpec(
        "oracle_next_use",
        baseline_oracle_future_reuse,
        reporting_only=True,
    ),
)


def _trial_row(trial: TrialMetrics) -> dict[str, object]:
    """Return the regret decomposition for one workload-capacity-seed group."""
    admission_regret = (
        trial.avoidable_admission_regret_tokens + trial.avoidable_rejection_regret_tokens
    )
    eviction_regret = trial.value_weighted_avoidable_eviction_regret_tokens
    total_regret = admission_regret + eviction_regret
    admission_regret_per_decision = admission_regret / max(1, trial.admission_score_count)
    eviction_regret_per_decision = eviction_regret / max(1, trial.eviction_count)
    return {
        "group": (
            f"{trial.split}/{trial.workload}/capacity_{trial.capacity_blocks}/seed_{trial.seed}"
        ),
        "split": trial.split,
        "workload": trial.workload,
        "capacity_blocks": trial.capacity_blocks,
        "seed": trial.seed,
        "invalid": trial.invalid,
        "invalid_reason": trial.invalid_reason,
        "block_hit_rate": trial.block_hit_rate,
        "token_hit_rate": trial.token_hit_rate,
        "prefill_tokens_saved": trial.prefill_tokens_saved,
        "p95_latency_proxy": trial.p95_latency_proxy,
        "admission_score_count": trial.admission_score_count,
        "admission_count": trial.admission_count,
        "admission_rejection_count": trial.admission_rejection_count,
        "eviction_count": trial.eviction_count,
        "avoidable_admission_count": trial.avoidable_admission_count,
        "avoidable_admission_regret_tokens": trial.avoidable_admission_regret_tokens,
        "avoidable_rejection_count": trial.avoidable_rejection_count,
        "avoidable_rejection_regret_tokens": trial.avoidable_rejection_regret_tokens,
        "value_weighted_avoidable_eviction_count": (trial.value_weighted_avoidable_eviction_count),
        "value_weighted_avoidable_eviction_regret_tokens": eviction_regret,
        "admission_side_regret_tokens": admission_regret,
        "eviction_side_regret_tokens": eviction_regret,
        "total_regret_tokens": total_regret,
        "admission_regret_share": admission_regret / total_regret if total_regret else 0.0,
        "admission_regret_tokens_per_decision": admission_regret_per_decision,
        "eviction_regret_tokens_per_decision": eviction_regret_per_decision,
        "admission_minus_eviction_regret_tokens": admission_regret - eviction_regret,
        "admission_dominates": admission_regret > eviction_regret,
        "decision_normalized_admission_dominates": (
            admission_regret_per_decision > eviction_regret_per_decision
        ),
    }


def _summarize_groups(groups: Iterable[dict[str, object]]) -> dict[str, object]:
    """Summarize strict admission dominance over valid regretful groups."""
    groups = list(groups)
    valid = [group for group in groups if not group["invalid"]]
    invalid = [group for group in groups if group["invalid"]]
    regretful = [group for group in valid if float(group["total_regret_tokens"]) > 0.0]
    zero_regret = [group for group in valid if float(group["total_regret_tokens"]) == 0.0]
    admission_dominant = [group for group in regretful if group["admission_dominates"]]
    decision_normalized_admission_dominant = [
        group for group in regretful if group["decision_normalized_admission_dominates"]
    ]
    eviction_dominant = [
        group
        for group in regretful
        if float(group["admission_side_regret_tokens"])
        < float(group["eviction_side_regret_tokens"])
    ]
    tied = [
        group
        for group in regretful
        if float(group["admission_side_regret_tokens"])
        == float(group["eviction_side_regret_tokens"])
    ]
    admission_regret = sum(float(group["admission_side_regret_tokens"]) for group in valid)
    eviction_regret = sum(float(group["eviction_side_regret_tokens"]) for group in valid)
    admission_decisions = sum(int(group["admission_score_count"]) for group in valid)
    eviction_decisions = sum(int(group["eviction_count"]) for group in valid)
    total_regret = admission_regret + eviction_regret
    margins = [float(group["admission_minus_eviction_regret_tokens"]) for group in regretful]
    mean_token_hit_rate = (
        sum(float(group["token_hit_rate"]) for group in valid) / len(valid) if valid else 0.0
    )
    mean_block_hit_rate = (
        sum(float(group["block_hit_rate"]) for group in valid) / len(valid) if valid else 0.0
    )
    mean_p95_latency = (
        sum(float(group["p95_latency_proxy"]) for group in valid) / len(valid) if valid else 0.0
    )
    uniform = bool(regretful) and len(admission_dominant) == len(regretful) and not invalid
    if invalid:
        verdict = "inconclusive_invalid_groups"
    elif not regretful:
        verdict = "inconclusive_no_regret"
    elif uniform:
        verdict = "supported"
    else:
        verdict = "falsified"
    return {
        "verdict": verdict,
        "group_count": len(groups),
        "valid_group_count": len(valid),
        "invalid_group_count": len(invalid),
        "regretful_group_count": len(regretful),
        "zero_regret_group_count": len(zero_regret),
        "admission_dominant_group_count": len(admission_dominant),
        "eviction_dominant_group_count": len(eviction_dominant),
        "tied_regretful_group_count": len(tied),
        "admission_dominance_rate": (
            len(admission_dominant) / len(regretful) if regretful else 0.0
        ),
        "decision_normalized_admission_dominant_group_count": len(
            decision_normalized_admission_dominant
        ),
        "decision_normalized_admission_dominance_rate": (
            len(decision_normalized_admission_dominant) / len(regretful) if regretful else 0.0
        ),
        "uniform_admission_dominance": uniform,
        "mean_token_hit_rate": mean_token_hit_rate,
        "mean_block_hit_rate": mean_block_hit_rate,
        "total_prefill_tokens_saved": sum(float(group["prefill_tokens_saved"]) for group in valid),
        "mean_p95_latency_proxy": mean_p95_latency,
        "aggregate_admission_decision_count": admission_decisions,
        "aggregate_eviction_decision_count": eviction_decisions,
        "aggregate_admission_side_regret_tokens": admission_regret,
        "aggregate_eviction_side_regret_tokens": eviction_regret,
        "aggregate_admission_regret_tokens_per_decision": (
            admission_regret / admission_decisions if admission_decisions else 0.0
        ),
        "aggregate_eviction_regret_tokens_per_decision": (
            eviction_regret / eviction_decisions if eviction_decisions else 0.0
        ),
        "aggregate_admission_regret_share": (
            admission_regret / total_regret if total_regret else 0.0
        ),
        "aggregate_admission_minus_eviction_regret_tokens": (admission_regret - eviction_regret),
        "median_group_admission_minus_eviction_regret_tokens": (
            median(margins) if margins else 0.0
        ),
    }


def _group_summaries(
    groups: list[dict[str, object]],
    key_fn: Callable[[dict[str, object]], str],
) -> dict[str, dict[str, object]]:
    """Summarize groups under stable labels."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for group in groups:
        grouped.setdefault(key_fn(group), []).append(group)
    return {key: _summarize_groups(values) for key, values in sorted(grouped.items())}


def run_analysis(
    config_path: Path,
    *,
    request_count: int | None = None,
    seeds: tuple[int, ...] | None = None,
    splits: tuple[str, ...] = _DEFAULT_SPLITS,
    workloads: tuple[str, ...] | None = None,
    factory: Callable[..., PrefixKVPolicy] = build_incumbent,
    policy_name: str = "pressure_aware_incumbent",
    fixed_admission_factory: Callable[..., PrefixKVPolicy] | None = None,
    expose_future_reuse: bool = False,
) -> dict[str, object]:
    """Run the local-oracle regret audit over workload-capacity-seed groups."""
    config = load_evaluator_config(config_path)
    updates: dict[str, object] = {}
    if request_count is not None:
        updates["request_count"] = request_count
    if seeds is not None:
        updates["seeds"] = seeds
    if workloads is not None:
        for split in splits:
            updates[f"{split}_families"] = workloads
        updates["family_request_multipliers"] = {}
    if updates:
        config = config.with_updates(**updates)

    result = PrefixKVCacheEvaluator(
        config,
        splits=splits,
        fixed_admission_factory=fixed_admission_factory,
        expose_future_reuse=expose_future_reuse,
    )(factory)
    groups = [
        {
            "verifier_version": result.verifier_version,
            "evaluation_context_sha256": result.evaluation_context_sha256,
            "panel_sha256": result.panel_sha256,
            **_trial_row(trial),
        }
        for trial in result.trials
    ]
    return {
        "schema": "prefix-kv-cache-admission-eviction-regret-audit-v1",
        "verifier_version": result.verifier_version,
        "evaluation_context_sha256": result.evaluation_context_sha256,
        "panel_sha256": result.panel_sha256,
        "config": str(config_path),
        "policy": policy_name,
        "request_count": config.request_count,
        "seeds": list(config.seeds),
        "capacity_blocks": list(config.effective_capacity_blocks()),
        "splits": list(splits),
        "workloads": list(workloads) if workloads is not None else None,
        "falsification_rule": (
            "The universal claim passes only when admission-side regret is strictly greater "
            "than eviction-side regret in every valid group with nonzero regret."
        ),
        "value_definition": (
            "The future-value surrogate upper bound equals block token count multiplied by "
            "remaining request occurrences after the current request."
        ),
        "summary": _summarize_groups(groups),
        "by_split": _group_summaries(groups, lambda group: str(group["split"])),
        "by_workload": _group_summaries(
            groups,
            lambda group: f"{group['split']}/{group['workload']}",
        ),
        "groups": groups,
    }


def _shadow_trial_row(trial: TrialMetrics) -> dict[str, object]:
    """Return calibrated water-level tracking diagnostics for one replay."""
    decisions = [decision for decision in trial.admission_decisions if decision.feasible]
    scale = trial.shadow_price_score_scale
    oracle_prices = [decision.oracle_shadow_price for decision in decisions]
    changes = [
        0.0,
        *(abs(current - previous) for previous, current in zip(oracle_prices, oracle_prices[1:])),
    ]
    positive_changes = [change for change in changes if change > 0.0]
    fast_threshold = percentile(positive_changes, 75) if positive_changes else float("inf")
    trajectory = [
        _shadow_decision_row(
            decision,
            scale=scale,
            oracle_change=change,
            fast_change=change >= fast_threshold,
        )
        for decision, change in zip(decisions, changes, strict=True)
    ]
    regretful = [decision for decision in trajectory if float(decision["regret_tokens"]) > 0.0]
    fast = [decision for decision in trajectory if decision["fast_change"]]
    fast_regret = sum(float(decision["regret_tokens"]) for decision in fast)
    total_regret = sum(float(decision["regret_tokens"]) for decision in trajectory)
    return {
        "group": (
            f"{trial.split}/{trial.workload}/capacity_{trial.capacity_blocks}/seed_{trial.seed}"
        ),
        "split": trial.split,
        "workload": trial.workload,
        "capacity_blocks": trial.capacity_blocks,
        "seed": trial.seed,
        "decision_count": len(trajectory),
        "regretful_decision_count": len(regretful),
        "score_scale": scale,
        "tracking_rmse": trial.shadow_price_tracking_rmse,
        "tracking_mae": trial.shadow_price_tracking_mae,
        "tracking_bias": trial.shadow_price_tracking_bias,
        "oracle_shadow_price_mean": trial.oracle_shadow_price_mean,
        "oracle_shadow_price_stddev": trial.oracle_shadow_price_stddev,
        "oracle_shadow_price_change_mean": trial.oracle_shadow_price_change_mean,
        "oracle_shadow_price_change_p95": trial.oracle_shadow_price_change_p95,
        "fast_change_threshold": (fast_threshold if fast_threshold != float("inf") else None),
        "fast_decision_count": len(fast),
        "fast_decision_fraction": len(fast) / len(trajectory) if trajectory else 0.0,
        "total_admission_regret_tokens": total_regret,
        "fast_admission_regret_tokens": fast_regret,
        "fast_regret_share": fast_regret / total_regret if total_regret else 0.0,
        "fast_regret_lift": trial.fast_shadow_price_regret_lift,
        "change_regret_correlation": trial.shadow_price_change_regret_correlation,
        "regret_concentrates_on_fast_changes": (
            bool(total_regret)
            and bool(fast)
            and fast_regret / total_regret > len(fast) / len(trajectory)
        ),
        "trajectory": trajectory,
    }


def _shadow_decision_row(
    decision: AdmissionDecisionDiagnostic,
    *,
    scale: float,
    oracle_change: float,
    fast_change: bool,
) -> dict[str, object]:
    """Calibrate one policy score into the oracle value-density units."""
    implied_shadow_price = decision.incoming_value_density - scale * decision.score
    return {
        "now": decision.now,
        "request_index": decision.request_index,
        "prefix_hash": decision.prefix_hash,
        "depth": decision.depth,
        "token_count": decision.token_count,
        "capacity_weight_tokens": decision.capacity_weight_tokens,
        "score": decision.score,
        "accepted": decision.accepted,
        "incoming_value_tokens": decision.incoming_value_tokens,
        "displaced_value_tokens": decision.displaced_value_tokens,
        "incoming_value_density": decision.incoming_value_density,
        "oracle_shadow_price": decision.oracle_shadow_price,
        "policy_implied_shadow_price": implied_shadow_price,
        "tracking_error": implied_shadow_price - decision.oracle_shadow_price,
        "oracle_shadow_price_change": oracle_change,
        "fast_change": fast_change,
        "regret_tokens": decision.regret_tokens,
    }


def _summarize_shadow_groups(groups: list[dict[str, object]]) -> dict[str, object]:
    """Summarize tracking quality and the fast-change regret prediction."""
    analyzable = [
        group
        for group in groups
        if int(group["decision_count"]) > 0 and int(group["fast_decision_count"]) > 0
    ]
    regretful = [
        group for group in analyzable if float(group["total_admission_regret_tokens"]) > 0.0
    ]
    concentrated = [group for group in regretful if group["regret_concentrates_on_fast_changes"]]
    decision_count = sum(int(group["decision_count"]) for group in analyzable)
    fast_decision_count = sum(int(group["fast_decision_count"]) for group in analyzable)
    total_regret = sum(float(group["total_admission_regret_tokens"]) for group in analyzable)
    fast_regret = sum(float(group["fast_admission_regret_tokens"]) for group in analyzable)
    slow_decision_count = decision_count - fast_decision_count
    slow_regret = total_regret - fast_regret
    fast_regret_rate = fast_regret / fast_decision_count if fast_decision_count else 0.0
    slow_regret_rate = slow_regret / slow_decision_count if slow_decision_count else 0.0
    aggregate_lift = (
        fast_regret_rate / slow_regret_rate
        if slow_regret_rate > 0.0
        else (None if fast_regret_rate > 0.0 else 1.0)
    )
    if not analyzable:
        verdict = "inconclusive_no_shadow_movement"
    elif not regretful:
        verdict = "inconclusive_no_admission_regret"
    elif len(concentrated) == len(regretful):
        verdict = "supported"
    else:
        verdict = "falsified"
    return {
        "verdict": verdict,
        "group_count": len(groups),
        "analyzable_group_count": len(analyzable),
        "regretful_group_count": len(regretful),
        "concentrated_group_count": len(concentrated),
        "concentration_rate": len(concentrated) / len(regretful) if regretful else 0.0,
        "mean_tracking_rmse": (
            mean(float(group["tracking_rmse"]) for group in analyzable) if analyzable else 0.0
        ),
        "mean_tracking_mae": (
            mean(float(group["tracking_mae"]) for group in analyzable) if analyzable else 0.0
        ),
        "mean_oracle_shadow_price_change": (
            mean(float(group["oracle_shadow_price_change_mean"]) for group in analyzable)
            if analyzable
            else 0.0
        ),
        "aggregate_fast_decision_fraction": (
            fast_decision_count / decision_count if decision_count else 0.0
        ),
        "aggregate_fast_regret_share": fast_regret / total_regret if total_regret else 0.0,
        "aggregate_fast_regret_lift": aggregate_lift,
        "mean_change_regret_correlation": (
            mean(float(group["change_regret_correlation"]) for group in analyzable)
            if analyzable
            else 0.0
        ),
    }


def run_shadow_price_analysis(
    config_path: Path,
    *,
    request_count: int | None = None,
    seeds: tuple[int, ...] | None = None,
    splits: tuple[str, ...] = _DEFAULT_SPLITS,
    workloads: tuple[str, ...] | None = None,
    capacity_blocks: tuple[int, ...] | None = None,
    factory: Callable[..., PrefixKVPolicy] = build_incumbent,
    policy_name: str = "pressure_aware_incumbent",
) -> dict[str, object]:
    """Measure oracle and policy-implied admission shadow-price trajectories."""
    config = load_evaluator_config(config_path)
    updates: dict[str, object] = {}
    if request_count is not None:
        updates["request_count"] = request_count
    if seeds is not None:
        updates["seeds"] = seeds
    if workloads is not None:
        for split in splits:
            updates[f"{split}_families"] = workloads
        updates["family_request_multipliers"] = {}
    if capacity_blocks is not None:
        updates["capacity_blocks"] = capacity_blocks[0]
        updates["capacity_sweep_blocks"] = capacity_blocks
    if updates:
        config = config.with_updates(**updates)

    result = PrefixKVCacheEvaluator(
        config,
        splits=splits,
        record_admission_diagnostics=True,
    )(factory)
    groups = [
        {
            "verifier_version": result.verifier_version,
            "evaluation_context_sha256": result.evaluation_context_sha256,
            "panel_sha256": result.panel_sha256,
            **_shadow_trial_row(trial),
        }
        for trial in result.trials
    ]
    summary_groups = [
        {key: value for key, value in group.items() if key != "trajectory"} for group in groups
    ]
    return {
        "schema": "prefix-kv-cache-shadow-price-tracking-v1",
        "verifier_version": result.verifier_version,
        "evaluation_context_sha256": result.evaluation_context_sha256,
        "panel_sha256": result.panel_sha256,
        "config": str(config_path),
        "policy": policy_name,
        "request_count": config.request_count,
        "seeds": list(config.seeds),
        "capacity_blocks": list(config.effective_capacity_blocks()),
        "block_size_tokens": config.block_size_tokens,
        "splits": list(splits),
        "workloads": list(workloads) if workloads is not None else None,
        "calibration": (
            "Per replay, fit nonnegative beta with no intercept so beta times policy "
            "admission score estimates oracle surplus density (F-mu)/w. The implied policy "
            "shadow price is F/w-beta*score; the oracle shadow price is mu/w."
        ),
        "capacity_weight": (
            "w is one physical cache slot, represented in token units by block_size_tokens; "
            "partial terminal blocks still consume one slot."
        ),
        "falsification_rule": (
            "The tracking-bandwidth claim passes only if every regretful group with "
            "measurable shadow-price movement places more admission regret in the fastest "
            "change quartile than that quartile's decision share."
        ),
        "summary": _summarize_shadow_groups(summary_groups),
        "by_workload": {
            workload: _summarize_shadow_groups(
                [
                    group
                    for group in summary_groups
                    if f"{group['split']}/{group['workload']}" == workload
                ]
            )
            for workload in sorted(
                {f"{group['split']}/{group['workload']}" for group in summary_groups}
            )
        },
        "by_capacity": {
            str(capacity): _summarize_shadow_groups(
                [group for group in summary_groups if int(group["capacity_blocks"]) == capacity]
            )
            for capacity in sorted({int(group["capacity_blocks"]) for group in summary_groups})
        },
        "groups": groups,
    }


_CAUSAL_CELLS = {
    "II": (False, False),
    "OI": (True, False),
    "IO": (False, True),
    "OO": (True, True),
}
_CAUSAL_OUTCOMES = (
    "group_score",
    "token_hit_rate",
    "block_hit_rate",
    "prefill_tokens_saved",
    "p95_latency_proxy",
    "cache_churn_per_1k",
)


def _causal_group_score(trial: TrialMetrics, config: EvaluatorConfig) -> float:
    """Return one trial's behavioral score without the shared complexity charge."""
    score = workload_base_score(
        [trial],
        token_weight=config.w_avg_tok,
        block_weight=config.w_avg_blk,
        request_tail_weight=config.request_tail_weight,
        worst_window_weight=config.worst_window_weight,
        priority_hit_weight=config.priority_hit_weight,
        wasted_admission_weight=config.wasted_admission_weight,
        admission_utility_weight=config.admission_utility_weight,
        # Keep the causal outcome independent of the future-aware audit surrogate.
        avoidable_eviction_weight=0.0,
        latency_weight=config.latency_weight,
        latency_cap=config.latency_cap,
        latency_norm=config.latency_norm,
    )
    score -= min(config.churn_cap, config.churn_weight * trial.cache_churn_per_1k)
    score -= min(config.underfill_cap, config.underfill_weight * trial.policy_underfill_rate)
    if trial.workload == "multi_tenant_skew":
        score -= min(
            config.fairness_cap,
            config.fairness_weight * trial.tenant_fairness_penalty,
        )
    return score


def _causal_trial_outcomes(
    trial: TrialMetrics,
    config: EvaluatorConfig,
) -> dict[str, float]:
    """Return realized outcomes used by the causal component factorial."""
    return {
        "group_score": _causal_group_score(trial, config),
        "token_hit_rate": trial.token_hit_rate,
        "block_hit_rate": trial.block_hit_rate,
        "prefill_tokens_saved": trial.prefill_tokens_saved,
        "p95_latency_proxy": trial.p95_latency_proxy,
        "cache_churn_per_1k": trial.cache_churn_per_1k,
    }


def _causal_group_key(trial: TrialMetrics) -> tuple[str, str, int, int]:
    """Return the stable workload-capacity-seed pairing key."""
    return (trial.split, trial.workload, trial.capacity_blocks, trial.seed)


def _causal_effects(
    cells: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute admission, eviction, and interaction effects for every outcome."""
    return {
        outcome: {
            "tau_A": cells["OI"][outcome] - cells["II"][outcome],
            "tau_E": cells["IO"][outcome] - cells["II"][outcome],
            "tau_AE": (
                cells["OO"][outcome]
                - cells["OI"][outcome]
                - cells["IO"][outcome]
                + cells["II"][outcome]
            ),
            "eviction_effect_after_oracle_admission": (cells["OO"][outcome] - cells["OI"][outcome]),
            "admission_effect_after_oracle_eviction": (cells["OO"][outcome] - cells["IO"][outcome]),
        }
        for outcome in _CAUSAL_OUTCOMES
    }


def _summarize_causal_groups(groups: list[dict[str, object]]) -> dict[str, object]:
    """Summarize paired causal effects over a stable group collection."""
    effect_tolerance = 1e-9
    effects = {
        outcome: {
            effect: sum(float(group["effects"][outcome][effect]) for group in groups)
            for effect in (
                "tau_A",
                "tau_E",
                "tau_AE",
                "eviction_effect_after_oracle_admission",
                "admission_effect_after_oracle_eviction",
            )
        }
        for outcome in _CAUSAL_OUTCOMES
    }
    score_effects = effects["group_score"]
    admission_dominant = [
        group
        for group in groups
        if float(group["effects"]["group_score"]["tau_A"])
        > float(group["effects"]["group_score"]["tau_E"])
    ]
    eviction_dominant = [
        group
        for group in groups
        if float(group["effects"]["group_score"]["tau_E"])
        > float(group["effects"]["group_score"]["tau_A"])
    ]
    interaction_absolute = sum(
        abs(float(group["effects"]["group_score"]["tau_AE"])) for group in groups
    )
    main_effect_absolute = sum(
        abs(float(group["effects"]["group_score"]["tau_A"]))
        + abs(float(group["effects"]["group_score"]["tau_E"]))
        for group in groups
    )
    eviction_before = score_effects["tau_E"]
    eviction_after = score_effects["eviction_effect_after_oracle_admission"]
    residual_eviction_groups = [
        group
        for group in groups
        if float(group["effects"]["group_score"]["eviction_effect_after_oracle_admission"])
        > effect_tolerance
    ]
    return {
        "group_count": len(groups),
        "effects": effects,
        "admission_dominant_group_count": len(admission_dominant),
        "eviction_dominant_group_count": len(eviction_dominant),
        "residual_eviction_value_group_count": len(residual_eviction_groups),
        "oracle_admission_dissolved_eviction_value_group_count": (
            len(groups) - len(residual_eviction_groups)
        ),
        "admission_dominance_rate": (len(admission_dominant) / len(groups) if groups else 0.0),
        "interaction_absolute_share": (
            interaction_absolute / main_effect_absolute if main_effect_absolute else 0.0
        ),
        "eviction_effect_retained_after_oracle_admission": (
            eviction_after / eviction_before if eviction_before else 0.0
        ),
        "oracle_admission_dissolves_aggregate_eviction_value": (
            eviction_before > 0.0 and eviction_after <= 0.0
        ),
    }


def run_causal_component_factorial(
    config_path: Path,
    *,
    request_count: int | None = None,
    seeds: tuple[int, ...] | None = None,
    splits: tuple[str, ...] = _DEFAULT_SPLITS,
    workloads: tuple[str, ...] | None = None,
    capacity_blocks: tuple[int, ...] | None = None,
    factory: Callable[..., PrefixKVPolicy] = build_production_incumbent,
    policy_name: str = "production_incumbent",
) -> dict[str, object]:
    """Run incumbent/oracle admission-by-eviction crossed replays."""
    config = load_evaluator_config(config_path)
    updates: dict[str, object] = {}
    if request_count is not None:
        updates["request_count"] = request_count
    if seeds is not None:
        updates["seeds"] = seeds
    if workloads is not None:
        for split in splits:
            updates[f"{split}_families"] = workloads
        updates["family_request_multipliers"] = {}
    if capacity_blocks is not None:
        updates["capacity_blocks"] = capacity_blocks[0]
        updates["capacity_sweep_blocks"] = capacity_blocks
    if updates:
        config = config.with_updates(**updates)

    cell_trials: dict[str, dict[tuple[str, str, int, int], TrialMetrics]] = {}
    cell_identities: dict[str, dict[str, str]] = {}
    for cell, (oracle_admission, oracle_eviction) in _CAUSAL_CELLS.items():
        result = PrefixKVCacheEvaluator(
            config,
            splits=splits,
            oracle_admission=oracle_admission,
            oracle_eviction=oracle_eviction,
        )(factory)
        cell_identities[cell] = {
            "verifier_version": result.verifier_version,
            "evaluation_context_sha256": result.evaluation_context_sha256,
            "panel_sha256": result.panel_sha256,
        }
        cell_trials[cell] = {_causal_group_key(trial): trial for trial in result.trials}
    identity = require_single_score_identity(
        cell_identities.values(),
        context="causal component replay",
    )

    common_keys = set.intersection(*(set(trials) for trials in cell_trials.values()))
    groups = []
    for key in sorted(common_keys):
        cells = {
            cell: {
                **cell_identities[cell],
                **_causal_trial_outcomes(trials[key], config),
            }
            for cell, trials in cell_trials.items()
        }
        incumbent_trial = cell_trials["II"][key]
        admission_regret = (
            incumbent_trial.avoidable_admission_regret_tokens
            + incumbent_trial.avoidable_rejection_regret_tokens
        )
        eviction_regret = incumbent_trial.value_weighted_avoidable_eviction_regret_tokens
        groups.append(
            {
                "verifier_version": identity.verifier_version,
                "evaluation_context_sha256": identity.evaluation_context_sha256,
                "panel_sha256": identity.panel_sha256,
                "group": (f"{key[0]}/{key[1]}/capacity_{key[2]}/seed_{key[3]}"),
                "split": key[0],
                "workload": key[1],
                "capacity_blocks": key[2],
                "seed": key[3],
                "audit_admission_regret_tokens": admission_regret,
                "audit_eviction_regret_tokens": eviction_regret,
                "audit_eviction_dominant": eviction_regret > admission_regret,
                "cells": cells,
                "effects": _causal_effects(cells),
            }
        )

    audit_eviction_dominant = [group for group in groups if group["audit_eviction_dominant"]]
    return {
        "schema": "prefix-kv-cache-causal-component-factorial-v1",
        "verifier_version": identity.verifier_version,
        "evaluation_context_sha256": identity.evaluation_context_sha256,
        "panel_sha256": identity.panel_sha256,
        "config": str(config_path),
        "policy": policy_name,
        "request_count": config.request_count,
        "seeds": list(config.seeds),
        "capacity_blocks": list(config.effective_capacity_blocks()),
        "splits": list(splits),
        "workloads": list(workloads) if workloads is not None else None,
        "primary_outcome": "group_score",
        "group_score_definition": (
            "Single-trial evaluator behavioral score with configured hit, tail, "
            "admission-waste, utility, latency, churn, underfill, and applicable fairness "
            "terms; future-aware avoidable-eviction and shared complexity charges are omitted."
        ),
        "oracle_definition": (
            "Oracle admission accepts iff future occurrence value F exceeds the "
            "cheapest legal displacement value mu. Oracle eviction selects the legal "
            "leaf with minimum F, breaking ties by furthest next use. Incumbent score "
            "functions still execute so only the selected action rule is intervened on."
        ),
        "summary": _summarize_causal_groups(groups),
        "audit_eviction_dominant_summary": _summarize_causal_groups(audit_eviction_dominant),
        "by_split": {
            split: _summarize_causal_groups([group for group in groups if group["split"] == split])
            for split in sorted({str(group["split"]) for group in groups})
        },
        "by_workload": {
            workload: _summarize_causal_groups(
                [group for group in groups if f"{group['split']}/{group['workload']}" == workload]
            )
            for workload in sorted({f"{group['split']}/{group['workload']}" for group in groups})
        },
        "groups": groups,
    }


def run_admission_policy_sweep(
    config_path: Path,
    *,
    request_count: int | None = None,
    seeds: tuple[int, ...] | None = None,
    splits: tuple[str, ...] = _DEFAULT_SPLITS,
    workloads: tuple[str, ...] | None = None,
    policy_specs: tuple[AdmissionPolicySpec, ...] = ADMISSION_POLICY_SPECS,
) -> dict[str, object]:
    """Evaluate every distinct admission rule with fixed LRU eviction."""
    policies: dict[str, dict[str, object]] = {}
    for specification in policy_specs:
        analysis = run_analysis(
            config_path,
            request_count=request_count,
            seeds=seeds,
            splits=splits,
            workloads=workloads,
            factory=baseline_lru_blocks,
            policy_name=f"{specification.name}+fixed_lru",
            fixed_admission_factory=specification.factory,
            expose_future_reuse=specification.reporting_only,
        )
        policies[specification.name] = {
            "aliases": list(specification.aliases),
            "reporting_only": specification.reporting_only,
            **analysis,
        }

    deployable = [policy for policy in policies.values() if not bool(policy["reporting_only"])]
    identity = require_single_score_identity(
        policies.values(),
        context="admission-policy sweep",
    )
    return {
        "schema": "prefix-kv-cache-admission-policy-regret-sweep-v1",
        "verifier_version": identity.verifier_version,
        "evaluation_context_sha256": identity.evaluation_context_sha256,
        "panel_sha256": identity.panel_sha256,
        "config": str(config_path),
        "eviction_policy": "lru",
        "policy_count": len(policies),
        "deployable_policy_count": len(deployable),
        "method": (
            "Hold legal-leaf LRU eviction fixed and vary each distinct admission "
            "implementation plus its lifecycle state."
        ),
        "policies": policies,
    }


def run_admission_eviction_matrix(
    config_path: Path,
    *,
    request_count: int | None = None,
    seeds: tuple[int, ...] | None = None,
    splits: tuple[str, ...] = _DEFAULT_SPLITS,
    workloads: tuple[str, ...] | None = None,
    admission_specs: tuple[AdmissionPolicySpec, ...] = ADMISSION_POLICY_SPECS,
    eviction_specs: tuple[EvictionPolicySpec, ...] = EVICTION_POLICY_SPECS,
) -> dict[str, object]:
    """Cross every distinct admission rule with representative eviction rules."""
    combinations: dict[str, dict[str, object]] = {}
    for admission in admission_specs:
        for eviction in eviction_specs:
            key = f"{admission.name}+{eviction.name}"
            analysis = run_analysis(
                config_path,
                request_count=request_count,
                seeds=seeds,
                splits=splits,
                workloads=workloads,
                factory=eviction.factory,
                policy_name=key,
                fixed_admission_factory=admission.factory,
                expose_future_reuse=admission.reporting_only or eviction.reporting_only,
            )
            combinations[key] = {
                "admission_policy": admission.name,
                "eviction_policy": eviction.name,
                "reporting_only": admission.reporting_only or eviction.reporting_only,
                **analysis,
            }

    identity = require_single_score_identity(
        combinations.values(),
        context="admission-eviction matrix",
    )
    return {
        "schema": "prefix-kv-cache-admission-eviction-policy-matrix-v1",
        "verifier_version": identity.verifier_version,
        "evaluation_context_sha256": identity.evaluation_context_sha256,
        "panel_sha256": identity.panel_sha256,
        "config": str(config_path),
        "admission_policies": [
            {
                "name": specification.name,
                "aliases": list(specification.aliases),
                "reporting_only": specification.reporting_only,
            }
            for specification in admission_specs
        ],
        "eviction_policies": [
            {
                "name": specification.name,
                "reporting_only": specification.reporting_only,
            }
            for specification in eviction_specs
        ],
        "combination_count": len(combinations),
        "method": (
            "Full factorial crossing of distinct admission implementations with "
            "representative eviction rules on identical workload-capacity-seed groups."
        ),
        "combinations": combinations,
    }


def _summary_row(label: str, summary: dict[str, object]) -> str:
    """Render one aggregate Markdown row."""
    return (
        f"| `{label}` | {summary['regretful_group_count']} | "
        f"{summary['admission_dominant_group_count']} | "
        f"{float(summary['admission_dominance_rate']):.1%} | "
        f"{float(summary['aggregate_admission_side_regret_tokens']):.1f} | "
        f"{float(summary['aggregate_eviction_side_regret_tokens']):.1f} | "
        f"{float(summary['aggregate_admission_minus_eviction_regret_tokens']):+.1f} |"
    )


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    """Write a human-readable falsification report."""
    summary = payload["summary"]
    groups = payload["groups"]
    identity = require_single_score_identity(
        groups,
        context="admission-eviction regret report",
    )
    verifier_version = identity.verifier_version
    if verifier_version != payload.get("verifier_version"):
        raise ValueError("regret report version does not match its score groups")
    counterexamples = sorted(
        (
            group
            for group in groups
            if not group["invalid"]
            and float(group["total_regret_tokens"]) > 0.0
            and not group["admission_dominates"]
        ),
        key=lambda group: float(group["admission_minus_eviction_regret_tokens"]),
    )
    negative_workloads = sorted(
        (
            (label, value)
            for label, value in payload["by_workload"].items()
            if float(value["aggregate_admission_minus_eviction_regret_tokens"]) < 0.0
        ),
        key=lambda item: float(item[1]["aggregate_admission_minus_eviction_regret_tokens"]),
    )
    verdict_text = {
        "supported": "Supported under the strict universal rule.",
        "falsified": "Falsified under the strict universal rule.",
        "inconclusive_invalid_groups": "Inconclusive because one or more groups were invalid.",
        "inconclusive_no_regret": "Inconclusive because no group had measurable regret.",
    }[summary["verdict"]]
    lines = [
        "# Admission vs Eviction Regret Audit",
        "",
        f"Policy: `{payload['policy']}`",
        "",
        f"Verifier: `{verifier_version}`",
        "",
        "## Falsifiable Claim",
        "",
        "Claim: admission-side regret dominates eviction-side regret across",
        "workload-capacity-seed groups.",
        "",
        f"**Verdict: {verdict_text}**",
        "",
        f"- Regretful groups: `{summary['regretful_group_count']}`.",
        f"- Admission-dominant groups: `{summary['admission_dominant_group_count']}` "
        f"(`{float(summary['admission_dominance_rate']):.1%}`).",
        f"- Aggregate admission-side regret: "
        f"`{float(summary['aggregate_admission_side_regret_tokens']):.1f}` future tokens.",
        f"- Aggregate eviction-side regret: "
        f"`{float(summary['aggregate_eviction_side_regret_tokens']):.1f}` future tokens.",
        f"- Aggregate admission share: `{float(summary['aggregate_admission_regret_share']):.1%}`.",
        f"- Admission decisions: `{summary['aggregate_admission_decision_count']}`; "
        f"surrogate regret per decision: "
        f"`{float(summary['aggregate_admission_regret_tokens_per_decision']):.2f}`.",
        f"- Eviction decisions: `{summary['aggregate_eviction_decision_count']}`; "
        f"surrogate regret per decision: "
        f"`{float(summary['aggregate_eviction_regret_tokens_per_decision']):.2f}`.",
        f"- Decision-normalized admission dominance: "
        f"`{summary['decision_normalized_admission_dominant_group_count']}` of "
        f"`{summary['regretful_group_count']}` regretful groups "
        f"(`{float(summary['decision_normalized_admission_dominance_rate']):.1%}`).",
        "",
        "The universal claim passes only if every valid group with nonzero regret has",
        "strictly greater admission-side regret. Zero-regret groups are reported but do not",
        "decide the claim.",
        "",
        "## Audit Definition",
        "",
        "- The future-value surrogate upper bound is block token count times remaining",
        "  request occurrences after the current request.",
        "- Avoidable admission regret is the value lost when an accepted incoming block is",
        "  worth less than the cheapest legal displacement.",
        "- Avoidable rejection regret is the value lost when a rejected incoming block is",
        "  worth more than the cheapest legal displacement; free space has displacement zero.",
        "- Value-weighted avoidable-eviction regret is the chosen victim's value minus the",
        "  cheapest legal victim's value.",
        "- Admission plus eviction surrogate regret exactly decomposes this local same-state",
        "  comparison. It is not realized causal regret or a full counterfactual replay.",
        "",
        "## By Split",
        "",
        "| Split | Regretful groups | Admission dominant | Dominance rate | "
        "Admission regret | Eviction regret | Margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_summary_row(label, value) for label, value in payload["by_split"].items())
    lines.extend(
        [
            "",
            "## By Workload",
            "",
            "| Workload | Regretful groups | Admission dominant | Dominance rate | "
            "Admission regret | Eviction regret | Margin |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(_summary_row(label, value) for label, value in payload["by_workload"].items())
    lines.extend(
        [
            "",
            "## Strongest Counterexamples",
            "",
            "| Group | Admission regret | Eviction regret | Margin |",
            "|---|---:|---:|---:|",
        ]
    )
    if counterexamples:
        for group in counterexamples[:20]:
            lines.append(
                f"| `{group['group']}` | "
                f"{float(group['admission_side_regret_tokens']):.1f} | "
                f"{float(group['eviction_side_regret_tokens']):.1f} | "
                f"{float(group['admission_minus_eviction_regret_tokens']):+.1f} |"
            )
    else:
        lines.append("| None | 0.0 | 0.0 | +0.0 |")
    negative_labels = ", ".join(f"`{label}`" for label, _ in negative_workloads[:5]) or "none"
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The strict universal claim is `{summary['verdict']}`: admission dominates in",
            f"`{summary['admission_dominant_group_count']}` of "
            f"`{summary['regretful_group_count']}` regretful groups, leaving",
            f"`{summary['eviction_dominant_group_count']}` eviction-dominant counterexamples.",
            f"The weaker aggregate-total claim is supported under this surrogate: admission "
            f"accounts for `{float(summary['aggregate_admission_regret_share']):.1%}` of total "
            "surrogate regret,",
            "and the median regretful group has a positive admission-minus-eviction margin.",
            "After normalizing each side by its own decision count, admission dominates in",
            f"`{summary['decision_normalized_admission_dominant_group_count']}` of "
            f"`{summary['regretful_group_count']}` regretful groups "
            f"(`{float(summary['decision_normalized_admission_dominance_rate']):.1%}`). "
            "The aggregate per-decision rate still favors admission, but groupwise",
            "per-decision dominance does not hold in a majority. Total contribution and",
            "per-decision severity answer different questions.",
            "",
            f"Workload-level counterexamples with negative aggregate margins include "
            f"{negative_labels}. These groups are concrete targets for eviction-specific",
            "follow-up rather than evidence for a uniform admission-first rule.",
        ]
    )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "The audit uses future occurrence count as a stable token-value surrogate upper",
            "bound. It does not model whether every future occurrence would remain",
            "root-contiguous after other decisions. It audits only explicit admission scores;",
            "descendants bypassed after a parent rejection are consequences of that scored",
            "rejection, not additional decisions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_admission_policy_markdown(path: Path, payload: dict[str, object]) -> None:
    """Write the controlled fixed-LRU admission-policy comparison."""
    verifier_version = require_single_score_identity(
        payload["policies"].values(),
        context="admission-policy report",
    ).verifier_version
    if verifier_version != payload.get("verifier_version"):
        raise ValueError("admission-policy report version does not match its score rows")
    rows = []
    for name, policy in payload["policies"].items():
        summary = policy["summary"]
        rows.append(
            (
                name,
                bool(policy["reporting_only"]),
                int(summary["regretful_group_count"]),
                int(summary["admission_dominant_group_count"]),
                float(summary["admission_dominance_rate"]),
                int(summary["decision_normalized_admission_dominant_group_count"]),
                float(summary["decision_normalized_admission_dominance_rate"]),
                float(summary["aggregate_admission_side_regret_tokens"]),
                float(summary["aggregate_eviction_side_regret_tokens"]),
                float(summary["aggregate_admission_regret_share"]),
            )
        )
    lines = [
        "# Admission-Policy Regret Sweep",
        "",
        f"Verifier: `{verifier_version}`",
        "",
        "Eviction is fixed to legal-leaf LRU. Each row changes only the admission",
        "implementation and the lifecycle state needed by that admission policy.",
        "Duplicate admit-all baselines are collapsed into one behavioral representative.",
        "",
        "| Admission policy | Scope | Regretful groups | Admission dominant | "
        "Dominance rate | Per-decision dominant | Per-decision rate | "
        "Admission regret | Eviction regret | Admission share |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        (
            name,
            reporting_only,
            regretful,
            admission_dominant,
            dominance_rate,
            normalized_dominant,
            normalized_rate,
            admission_regret,
            eviction_regret,
            admission_share,
        ) = row
        scope = "reporting-only" if reporting_only else "deployable"
        lines.append(
            f"| `{name}` | {scope} | {regretful} | {admission_dominant} | "
            f"{dominance_rate:.1%} | {normalized_dominant} | {normalized_rate:.1%} | "
            f"{admission_regret:.1f} | {eviction_regret:.1f} | {admission_share:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Behavioral Aliases",
            "",
            "These policies have the same admission decisions under the simulator contract",
            "and are represented once in the table:",
            "",
        ]
    )
    for name, policy in payload["policies"].items():
        aliases = policy["aliases"]
        if aliases:
            lines.append(f"- `{name}`: " + ", ".join(f"`{alias}`" for alias in aliases) + ".")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a controlled local-regret audit, not a causal replay. It tests whether",
            "LRU leaves substantial hindsight eviction value on the table after each admission",
            "rule shapes the resident set. It does not yet measure the realized gain from",
            "replacing LRU with a more complex eviction policy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_admission_eviction_matrix_markdown(
    path: Path,
    payload: dict[str, object],
) -> None:
    """Write realized and local-regret results for the policy matrix."""
    combinations = payload["combinations"]
    verifier_version = require_single_score_identity(
        combinations.values(),
        context="admission-eviction matrix report",
    ).verifier_version
    if verifier_version != payload.get("verifier_version"):
        raise ValueError("matrix report version does not match its score rows")
    eviction_names = [item["name"] for item in payload["eviction_policies"]]
    lines = [
        "# Admission-Eviction Policy Matrix",
        "",
        f"Verifier: `{verifier_version}`",
        "",
        "Every distinct admission implementation is crossed with four deployable",
        "eviction rules and one reporting-only constrained next-use control.",
        "",
        "| Admission | Eviction | Scope | Mean token hit | Saved tokens | "
        "Eviction regret | Admission share |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for admission in payload["admission_policies"]:
        for eviction_name in eviction_names:
            combination = combinations[f"{admission['name']}+{eviction_name}"]
            summary = combination["summary"]
            scope = "reporting-only" if combination["reporting_only"] else "deployable"
            lines.append(
                f"| `{admission['name']}` | `{eviction_name}` | {scope} | "
                f"{float(summary['mean_token_hit_rate']):.4f} | "
                f"{float(summary['total_prefill_tokens_saved']):.0f} | "
                f"{float(summary['aggregate_eviction_side_regret_tokens']):.0f} | "
                f"{float(summary['aggregate_admission_regret_share']):.1%} |"
            )

    lines.extend(
        [
            "",
            "## Best Eviction Per Admission",
            "",
            "| Admission | LRU token hit | Best deployable eviction | Best token hit | "
            "Gain over LRU | Oracle token hit | Oracle headroom |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    deployable_evictions = [
        item["name"] for item in payload["eviction_policies"] if not item["reporting_only"]
    ]
    for admission in payload["admission_policies"]:
        name = admission["name"]
        lru_hit = float(combinations[f"{name}+lru"]["summary"]["mean_token_hit_rate"])
        deployable = max(
            (
                (
                    eviction_name,
                    float(
                        combinations[f"{name}+{eviction_name}"]["summary"]["mean_token_hit_rate"]
                    ),
                )
                for eviction_name in deployable_evictions
            ),
            key=lambda item: item[1],
        )
        oracle_hit = float(
            combinations[f"{name}+oracle_next_use"]["summary"]["mean_token_hit_rate"]
        )
        lines.append(
            f"| `{name}` | {lru_hit:.4f} | `{deployable[0]}` | {deployable[1]:.4f} | "
            f"{deployable[1] - lru_hit:+.4f} | {oracle_hit:.4f} | "
            f"{oracle_hit - deployable[1]:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Paired Group Comparisons Against LRU",
            "",
            "Win/tie/loss counts compare token hit rate on the same",
            "workload-capacity-seed group.",
            "",
            "| Admission | Eviction | Wins | Ties | Losses | Mean token-hit delta |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for admission in payload["admission_policies"]:
        name = admission["name"]
        if name == "reject_all":
            continue
        lru_groups = {
            group["group"]: float(group["token_hit_rate"])
            for group in combinations[f"{name}+lru"]["groups"]
        }
        comparison_evictions = [
            eviction_name
            for eviction_name in ("lfu", "incumbent_value_aware", "oracle_next_use")
            if f"{name}+{eviction_name}" in combinations
        ]
        for eviction_name in comparison_evictions:
            deltas = [
                float(group["token_hit_rate"]) - lru_groups[str(group["group"])]
                for group in combinations[f"{name}+{eviction_name}"]["groups"]
            ]
            wins = sum(delta > 1e-12 for delta in deltas)
            ties = sum(abs(delta) <= 1e-12 for delta in deltas)
            losses = sum(delta < -1e-12 for delta in deltas)
            mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
            lines.append(
                f"| `{name}` | `{eviction_name}` | {wins} | {ties} | {losses} | {mean_delta:+.4f} |"
            )

    deployable_combinations = [
        combination for combination in combinations.values() if not combination["reporting_only"]
    ]
    best_deployable = max(
        deployable_combinations,
        key=lambda combination: float(combination["summary"]["mean_token_hit_rate"]),
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The best deployable pairing is `{best_deployable['admission_policy']}+"
            f"{best_deployable['eviction_policy']}` at mean token hit "
            f"`{float(best_deployable['summary']['mean_token_hit_rate']):.4f}`.",
            "LFU and the compact incumbent value-aware eviction rule deliver similar",
            "realized gains across the selective admission policies; neither uniformly",
            "wins every group. This supports a simple frequency-aware eviction rule,",
            "not the stronger claim that eviction choice is unimportant.",
            "",
            "Mean token-hit differences are realized outcomes on identical generated",
            "request panels. Eviction regret remains a local future-count surrogate.",
            "The oracle is constrained to legal resident leaves and is reporting-only;",
            "it is not an unconstrained globally optimal cache replay.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_shadow_lift(value: object) -> str:
    """Render a finite regret lift or an explicit infinite result."""
    return "infinite" if value is None else f"{float(value):.2f}x"


def _write_shadow_price_markdown(path: Path, payload: dict[str, object]) -> None:
    """Write the theory-grounded shadow-price tracking report."""
    verifier_version = require_single_score_identity(
        payload["groups"],
        context="shadow-price report",
    ).verifier_version
    if verifier_version != payload.get("verifier_version"):
        raise ValueError("shadow-price report version does not match its score groups")
    summary = payload["summary"]
    verdict_text = {
        "supported": "Supported under the strict groupwise concentration rule.",
        "falsified": "Falsified under the strict groupwise concentration rule.",
        "inconclusive_no_shadow_movement": "Inconclusive because no group moved its oracle price.",
        "inconclusive_no_admission_regret": "Inconclusive because no moving group had regret.",
    }[summary["verdict"]]
    lines = [
        "# Admission Shadow-Price Tracking",
        "",
        f"Verifier: `{verifier_version}`",
        "",
        f"Policy: `{payload['policy']}`",
        "",
        f"Requests per workload: `{payload['request_count']}`; seeds: "
        f"`{', '.join(str(seed) for seed in payload['seeds'])}`; capacities: "
        f"`{', '.join(str(value) for value in payload['capacity_blocks'])}` blocks.",
        "",
        "## Result",
        "",
        f"**{verdict_text}**",
        "",
        f"- Analyzable groups: `{summary['analyzable_group_count']}`.",
        f"- Groups with no oracle-price movement: "
        f"`{int(summary['group_count']) - int(summary['analyzable_group_count'])}`.",
        f"- Regretful groups: `{summary['regretful_group_count']}`.",
        f"- Groups with fast-change regret concentration: "
        f"`{summary['concentrated_group_count']}` "
        f"(`{float(summary['concentration_rate']):.1%}`).",
        f"- Fast-change decision share: "
        f"`{float(summary['aggregate_fast_decision_fraction']):.1%}`.",
        f"- Fast-change admission-regret share: "
        f"`{float(summary['aggregate_fast_regret_share']):.1%}`.",
        f"- Aggregate fast/slow regret-density lift: "
        f"`{_format_shadow_lift(summary['aggregate_fast_regret_lift'])}`.",
        f"- Mean calibrated tracking RMSE: "
        f"`{float(summary['mean_tracking_rmse']):.4f}` future-value units per cache token.",
        f"- Mean absolute tracking error: `{float(summary['mean_tracking_mae']):.4f}`.",
        f"- Mean shadow-movement/regret correlation: "
        f"`{float(summary['mean_change_regret_correlation']):+.3f}`.",
        "",
        "## Definition",
        "",
        "The audit computes the oracle water level as `lambda*=mu/w`, where `mu` is",
        "the cheapest legal displacement value and `w` is one physical cache slot in",
        "token units. The policy score has arbitrary heuristic units, so each replay",
        "fits a nonnegative no-intercept scale `beta` from score to oracle surplus",
        "density. The policy-implied water level is then `F/w - beta*score`. This keeps",
        "the policy's zero crossing fixed while making tracking error dimensionally",
        "comparable.",
        "",
        "Fast movement means the top quartile of positive absolute oracle-price changes",
        "within one workload-capacity-seed replay.",
        "",
        "## By Workload",
        "",
        "| Workload | Regretful groups | Concentrated | Rate | Fast decisions | Fast regret | "
        "Lift | Tracking RMSE | Correlation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for workload, values in payload["by_workload"].items():
        lines.append(
            f"| `{workload}` | {values['regretful_group_count']} | "
            f"{values['concentrated_group_count']} | "
            f"{float(values['concentration_rate']):.1%} | "
            f"{float(values['aggregate_fast_decision_fraction']):.1%} | "
            f"{float(values['aggregate_fast_regret_share']):.1%} | "
            f"{_format_shadow_lift(values['aggregate_fast_regret_lift'])} | "
            f"{float(values['mean_tracking_rmse']):.4f} | "
            f"{float(values['mean_change_regret_correlation']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## By Capacity",
            "",
            "| Capacity blocks | Regretful groups | Concentrated | Rate | "
            "Fast decisions | Fast regret | Lift |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for capacity, values in payload["by_capacity"].items():
        lines.append(
            f"| {capacity} | {values['regretful_group_count']} | "
            f"{values['concentrated_group_count']} | "
            f"{float(values['concentration_rate']):.1%} | "
            f"{float(values['aggregate_fast_decision_fraction']):.1%} | "
            f"{float(values['aggregate_fast_regret_share']):.1%} | "
            f"{_format_shadow_lift(values['aggregate_fast_regret_lift'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A tracking RMSE is not a deployable estimate of future value. It is a",
            "quarantined replay diagnostic that asks whether a candidate's signed admission",
            "score behaves like an online estimate of oracle surplus after only a scalar",
            "unit calibration. It is suitable as a mechanism-diversity archive coordinate,",
            "not as a direct selection reward.",
            "",
            "The strict bandwidth prediction is stronger than an aggregate association:",
            "every regretful moving group must concentrate regret in its own fastest-change",
            "quartile. The JSON artifact contains each calibrated decision trajectory for",
            "checking phase boundaries and individual failures.",
            "",
            "A zero-movement group is not evidence of successful tracking. It means this",
            "local oracle found a zero-value legal victim at every scored decision, pinning",
            "the displacement price at zero. The capacity table therefore matters: the",
            "bandwidth signal weakens as disposable capacity increases.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_causal_component_markdown(
    path: Path,
    payload: dict[str, object],
) -> None:
    """Write the incumbent/oracle component-factorial report."""
    verifier_version = require_single_score_identity(
        payload["groups"],
        context="causal component report",
    ).verifier_version
    if verifier_version != payload.get("verifier_version"):
        raise ValueError("causal report version does not match its score groups")
    summary = payload["summary"]
    counterexamples = payload["audit_eviction_dominant_summary"]
    score = summary["effects"]["group_score"]
    counterexample_score = counterexamples["effects"]["group_score"]
    eviction_main_effect = float(score["tau_E"])
    if abs(eviction_main_effect) > 1e-9:
        main_effect_comparison = (
            "On this panel, admission has "
            f"`{float(score['tau_A']) / eviction_main_effect:.2f}x` the aggregate main "
            "effect of eviction."
        )
    else:
        main_effect_comparison = (
            "On this panel, the aggregate eviction main effect is zero, so the "
            "admission-to-eviction ratio is undefined."
        )
    if int(counterexamples["group_count"]) > 0:
        counterexample_eviction_removed = 1.0 - float(
            counterexamples["eviction_effect_retained_after_oracle_admission"]
        )
        counterexample_interpretation = [
            "Oracle admission removes",
            f"`{counterexample_eviction_removed:.1%}` "
            "of the eviction main effect inside the groups that the local audit labeled",
            "eviction-dominant.",
        ]
    else:
        counterexample_interpretation = [
            "No groups in this slice were labeled eviction-dominant by the local audit."
        ]
    lines = [
        "# Causal Admission-Eviction Component Factorial",
        "",
        f"Verifier: `{verifier_version}`",
        "",
        f"Policy: `{payload['policy']}`",
        "",
        f"Requests per workload: `{payload['request_count']}`; seeds: "
        f"`{', '.join(str(seed) for seed in payload['seeds'])}`; capacities: "
        f"`{', '.join(str(value) for value in payload['capacity_blocks'])}` blocks.",
        "",
        "## Design",
        "",
        "Each workload-capacity-seed group is replayed in four cells:",
        "",
        "- `II`: incumbent admission and incumbent eviction.",
        "- `OI`: oracle admission and incumbent eviction.",
        "- `IO`: incumbent admission and oracle eviction.",
        "- `OO`: oracle admission and oracle eviction.",
        "",
        "Oracle admission accepts exactly when future occurrence value exceeds the",
        "cheapest legal displacement value. Oracle eviction chooses the legal leaf",
        "with minimum future occurrence value. The incumbent score functions still",
        "execute, so each intervention replaces only the named action rule.",
        "",
        "## Aggregate Effects",
        "",
        f"- Groups: `{summary['group_count']}`.",
        f"- `sum tau_A`: `{float(score['tau_A']):+.3f}` group-score points.",
        f"- `sum tau_E`: `{float(score['tau_E']):+.3f}` group-score points.",
        f"- `sum tau_AE`: `{float(score['tau_AE']):+.3f}` group-score points.",
        f"- Eviction effect after oracle admission: "
        f"`{float(score['eviction_effect_after_oracle_admission']):+.3f}`.",
        f"- Admission-dominant causal groups: "
        f"`{summary['admission_dominant_group_count']}` "
        f"(`{float(summary['admission_dominance_rate']):.1%}`).",
        f"- Absolute interaction/main-effect ratio: "
        f"`{float(summary['interaction_absolute_share']):.1%}`.",
        f"- Aggregate eviction value retained after oracle admission: "
        f"`{float(summary['eviction_effect_retained_after_oracle_admission']):.1%}`.",
        f"- Groups with residual eviction value after oracle admission: "
        f"`{summary['residual_eviction_value_group_count']}/{summary['group_count']}`.",
        "",
        "## Audit-Identified Eviction-Dominant Groups",
        "",
        f"- Groups: `{counterexamples['group_count']}`.",
        f"- `sum tau_A`: `{float(counterexample_score['tau_A']):+.3f}`.",
        f"- `sum tau_E`: `{float(counterexample_score['tau_E']):+.3f}`.",
        f"- `sum tau_AE`: `{float(counterexample_score['tau_AE']):+.3f}`.",
        f"- Eviction effect after oracle admission: "
        f"`{float(counterexample_score['eviction_effect_after_oracle_admission']):+.3f}`.",
        f"- Eviction value retained after oracle admission: "
        f"`{float(counterexamples['eviction_effect_retained_after_oracle_admission']):.1%}`.",
        f"- Groups with residual eviction value after oracle admission: "
        f"`{counterexamples['residual_eviction_value_group_count']}/"
        f"{counterexamples['group_count']}`.",
        "",
        "## By Split",
        "",
        "| Split | Groups | tau_A | tau_E | tau_AE | Eviction after oracle admission |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, values in payload["by_split"].items():
        effects = values["effects"]["group_score"]
        lines.append(
            f"| `{split}` | {values['group_count']} | "
            f"{float(effects['tau_A']):+.3f} | "
            f"{float(effects['tau_E']):+.3f} | "
            f"{float(effects['tau_AE']):+.3f} | "
            f"{float(effects['eviction_effect_after_oracle_admission']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## By Workload",
            "",
            "| Workload | Groups | tau_A | tau_E | tau_AE | Eviction after oracle admission |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for workload, values in payload["by_workload"].items():
        effects = values["effects"]["group_score"]
        lines.append(
            f"| `{workload}` | {values['group_count']} | "
            f"{float(effects['tau_A']):+.3f} | "
            f"{float(effects['tau_E']):+.3f} | "
            f"{float(effects['tau_AE']):+.3f} | "
            f"{float(effects['eviction_effect_after_oracle_admission']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The interaction term measures whether admission changes the value of oracle",
            "eviction by reshaping the later legal-victim distribution. If the eviction",
            "effect after oracle admission is near zero, admission repairs dissolve the",
            "case for a hybrid. If it remains material, admission and eviction retain",
            "separable causal value and a regime-gated hybrid remains justified.",
            "",
            main_effect_comparison,
            *counterexample_interpretation,
            "Residual eviction value is therefore real but sparse:",
            f"`{summary['residual_eviction_value_group_count']}` of "
            f"`{summary['group_count']}` groups retain it. This supports a selective",
            "regime gate rather than unconditional eviction complexity.",
            "",
            "The primary group score omits the future-aware avoidable-eviction penalty",
            "and the common candidate complexity charge.",
            "The JSON artifact also reports paired effects for token hit, block hit,",
            "saved tokens, p95 latency, and churn.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@click.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    default=Path("configs/prefix_kv_cache.yaml"),
    show_default=True,
)
@click.option("--candidate-program", type=click.Path(path_type=Path))
@click.option("--request-count", type=click.IntRange(min=1))
@click.option("--seeds", type=int, multiple=True)
@click.option("--splits", type=click.Choice(_DEFAULT_SPLITS), multiple=True)
@click.option("--workloads", multiple=True)
@click.option("--capacity-blocks", type=click.IntRange(min=1), multiple=True)
@click.option(
    "--all-admission-policies",
    is_flag=True,
    help="Cross every distinct admission rule with representative eviction rules.",
)
@click.option(
    "--shadow-price",
    is_flag=True,
    help="Measure oracle and policy-implied admission shadow-price trajectories.",
)
@click.option(
    "--causal-components",
    is_flag=True,
    help="Run incumbent/oracle admission-by-eviction crossed replays.",
)
@click.option("--output", type=click.Path(path_type=Path))
@click.option(
    "--markdown",
    type=click.Path(path_type=Path),
    help="Optionally write a Markdown summary; mechanism runs default to JSON only.",
)
def main(
    config: Path,
    candidate_program: Path | None,
    request_count: int | None,
    seeds: tuple[int, ...],
    splits: tuple[str, ...],
    workloads: tuple[str, ...],
    capacity_blocks: tuple[int, ...],
    all_admission_policies: bool,
    shadow_price: bool,
    causal_components: bool,
    output: Path | None,
    markdown: Path | None,
) -> None:
    """Audit admission and eviction regret."""
    selected_splits = splits or _DEFAULT_SPLITS
    selected_modes = sum((all_admission_policies, shadow_price, causal_components))
    if selected_modes > 1:
        raise click.UsageError(
            "--all-admission-policies, --shadow-price, and --causal-components "
            "are mutually exclusive"
        )
    if capacity_blocks and not (shadow_price or causal_components):
        raise click.UsageError("--capacity-blocks requires --shadow-price or --causal-components")
    if all_admission_policies:
        output_path = output or Path(
            "artifacts/prefix_kv_cache_admission_eviction_policy_matrix.json"
        )
        markdown_path = markdown or Path(
            "artifacts/prefix_kv_cache_admission_eviction_policy_matrix.md"
        )
        payload = run_admission_eviction_matrix(
            config,
            request_count=request_count,
            seeds=seeds or None,
            splits=selected_splits,
            workloads=workloads or None,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        _write_admission_eviction_matrix_markdown(markdown_path, payload)
        click.echo(output_path)
        click.echo(markdown_path)
        return

    if causal_components:
        output_path = output or Path("artifacts/prefix_kv_cache_causal_component_factorial.json")
        factory = build_production_incumbent
        policy_name = "production_incumbent"
        if candidate_program is not None:
            factory = load_candidate_factory(str(candidate_program))
            policy_name = str(candidate_program)
        payload = run_causal_component_factorial(
            config,
            request_count=request_count,
            seeds=seeds or None,
            splits=selected_splits,
            workloads=workloads or None,
            capacity_blocks=capacity_blocks or None,
            factory=factory,
            policy_name=policy_name,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        click.echo(output_path)
        if markdown is not None:
            markdown.parent.mkdir(parents=True, exist_ok=True)
            _write_causal_component_markdown(markdown, payload)
            click.echo(markdown)
        return

    if shadow_price:
        output_path = output or Path("artifacts/prefix_kv_cache_shadow_price_tracking.json")
        factory = build_production_incumbent
        policy_name = "production_incumbent"
        if candidate_program is not None:
            factory = load_candidate_factory(str(candidate_program))
            policy_name = str(candidate_program)
        payload = run_shadow_price_analysis(
            config,
            request_count=request_count,
            seeds=seeds or None,
            splits=selected_splits,
            workloads=workloads or None,
            capacity_blocks=capacity_blocks or None,
            factory=factory,
            policy_name=policy_name,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        click.echo(output_path)
        if markdown is not None:
            markdown.parent.mkdir(parents=True, exist_ok=True)
            _write_shadow_price_markdown(markdown, payload)
            click.echo(markdown)
        return

    output_path = output or Path("artifacts/prefix_kv_cache_admission_eviction_regret_audit.json")
    markdown_path = markdown or Path("artifacts/prefix_kv_cache_admission_eviction_regret_audit.md")
    factory = build_incumbent
    policy_name = "pressure_aware_incumbent"
    if candidate_program is not None:
        factory = load_candidate_factory(str(candidate_program))
        policy_name = str(candidate_program)
    payload = run_analysis(
        config,
        request_count=request_count,
        seeds=seeds or None,
        splits=selected_splits,
        workloads=workloads or None,
        factory=factory,
        policy_name=policy_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(markdown_path, payload)
    click.echo(output_path)
    click.echo(markdown_path)


if __name__ == "__main__":
    main()
