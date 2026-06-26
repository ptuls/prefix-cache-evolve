"""Metric aggregation and scoring helpers for prefix KV-cache evaluation."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Iterable

from prefix_cache_evolve.evaluators.results import (
    MAX_AGGREGATION_FIELDS,
    MEAN_AGGREGATION_FIELDS,
)
from prefix_cache_evolve.evaluators.utilities import percentile as _percentile


def aggregate_by(
    keys: Iterable[str],
    trials: list[Any],
) -> dict[str, dict[str, float | int | bool | str]]:
    """Aggregate trials after grouping them by corresponding keys."""
    grouped: dict[str, list[Any]] = {}
    for key, trial in zip(keys, trials):
        grouped.setdefault(key, []).append(trial)
    return {key: aggregate_trials(value) for key, value in grouped.items()}


def aggregate_trials(
    trials: list[Any],
) -> dict[str, float | int | bool | str]:
    """Aggregate trial metrics into one summary mapping."""
    if not trials:
        return {}
    result: dict[str, float | int | bool | str] = {
        field: mean(float(getattr(trial, field)) for trial in trials)
        for field in MEAN_AGGREGATION_FIELDS
    }
    result.update(
        {
            field: max(int(getattr(trial, field)) for trial in trials)
            for field in MAX_AGGREGATION_FIELDS
        }
    )
    token_hit_rates = [trial.token_hit_rate for trial in trials]
    result["token_hit_rate_worst_trial"] = min(token_hit_rates)
    result["token_hit_rate_p10_across_trials"] = _percentile(token_hit_rates, 10)
    result["token_hit_rate_stddev_across_trials"] = (
        pstdev(token_hit_rates) if len(token_hit_rates) > 1 else 0.0
    )
    result["p95_latency_proxy_worst_trial"] = max(trial.p95_latency_proxy for trial in trials)
    result["cache_churn_per_1k_worst_trial"] = max(trial.cache_churn_per_1k for trial in trials)
    result["invalid_fraction"] = sum(1 for trial in trials if trial.invalid) / len(trials)
    result["invalid"] = any(trial.invalid for trial in trials)
    result["invalid_reason"] = "; ".join(
        sorted({trial.invalid_reason for trial in trials if trial.invalid_reason})
    )
    structural_keys = sorted({key for trial in trials for key in trial.structural_metrics})
    for key in structural_keys:
        result[key] = mean(float(trial.structural_metrics.get(key, 0.0)) for trial in trials)
    return result


def workload_base_score(
    trials: list[Any],
    *,
    token_weight: float,
    block_weight: float,
    request_tail_weight: float,
    worst_window_weight: float,
    priority_hit_weight: float,
    wasted_admission_weight: float,
    admission_utility_weight: float,
    avoidable_eviction_weight: float,
    latency_weight: float,
    latency_cap: float,
    latency_norm: float,
) -> float:
    """Compute the unnormalized behavioral score for one workload."""
    token_score = token_weight * mean(trial.token_hit_rate for trial in trials)
    block_score = block_weight * mean(trial.block_hit_rate for trial in trials)
    request_tail_score = request_tail_weight * mean(
        trial.request_token_hit_rate_p10 for trial in trials
    )
    worst_window_score = worst_window_weight * mean(
        trial.worst_quarter_token_hit_rate for trial in trials
    )
    priority_trials = [trial for trial in trials if trial.priority_request_fraction > 0.0]
    priority_score = (
        priority_hit_weight * mean(trial.high_priority_token_hit_rate for trial in priority_trials)
        if priority_trials
        else 0.0
    )
    wasted_admission_cost = wasted_admission_weight * mean(
        trial.wasted_admission_token_rate for trial in trials
    )
    admission_utility_score = admission_utility_weight * mean(
        math.log1p(trial.admission_token_utility) for trial in trials
    )
    avoidable_eviction_cost = avoidable_eviction_weight * mean(
        trial.avoidable_eviction_rate for trial in trials
    )
    latency = mean(trial.p95_latency_proxy for trial in trials)
    if latency_norm <= 0.0:
        latency_norm = max(
            (trial.max_prefill_cost for trial in trials),
            default=1.0,
        )
    latency_cost = min(
        latency_cap,
        latency_weight * latency / max(latency_norm, 1.0),
    )
    return (
        token_score
        + block_score
        + request_tail_score
        + worst_window_score
        + priority_score
        + admission_utility_score
        - latency_cost
        - wasted_admission_cost
        - avoidable_eviction_cost
    )
