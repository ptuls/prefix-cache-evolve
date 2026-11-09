"""Shared builders for concise score-bearing test records."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from prefix_cache_evolve.evaluators.results import EvaluationResult
from prefix_cache_evolve.evaluators.verifier import VERIFIER_VERSION

TEST_CONTEXT_SHA = "b" * 64
TEST_PANEL_SHA = "a" * 64


def score_identity(
    *,
    verifier_version: str = VERIFIER_VERSION,
    evaluation_context_sha256: str = TEST_CONTEXT_SHA,
    panel_sha256: str = TEST_PANEL_SHA,
) -> dict[str, str]:
    """Return a complete synthetic score identity."""
    return {
        "verifier_version": verifier_version,
        "evaluation_context_sha256": evaluation_context_sha256,
        "panel_sha256": panel_sha256,
    }


def score_record(
    combined_score: float,
    *,
    verifier_version: str = VERIFIER_VERSION,
    evaluation_context_sha256: str = TEST_CONTEXT_SHA,
    panel_sha256: str = TEST_PANEL_SHA,
    **fields: Any,
) -> SimpleNamespace:
    """Return a namespace representing one score-bearing result."""
    return SimpleNamespace(
        **score_identity(
            verifier_version=verifier_version,
            evaluation_context_sha256=evaluation_context_sha256,
            panel_sha256=panel_sha256,
        ),
        combined_score=combined_score,
        **fields,
    )


def report_result(
    score: float,
    *,
    capacities: tuple[int, ...] = (8, 16),
) -> EvaluationResult:
    split_metrics = {
        "block_hit_rate": 0.4,
        "token_hit_rate": 0.5,
        "worst_quarter_token_hit_rate": 0.3,
        "request_token_hit_rate_p10": 0.2,
        "wasted_admission_token_rate": 0.1,
        "admission_token_utility": 1.5,
        "avoidable_eviction_rate": 0.05,
        "policy_underfill_rate": 0.04,
        "cache_churn_per_1k": 10.0,
    }
    workload_metrics = {
        "token_hit_rate": 0.5,
        "block_hit_rate": 0.4,
        "cache_churn_per_1k": 10.0,
        "priority_weighted_token_hit_rate": 0.55,
    }
    return EvaluationResult(
        **score_identity(),
        combined_score=score,
        success=True,
        invalid_fraction=0.0,
        split_metrics={
            "validation": dict(split_metrics),
            "probe": dict(split_metrics),
        },
        workload_metrics={
            "validation/priority_burst_recovery": dict(workload_metrics),
            "validation/priority_one_off_noise": dict(workload_metrics),
            "probe/agent_trace_branching": dict(workload_metrics),
        },
        capacity_metrics={
            f"capacity_{capacity}": dict(workload_metrics) for capacity in capacities
        },
        candidate_metadata={"scoring_fn_complexity": 3},
        score_breakdown={
            "combined_score": score,
            "mean_workload_score": score,
            "min_workload_contribution": 0.0,
            "churn_cost": 0.0,
            "underfill_cost": 0.0,
            "fairness_cost": 0.0,
            "complexity_cost": 0.0,
        },
    )
