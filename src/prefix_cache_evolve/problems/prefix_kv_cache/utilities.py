"""Secondary utilities for prefix KV-cache problem orchestration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from prefix_cache_evolve.evaluators.prefix_kv_cache import EvaluationResult

AGENTIC_SURROGATE_WORKLOAD = "train/agentic_tool_workflows"
AGENTIC_PROBE_WORKLOAD = "probe/agent_trace_branching"
AGENTIC_SURROGATE_PROBE_DIVERGENCE_THRESHOLD = 0.12


def write_json(path: Path, payload: Any) -> None:
    """Write stable, human-reviewable JSON and create parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def evaluation_result_summary(result: EvaluationResult) -> dict[str, Any]:
    """Return the serializable summary fields used by reports and artifacts."""
    return {
        "combined_score": result.combined_score,
        "success": result.success,
        "invalid_fraction": result.invalid_fraction,
        "split_metrics": result.split_metrics,
        "workload_metrics": result.workload_metrics,
        "capacity_metrics": result.capacity_metrics,
        "candidate_metadata": result.candidate_metadata,
        "score_breakdown": result.score_breakdown,
    }


def parse_positive_int_csv(value: str, *, option_name: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive integers."""
    if not value.strip():
        return ()
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        return ()
    if any(item <= 0 for item in values):
        raise ValueError(f"{option_name} values must be positive")
    return values


def parse_unique_positive_int_csv(value: str, *, option_name: str) -> tuple[int, ...]:
    """Parse positive integers while preserving order and removing duplicates."""
    values = parse_positive_int_csv(value, option_name=option_name)
    return tuple(dict.fromkeys(values))


def capacity_blocks_for_token_tiers(
    capacity_tokens: tuple[int, ...],
    *,
    block_size_tokens: int,
) -> tuple[int, ...]:
    """Convert fixed token-capacity tiers into exact block counts."""
    if block_size_tokens <= 0:
        raise ValueError("block size tokens must be positive")
    if not capacity_tokens or any(capacity <= 0 for capacity in capacity_tokens):
        raise ValueError("capacity token tiers must be positive")
    if any(capacity % block_size_tokens for capacity in capacity_tokens):
        raise ValueError("capacity token tiers must be divisible by every block size")
    return tuple(capacity // block_size_tokens for capacity in capacity_tokens)


def format_int_tuple(values: tuple[int, ...]) -> str:
    """Format a tuple of integers for compact Markdown tables."""
    return " / ".join(str(value) for value in values)


def agentic_surrogate_probe_tripwire(
    workload_metrics: object,
    *,
    threshold: float = AGENTIC_SURROGATE_PROBE_DIVERGENCE_THRESHOLD,
) -> dict[str, Any]:
    """Flag excessive agentic surrogate-to-held-out-probe divergence."""
    payload: dict[str, Any] = {
        "schema": "prefix-kv-cache-agentic-surrogate-probe-tripwire-v1",
        "selection_score_excludes_probe": True,
        "metric": "token_hit_rate",
        "surrogate_workload": AGENTIC_SURROGATE_WORKLOAD,
        "probe_workload": AGENTIC_PROBE_WORKLOAD,
        "surrogate_value": None,
        "probe_value": None,
        "surrogate_minus_probe": None,
        "absolute_gap": None,
        "threshold": threshold,
    }
    surrogate_value = tripwire_metric_value(workload_metrics, AGENTIC_SURROGATE_WORKLOAD)
    probe_value = tripwire_metric_value(workload_metrics, AGENTIC_PROBE_WORKLOAD)
    payload["surrogate_value"] = surrogate_value
    payload["probe_value"] = probe_value
    if surrogate_value is None or probe_value is None:
        payload.update(
            {
                "status": "flagged",
                "flagged": True,
                "flag_reason": "missing_or_invalid_metric",
                "interpretation": (
                    "The tripwire could not compare both agentic workloads and failed closed."
                ),
            }
        )
        return payload

    signed_gap = surrogate_value - probe_value
    absolute_gap = abs(signed_gap)
    flagged = absolute_gap > threshold
    payload.update(
        {
            "status": "flagged" if flagged else "pass",
            "flagged": flagged,
            "flag_reason": "divergence_exceeds_threshold" if flagged else None,
            "surrogate_minus_probe": signed_gap,
            "absolute_gap": absolute_gap,
            "interpretation": (
                "The surrogate and held-out agentic probe diverge beyond the allowed threshold."
                if flagged
                else "The surrogate and held-out agentic probe remain within the allowed threshold."
            ),
        }
    )
    return payload


def tripwire_metric_value(workload_metrics: object, workload: str) -> float | None:
    """Return a finite workload token-hit rate, or None when unavailable."""
    if not isinstance(workload_metrics, dict):
        return None
    metrics = workload_metrics.get(workload)
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("token_hit_rate")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric_value = float(value)
    return numeric_value if math.isfinite(numeric_value) else None


def write_agentic_surrogate_probe_tripwire_report(
    path: Path,
    tripwire: dict[str, Any],
) -> None:
    """Write a compact human-readable agentic divergence tripwire report."""
    lines = [
        "# Agentic Surrogate-to-Probe Tripwire",
        "",
        f"**Status: {str(tripwire['status']).upper()}**",
        "",
        (
            "This check compares token hit rate on the non-quarantined agentic surrogate "
            "with the held-out agentic probe. The probe remains excluded from selection."
        ),
        "",
    ]
    if tripwire["absolute_gap"] is None:
        lines.append(
            "The check failed closed because one or both required workload metrics were "
            "missing or invalid."
        )
    else:
        lines.extend(
            [
                "| Surrogate | Held-out probe | Surrogate - probe | Absolute gap | Threshold |",
                "|---:|---:|---:|---:|---:|",
                (
                    f"| {tripwire['surrogate_value']:.4f} | {tripwire['probe_value']:.4f} | "
                    f"{tripwire['surrogate_minus_probe']:.4f} | "
                    f"{tripwire['absolute_gap']:.4f} | {tripwire['threshold']:.4f} |"
                ),
                "",
                str(tripwire["interpretation"]),
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def promotion_check(passed: bool, candidate: object, incumbent: object) -> dict[str, Any]:
    """Build one serializable promotion-check result."""
    return {
        "passed": passed,
        "candidate": candidate,
        "incumbent_or_limit": incumbent,
    }


def score_non_regression(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    *,
    panel: str,
) -> dict[str, Any]:
    """Check a candidate panel score against the incumbent."""
    candidate_score = float(candidate[panel]["combined_score"])
    incumbent_score = float(incumbent[panel]["combined_score"])
    return promotion_check(
        math.isfinite(candidate_score)
        and math.isfinite(incumbent_score)
        and candidate_score >= incumbent_score,
        candidate_score,
        incumbent_score,
    )


def raw_selection_improvement(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
) -> dict[str, Any]:
    """Require strictly better selection behavior before complexity accounting."""
    candidate_selection = candidate["selection"]
    incumbent_selection = incumbent["selection"]
    candidate_raw = float(candidate_selection["combined_score"]) + float(
        candidate_selection["score_breakdown"].get("complexity_cost", 0.0)
    )
    incumbent_raw = float(incumbent_selection["combined_score"]) + float(
        incumbent_selection["score_breakdown"].get("complexity_cost", 0.0)
    )
    return promotion_check(
        math.isfinite(candidate_raw)
        and math.isfinite(incumbent_raw)
        and candidate_raw > incumbent_raw,
        candidate_raw,
        incumbent_raw,
    )


def split_metric_non_regression(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    *,
    split: str,
    metric: str,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    """Check one aggregate split metric against the incumbent."""
    candidate_value = float(candidate["selection"]["split_metrics"][split][metric])
    incumbent_value = float(incumbent["selection"]["split_metrics"][split][metric])
    passed = (
        candidate_value <= incumbent_value
        if lower_is_better
        else candidate_value >= incumbent_value
    )
    return promotion_check(
        math.isfinite(candidate_value) and math.isfinite(incumbent_value) and passed,
        candidate_value,
        incumbent_value,
    )


def workload_metric_non_regression(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    *,
    panel: str,
    workload: str,
    metric: str,
) -> dict[str, Any]:
    """Check one candidate workload metric against the incumbent."""
    candidate_value = float(candidate[panel]["workload_metrics"][workload][metric])
    incumbent_value = float(incumbent[panel]["workload_metrics"][workload][metric])
    return promotion_check(
        math.isfinite(candidate_value)
        and math.isfinite(incumbent_value)
        and candidate_value >= incumbent_value,
        candidate_value,
        incumbent_value,
    )


def promotion_tripwire_check(candidate: dict[str, Any]) -> dict[str, Any]:
    """Apply the existing agentic surrogate-to-probe tripwire to a candidate."""
    workloads = {
        **candidate["selection"]["workload_metrics"],
        **candidate["probe"]["workload_metrics"],
    }
    tripwire = agentic_surrogate_probe_tripwire(workloads)
    return {
        "passed": not tripwire["flagged"],
        "candidate": tripwire.get("absolute_gap"),
        "incumbent_or_limit": tripwire["threshold"],
        "tripwire": tripwire,
    }


def write_specialist_promotion_adjudication_report(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write a compact specialist promotion report."""
    lines = [
        "# Specialist Promotion Adjudication",
        "",
        f"**Status: {str(payload['status']).upper()}**",
        "",
        str(payload["interpretation"]),
        "",
    ]
    checks = payload.get("checks")
    if isinstance(checks, dict):
        lines.extend(
            [
                "| Check | Passed | Candidate | Incumbent or limit |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, check in checks.items():
            lines.append(
                f"| `{name}` | {check['passed']} | "
                f"{format_promotion_value(check.get('candidate'))} | "
                f"{format_promotion_value(check.get('incumbent_or_limit'))} |"
            )
    elif payload.get("error_message"):
        lines.append(f"Adjudication error: `{payload['error_type']}: {payload['error_message']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def format_promotion_value(value: object) -> str:
    """Format one promotion result value for Markdown."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def elite_source(elite: dict[str, Any]) -> str:
    """Return source text from a Levi snapshot elite."""
    return str(elite.get("content") or elite.get("code") or "")


def normalized_source(source: str) -> str:
    """Normalize source for exact seed-versus-generated identity checks."""
    return source.strip()


def write_generated_mutation_report(
    path: Path,
    decomposition: dict[str, Any],
) -> None:
    """Write a compact incumbent-versus-generated decomposition table."""
    rows = [
        ("Seed", decomposition["seed"]),
        ("Best generated mutation", decomposition["best_generated_mutation"]),
    ]
    lines = [
        "# Best Generated Mutation Decomposition",
        "",
        (
            "The recurrence-heavy probe and hidden panel are reporting-only and "
            "do not affect the selection combined score."
        ),
        "",
        "| Candidate | Selection | Raw before cx | Mean | Min contrib. | "
        "Churn cost | Underfill cost | Cx | Cx subsidy | Probe | Agent hit | "
        "Cyclic hit | Hidden |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, candidate in rows:
        selection = candidate["selection"]
        selection_breakdown = selection["score_breakdown"]
        probe = candidate["probe"]
        probe_workloads = probe["workload_metrics"]
        hidden = candidate["hidden"]
        raw_before_complexity = selection["combined_score"] + selection_breakdown.get(
            "complexity_cost", 0.0
        )
        lines.append(
            f"| {label} | {selection['combined_score']:.3f} | "
            f"{raw_before_complexity:.3f} | "
            f"{selection_breakdown.get('mean_workload_score', 0.0):.3f} | "
            f"{selection_breakdown.get('min_workload_contribution', 0.0):.3f} | "
            f"{selection_breakdown.get('churn_cost', 0.0):.3f} | "
            f"{selection_breakdown.get('underfill_cost', 0.0):.3f} | "
            f"{candidate['effective_complexity']} | "
            f"{candidate['primitive_subsidy_nodes']} | "
            f"{probe['combined_score']:.3f} | "
            f"{probe_workloads['probe/agent_trace_branching']['token_hit_rate']:.4f} | "
            f"{probe_workloads['probe/cyclic_working_set_pressure']['token_hit_rate']:.4f} | "
            f"{hidden['combined_score']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
