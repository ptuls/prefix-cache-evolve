"""Secondary utilities for prefix KV-cache problem orchestration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from prefix_cache_evolve.artifacts import write_json as write_json
from prefix_cache_evolve.evaluators.results import EvaluationResult
from prefix_cache_evolve.evaluators.verifier import (
    require_single_score_identity,
    require_single_verifier_version,
)

AGENTIC_SURROGATE_WORKLOAD = "train/agentic_tool_workflows"
AGENTIC_PROBE_WORKLOAD = "probe/agent_trace_branching"
AGENTIC_SURROGATE_PROBE_DIVERGENCE_THRESHOLD = 0.12
AGENTIC_SURROGATE_GATE_ABSOLUTE_GAP_THRESHOLDS = {
    "request_token_hit_rate_p10": 0.15,
    "worst_quarter_token_hit_rate": 0.20,
    "wasted_admission_token_rate": 0.20,
    "policy_underfill_rate": 0.16,
    "short_reuse_after_eviction_missed_token_rate": 0.10,
}
AGENTIC_SURROGATE_GATE_NORMALIZED_GAP_THRESHOLDS = {
    "cache_churn_per_1k": {
        "threshold": 0.75,
        "scale_floor": 100.0,
    },
}
CYCLIC_SURROGATE_WORKLOAD = "validation/hotset_cold_scan"
CYCLIC_PROBE_WORKLOAD = "probe/cyclic_working_set_pressure"
CYCLIC_SURROGATE_PROBE_DIVERGENCE_THRESHOLD = 0.25
SURROGATE_PROBE_TRIPWIRE_SPECS = (
    {
        "name": "cyclic_working_set",
        "surrogate_workload": CYCLIC_SURROGATE_WORKLOAD,
        "probe_workload": CYCLIC_PROBE_WORKLOAD,
        "threshold": CYCLIC_SURROGATE_PROBE_DIVERGENCE_THRESHOLD,
    },
)


def evaluation_result_summary(result: EvaluationResult) -> dict[str, Any]:
    """Return the serializable summary fields used by reports and artifacts."""
    return {
        "verifier_version": result.verifier_version,
        "evaluation_context_sha256": result.evaluation_context_sha256,
        "panel_sha256": result.panel_sha256,
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
    payload = surrogate_probe_tripwire(
        workload_metrics,
        name="agentic_branching",
        surrogate_workload=AGENTIC_SURROGATE_WORKLOAD,
        probe_workload=AGENTIC_PROBE_WORKLOAD,
        threshold=threshold,
    )
    payload["schema"] = "prefix-kv-cache-agentic-surrogate-probe-tripwire-v1"
    return payload


def agentic_surrogate_probe_gate(
    workload_metrics: object,
    *,
    token_hit_threshold: float = AGENTIC_SURROGATE_PROBE_DIVERGENCE_THRESHOLD,
) -> dict[str, Any]:
    """Apply a fail-closed multi-metric agentic surrogate gate."""
    checks = {
        "token_hit_rate": surrogate_gate_metric_check(
            workload_metrics,
            metric="token_hit_rate",
            comparison="absolute_gap",
            threshold=token_hit_threshold,
        )
    }
    for metric, threshold in AGENTIC_SURROGATE_GATE_ABSOLUTE_GAP_THRESHOLDS.items():
        checks[metric] = surrogate_gate_metric_check(
            workload_metrics,
            metric=metric,
            comparison="absolute_gap",
            threshold=threshold,
        )
    for metric, spec in AGENTIC_SURROGATE_GATE_NORMALIZED_GAP_THRESHOLDS.items():
        checks[metric] = surrogate_gate_metric_check(
            workload_metrics,
            metric=metric,
            comparison="normalized_absolute_gap",
            threshold=spec["threshold"],
            scale_floor=spec["scale_floor"],
        )

    failed_metrics = [metric for metric, check in checks.items() if check["flagged"]]
    missing_metrics = [
        metric
        for metric, check in checks.items()
        if check["flag_reason"] == "missing_or_invalid_metric"
    ]
    max_threshold_ratio = max(
        (
            float(check["comparison_value"]) / float(check["threshold"])
            for check in checks.values()
            if check["comparison_value"] is not None and float(check["threshold"]) > 0.0
        ),
        default=None,
    )
    return {
        "schema": "prefix-kv-cache-agentic-surrogate-probe-gate-v1",
        "name": "agentic_branching",
        "selection_score_excludes_probe": True,
        "surrogate_workload": AGENTIC_SURROGATE_WORKLOAD,
        "probe_workload": AGENTIC_PROBE_WORKLOAD,
        "status": "flagged" if failed_metrics else "pass",
        "flagged": bool(failed_metrics),
        "flag_reason": (
            "missing_or_invalid_metric"
            if missing_metrics
            else "one_or_more_metric_gaps_exceed_threshold"
            if failed_metrics
            else None
        ),
        "checked_metric_count": len(checks),
        "failed_metric_count": len(failed_metrics),
        "failed_metrics": failed_metrics,
        "missing_metrics": missing_metrics,
        "max_threshold_ratio": max_threshold_ratio,
        "checks": checks,
    }


def surrogate_gate_metric_check(
    workload_metrics: object,
    *,
    metric: str,
    comparison: str,
    threshold: float,
    scale_floor: float | None = None,
) -> dict[str, Any]:
    """Build one fail-closed agentic surrogate-to-probe metric check."""
    surrogate_value = workload_metric_value(
        workload_metrics,
        AGENTIC_SURROGATE_WORKLOAD,
        metric,
    )
    probe_value = workload_metric_value(
        workload_metrics,
        AGENTIC_PROBE_WORKLOAD,
        metric,
    )
    check: dict[str, Any] = {
        "comparison": comparison,
        "surrogate_value": surrogate_value,
        "probe_value": probe_value,
        "surrogate_minus_probe": None,
        "absolute_gap": None,
        "comparison_value": None,
        "threshold": threshold,
    }
    if scale_floor is not None:
        check["scale_floor"] = scale_floor
    if surrogate_value is None or probe_value is None:
        check.update(
            {
                "status": "flagged",
                "flagged": True,
                "flag_reason": "missing_or_invalid_metric",
            }
        )
        return check

    signed_gap = surrogate_value - probe_value
    absolute_gap = abs(signed_gap)
    comparison_value = absolute_gap
    if comparison == "normalized_absolute_gap":
        if scale_floor is None or scale_floor <= 0:
            raise ValueError("normalized absolute gap requires a positive scale floor")
        comparison_value = absolute_gap / max(
            abs(surrogate_value),
            abs(probe_value),
            scale_floor,
        )
    elif comparison != "absolute_gap":
        raise ValueError(f"unsupported surrogate gate comparison: {comparison}")
    flagged = comparison_value > threshold
    check.update(
        {
            "status": "flagged" if flagged else "pass",
            "flagged": flagged,
            "flag_reason": "gap_exceeds_threshold" if flagged else None,
            "surrogate_minus_probe": signed_gap,
            "absolute_gap": absolute_gap,
            "comparison_value": comparison_value,
        }
    )
    return check


def surrogate_probe_tripwire(
    workload_metrics: object,
    *,
    name: str,
    surrogate_workload: str,
    probe_workload: str,
    threshold: float,
) -> dict[str, Any]:
    """Flag excessive divergence for one non-quarantined-to-probe workload pair."""
    payload: dict[str, Any] = {
        "schema": "prefix-kv-cache-surrogate-probe-tripwire-channel-v1",
        "name": name,
        "selection_score_excludes_probe": True,
        "metric": "token_hit_rate",
        "surrogate_workload": surrogate_workload,
        "probe_workload": probe_workload,
        "surrogate_value": None,
        "probe_value": None,
        "surrogate_minus_probe": None,
        "absolute_gap": None,
        "threshold": threshold,
    }
    surrogate_value = tripwire_metric_value(workload_metrics, surrogate_workload)
    probe_value = tripwire_metric_value(workload_metrics, probe_workload)
    payload["surrogate_value"] = surrogate_value
    payload["probe_value"] = probe_value
    if surrogate_value is None or probe_value is None:
        payload.update(
            {
                "status": "flagged",
                "flagged": True,
                "flag_reason": "missing_or_invalid_metric",
                "interpretation": (
                    f"The {name} tripwire could not compare both workloads and failed closed."
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
                f"The {name} surrogate and held-out probe diverge beyond the allowed threshold."
                if flagged
                else f"The {name} surrogate and held-out probe remain within the allowed threshold."
            ),
        }
    )
    return payload


def surrogate_probe_tripwire_suite(
    workload_metrics: object,
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate every configured surrogate-to-probe generalization channel."""
    thresholds = thresholds or {}
    channels = {
        "agentic_branching": agentic_surrogate_probe_gate(
            workload_metrics,
            token_hit_threshold=float(
                thresholds.get(
                    "agentic_branching",
                    AGENTIC_SURROGATE_PROBE_DIVERGENCE_THRESHOLD,
                )
            ),
        ),
        **{
            str(spec["name"]): surrogate_probe_tripwire(
                workload_metrics,
                name=str(spec["name"]),
                surrogate_workload=str(spec["surrogate_workload"]),
                probe_workload=str(spec["probe_workload"]),
                threshold=float(thresholds.get(str(spec["name"]), spec["threshold"])),
            )
            for spec in SURROGATE_PROBE_TRIPWIRE_SPECS
        },
    }
    flagged_channels = [name for name, channel in channels.items() if bool(channel["flagged"])]
    valid_ratios = [
        float(channel["max_threshold_ratio"])
        if "max_threshold_ratio" in channel
        else float(channel["absolute_gap"]) / float(channel["threshold"])
        for channel in channels.values()
        if (
            channel.get("max_threshold_ratio") is not None
            or channel.get("absolute_gap") is not None
            and float(channel["threshold"]) > 0.0
        )
    ]
    return {
        "schema": "prefix-kv-cache-surrogate-probe-gate-suite-v2",
        "selection_score_excludes_probe": True,
        "status": "flagged" if flagged_channels else "pass",
        "flagged": bool(flagged_channels),
        "flagged_channels": flagged_channels,
        "passed_channels": [name for name in channels if name not in flagged_channels],
        "max_threshold_ratio": max(valid_ratios) if valid_ratios else None,
        "channels": channels,
    }


def tripwire_metric_value(workload_metrics: object, workload: str) -> float | None:
    """Return a finite workload token-hit rate, or None when unavailable."""
    return workload_metric_value(workload_metrics, workload, "token_hit_rate")


def workload_metric_value(
    workload_metrics: object,
    workload: str,
    metric: str,
) -> float | None:
    """Return a finite workload metric, or None when unavailable."""
    if not isinstance(workload_metrics, dict):
        return None
    metrics = workload_metrics.get(workload)
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric_value = float(value)
    return numeric_value if math.isfinite(numeric_value) else None


def write_agentic_surrogate_probe_gate_report(
    path: Path,
    gate: dict[str, Any],
) -> None:
    """Write a compact human-readable multi-metric agentic surrogate gate report."""
    lines = [
        "# Agentic Surrogate-to-Probe Gate",
        "",
        f"**Status: {str(gate['status']).upper()}**",
        "",
        (
            "This fail-closed gate compares aggregate, tail, admission, utilization, "
            "eviction, and churn behavior on the non-quarantined agentic surrogate "
            "with the held-out agentic probe. The probe remains excluded from selection."
        ),
        "",
        "| Metric | Comparison | Surrogate | Probe | Gate value | Limit | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for metric, check in gate["checks"].items():
        lines.append(
            f"| `{metric}` | `{check['comparison']}` | "
            f"{format_promotion_value(check['surrogate_value'])} | "
            f"{format_promotion_value(check['probe_value'])} | "
            f"{format_promotion_value(check['comparison_value'])} | "
            f"{float(check['threshold']):.6f} | {check['status']} |"
        )
    lines.extend(
        [
            "",
            (
                "Failed metrics: "
                + (
                    ", ".join(f"`{metric}`" for metric in gate["failed_metrics"])
                    if gate["failed_metrics"]
                    else "none"
                )
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_surrogate_probe_tripwire_report(
    path: Path,
    suite: dict[str, Any],
) -> None:
    """Write a compact report for all surrogate-to-probe tripwire channels."""
    lines = [
        "# Surrogate-to-Probe Gate Suite",
        "",
        f"**Status: {str(suite['status']).upper()}**",
        "",
        (
            "Each channel compares a non-quarantined workload with a related held-out "
            "probe. Probe metrics remain excluded from selection and mutation feedback."
        ),
        "",
        "| Channel | Comparison | Failed checks | Max threshold ratio | Status |",
        "|---|---|---|---:|---|",
    ]
    for name, channel in suite["channels"].items():
        if "checks" in channel:
            comparison = "multi-metric"
            failed = ", ".join(channel["failed_metrics"]) or "-"
            ratio = channel["max_threshold_ratio"]
        else:
            comparison = str(channel["metric"])
            failed = str(channel["flag_reason"] or "-")
            ratio = (
                float(channel["absolute_gap"]) / float(channel["threshold"])
                if channel["absolute_gap"] is not None and float(channel["threshold"]) > 0.0
                else None
            )
        lines.append(
            f"| `{name}` | `{comparison}` | {failed} | "
            f"{format_promotion_value(ratio)} | {channel['status']} |"
        )
    lines.extend(
        [
            "",
            (
                "Flagged channels: "
                + (
                    ", ".join(f"`{name}`" for name in suite["flagged_channels"])
                    if suite["flagged_channels"]
                    else "none"
                )
            ),
            "",
        ]
    )
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
    require_single_score_identity(
        (candidate[panel], incumbent[panel]),
        context=f"{panel} score comparison",
    )
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
    require_single_score_identity(
        (candidate["selection"], incumbent["selection"]),
        context="raw selection comparison",
    )
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
    require_single_score_identity(
        (candidate["selection"], incumbent["selection"]),
        context=f"{split} metric comparison",
    )
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
    require_single_score_identity(
        (candidate[panel], incumbent[panel]),
        context=f"{panel} workload comparison",
    )
    candidate_value = float(candidate[panel]["workload_metrics"][workload][metric])
    incumbent_value = float(incumbent[panel]["workload_metrics"][workload][metric])
    return promotion_check(
        math.isfinite(candidate_value)
        and math.isfinite(incumbent_value)
        and candidate_value >= incumbent_value,
        candidate_value,
        incumbent_value,
    )


def promotion_tripwire_check(
    candidate: dict[str, Any],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply all surrogate-to-probe tripwires to a promotion candidate."""
    workloads = {
        **candidate["selection"]["workload_metrics"],
        **candidate["probe"]["workload_metrics"],
    }
    tripwire = surrogate_probe_tripwire_suite(workloads, thresholds=thresholds)
    return {
        "passed": not tripwire["flagged"],
        "candidate": tripwire.get("max_threshold_ratio"),
        "incumbent_or_limit": 1.0,
        "tripwire": tripwire,
    }


def write_specialist_promotion_adjudication_report(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write a compact specialist promotion report."""
    verifier_version = str(payload["verifier_version"])
    lines = [
        "# Specialist Promotion Adjudication",
        "",
        f"Verifier: `{verifier_version}`",
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
    records = tuple(
        candidate[panel] for _, candidate in rows for panel in ("selection", "probe", "hidden")
    )
    verifier_version = require_single_verifier_version(
        records,
        context="generated mutation report",
    )
    identities = {
        panel: require_single_score_identity(
            (candidate[panel] for _, candidate in rows),
            context=f"generated mutation {panel} comparison",
        )
        for panel in ("selection", "probe", "hidden")
    }
    lines = [
        "# Best Generated Mutation Decomposition",
        "",
        f"Verifier: `{verifier_version}`",
        "",
        "Evaluation contexts: "
        + ", ".join(
            f"`{panel}={identity.evaluation_context_sha256}`"
            for panel, identity in identities.items()
        ),
        "",
        "Panels: "
        + ", ".join(f"`{panel}={identity.panel_sha256}`" for panel, identity in identities.items()),
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
