"""Paired robustness diagnostics for the headline score gap.

The repository headline compares the production incumbent against a named
baseline (TinyLFU-LRU) by a single aggregate combined score. That aggregate
hides how consistently the candidate leads across workload families and generated
panel realizations. This tool decomposes the operative validation panel into paired
per-group behavioral scores and reports panel-level, family-level, and seed-level
diagnostics without treating nominal trial rows as independent observations.

Pairing unit
------------
The uncertainty unit is one complete validation panel generated from one base
seed. Both policies replay the *same* request streams (identical panel fingerprint),
and their fully charged scores are differenced once per base seed. Per-workload and
per-capacity trial differences remain useful for descriptive decomposition but are
not treated as independent uncertainty observations.

The global churn, underfill, fairness, and complexity charges are policy-level
offsets applied once to the combined score, not per-group quantities, so they
are excluded from the paired unit and reported separately as context.

Interpretation and pseudo-replication
-------------------------------------
The workload generator does not vary every family's request stream with the
seed: several families produce byte-identical per-seed scores, so the
``(family, capacity, seed)`` groups are *not* mutually independent. Treating all
of them as independent units would inflate the effective sample size
(pseudo-replication). With only three configured base seeds, neither a bootstrap
nor a t interval can reliably estimate seed uncertainty. The default report therefore
uses the charged score difference, family wins and losses, the range across paired
whole-panel seed realizations, and leave-one-family-out sensitivity. A family bootstrap
is retained only as a descriptive stability interval, not a population-level
confidence interval.

For future experiments, configure at least 20 independent outer seeds. The tool then
reports a paired bootstrap confidence interval over whole-panel charged-score
differences, conditional on the fixed workload-family panel. Generalization beyond
that panel requires a separately defined workload-family sampling distribution.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

import click

from prefix_cache_evolve.artifacts import write_json
from prefix_cache_evolve.evaluator_entry import load_candidate_factory
from prefix_cache_evolve.evaluators.baselines import BASELINE_REGISTRY
from prefix_cache_evolve.evaluators.complexity import scoring_fn_complexity
from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.evaluators.contracts import PrefixKVPolicy
from prefix_cache_evolve.evaluators.prefix_kv_cache import PrefixKVCacheEvaluator
from prefix_cache_evolve.evaluators.results import EvaluationResult, TrialMetrics
from prefix_cache_evolve.evaluators.scoring import workload_base_score
from prefix_cache_evolve.evaluators.verifier import require_single_score_identity
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import (
    DEFAULT_CONFIG_PATH,
    load_evaluator_config,
)
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents import (
    build_current_incumbent as build_production_incumbent,
)
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents.registry import (
    current_incumbent,
)

_DEFAULT_SPLITS = ("validation",)
_DEFAULT_BASELINE = "tinylfu_lru"
_SCHEMA = "prefix-kv-cache-score-gap-robustness-v2"
_MIN_OUTER_SEEDS_FOR_INTERVAL = 20
_PRODUCTION_COMPLEXITY = current_incumbent("production").effective_complexity


@dataclass(frozen=True, slots=True)
class PairedUnit:
    """One paired per-group behavioral score for candidate and baseline."""

    group: str
    split: str
    workload: str
    capacity_blocks: int
    base_seed: int
    seed: int
    candidate_score: float
    baseline_score: float

    @property
    def difference(self) -> float:
        """Return the candidate-minus-baseline paired difference."""
        return self.candidate_score - self.baseline_score


def group_score(trial: TrialMetrics, config: EvaluatorConfig) -> float:
    """Return the single-trial behavioral score under configured weights.

    This is the evaluator building block: the combined score averages this
    quantity over a seed group (blended with the per-group seed floor) before
    applying policy-level penalty offsets.
    """
    return workload_base_score(
        [trial],
        token_weight=config.w_avg_tok,
        block_weight=config.w_avg_blk,
        request_tail_weight=config.request_tail_weight,
        worst_window_weight=config.worst_window_weight,
        priority_hit_weight=config.priority_hit_weight,
        wasted_admission_weight=config.wasted_admission_weight,
        admission_utility_weight=config.admission_utility_weight,
        avoidable_eviction_weight=config.avoidable_eviction_weight,
        latency_weight=config.latency_weight,
        latency_cap=config.latency_cap,
        latency_norm=config.latency_norm,
    )


def _quantile(sorted_values: list[float], quantile: float) -> float:
    """Return a linearly interpolated quantile of a pre-sorted sample."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * fraction


def bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Return a deterministic percentile bootstrap CI for the mean.

    The paired differences are resampled with replacement using a seeded
    ``random.Random`` so the interval is reproducible for fixed inputs.
    """
    if not values:
        raise ValueError("bootstrap requires at least one paired difference")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    count = len(values)
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    lower_quantile = (1.0 - confidence) / 2.0
    return {
        "lower": _quantile(means, lower_quantile),
        "upper": _quantile(means, 1.0 - lower_quantile),
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
    }


def cluster_mean_differences(
    units: list[PairedUnit],
    key: Callable[[PairedUnit], tuple[object, ...]],
) -> dict[tuple[object, ...], float]:
    """Return the mean paired difference within each cluster.

    Collapsing correlated groups avoids counting seed-invariant rows repeatedly.
    It does not make fixed, hand-designed workload families a random population.
    """
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for unit in units:
        grouped[key(unit)].append(unit.difference)
    return {cluster: mean(diffs) for cluster, diffs in grouped.items()}


def descriptive_family_summary(
    units: list[PairedUnit],
    *,
    confidence: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Summarize paired differences across the fixed workload families."""
    clustered = cluster_mean_differences(
        units,
        lambda unit: (unit.split, unit.workload),
    )
    cluster_diffs = list(clustered.values())
    wins = sum(value > 1e-12 for value in cluster_diffs)
    losses = sum(value < -1e-12 for value in cluster_diffs)
    mean_difference = mean(cluster_diffs)
    return {
        "family_count": len(cluster_diffs),
        "family_wins": wins,
        "family_losses": losses,
        "family_ties": len(cluster_diffs) - wins - losses,
        "mean_difference": mean_difference,
        "descriptive_family_bootstrap_interval": bootstrap_mean_ci(
            cluster_diffs,
            confidence=confidence,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "family_differences": {
            f"{split}/{workload}": difference
            for (split, workload), difference in sorted(clustered.items())
        },
        "note": (
            "The bootstrap resamples the fixed, hand-designed workload families. "
            "It is a descriptive stability interval, not a population-level "
            "confidence interval."
        ),
    }


def seed_degeneracy_report(units: list[PairedUnit]) -> dict[str, object]:
    """Report how many ``(family, capacity)`` cells ignore the seed.

    A cell whose per-seed differences are all identical contributes no
    independent seed replicates, which is why per-group statistics overstate the
    effective sample size.
    """
    cells: dict[tuple[str, int], list[float]] = defaultdict(list)
    for unit in units:
        cells[(unit.workload, unit.capacity_blocks)].append(round(unit.difference, 9))
    invariant = [
        f"{workload}/capacity_{capacity}"
        for (workload, capacity), diffs in cells.items()
        if len(diffs) > 1 and len(set(diffs)) == 1
    ]
    return {
        "cell_count": len(cells),
        "seed_invariant_cell_count": len(invariant),
        "seed_invariant_cells": sorted(invariant),
        "note": (
            "Cells whose per-seed paired differences are identical contribute no "
            "seed variation. Nominal trial rows must not be treated as independent "
            "replicates."
        ),
    }


def _score_map(
    factory: Callable[..., PrefixKVPolicy],
    config: EvaluatorConfig,
    splits: tuple[str, ...],
    *,
    complexity: int,
) -> tuple[dict[tuple[str, str, int, int], float], dict[str, object], EvaluationResult]:
    """Return per-group behavioral scores, panel identity, and the full result."""
    result = PrefixKVCacheEvaluator(config, splits=splits)(
        factory,
        scoring_fn_complexity=complexity,
    )
    scores = {
        (trial.split, trial.workload, trial.capacity_blocks, trial.seed): group_score(trial, config)
        for trial in result.trials
    }
    identity = {
        "verifier_version": result.verifier_version,
        "evaluation_context_sha256": result.evaluation_context_sha256,
        "panel_sha256": result.panel_sha256,
    }
    return scores, identity, result


def _base_seed_offsets(
    config: EvaluatorConfig,
    splits: tuple[str, ...],
) -> dict[tuple[str, str], int]:
    """Return the configured seed offset for each split and workload family."""
    return {
        (workload.split, workload.family): workload.seed_offset
        for workload in config.workload_configs(splits)
    }


def _rescored_pair(
    evaluator: PrefixKVCacheEvaluator,
    candidate_trials: list[TrialMetrics],
    baseline_trials: list[TrialMetrics],
    *,
    candidate_complexity: int,
) -> dict[str, float]:
    """Return charged scores for one matched subset of candidate and baseline trials."""
    candidate = evaluator.rescore_trials(
        candidate_trials,
        scoring_fn_complexity=candidate_complexity,
    )
    baseline = evaluator.rescore_trials(baseline_trials, scoring_fn_complexity=0)
    return {
        "candidate_charged_score": candidate.combined_score,
        "baseline_charged_score": baseline.combined_score,
        "charged_score_difference": candidate.combined_score - baseline.combined_score,
    }


def whole_panel_seed_report(
    units: list[PairedUnit],
    candidate_result: EvaluationResult,
    baseline_result: EvaluationResult,
    config: EvaluatorConfig,
    splits: tuple[str, ...],
    *,
    candidate_complexity: int,
    confidence: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Report one paired charged-score difference per independent base seed."""
    offsets = _base_seed_offsets(config, splits)
    evaluator = PrefixKVCacheEvaluator(config, splits=splits)
    rows = []
    for base_seed in config.seeds:
        candidate_trials = [
            trial
            for trial in candidate_result.trials
            if trial.seed - offsets[(trial.split, trial.workload)] == base_seed
        ]
        baseline_trials = [
            trial
            for trial in baseline_result.trials
            if trial.seed - offsets[(trial.split, trial.workload)] == base_seed
        ]
        scores = _rescored_pair(
            evaluator,
            candidate_trials,
            baseline_trials,
            candidate_complexity=candidate_complexity,
        )
        behavioral_differences = [unit.difference for unit in units if unit.base_seed == base_seed]
        rows.append(
            {
                "base_seed": base_seed,
                **scores,
                "mean_uncharged_behavioral_difference": mean(behavioral_differences),
            }
        )

    charged_differences = [float(row["charged_score_difference"]) for row in rows]
    inference_ready = len(rows) >= _MIN_OUTER_SEEDS_FOR_INTERVAL
    interval = (
        bootstrap_mean_ci(
            charged_differences,
            confidence=confidence,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        if inference_ready
        else None
    )
    return {
        "outer_seed_count": len(rows),
        "minimum_outer_seeds_for_interval": _MIN_OUTER_SEEDS_FOR_INTERVAL,
        "paired_whole_panel_differences": rows,
        "charged_score_difference_range": {
            "minimum": min(charged_differences),
            "maximum": max(charged_differences),
        },
        "paired_outer_seed_bootstrap_confidence_interval": interval,
        "inference_ready": inference_ready,
        "note": (
            "The interval is omitted because three outer seeds cannot support a "
            "reliable uncertainty estimate. Report the observed range descriptively."
            if not inference_ready
            else "The paired bootstrap resamples whole-panel outer-seed differences "
            "and is conditional on the fixed workload-family panel."
        ),
    }


def leave_one_family_out_report(
    units: list[PairedUnit],
    candidate_result: EvaluationResult,
    baseline_result: EvaluationResult,
    config: EvaluatorConfig,
    splits: tuple[str, ...],
    *,
    candidate_complexity: int,
) -> dict[str, object]:
    """Report charged-score sensitivity to omitting each workload family."""
    evaluator = PrefixKVCacheEvaluator(config, splits=splits)
    families = sorted({unit.workload for unit in units})
    rows = []
    for family in families:
        candidate_trials = [trial for trial in candidate_result.trials if trial.workload != family]
        baseline_trials = [trial for trial in baseline_result.trials if trial.workload != family]
        if not candidate_trials or not baseline_trials:
            continue
        rows.append(
            {
                "omitted_family": family,
                **_rescored_pair(
                    evaluator,
                    candidate_trials,
                    baseline_trials,
                    candidate_complexity=candidate_complexity,
                ),
            }
        )
    differences = [float(row["charged_score_difference"]) for row in rows]
    return {
        "omissions": rows,
        "charged_score_difference_range": (
            {"minimum": min(differences), "maximum": max(differences)} if differences else None
        ),
        "candidate_leads_for_every_omission": bool(differences)
        and all(difference > 0.0 for difference in differences),
        "note": (
            "Leave-one-family-out values are deterministic sensitivity checks on the "
            "fixed panel, not confidence bounds."
        ),
    }


def run_significance_analysis(
    config_path: Path,
    *,
    candidate_factory: Callable[..., PrefixKVPolicy] = build_production_incumbent,
    candidate_name: str = "production_incumbent",
    baseline_name: str = _DEFAULT_BASELINE,
    request_count: int | None = None,
    seeds: tuple[int, ...] | None = None,
    splits: tuple[str, ...] = _DEFAULT_SPLITS,
    workloads: tuple[str, ...] | None = None,
    candidate_complexity: int = _PRODUCTION_COMPLEXITY,
    confidence: float = 0.95,
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Run paired robustness diagnostics on the configured validation panel."""
    baseline_factory = BASELINE_REGISTRY.factories(include_reporting=True).get(baseline_name)
    if baseline_factory is None:
        available = ", ".join(sorted(BASELINE_REGISTRY.factories(include_reporting=True)))
        raise ValueError(f"unknown baseline {baseline_name!r}; choose from: {available}")

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

    candidate_scores, candidate_identity, candidate_result = _score_map(
        candidate_factory,
        config,
        splits,
        complexity=candidate_complexity,
    )
    baseline_scores, baseline_identity, baseline_result = _score_map(
        baseline_factory,
        config,
        splits,
        complexity=0,
    )
    identity = require_single_score_identity(
        (candidate_identity, baseline_identity),
        context="score-gap robustness analysis",
    )
    shared_keys = sorted(set(candidate_scores) & set(baseline_scores))
    if not shared_keys:
        raise ValueError("candidate and baseline share no evaluation groups")

    offsets = _base_seed_offsets(config, splits)
    units = [
        PairedUnit(
            group=f"{split}/{workload}/capacity_{capacity}/seed_{seed}",
            split=split,
            workload=workload,
            capacity_blocks=capacity,
            base_seed=seed - offsets[(split, workload)],
            seed=seed,
            candidate_score=candidate_scores[key],
            baseline_score=baseline_scores[key],
        )
        for key in shared_keys
        for (split, workload, capacity, seed) in (key,)
    ]
    differences = [unit.difference for unit in units]
    mean_difference = mean(differences)
    family_summary = descriptive_family_summary(
        units,
        confidence=confidence,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    whole_panel_seeds = whole_panel_seed_report(
        units,
        candidate_result,
        baseline_result,
        config,
        splits,
        candidate_complexity=candidate_complexity,
        confidence=confidence,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    leave_one_family_out = leave_one_family_out_report(
        units,
        candidate_result,
        baseline_result,
        config,
        splits,
        candidate_complexity=candidate_complexity,
    )
    seed_degeneracy = seed_degeneracy_report(units)

    return {
        "schema": _SCHEMA,
        "verifier_version": identity.verifier_version,
        "evaluation_context_sha256": identity.evaluation_context_sha256,
        "panel_sha256": identity.panel_sha256,
        "config": str(config_path),
        "candidate": candidate_name,
        "baseline": baseline_name,
        "splits": list(splits),
        "workloads": list(workloads) if workloads is not None else None,
        "request_count": config.request_count,
        "outer_seeds": list(config.seeds),
        "capacity_blocks": list(config.effective_capacity_blocks()),
        "candidate_complexity": candidate_complexity,
        "pairing_unit": "paired whole-panel outer-seed realization",
        "unit_score_definition": (
            "workload_base_score([trial]) under configured scoring weights; the "
            "single-trial behavioral score the evaluator aggregates into the combined "
            "score. Global churn, underfill, fairness, and complexity offsets are excluded."
        ),
        "charged_score": {
            "candidate": candidate_result.combined_score,
            "baseline": baseline_result.combined_score,
            "difference": candidate_result.combined_score - baseline_result.combined_score,
            "note": "Candidate complexity and all configured global charges are included.",
        },
        "paired_unit_count": len(units),
        "mean_candidate_score": mean(unit.candidate_score for unit in units),
        "mean_baseline_score": mean(unit.baseline_score for unit in units),
        "mean_difference": mean_difference,
        "seed_degeneracy": seed_degeneracy,
        "descriptive_family_summary": family_summary,
        "whole_panel_outer_seed_summary": whole_panel_seeds,
        "leave_one_family_out_sensitivity": leave_one_family_out,
        "interpretation": {
            "current_panel": (
                "Descriptive robustness only: three outer seeds are insufficient for a "
                "reliable confidence interval."
            ),
            "future_panel": (
                "At 20 or more independent outer seeds, use the paired whole-panel "
                "bootstrap interval, conditional on the fixed family panel."
            ),
            "population_scope": (
                "Generalization beyond the fixed families requires a declared "
                "workload-family sampling distribution and hierarchical resampling."
            ),
        },
        "units": [
            {
                "group": unit.group,
                "split": unit.split,
                "workload": unit.workload,
                "capacity_blocks": unit.capacity_blocks,
                "base_seed": unit.base_seed,
                "seed": unit.seed,
                "candidate_score": unit.candidate_score,
                "baseline_score": unit.baseline_score,
                "difference": unit.difference,
            }
            for unit in units
        ],
    }


def format_report(payload: dict[str, object]) -> str:
    """Return a human-readable summary of the robustness payload."""
    charged = payload["charged_score"]
    family = payload["descriptive_family_summary"]
    degeneracy = payload["seed_degeneracy"]
    seed_summary = payload["whole_panel_outer_seed_summary"]
    seed_range = seed_summary["charged_score_difference_range"]
    leave_one_out = payload["leave_one_family_out_sensitivity"]
    leave_one_out_range = leave_one_out["charged_score_difference_range"]
    lines = [
        "Score-gap robustness analysis",
        f"  candidate:            {payload['candidate']}",
        f"  baseline:             {payload['baseline']}",
        f"  verifier:             {payload['verifier_version']}",
        f"  panel:                {payload['panel_sha256']}",
        f"  splits:               {', '.join(payload['splits'])}",
        f"  charged score:        candidate {float(charged['candidate']):.3f} "
        f"vs baseline {float(charged['baseline']):.3f} "
        f"(delta {float(charged['difference']):+.3f})",
        f"  family outcomes:      {family['family_wins']} wins, "
        f"{family['family_losses']} losses, {family['family_ties']} ties",
        f"  whole-panel seeds:    {seed_summary['outer_seed_count']} "
        f"(charged delta range {float(seed_range['minimum']):+.3f} to "
        f"{float(seed_range['maximum']):+.3f})",
        f"  seed-invariant cells: {degeneracy['seed_invariant_cell_count']}"
        f"/{degeneracy['cell_count']}",
    ]
    if leave_one_out_range is not None:
        lines.append(
            f"  leave-one-family-out: charged delta range "
            f"{float(leave_one_out_range['minimum']):+.3f} to "
            f"{float(leave_one_out_range['maximum']):+.3f}"
        )
    interval = seed_summary["paired_outer_seed_bootstrap_confidence_interval"]
    if interval is None:
        lines.append(
            f"  confidence interval: omitted; at least "
            f"{seed_summary['minimum_outer_seeds_for_interval']} outer seeds required"
        )
    else:
        lines.append(
            f"  paired seed CI:       [{float(interval['lower']):+.3f}, "
            f"{float(interval['upper']):+.3f}]"
        )
    lines.append("  interpretation:       fixed-panel descriptive robustness")
    return "\n".join(lines)


@click.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    default=DEFAULT_CONFIG_PATH,
    show_default=True,
)
@click.option(
    "--candidate-program",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    help="Candidate policy source; defaults to the production incumbent.",
)
@click.option(
    "--baseline",
    default=_DEFAULT_BASELINE,
    show_default=True,
    help="Named baseline factory to compare against.",
)
@click.option("--request-count", type=click.IntRange(min=1))
@click.option("--seeds", type=int, multiple=True, help="Explicit outer workload seeds.")
@click.option(
    "--outer-seed-count",
    type=click.IntRange(min=1),
    help="Generate this many consecutive outer seeds for a future-panel run.",
)
@click.option("--outer-seed-start", type=int, default=0, show_default=True)
@click.option(
    "--splits", type=click.Choice(("train", "validation", "probe", "hidden")), multiple=True
)
@click.option("--workloads", multiple=True)
@click.option(
    "--confidence", type=click.FloatRange(min=0.5, max=0.999), default=0.95, show_default=True
)
@click.option(
    "--bootstrap-resamples", type=click.IntRange(min=1000), default=10000, show_default=True
)
@click.option("--bootstrap-seed", type=int, default=0, show_default=True)
@click.option("--output", type=click.Path(path_type=Path))
def main(
    config: Path,
    candidate_program: Path | None,
    baseline: str,
    request_count: int | None,
    seeds: tuple[int, ...],
    outer_seed_count: int | None,
    outer_seed_start: int,
    splits: tuple[str, ...],
    workloads: tuple[str, ...],
    confidence: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    output: Path | None,
) -> None:
    """Report paired robustness diagnostics for the headline score gap."""
    if seeds and outer_seed_count is not None:
        raise click.UsageError("use either --seeds or --outer-seed-count, not both")
    selected_seeds = seeds
    if outer_seed_count is not None:
        selected_seeds = tuple(range(outer_seed_start, outer_seed_start + outer_seed_count))

    candidate_factory: Callable[..., PrefixKVPolicy] = build_production_incumbent
    candidate_name = "production_incumbent"
    candidate_complexity = _PRODUCTION_COMPLEXITY
    if candidate_program is not None:
        candidate_factory = load_candidate_factory(str(candidate_program))
        candidate_name = str(candidate_program)
        source = candidate_program.read_text(encoding="utf-8")
        evaluator_config = load_evaluator_config(config)
        candidate_complexity = scoring_fn_complexity(
            source,
            form_aware=evaluator_config.form_aware_complexity,
        )
    payload = run_significance_analysis(
        config,
        candidate_factory=candidate_factory,
        candidate_name=candidate_name,
        candidate_complexity=candidate_complexity,
        baseline_name=baseline,
        request_count=request_count,
        seeds=selected_seeds or None,
        splits=splits or _DEFAULT_SPLITS,
        workloads=workloads or None,
        confidence=confidence,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    output_path = output or Path("artifacts/prefix_kv_cache_score_gap_significance.json")
    write_json(output_path, payload)
    click.echo(format_report(payload))
    click.echo(str(output_path))


if __name__ == "__main__":
    main()
