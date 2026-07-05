"""Statistical verification of the headline score gap over seed noise.

The repository headline compares the production incumbent against a named
baseline (TinyLFU-LRU) by a single aggregate combined score. That aggregate
hides whether the observed gap is larger than the run-to-run noise across the
generated workload panel. This tool decomposes the operative validation panel
into paired per-group behavioral scores, then reports the mean paired
difference, a seeded percentile bootstrap confidence interval, and paired
permutation and sign-test p-values.

Pairing unit
------------
The paired unit is one validation ``(workload_family, capacity_blocks, seed)``
group. Both policies replay the *same* generated request stream for each group
(identical panel fingerprint), so the difference isolates policy behavior from
workload noise. The per-unit score is ``workload_base_score([trial])`` under the
configured scoring weights. That single-trial behavioral score is the exact
building block the evaluator aggregates (mean plus a min-seed blend) into the
headline combined score, so a positive, tight interval on these paired
differences is direct evidence that the combined-score lead exceeds seed noise.

The global churn, underfill, fairness, and complexity charges are policy-level
offsets applied once to the combined score, not per-group quantities, so they
are excluded from the paired unit and reported separately as context.

Clustering and pseudo-replication
---------------------------------
The workload generator does not vary every family's request stream with the
seed: several families produce byte-identical per-seed scores, so the
``(family, capacity, seed)`` groups are *not* mutually independent. Treating all
of them as independent units would inflate the effective sample size and
understate the p-value (pseudo-replication). The *primary* verdict is therefore
computed on cluster-mean differences at the workload-family level, which is the
coarsest defensible independent unit. Per-group and family-by-capacity results
are still reported for transparency, alongside a seed-degeneracy diagnostic.
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
from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.evaluators.contracts import PrefixKVPolicy
from prefix_cache_evolve.evaluators.prefix_kv_cache import PrefixKVCacheEvaluator
from prefix_cache_evolve.evaluators.results import TrialMetrics
from prefix_cache_evolve.evaluators.scoring import workload_base_score
from prefix_cache_evolve.evaluators.verifier import require_single_score_identity
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import (
    DEFAULT_CONFIG_PATH,
    load_evaluator_config,
)
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents import (
    build_current_incumbent as build_production_incumbent,
)

_DEFAULT_SPLITS = ("validation",)
_DEFAULT_BASELINE = "tinylfu_lru"
_SCHEMA = "prefix-kv-cache-score-gap-significance-v1"


@dataclass(frozen=True, slots=True)
class PairedUnit:
    """One paired per-group behavioral score for candidate and baseline."""

    group: str
    split: str
    workload: str
    capacity_blocks: int
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


def sign_test_p_value(values: list[float], *, tolerance: float = 1e-12) -> dict[str, float | int]:
    """Return an exact two-sided sign-test p-value for paired differences.

    Zero differences (within ``tolerance``) are dropped as ties. The p-value is
    the exact two-sided binomial tail under a fair coin.
    """
    positives = sum(1 for value in values if value > tolerance)
    negatives = sum(1 for value in values if value < -tolerance)
    effective = positives + negatives
    if effective == 0:
        return {"p_value": 1.0, "positive": 0, "negative": 0, "ties": len(values)}
    extreme = min(positives, negatives)
    tail = 0.0
    for successes in range(0, extreme + 1):
        tail += _binomial_coefficient(effective, successes)
    tail *= 0.5**effective
    p_value = min(1.0, 2.0 * tail)
    return {
        "p_value": p_value,
        "positive": positives,
        "negative": negatives,
        "ties": len(values) - effective,
    }


def _binomial_coefficient(total: int, successes: int) -> float:
    """Return the binomial coefficient as a float."""
    successes = min(successes, total - successes)
    if successes < 0:
        return 0.0
    result = 1.0
    for step in range(1, successes + 1):
        result = result * (total - successes + step) / step
    return result


def permutation_p_value(
    values: list[float],
    *,
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Return a seeded two-sided paired sign-flip permutation p-value.

    Under the paired null, each difference's sign is exchangeable. Random sign
    flips build the null distribution of the mean difference; the p-value counts
    how often the permuted absolute mean reaches the observed absolute mean.
    """
    if not values:
        raise ValueError("permutation test requires at least one paired difference")
    observed = abs(mean(values))
    count = len(values)
    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(resamples):
        total = 0.0
        for value in values:
            total += value if rng.random() < 0.5 else -value
        if abs(total / count) >= observed - 1e-12:
            at_least_as_extreme += 1
    # Add-one correction keeps the estimate conservative and never reports zero.
    p_value = (at_least_as_extreme + 1) / (resamples + 1)
    return {
        "p_value": p_value,
        "at_least_as_extreme": at_least_as_extreme,
        "resamples": resamples,
        "seed": seed,
    }


def cluster_mean_differences(
    units: list[PairedUnit],
    key: Callable[[PairedUnit], tuple[object, ...]],
) -> dict[tuple[object, ...], float]:
    """Return the mean paired difference within each cluster.

    Collapsing correlated groups (for example the seed-invariant replicates of a
    single workload family) to one mean-difference per cluster is the standard
    cluster-robust correction for pseudo-replication: each returned value is one
    approximately independent observation.
    """
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for unit in units:
        grouped[key(unit)].append(unit.difference)
    return {cluster: mean(diffs) for cluster, diffs in grouped.items()}


def clustered_significance(
    units: list[PairedUnit],
    key: Callable[[PairedUnit], tuple[object, ...]],
    *,
    label: str,
    confidence: float,
    bootstrap_resamples: int,
    permutation_resamples: int,
    bootstrap_seed: int,
    permutation_seed: int,
) -> dict[str, object]:
    """Return significance statistics on cluster-mean paired differences."""
    cluster_diffs = list(cluster_mean_differences(units, key).values())
    wins = sum(1 for value in cluster_diffs if value > 1e-12)
    losses = sum(1 for value in cluster_diffs if value < -1e-12)
    mean_difference = mean(cluster_diffs)
    bootstrap = bootstrap_mean_ci(
        cluster_diffs,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    permutation = permutation_p_value(
        cluster_diffs,
        resamples=permutation_resamples,
        seed=permutation_seed,
    )
    sign_test = sign_test_p_value(cluster_diffs)
    ci_excludes_zero = bool(float(bootstrap["lower"]) > 0.0 or float(bootstrap["upper"]) < 0.0)
    return {
        "clustering": label,
        "cluster_count": len(cluster_diffs),
        "cluster_wins": wins,
        "cluster_losses": losses,
        "cluster_ties": len(cluster_diffs) - wins - losses,
        "mean_difference": mean_difference,
        "bootstrap_confidence_interval": bootstrap,
        "confidence_interval_excludes_zero": ci_excludes_zero,
        "permutation_test": permutation,
        "sign_test": sign_test,
        "verdict": _verdict(mean_difference, ci_excludes_zero, float(permutation["p_value"])),
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
            "independent seed replicates; per-group statistics therefore overstate "
            "the effective sample size. The primary verdict uses workload-family "
            "clusters to avoid this pseudo-replication."
        ),
    }


def _score_map(
    factory: Callable[..., PrefixKVPolicy],
    config: EvaluatorConfig,
    splits: tuple[str, ...],
) -> tuple[dict[tuple[str, str, int, int], float], dict[str, object], float]:
    """Return per-group behavioral scores plus panel identity and combined score."""
    result = PrefixKVCacheEvaluator(config, splits=splits)(factory)
    scores = {
        (trial.split, trial.workload, trial.capacity_blocks, trial.seed): group_score(trial, config)
        for trial in result.trials
    }
    identity = {
        "verifier_version": result.verifier_version,
        "evaluation_context_sha256": result.evaluation_context_sha256,
        "panel_sha256": result.panel_sha256,
    }
    return scores, identity, result.combined_score


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
    confidence: float = 0.95,
    bootstrap_resamples: int = 10000,
    permutation_resamples: int = 10000,
    bootstrap_seed: int = 0,
    permutation_seed: int = 0,
) -> dict[str, object]:
    """Run the paired per-group significance analysis on the validation panel."""
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

    candidate_scores, candidate_identity, candidate_combined = _score_map(
        candidate_factory, config, splits
    )
    baseline_scores, baseline_identity, baseline_combined = _score_map(
        baseline_factory, config, splits
    )
    identity = require_single_score_identity(
        (candidate_identity, baseline_identity),
        context="score-gap significance analysis",
    )
    shared_keys = sorted(set(candidate_scores) & set(baseline_scores))
    if not shared_keys:
        raise ValueError("candidate and baseline share no evaluation groups")

    units = [
        PairedUnit(
            group=f"{split}/{workload}/capacity_{capacity}/seed_{seed}",
            split=split,
            workload=workload,
            capacity_blocks=capacity,
            seed=seed,
            candidate_score=candidate_scores[key],
            baseline_score=baseline_scores[key],
        )
        for key in shared_keys
        for (split, workload, capacity, seed) in (key,)
    ]
    differences = [unit.difference for unit in units]
    wins = sum(1 for value in differences if value > 1e-12)
    losses = sum(1 for value in differences if value < -1e-12)
    ties = len(differences) - wins - losses
    mean_difference = mean(differences)
    bootstrap = bootstrap_mean_ci(
        differences,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    permutation = permutation_p_value(
        differences,
        resamples=permutation_resamples,
        seed=permutation_seed,
    )
    sign_test = sign_test_p_value(differences)
    ci_excludes_zero = bool(float(bootstrap["lower"]) > 0.0 or float(bootstrap["upper"]) < 0.0)

    by_workload = clustered_significance(
        units,
        lambda unit: (unit.split, unit.workload),
        label="workload_family",
        confidence=confidence,
        bootstrap_resamples=bootstrap_resamples,
        permutation_resamples=permutation_resamples,
        bootstrap_seed=bootstrap_seed,
        permutation_seed=permutation_seed,
    )
    by_workload_capacity = clustered_significance(
        units,
        lambda unit: (unit.split, unit.workload, unit.capacity_blocks),
        label="workload_family_capacity",
        confidence=confidence,
        bootstrap_resamples=bootstrap_resamples,
        permutation_resamples=permutation_resamples,
        bootstrap_seed=bootstrap_seed,
        permutation_seed=permutation_seed,
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
        "seeds": list(config.seeds),
        "capacity_blocks": list(config.effective_capacity_blocks()),
        "pairing_unit": "validation (workload_family, capacity_blocks, seed) group",
        "unit_score_definition": (
            "workload_base_score([trial]) under configured scoring weights; the "
            "single-trial behavioral score the evaluator aggregates into the combined "
            "score. Global churn, underfill, fairness, and complexity offsets are excluded."
        ),
        "combined_score_context": {
            "candidate_combined_score": candidate_combined,
            "baseline_combined_score": baseline_combined,
            "combined_score_difference": candidate_combined - baseline_combined,
            "note": (
                "Combined scores here omit the candidate complexity charge "
                "(scoring_fn_complexity=0) and are reported for context only; the "
                "significance statistics use the paired per-group behavioral scores."
            ),
        },
        "paired_unit_count": len(units),
        "candidate_wins": wins,
        "candidate_losses": losses,
        "ties": ties,
        "mean_candidate_score": mean(unit.candidate_score for unit in units),
        "mean_baseline_score": mean(unit.baseline_score for unit in units),
        "mean_difference": mean_difference,
        "primary_clustering": "workload_family",
        "verdict": by_workload["verdict"],
        "seed_degeneracy": seed_degeneracy,
        "clustered": {
            "by_workload_family": by_workload,
            "by_workload_family_capacity": by_workload_capacity,
        },
        "per_group_naive": {
            "note": (
                "Per-group statistics treat every (family, capacity, seed) group as "
                "independent. Because several families are seed-invariant (see "
                "seed_degeneracy), these overstate the effective sample size and are "
                "retained for transparency only; the primary verdict uses the "
                "workload-family clustering."
            ),
            "bootstrap_confidence_interval": bootstrap,
            "confidence_interval_excludes_zero": ci_excludes_zero,
            "permutation_test": permutation,
            "sign_test": sign_test,
        },
        "units": [
            {
                "group": unit.group,
                "split": unit.split,
                "workload": unit.workload,
                "capacity_blocks": unit.capacity_blocks,
                "seed": unit.seed,
                "candidate_score": unit.candidate_score,
                "baseline_score": unit.baseline_score,
                "difference": unit.difference,
            }
            for unit in units
        ],
    }


def _verdict(mean_difference: float, ci_excludes_zero: bool, permutation_p: float) -> str:
    """Return a compact verdict label for the paired comparison."""
    if not ci_excludes_zero or permutation_p >= 0.05:
        return "inconclusive_within_seed_noise"
    return "candidate_favored" if mean_difference > 0.0 else "baseline_favored"


def _format_clustered(block: dict[str, object]) -> list[str]:
    """Return report lines for one clustered significance block."""
    bootstrap = block["bootstrap_confidence_interval"]
    permutation = block["permutation_test"]
    sign_test = block["sign_test"]
    confidence_percent = float(bootstrap["confidence"]) * 100.0
    return [
        f"  [{block['clustering']}] clusters: {block['cluster_count']} "
        f"(wins {block['cluster_wins']}, losses {block['cluster_losses']}, "
        f"ties {block['cluster_ties']})",
        f"    mean difference:   {float(block['mean_difference']):+.4f}",
        f"    {confidence_percent:.0f}% bootstrap CI: "
        f"[{float(bootstrap['lower']):+.4f}, {float(bootstrap['upper']):+.4f}]",
        f"    permutation p:     {float(permutation['p_value']):.5f}",
        f"    sign-test p:       {float(sign_test['p_value']):.5f} "
        f"(+{sign_test['positive']} / -{sign_test['negative']} / ={sign_test['ties']})",
    ]


def format_report(payload: dict[str, object]) -> str:
    """Return a human-readable summary of the significance payload."""
    context = payload["combined_score_context"]
    clustered = payload["clustered"]
    degeneracy = payload["seed_degeneracy"]
    naive = payload["per_group_naive"]
    naive_permutation = naive["permutation_test"]
    lines = [
        "Score-gap significance analysis",
        f"  candidate:            {payload['candidate']}",
        f"  baseline:             {payload['baseline']}",
        f"  verifier:             {payload['verifier_version']}",
        f"  panel:                {payload['panel_sha256']}",
        f"  splits:               {', '.join(payload['splits'])}",
        f"  paired units:         {payload['paired_unit_count']} "
        f"(wins {payload['candidate_wins']}, losses {payload['candidate_losses']}, "
        f"ties {payload['ties']})",
        f"  combined score:       candidate {float(context['candidate_combined_score']):.3f} "
        f"vs baseline {float(context['baseline_combined_score']):.3f} "
        f"(delta {float(context['combined_score_difference']):+.3f})",
        f"  mean per-unit score:  candidate {float(payload['mean_candidate_score']):.3f} "
        f"vs baseline {float(payload['mean_baseline_score']):.3f}",
        f"  seed-invariant cells: {degeneracy['seed_invariant_cell_count']}"
        f"/{degeneracy['cell_count']} "
        "(seeds ignored -> per-group stats overstate significance)",
        "Cluster-robust significance (primary = workload_family):",
        *_format_clustered(clustered["by_workload_family"]),
        *_format_clustered(clustered["by_workload_family_capacity"]),
        f"  per-group naive p:    {float(naive_permutation['p_value']):.5f} "
        "(overstated; transparency only)",
        f"  primary verdict:      {payload['verdict']} "
        f"(clustering: {payload['primary_clustering']})",
    ]
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
@click.option("--seeds", type=int, multiple=True)
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
@click.option(
    "--permutation-resamples", type=click.IntRange(min=1000), default=10000, show_default=True
)
@click.option("--bootstrap-seed", type=int, default=0, show_default=True)
@click.option("--permutation-seed", type=int, default=0, show_default=True)
@click.option("--output", type=click.Path(path_type=Path))
def main(
    config: Path,
    candidate_program: Path | None,
    baseline: str,
    request_count: int | None,
    seeds: tuple[int, ...],
    splits: tuple[str, ...],
    workloads: tuple[str, ...],
    confidence: float,
    bootstrap_resamples: int,
    permutation_resamples: int,
    bootstrap_seed: int,
    permutation_seed: int,
    output: Path | None,
) -> None:
    """Test whether the headline score gap exceeds seed noise."""
    candidate_factory: Callable[..., PrefixKVPolicy] = build_production_incumbent
    candidate_name = "production_incumbent"
    if candidate_program is not None:
        candidate_factory = load_candidate_factory(str(candidate_program))
        candidate_name = str(candidate_program)
    payload = run_significance_analysis(
        config,
        candidate_factory=candidate_factory,
        candidate_name=candidate_name,
        baseline_name=baseline,
        request_count=request_count,
        seeds=seeds or None,
        splits=splits or _DEFAULT_SPLITS,
        workloads=workloads or None,
        confidence=confidence,
        bootstrap_resamples=bootstrap_resamples,
        permutation_resamples=permutation_resamples,
        bootstrap_seed=bootstrap_seed,
        permutation_seed=permutation_seed,
    )
    output_path = output or Path("artifacts/prefix_kv_cache_score_gap_significance.json")
    write_json(output_path, payload)
    click.echo(format_report(payload))
    click.echo(str(output_path))


if __name__ == "__main__":
    main()
