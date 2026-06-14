#!/usr/bin/env python3
"""Sweep registered prefix-cache baselines across cache geometries."""

from __future__ import annotations

import concurrent.futures
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from prefix_cache_evolve.evaluators.baselines import (
    ALL_REPORTING_BASELINES,
    BASELINE_REGISTRY,
)
from prefix_cache_evolve.evaluators.complexity import scoring_fn_complexity
from prefix_cache_evolve.evaluators.prefix_kv_cache import PrefixKVCacheEvaluator
from prefix_cache_evolve.evaluators.verifier import (
    require_single_score_identity,
    require_single_verifier_version,
)
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.production_incumbent import (
    build_candidate,
)

_DEFAULT_BLOCK_SIZES = (16, 24, 32, 48, 64)
_DEFAULT_CAPACITIES = (24, 48, 96, 128)
_DEFAULT_CONFIG = Path("configs/prefix_kv_cache.yaml")
_DEFAULT_OUTPUT = Path("docs/results/baseline_geometry_sweep.json")
_INCUMBENT_NAME = "evolved_incumbent"
_INCUMBENT_PATH = Path("src/prefix_cache_evolve/problems/prefix_kv_cache/production_incumbent.py")


@dataclass(frozen=True)
class SweepJob:
    """One policy and physical block-size evaluation."""

    policy: str
    block_size_tokens: int
    capacities: tuple[int, ...]
    config_path: str


def _parse_positive_ints(
    _context: click.Context,
    _parameter: click.Parameter,
    value: str,
) -> tuple[int, ...]:
    """Parse a comma-separated list of unique positive integers."""
    values: list[int] = []
    try:
        numbers = (int(item.strip()) for item in value.split(","))
        for number in numbers:
            if number <= 0:
                raise click.BadParameter("values must be positive")
            if number not in values:
                values.append(number)
    except ValueError as error:
        raise click.BadParameter("values must be integers") from error
    if not values:
        raise click.BadParameter("at least one value is required")
    return tuple(values)


def _metric(metrics: dict[str, Any], name: str) -> float:
    """Return one aggregate metric as a float."""
    return float(metrics[name])


def _run_job(job: SweepJob) -> dict[str, Any]:
    """Evaluate one policy at one block size."""
    config = load_evaluator_config(Path(job.config_path)).with_updates(
        capacity_blocks=job.capacities[0],
        capacity_sweep_blocks=job.capacities,
        block_size_tokens=job.block_size_tokens,
    )
    if job.policy == _INCUMBENT_NAME:
        factory = build_candidate
        source = _INCUMBENT_PATH.read_text(encoding="utf-8")
        complexity = scoring_fn_complexity(source, form_aware=config.form_aware_complexity)
        group = "deployable"
    else:
        factory = ALL_REPORTING_BASELINES[job.policy]
        complexity = 0
        group = BASELINE_REGISTRY.group(job.policy)

    evaluator = PrefixKVCacheEvaluator(
        config,
        splits=("train", "validation"),
        expose_future_reuse=BASELINE_REGISTRY.requires_future_reuse(job.policy),
    )
    result = evaluator(factory, scoring_fn_complexity=complexity)
    validation = result.split_metrics["validation"]
    capacities = {
        name.removeprefix("capacity_"): {
            "token_hit_rate": _metric(metrics, "token_hit_rate"),
            "request_token_hit_rate_p10": _metric(metrics, "request_token_hit_rate_p10"),
            "cache_churn_per_1k": _metric(metrics, "cache_churn_per_1k"),
            "policy_underfill_rate": _metric(metrics, "policy_underfill_rate"),
        }
        for name, metrics in result.capacity_metrics.items()
    }
    return {
        "verifier_version": result.verifier_version,
        "evaluation_context_sha256": result.evaluation_context_sha256,
        "panel_sha256": result.panel_sha256,
        "policy": job.policy,
        "group": group,
        "block_size_tokens": job.block_size_tokens,
        "combined_score": result.combined_score,
        "score_before_complexity": (
            result.combined_score + result.score_breakdown["complexity_cost"]
        ),
        "complexity": complexity,
        "complexity_cost": result.score_breakdown["complexity_cost"],
        "validation": {
            "token_hit_rate": _metric(validation, "token_hit_rate"),
            "request_token_hit_rate_p10": _metric(validation, "request_token_hit_rate_p10"),
            "cache_churn_per_1k": _metric(validation, "cache_churn_per_1k"),
            "policy_underfill_rate": _metric(validation, "policy_underfill_rate"),
            "avoidable_rejection_regret_token_rate": _metric(
                validation, "avoidable_rejection_regret_token_rate"
            ),
            "value_weighted_avoidable_eviction_rate": _metric(
                validation, "value_weighted_avoidable_eviction_rate"
            ),
        },
        "capacities": capacities,
    }


@click.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    default=_DEFAULT_CONFIG,
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=_DEFAULT_OUTPUT,
    show_default=True,
)
@click.option(
    "--block-sizes",
    callback=_parse_positive_ints,
    default=",".join(str(value) for value in _DEFAULT_BLOCK_SIZES),
    show_default=True,
)
@click.option(
    "--capacities",
    callback=_parse_positive_ints,
    default=",".join(str(value) for value in _DEFAULT_CAPACITIES),
    show_default=True,
)
@click.option("--workers", type=click.IntRange(min=1), default=4, show_default=True)
def main(
    config: Path,
    output: Path,
    block_sizes: tuple[int, ...],
    capacities: tuple[int, ...],
    workers: int,
) -> None:
    """Run the complete geometry sweep and write its JSON artifact."""
    policies = (_INCUMBENT_NAME, *ALL_REPORTING_BASELINES)
    jobs = [
        SweepJob(
            policy=policy,
            block_size_tokens=block_size,
            capacities=capacities,
            config_path=str(config),
        )
        for policy in policies
        for block_size in block_sizes
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_run_job, jobs))

    results.sort(key=lambda item: (item["block_size_tokens"], item["policy"]))
    try:
        verifier_version = require_single_verifier_version(
            results,
            context="geometry sweep",
        )
        identities = {
            str(block_size): require_single_score_identity(
                (result for result in results if result["block_size_tokens"] == block_size),
                context=f"geometry sweep block size {block_size}",
            )
            for block_size in block_sizes
        }
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    payload = {
        "verifier_version": verifier_version,
        "evaluation_contexts": {
            block_size: identity.evaluation_context_sha256
            for block_size, identity in identities.items()
        },
        "panel_sha256s": {
            block_size: identity.panel_sha256 for block_size, identity in identities.items()
        },
        "generated_at": datetime.now(UTC).isoformat(),
        "config": str(config),
        "block_sizes": list(block_sizes),
        "capacities": list(capacities),
        "splits": ["train", "validation"],
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    click.echo(f"Wrote {len(results)} evaluations to {output}")


if __name__ == "__main__":
    main()
