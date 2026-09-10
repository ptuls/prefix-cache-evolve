"""Compare behavior, implementation size, and measured policy costs separately."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click

from prefix_cache_evolve.evaluators.baselines import BASELINE_REGISTRY
from prefix_cache_evolve.evaluators.complexity import scoring_fn_complexity
from prefix_cache_evolve.evaluators.policy_costs import baseline_source, equivalent_behavior
from prefix_cache_evolve.evaluators.prefix_kv_cache import EvaluationResult
from prefix_cache_evolve.evaluators.verifier import require_single_score_identity
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.sandbox import evaluate_in_docker


def assess_budgets(
    profiles: list[dict[str, Any]],
    *,
    callback_p99_us: float | None,
    scan_p99_us: float | None,
    state_budget_bytes: int | None,
) -> dict[str, Any]:
    """Check declared budgets across every repeat and trial, without score penalties."""
    limits = {
        "callback_p99_us": callback_p99_us,
        "scan_p99_us": scan_p99_us,
        "sampled_retained_bytes": state_budget_bytes,
    }
    trials = [trial for profile in profiles for trial in profile["trials"]]
    measured = {
        "callback_p99_us": max(
            (
                callback["wall_ns"]["p99_upper"] / 1000
                for trial in trials
                for callback in trial["callbacks"].values()
            ),
            default=0.0,
        ),
        "scan_p99_us": max(
            (trial["eviction_scans"]["wall_ns"]["p99_upper"] / 1000 for trial in trials),
            default=0.0,
        ),
        "sampled_retained_bytes": max(
            (sample["retained_bytes"] for trial in trials for sample in trial["state_samples"]),
            default=0,
        ),
    }
    failures = [key for key, limit in limits.items() if limit is not None and measured[key] > limit]
    supplied = [key for key, limit in limits.items() if limit is not None]
    return {
        "status": "exceeded"
        if failures
        else "within_declared_limits"
        if supplied
        else "unassessed",
        "limits": limits,
        "measured": measured,
        "failed": failures,
        "deployment_approved": False,
    }


def pareto_names(rows: dict[str, dict[str, Any]]) -> list[str]:
    """Keep every nondominated behavioral-score/source-size alternative."""
    return sorted(
        name
        for name, row in rows.items()
        if not any(
            other["raw_before_complexity"] >= row["raw_before_complexity"]
            and other["implementation_ast_nodes"] <= row["implementation_ast_nodes"]
            and (
                other["raw_before_complexity"] > row["raw_before_complexity"]
                or other["implementation_ast_nodes"] < row["implementation_ast_nodes"]
            )
            for other_name, other in rows.items()
            if other_name != name
        )
    )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Publish behavior, source size, callback time and state in adjacent columns."""
    rows = payload["policies"]
    lines = [
        "# Policy behavior and cost comparison",
        "",
        "Raw score removes only the AST charge. Timing is diagnostic and never enters selection.",
        "All policies use the same container, visible panel and measurement convention.",
        "",
        *(
            f"`{name}`: `{row['source_path']}`"
            for name, row in rows.items()
            if row.get("source_path")
        ),
        "",
        "| Policy | Raw score | Historical charged score | Implementation AST | "
        "Callback p99 upper (us) | Scan p99 upper (us) | Sampled state max (bytes) | Budgets |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in sorted(rows, key=lambda key: (-rows[key]["raw_before_complexity"], key)):
        row = rows[name]
        measured = row["budgets"]["measured"]
        lines.append(
            f"| {name} | {row['raw_before_complexity']:.3f} | {row['combined_score']:.3f} | "
            f"{row['implementation_ast_nodes']} | {measured['callback_p99_us']:.3f} | "
            f"{measured['scan_p99_us']:.3f} | {measured['sampled_retained_bytes']} | "
            f"{row['budgets']['status']} |"
        )
    lines.extend(
        [
            "",
            "Retained behavior/source-size alternatives: " + ", ".join(payload["pareto_policies"]),
            "",
            "Implementation AST uses uncredited policy classes, bases and module helpers; "
            "shared library code and thin factories are excluded for both baselines "
            "and candidates.",
            "Historical charges remain asymmetric and are shown for continuity only.",
            "",
            "Times include Python call/timer overhead; eviction scan time sums only policy "
            "ranking callbacks. It excludes simulator scan construction, sorting and bookkeeping.",
            "p99 values are approximate upper bounds from logarithmic histograms, and the table "
            "reports the worst trial/repeat. CPU distributions and capacity/state trajectories "
            "are in the JSON report. These are instrumented Python measurements, "
            "not GPU throughput.",
            "",
            "State is sampled at initialization, powers of two requests and the final request. "
            "It estimates the reachable policy object graph, including primitive delegates, "
            "excluding code/classes, profiler storage and simulator KV. "
            "It is not a peak/RSS budget.",
            "",
            "No hidden or probe inputs were opened. Suitability for deployment and fresh "
            "holdout adjudication remain separate decisions.",
        ]
    )
    if payload.get("simplification") is not None:
        lines.extend(["", "Simplification check: `" + json.dumps(payload["simplification"]) + "`."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@click.command()
@click.option("--config", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--candidate", type=click.Path(path_type=Path, exists=True, dir_okay=False), multiple=True
)
@click.option("--reference", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option(
    "--baseline",
    type=click.Choice(tuple(BASELINE_REGISTRY.factories())),
    multiple=True,
    default=("lru", "tinylfu_lru", "vllm_apc"),
)
@click.option("--sandbox-image", help="Rebuilt, immutable Docker image ID (sha256:...).")
@click.option("--repeats", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--callback-p99-us", type=click.FloatRange(min=0, min_open=True))
@click.option("--scan-p99-us", type=click.FloatRange(min=0, min_open=True))
@click.option("--state-budget-bytes", type=click.IntRange(min=1))
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
def main(
    config: Path,
    candidate: tuple[Path, ...],
    reference: Path | None,
    baseline: tuple[str, ...],
    sandbox_image: str | None,
    repeats: int,
    callback_p99_us: float | None,
    scan_p99_us: float | None,
    state_budget_bytes: int | None,
    output_dir: Path,
) -> None:
    """Profile visible panels in Docker and retain smaller behavioral alternatives.

    --reference enables strict simplification adjudication: all deterministic
    train/validation trial metrics must match exactly, with a smaller effective
    source and the configured promotion size limit. This never promotes a policy.
    """
    if any(
        value is not None and not math.isfinite(value) for value in (callback_p99_us, scan_p99_us)
    ):
        raise click.BadParameter("budgets must be finite")
    settings = load_evaluator_config(config)
    if sandbox_image is not None:
        settings = settings.with_updates(sandbox_image=sandbox_image)
    if not settings.sandbox_image or not settings.sandbox_image.startswith("sha256:"):
        raise click.UsageError("profile with a rebuilt, pinned --sandbox-image sha256:...")
    if settings.candidate_policy_surface != "full" or settings.fixed_admission_policy is not None:
        raise click.UsageError("profiling requires full policies without fixed admission")
    if output_dir.exists():
        raise click.UsageError("--output-dir must be new to preserve previous measurements")
    paths = tuple(
        dict.fromkeys(path.resolve() for path in (*candidate, *((reference,) if reference else ())))
    )
    sources = {
        f"candidate_{index}": path.read_text(encoding="utf-8") for index, path in enumerate(paths)
    }
    sources.update({name: baseline_source(name) for name in baseline})
    output_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text(settings.model_dump_json(indent=2) + "\n")
    payload: dict[str, Any] = {
        "schema": "prefix-kv-cache-policy-cost-comparison-v1",
        "config": str(config.resolve()),
        "sandbox_image": settings.sandbox_image,
        "splits": ["train", "validation"],
        "repeats": repeats,
        "simplification_rule": (
            "exact visible-trial behavior; smaller effective AST; promotion size limit"
        ),
        "declared_budgets": {
            "callback_p99_us": callback_p99_us,
            "scan_p99_us": scan_p99_us,
            "state_budget_bytes": state_budget_bytes,
        },
        "policies": {},
    }
    results: dict[str, EvaluationResult] = {}
    for name, source in sources.items():
        (output_dir / f"{name}.py").write_text(source, encoding="utf-8")
        profiles = []
        for repeat in range(repeats):
            click.echo(f"Profiling {name}, repeat {repeat + 1}/{repeats}", err=True)
            try:
                result = evaluate_in_docker(
                    source if name.startswith("candidate_") else "",
                    settings,
                    splits=("train", "validation"),
                    profile=True,
                    baseline=None if name.startswith("candidate_") else name,
                )
            except (RuntimeError, ValueError, OSError) as exc:
                (output_dir / "failure.json").write_text(
                    json.dumps(
                        {
                            "policy": name,
                            "repeat": repeat,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                raise click.ClickException(f"{name}: {exc}; partial evidence retained") from exc
            (output_dir / f"{name}-{repeat}.json").write_text(
                json.dumps(asdict(result), indent=2) + "\n"
            )
            if not result.success or result.invalid_fraction or not result.trials:
                raise click.ClickException(
                    f"{name}: invalid or empty profile; partial evidence retained"
                )
            if (
                result.candidate_metadata.get("profile_source_sha256")
                != hashlib.sha256(source.encode()).hexdigest()
            ):
                raise click.ClickException(
                    f"{name}: container policy source differs from this checkout"
                )
            if repeat and not equivalent_behavior(results[name], result):
                raise click.ClickException(f"{name}: behavior changed across profile repeats")
            results[name] = result
            profiles.append(json.loads(str(result.candidate_metadata["policy_cost_profile"])))
        complexity = scoring_fn_complexity(source, form_aware=settings.form_aware_complexity)
        payload["policies"][name] = {
            "source_path": str(paths[int(name.removeprefix("candidate_"))])
            if name.startswith("candidate_")
            else None,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "implementation_ast_nodes": scoring_fn_complexity(source),
            "effective_ast_nodes": complexity,
            "combined_score": result.combined_score,
            "raw_before_complexity": result.combined_score
            + result.score_breakdown["complexity_cost"],
            "budgets": assess_budgets(
                profiles,
                callback_p99_us=callback_p99_us,
                scan_p99_us=scan_p99_us,
                state_budget_bytes=state_budget_bytes,
            ),
            "profiles": profiles,
        }
    identity = require_single_score_identity(results.values(), context="policy cost comparison")
    payload["identity"] = asdict(identity)
    payload["pareto_policies"] = pareto_names(payload["policies"])
    if reference is not None:
        name = f"candidate_{paths.index(reference.resolve())}"
        limit = settings.promotion_max_candidate_complexity or settings.max_candidate_complexity
        eligible = [
            key
            for key in results
            if key.startswith("candidate_")
            and key != name
            and payload["policies"][key]["effective_ast_nodes"]
            < payload["policies"][name]["effective_ast_nodes"]
            and (limit is None or payload["policies"][key]["effective_ast_nodes"] <= limit)
            and equivalent_behavior(results[name], results[key])
        ]
        payload["simplification"] = {
            "reference": name,
            "eligible": eligible,
            "promotion_size_limit": limit,
            "promoted": False,
        }
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_report(output_dir / "report.md", payload)
    click.echo(output_dir / "report.md")


if __name__ == "__main__":
    main()
