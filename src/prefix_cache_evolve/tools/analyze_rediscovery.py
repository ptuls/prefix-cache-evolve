"""Adjudicate whether weak-seed evolution gets close to the incumbent."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import click

from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents.registry import (
    current_incumbent,
    incumbent_record,
)
from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import file_sha256
from prefix_cache_evolve.problems.prefix_kv_cache.runner import _candidate_panel_decomposition
from prefix_cache_evolve.problems.prefix_kv_cache.utilities import (
    agentic_surrogate_probe_gate,
    normalized_source,
    write_json,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _REPOSITORY_ROOT / "configs/prefix_kv_cache_rediscovery.yaml"
_DEFAULT_ADJUDICATION_CONFIG_PATH = _REPOSITORY_ROOT / "configs/prefix_kv_cache.yaml"
_DEFAULT_OUTPUT_PATH = _REPOSITORY_ROOT / "artifacts/prefix_kv_cache_rediscovery_analysis.json"
_POLICY_ROOT = _REPOSITORY_ROOT / "src/prefix_cache_evolve/problems/prefix_kv_cache"
_REFERENCE_PATHS = {
    "weak_initial": _POLICY_ROOT / "seeds/weak_initial.py",
    "intermediate_compact": incumbent_record("historical_compact_20260607").source_path,
    "incumbent": current_incumbent("production").source_path,
}
_PANELS = ("selection", "probe", "hidden")


def _raw_score(panel: dict[str, Any]) -> float:
    """Return a panel score before the source-complexity charge."""
    return float(panel["combined_score"]) + float(
        panel["score_breakdown"].get("complexity_cost", 0.0)
    )


def _gap_recovery(candidate: float, seed: float, incumbent: float) -> float:
    """Return the fraction of the seed-to-incumbent score gap recovered."""
    gap = incumbent - seed
    if gap <= 0.0:
        return 1.0 if candidate >= incumbent else 0.0
    return (candidate - seed) / gap


def _gap_target(seed: float, incumbent: float, recovery_threshold: float) -> float:
    """Return the score required to recover a fraction of the incumbent gap."""
    return seed + recovery_threshold * (incumbent - seed)


def _panel_recovery(
    candidate: dict[str, Any],
    seed: dict[str, Any],
    incumbent: dict[str, Any],
    panel: str,
) -> dict[str, float]:
    """Return charged and raw recovery for one evaluation panel."""
    candidate_panel = candidate[panel]
    seed_panel = seed[panel]
    incumbent_panel = incumbent[panel]
    charged_candidate = float(candidate_panel["combined_score"])
    charged_seed = float(seed_panel["combined_score"])
    charged_incumbent = float(incumbent_panel["combined_score"])
    raw_candidate = _raw_score(candidate_panel)
    raw_seed = _raw_score(seed_panel)
    raw_incumbent = _raw_score(incumbent_panel)
    return {
        "candidate_charged_score": charged_candidate,
        "seed_charged_score": charged_seed,
        "incumbent_charged_score": charged_incumbent,
        "charged_gap_recovery": _gap_recovery(
            charged_candidate,
            charged_seed,
            charged_incumbent,
        ),
        "candidate_raw_score": raw_candidate,
        "seed_raw_score": raw_seed,
        "incumbent_raw_score": raw_incumbent,
        "raw_gap_recovery": _gap_recovery(raw_candidate, raw_seed, raw_incumbent),
    }


def _reference_label(source: str) -> str:
    """Identify whether source exactly matches a registered starting policy."""
    normalized = normalized_source(source)
    for label, path in _REFERENCE_PATHS.items():
        if normalized_source(path.read_text(encoding="utf-8")) == normalized:
            return label
    return "unregistered"


def _resolve_run_program(run_dir: Path) -> tuple[Path, str]:
    """Select the strongest generated mutation when available."""
    generated = run_dir / "best_generated_mutation.py"
    if generated.is_file():
        return generated, "best_generated_mutation"
    winner = run_dir / "best_program.py"
    if winner.is_file():
        return winner, "best_program"
    raise FileNotFoundError(
        f"{run_dir} contains neither best_generated_mutation.py nor best_program.py"
    )


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object when present."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_run_seed(run_dir: Path, summary: dict[str, Any]) -> Path:
    """Resolve the exact source used to initialize a saved evolution run."""
    persisted = run_dir / "seed_program.py"
    if persisted.is_file():
        return persisted
    seed_label = summary.get("seed_program")
    if isinstance(seed_label, str):
        path = Path(seed_label)
        if path.is_file():
            return path
        repository_path = _REPOSITORY_ROOT / path
        if repository_path.is_file():
            return repository_path
    raise FileNotFoundError(
        f"{run_dir} does not contain seed_program.py and its run summary seed is unavailable"
    )


def _evaluate_path(
    path: Path,
    *,
    config: Any,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one source path once per analysis."""
    key = normalized_source(path.read_text(encoding="utf-8"))
    if key not in cache:
        cache[key] = _candidate_panel_decomposition(config, path)
    return cache[key]


def _run_adjudication(
    run_dir: Path,
    *,
    search_config_path: Path,
    config: Any,
    references: dict[str, dict[str, Any]],
    recovery_threshold: float,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Adjudicate one weak-seed evolution run."""
    summary = _read_json(run_dir / "run_summary.json")
    metadata = _read_json(run_dir / "metadata.json")
    candidate_path, candidate_kind = _resolve_run_program(run_dir)
    seed_path = _resolve_run_seed(run_dir, summary)
    candidate_source = candidate_path.read_text(encoding="utf-8")
    seed_source = seed_path.read_text(encoding="utf-8")
    search_winner_path = run_dir / "best_program.py"
    search_winner_generated = (
        normalized_source(search_winner_path.read_text(encoding="utf-8"))
        != normalized_source(seed_source)
        if search_winner_path.is_file()
        else None
    )
    candidate = _evaluate_path(candidate_path, config=config, cache=cache)
    seed = _evaluate_path(seed_path, config=config, cache=cache)
    incumbent = references["incumbent"]["evaluation"]
    recovery = {panel: _panel_recovery(candidate, seed, incumbent, panel) for panel in _PANELS}
    promotion_limit = (
        config.promotion_max_candidate_complexity
        if config.promotion_max_candidate_complexity is not None
        else config.max_candidate_complexity
    )
    snapshot = run_dir / "config_snapshot.yaml"
    snapshot_matches = snapshot.is_file() and file_sha256(snapshot) == file_sha256(
        search_config_path
    )
    generated = normalized_source(candidate_source) != normalized_source(seed_source)
    valid = all(bool(candidate[panel]["success"]) for panel in _PANELS)
    deployable = promotion_limit is None or candidate["effective_complexity"] <= promotion_limit
    charged_recovery_pass = all(
        recovery[panel]["charged_gap_recovery"] >= recovery_threshold for panel in _PANELS
    )
    surrogate_gate = agentic_surrogate_probe_gate(
        {
            **candidate["selection"]["workload_metrics"],
            **candidate["probe"]["workload_metrics"],
        }
    )
    surrogate_gate_pass = not surrogate_gate["flagged"]
    behaviorally_close = (
        snapshot_matches
        and generated
        and valid
        and deployable
        and charged_recovery_pass
        and surrogate_gate_pass
    )
    return {
        "run_dir": str(run_dir),
        "candidate": str(candidate_path),
        "candidate_kind": candidate_kind,
        "seed": str(seed_path),
        "seed_tier": _reference_label(seed_source),
        "search_seed": metadata.get("search_seed"),
        "iterations": summary.get("iterations"),
        "total_evaluations": summary.get("total_evaluations"),
        "total_cost": summary.get("total_cost"),
        "search_best_score": summary.get("best_score"),
        "search_winner": str(search_winner_path) if search_winner_path.is_file() else None,
        "search_winner_source_differs_from_seed": search_winner_generated,
        "config_snapshot_matches_neutral_config": snapshot_matches,
        "generated_source_differs_from_seed": generated,
        "effective_complexity": candidate["effective_complexity"],
        "promotion_complexity_limit": promotion_limit,
        "recovery": recovery,
        "agentic_surrogate_probe_gate": surrogate_gate,
        "checks": {
            "neutral_config_snapshot": snapshot_matches,
            "valid_all_panels": valid,
            "deployable": deployable,
            "charged_gap_recovery_at_least_threshold_all_panels": charged_recovery_pass,
            "agentic_surrogate_probe_gate": surrogate_gate_pass,
            "behaviorally_close": behaviorally_close,
        },
        "evaluation": candidate,
    }


def _experiment_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize repeated-run evidence without overclaiming one success."""
    close_runs = [run for run in runs if run["checks"]["behaviorally_close"]]
    weak_close_runs = [run for run in close_runs if run["seed_tier"] == "weak_initial"]
    distinct_search_seeds = {
        run["search_seed"] for run in weak_close_runs if run["search_seed"] is not None
    }
    if not runs:
        verdict = "not_run"
        interpretation = "No weak-seed evolution runs were supplied for adjudication."
    elif len(weak_close_runs) >= 2 and len(distinct_search_seeds) >= 2:
        verdict = "supported"
        interpretation = (
            "At least two independent weak-initial-seed runs recovered a deployable "
            "policy close to the incumbent across selection, probe, and hidden panels."
        )
    elif weak_close_runs:
        verdict = "preliminary"
        interpretation = (
            "One weak-initial-seed run recovered a policy close to the incumbent, "
            "but repeated evidence is still insufficient."
        )
    else:
        verdict = "not_supported"
        interpretation = (
            "The supplied runs did not independently recover a deployable policy "
            "close to the incumbent across all panels."
        )
    return {
        "verdict": verdict,
        "interpretation": interpretation,
        "run_count": len(runs),
        "behaviorally_close_count": len(close_runs),
        "weak_initial_behaviorally_close_count": len(weak_close_runs),
        "distinct_weak_initial_close_search_seed_count": len(distinct_search_seeds),
    }


def _rediscovery_targets(
    references: dict[str, dict[str, Any]],
    recovery_threshold: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return charged and raw pass thresholds for each starting policy."""
    incumbent = references["incumbent"]["evaluation"]
    targets = {}
    for label in ("weak_initial", "intermediate_compact"):
        seed = references[label]["evaluation"]
        targets[label] = {}
        for panel in _PANELS:
            targets[label][panel] = {
                "charged_score": _gap_target(
                    float(seed[panel]["combined_score"]),
                    float(incumbent[panel]["combined_score"]),
                    recovery_threshold,
                ),
                "raw_score": _gap_target(
                    _raw_score(seed[panel]),
                    _raw_score(incumbent[panel]),
                    recovery_threshold,
                ),
            }
    return targets


def run_analysis(
    config_path: Path,
    *,
    adjudication_config_path: Path = _DEFAULT_ADJUDICATION_CONFIG_PATH,
    run_dirs: tuple[Path, ...] = (),
    quick: bool = False,
    recovery_threshold: float = 0.8,
) -> dict[str, Any]:
    """Evaluate reference policies and adjudicate supplied evolution runs."""
    search_config = load_evaluator_config(config_path)
    config = load_evaluator_config(adjudication_config_path)
    if quick:
        config = config.with_updates(
            request_count=36,
            seeds=(3,),
            family_request_multipliers={},
        )
    cache: dict[str, dict[str, Any]] = {}
    references = {
        label: {
            "path": str(path),
            "evaluation": _evaluate_path(path, config=config, cache=cache),
        }
        for label, path in _REFERENCE_PATHS.items()
    }
    runs = [
        _run_adjudication(
            run_dir,
            search_config_path=config_path,
            config=config,
            references=references,
            recovery_threshold=recovery_threshold,
            cache=cache,
        )
        for run_dir in run_dirs
    ]
    return {
        "schema": "prefix-kv-cache-behavioral-rediscovery-v2",
        "search_config": str(config_path),
        "adjudication_config": str(adjudication_config_path),
        "quick": quick,
        "recovery_threshold": recovery_threshold,
        "neutrality_contract": {
            "incumbent_source_used_as_search_seed": False,
            "incumbent_scores_or_coefficients_in_prompt": False,
            "incumbent_mechanism_preservation_in_prompt": False,
            "selection_uses_probe_or_hidden": False,
            "search_uses_non_quarantined_guidance_floor": (
                search_config.search_score_mode == "robust_min"
            ),
            "adjudication_uses_canonical_combined_score": (config.search_score_mode == "combined"),
            "adjudication_prefers_best_generated_mutation": True,
        },
        "criterion": (
            "Generated, valid source at or below the promotion complexity cap must "
            "recover the configured fraction of the charged weak-seed-to-incumbent "
            "gap on selection, probe, and hidden panels while passing the agentic gate."
        ),
        "supported_verdict": (
            "At least two distinct-search-seed behavioral rediscoveries from weak_initial."
        ),
        "references": references,
        "targets": _rediscovery_targets(references, recovery_threshold),
        "runs": runs,
        "summary": _experiment_summary(runs),
    }


@click.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    default=_DEFAULT_CONFIG_PATH,
    show_default=True,
)
@click.option(
    "--adjudication-config",
    type=click.Path(path_type=Path),
    default=_DEFAULT_ADJUDICATION_CONFIG_PATH,
    show_default=True,
    help="Canonical config used for final selection, probe, and hidden scoring.",
)
@click.option(
    "--run",
    "run_dirs",
    type=click.Path(path_type=Path, file_okay=False),
    multiple=True,
    help="Saved weak-seed evolution run directory. May be repeated.",
)
@click.option(
    "--quick",
    is_flag=True,
    help="Use a smoke-only evaluator slice; never use it for the final verdict.",
)
@click.option(
    "--recovery-threshold",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.8,
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=_DEFAULT_OUTPUT_PATH,
    show_default=True,
)
def main(
    config: Path,
    adjudication_config: Path,
    run_dirs: tuple[Path, ...],
    quick: bool,
    recovery_threshold: float,
    output: Path,
) -> None:
    """Adjudicate weak-seed incumbent rediscovery runs."""
    if not math.isfinite(recovery_threshold):
        raise click.BadParameter("recovery threshold must be finite")
    payload = run_analysis(
        config,
        adjudication_config_path=adjudication_config,
        run_dirs=run_dirs,
        quick=quick,
        recovery_threshold=recovery_threshold,
    )
    write_json(output, payload)
    click.echo(output)


if __name__ == "__main__":
    main()
