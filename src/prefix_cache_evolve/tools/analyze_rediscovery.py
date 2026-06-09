"""Adjudicate whether weak-seed evolution independently rediscovers the incumbent."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import click

from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import file_sha256
from prefix_cache_evolve.problems.prefix_kv_cache.runner import _candidate_panel_decomposition
from prefix_cache_evolve.problems.prefix_kv_cache.utilities import (
    agentic_surrogate_probe_gate,
    normalized_source,
    write_json,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _REPOSITORY_ROOT / "configs/prefix_kv_cache_rediscovery.yaml"
_DEFAULT_OUTPUT_PATH = _REPOSITORY_ROOT / "artifacts/prefix_kv_cache_rediscovery_analysis.json"
_DEFAULT_MARKDOWN_PATH = _REPOSITORY_ROOT / "docs/results/incumbent_rediscovery.md"
_POLICY_ROOT = _REPOSITORY_ROOT / "src/prefix_cache_evolve/problems/prefix_kv_cache"
_REFERENCE_PATHS = {
    "weak_initial": _POLICY_ROOT / "initial_program.py",
    "intermediate_compact": _POLICY_ROOT / "compact_seed.py",
    "incumbent": _POLICY_ROOT / "pressure_aware_incumbent.py",
}
_PANELS = ("selection", "probe", "hidden")
_DESIGN_FAMILY_SIGNALS = (
    "pressure_conditioned_admission",
    "observed_reuse_state",
    "bounded_decay_state",
    "structural_admission",
    "recurrence_admission",
    "priority_modulation",
)


def _method(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the first method or function with a given name."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _attribute_names(node: ast.AST | None) -> set[str]:
    """Return all referenced attribute names below one AST node."""
    if node is None:
        return set()
    return {child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)}


def _name_tokens(node: ast.AST | None) -> set[str]:
    """Return identifier and attribute tokens below one AST node."""
    if node is None:
        return set()
    tokens = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    tokens.update(_attribute_names(node))
    return tokens


def _self_attributes(node: ast.AST | None) -> set[str]:
    """Return attributes referenced directly from self below one AST node."""
    if node is None:
        return set()
    return {
        child.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and isinstance(child.value, ast.Name)
        and child.value.id == "self"
    }


def _expanded_self_attributes(tree: ast.AST, node: ast.AST | None) -> set[str]:
    """Return self attributes referenced directly or through self helper calls."""
    if node is None:
        return set()
    methods = {
        child.name: child
        for child in ast.walk(tree)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    pending = [node]
    visited: set[str] = set()
    attributes: set[str] = set()
    while pending:
        current = pending.pop()
        attributes.update(_self_attributes(current))
        for child in ast.walk(current):
            if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
                continue
            if not isinstance(child.func.value, ast.Name) or child.func.value.id != "self":
                continue
            called = child.func.attr
            if called in methods and called not in visited:
                visited.add(called)
                pending.append(methods[called])
    return attributes - methods.keys()


def source_design_signals(source: str) -> dict[str, bool | int | list[str]]:
    """Extract coarse, source-level signals for the incumbent's design family."""
    tree = ast.parse(source)
    request_start = _method(tree, "on_request_start")
    cache_hit = _method(tree, "on_cache_hit")
    cache_miss = _method(tree, "on_cache_miss")
    admission = _method(tree, "score_admission")
    source_tokens = _name_tokens(tree)
    request_tokens = _name_tokens(request_start)
    admission_tokens = _name_tokens(admission)
    request_state = _expanded_self_attributes(tree, request_start)
    admission_state = _expanded_self_attributes(tree, admission)
    observation_state = _expanded_self_attributes(
        tree,
        cache_hit,
    ) | _expanded_self_attributes(tree, cache_miss)

    pressure_fields = {"recent_admission_pressure", "recent_miss_rate"}
    structural_fields = {"depth", "descendant_count", "estimated_recompute_cost", "token_count"}
    recurrence_fields = {
        "last_access_gap",
        "access_gap_mean",
        "access_gap_var",
        "subtree_hit_rate",
        "subtree_active_ref_count",
        "recency",
    }
    pressure_state = {
        name for name in request_state | admission_state if "pressure" in name or "miss" in name
    }
    pressure_context = bool(request_tokens & pressure_fields)
    pressure_conditioned_admission = pressure_context and bool(
        admission_state & (request_state | pressure_state)
    )
    observed_reuse_state = bool(observation_state & admission_state)
    bounded_decay_state = bool(
        {"MultiTimescaleDecay", "decay_vector"} & source_tokens
        or any("half_life" in token or "decay" in token for token in source_tokens)
    )
    structural_admission = bool(admission_tokens & structural_fields)
    recurrence_admission = bool(admission_tokens & recurrence_fields)
    priority_modulation = "priority" in admission_tokens and (
        "priority" in request_tokens or bool(observation_state & admission_state)
    )
    signals = {
        "pressure_conditioned_admission": pressure_conditioned_admission,
        "observed_reuse_state": observed_reuse_state,
        "bounded_decay_state": bounded_decay_state,
        "structural_admission": structural_admission,
        "recurrence_admission": recurrence_admission,
        "priority_modulation": priority_modulation,
    }
    matched = [name for name in _DESIGN_FAMILY_SIGNALS if signals[name]]
    signals["matched_signals"] = matched
    signals["matched_signal_count"] = len(matched)
    signals["incumbent_design_family"] = bool(
        pressure_conditioned_admission
        and observed_reuse_state
        and structural_admission
        and len(matched) >= 4
    )
    return signals


def _raw_score(panel: dict[str, Any]) -> float:
    """Return panel score before the source-complexity charge."""
    return float(panel["combined_score"]) + float(
        panel["score_breakdown"].get("complexity_cost", 0)
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
    """Identify whether source exactly matches a registered reference policy."""
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


def _read_run_summary(run_dir: Path) -> dict[str, Any]:
    """Read a saved run summary when present."""
    path = run_dir / "run_summary.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_run_metadata(run_dir: Path) -> dict[str, Any]:
    """Read saved run metadata when present."""
    path = run_dir / "metadata.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    config,
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
    config_path: Path,
    config,
    references: dict[str, dict[str, Any]],
    recovery_threshold: float,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Adjudicate one weak-seed evolution run."""
    summary = _read_run_summary(run_dir)
    metadata = _read_run_metadata(run_dir)
    candidate_path, candidate_kind = _resolve_run_program(run_dir)
    seed_path = _resolve_run_seed(run_dir, summary)
    candidate_source = candidate_path.read_text(encoding="utf-8")
    seed_source = seed_path.read_text(encoding="utf-8")
    seed_label = _reference_label(seed_source)
    candidate = _evaluate_path(candidate_path, config=config, cache=cache)
    seed = _evaluate_path(seed_path, config=config, cache=cache)
    incumbent = references["incumbent"]["evaluation"]
    recovery = {panel: _panel_recovery(candidate, seed, incumbent, panel) for panel in _PANELS}
    signals = source_design_signals(candidate_source)
    promotion_limit = (
        config.promotion_max_candidate_complexity
        if config.promotion_max_candidate_complexity is not None
        else config.max_candidate_complexity
    )
    generated = normalized_source(candidate_source) != normalized_source(seed_source)
    valid = all(bool(candidate[panel]["success"]) for panel in _PANELS)
    deployable = promotion_limit is None or candidate["effective_complexity"] <= promotion_limit
    charged_recovery_pass = all(
        recovery[panel]["charged_gap_recovery"] >= recovery_threshold for panel in _PANELS
    )
    raw_recovery_pass = all(
        recovery[panel]["raw_gap_recovery"] >= recovery_threshold for panel in _PANELS
    )
    surrogate_gate = agentic_surrogate_probe_gate(
        {
            **candidate["selection"]["workload_metrics"],
            **candidate["probe"]["workload_metrics"],
        }
    )
    surrogate_gate_pass = not surrogate_gate["flagged"]
    behaviorally_rediscovered = (
        generated and valid and deployable and charged_recovery_pass and surrogate_gate_pass
    )
    mechanism_rediscovered = behaviorally_rediscovered and bool(signals["incumbent_design_family"])
    equal_or_better = all(
        recovery[panel]["candidate_charged_score"] >= recovery[panel]["incumbent_charged_score"]
        for panel in _PANELS
    )
    snapshot = run_dir / "config_snapshot.yaml"
    snapshot_matches = snapshot.is_file() and file_sha256(snapshot) == file_sha256(config_path)
    return {
        "run_dir": str(run_dir),
        "candidate": str(candidate_path),
        "candidate_kind": candidate_kind,
        "seed": str(seed_path),
        "seed_tier": seed_label,
        "search_seed": metadata.get("search_seed"),
        "iterations": summary.get("iterations"),
        "total_evaluations": summary.get("total_evaluations"),
        "total_cost": summary.get("total_cost"),
        "config_snapshot_matches_neutral_config": snapshot_matches,
        "generated_source_differs_from_seed": generated,
        "effective_complexity": candidate["effective_complexity"],
        "promotion_complexity_limit": promotion_limit,
        "source_signals": signals,
        "recovery": recovery,
        "agentic_surrogate_probe_gate": surrogate_gate,
        "checks": {
            "valid_all_panels": valid,
            "deployable": deployable,
            "charged_gap_recovery_at_least_threshold_all_panels": charged_recovery_pass,
            "raw_gap_recovery_at_least_threshold_all_panels": raw_recovery_pass,
            "agentic_surrogate_probe_gate": surrogate_gate_pass,
            "incumbent_design_family": bool(signals["incumbent_design_family"]),
            "behaviorally_rediscovered": behaviorally_rediscovered,
            "mechanism_rediscovered": mechanism_rediscovered,
            "equal_or_better_than_incumbent_all_panels": equal_or_better,
        },
        "evaluation": candidate,
    }


def _experiment_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize repeated-run evidence without overclaiming a single success."""
    mechanism_runs = [run for run in runs if run["checks"]["mechanism_rediscovered"]]
    behavior_runs = [run for run in runs if run["checks"]["behaviorally_rediscovered"]]
    weak_mechanism_runs = [run for run in mechanism_runs if run["seed_tier"] == "weak_initial"]
    distinct_search_seeds = {
        run["search_seed"] for run in mechanism_runs if run["search_seed"] is not None
    }
    if not runs:
        verdict = "not_run"
        interpretation = "No weak-seed evolution runs were supplied for adjudication."
    elif len(weak_mechanism_runs) >= 2 and len(distinct_search_seeds) >= 2:
        verdict = "supported"
        interpretation = (
            "At least two independent weak-initial-seed runs recovered a deployable "
            "incumbent-family policy across selection, probe, and hidden panels."
        )
    elif mechanism_runs:
        verdict = "preliminary"
        interpretation = (
            "At least one run recovered an incumbent-family policy, but repeated "
            "weak-initial-seed evidence is still insufficient."
        )
    else:
        verdict = "not_supported"
        interpretation = (
            "The supplied runs did not independently recover a deployable "
            "incumbent-family policy across all panels."
        )
    return {
        "verdict": verdict,
        "interpretation": interpretation,
        "run_count": len(runs),
        "behavioral_rediscovery_count": len(behavior_runs),
        "mechanism_rediscovery_count": len(mechanism_runs),
        "weak_initial_mechanism_rediscovery_count": len(weak_mechanism_runs),
        "distinct_mechanism_rediscovery_search_seed_count": len(distinct_search_seeds),
    }


def _rediscovery_targets(
    references: dict[str, dict[str, Any]],
    recovery_threshold: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return fixed charged and raw pass thresholds for each non-incumbent seed."""
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
    run_dirs: tuple[Path, ...] = (),
    quick: bool = False,
    recovery_threshold: float = 0.8,
) -> dict[str, Any]:
    """Evaluate reference seeds and adjudicate any supplied evolution runs."""
    config = load_evaluator_config(config_path)
    if quick:
        config = config.with_updates(
            request_count=36,
            seeds=(3,),
            family_request_multipliers={},
        )
    cache: dict[str, dict[str, Any]] = {}
    references = {}
    for label, path in _REFERENCE_PATHS.items():
        source = path.read_text(encoding="utf-8")
        references[label] = {
            "path": str(path),
            "source_signals": source_design_signals(source),
            "evaluation": _evaluate_path(path, config=config, cache=cache),
        }
    runs = [
        _run_adjudication(
            run_dir,
            config_path=config_path,
            config=config,
            references=references,
            recovery_threshold=recovery_threshold,
            cache=cache,
        )
        for run_dir in run_dirs
    ]
    return {
        "schema": "prefix-kv-cache-incumbent-rediscovery-v2",
        "config": str(config_path),
        "quick": quick,
        "recovery_threshold": recovery_threshold,
        "neutrality_contract": {
            "incumbent_source_used_as_search_seed": False,
            "incumbent_scores_or_coefficients_in_prompt": False,
            "incumbent_mechanism_preservation_in_prompt": False,
            "selection_uses_probe_or_hidden": False,
            "adjudication_prefers_best_generated_mutation": True,
        },
        "criteria": {
            "behavioral_rediscovery": (
                "Generated, valid, deployable source recovers at least the configured "
                "fraction of the charged seed-to-incumbent gap on selection, probe, and "
                "hidden while passing the agentic surrogate gate."
            ),
            "mechanism_rediscovery": (
                "Behavioral rediscovery plus source-level pressure-conditioned admission, "
                "observed reuse state, structural admission, and at least four incumbent-family "
                "signals."
            ),
            "supported_verdict": (
                "At least two distinct-search-seed mechanism rediscoveries from weak_initial."
            ),
        },
        "references": references,
        "targets": _rediscovery_targets(references, recovery_threshold),
        "runs": runs,
        "summary": _experiment_summary(runs),
    }


def _score_cell(reference: dict[str, Any], panel: str) -> str:
    """Format one reference panel score cell."""
    result = reference["evaluation"][panel]
    return f"{float(result['combined_score']):.3f} / {_raw_score(result):.3f}"


def _run_cost_text(run: dict[str, Any]) -> str:
    """Format reported model cost when available."""
    value = run.get("total_cost")
    return f"${float(value):.3f}" if isinstance(value, (int, float)) else "unknown"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    """Write a compact human-readable rediscovery report."""
    lines = [
        "# Incumbent Rediscovery Experiment",
        "",
        f"**Verdict: `{payload['summary']['verdict']}`.** {payload['summary']['interpretation']}",
        "",
        "This experiment tests whether neutral evolution from a weaker base independently",
        "recovers the retained incumbent's behavior or design family. Probe and hidden",
        "panels are used only after search.",
        "",
        "## Reference Starting Conditions",
        "",
        "| Reference | Selection charged / raw | Probe charged / raw | "
        "Hidden charged / raw | Effective complexity | Design-family signals |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, reference in payload["references"].items():
        lines.append(
            f"| `{label}` | {_score_cell(reference, 'selection')} | "
            f"{_score_cell(reference, 'probe')} | {_score_cell(reference, 'hidden')} | "
            f"{reference['evaluation']['effective_complexity']} | "
            f"{reference['source_signals']['matched_signal_count']} |"
        )
    lines.extend(
        [
            "",
            f"## Required {float(payload['recovery_threshold']):.0%} Gap-Recovery Scores",
            "",
            "| Starting seed | Selection charged / raw | Probe charged / raw | "
            "Hidden charged / raw |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, targets in payload["targets"].items():
        lines.append(
            f"| `{label}` | "
            f"{targets['selection']['charged_score']:.3f} / "
            f"{targets['selection']['raw_score']:.3f} | "
            f"{targets['probe']['charged_score']:.3f} / "
            f"{targets['probe']['raw_score']:.3f} | "
            f"{targets['hidden']['charged_score']:.3f} / "
            f"{targets['hidden']['raw_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Run Adjudication",
            "",
            "| Run | Seed | Selection charged / raw recovery | Probe charged / raw recovery | "
            "Hidden charged / raw recovery | Agentic gate | Complexity | Family | Verdict |",
            "|---|---|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    if payload["runs"]:
        for run in payload["runs"]:
            run_name = Path(run["run_dir"]).name
            checks = run["checks"]
            verdict = (
                "mechanism"
                if checks["mechanism_rediscovered"]
                else "behavioral"
                if checks["behaviorally_rediscovered"]
                else "no"
            )
            lines.append(
                f"| `{run_name}` | `{run['seed_tier']}` | "
                f"{run['recovery']['selection']['charged_gap_recovery']:.1%} / "
                f"{run['recovery']['selection']['raw_gap_recovery']:.1%} | "
                f"{run['recovery']['probe']['charged_gap_recovery']:.1%} / "
                f"{run['recovery']['probe']['raw_gap_recovery']:.1%} | "
                f"{run['recovery']['hidden']['charged_gap_recovery']:.1%} / "
                f"{run['recovery']['hidden']['raw_gap_recovery']:.1%} | "
                f"`{'pass' if checks['agentic_surrogate_probe_gate'] else 'fail'}` | "
                f"{run['effective_complexity']} | "
                f"{run['source_signals']['matched_signal_count']} | `{verdict}` |"
            )
    else:
        lines.append("| No runs supplied | - | - | - | - | - | - | - | `not_run` |")
    if payload["runs"]:
        lines.extend(["", "## Run Interpretation", ""])
        for run in payload["runs"]:
            run_name = Path(run["run_dir"]).name
            checks = run["checks"]
            matched = run["source_signals"]["matched_signal_count"]
            lines.append(
                f"- `{run_name}` used search seed `{run['search_seed']}`, completed "
                f"`{run['total_evaluations']}` evaluations, and reported cost "
                f"`{_run_cost_text(run)}`."
            )
            if checks["incumbent_design_family"] and not checks["behaviorally_rediscovered"]:
                lines.append(
                    f"- The run independently assembled `{matched}` of "
                    f"`{len(_DESIGN_FAMILY_SIGNALS)}` incumbent-family signals, but this is "
                    "mechanism emergence rather than behavioral rediscovery."
                )
            if not checks["deployable"]:
                lines.append(
                    f"- Its strongest generated policy has effective complexity "
                    f"`{run['effective_complexity']}`, above the "
                    f"`{run['promotion_complexity_limit']}` deployability limit."
                )
            if not checks["charged_gap_recovery_at_least_threshold_all_panels"]:
                lines.append(
                    "- It misses the all-panel charged recovery rule; notably, probe "
                    f"recovery is `{run['recovery']['probe']['charged_gap_recovery']:.1%}`."
                )
            if not checks["agentic_surrogate_probe_gate"]:
                failed_metrics = run["agentic_surrogate_probe_gate"]["failed_metrics"]
                lines.append(
                    f"- It fails the agentic surrogate gate on `{', '.join(failed_metrics)}`."
                )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            f"- A behavioral rediscovery must recover at least "
            f"`{float(payload['recovery_threshold']):.0%}` of the charged seed-to-incumbent "
            "gap on selection, probe, and hidden panels while remaining valid, deployable,",
            "  and within every agentic surrogate-gate limit.",
            "- A mechanism rediscovery must also reproduce the incumbent design family under",
            "  coarse AST-derived signals. This is a diagnostic classification, not proof of",
            "  semantic equivalence.",
            "- The overall claim is supported only after at least two independent search seeds",
            "  rediscover an incumbent-family policy from the weakest initial seed.",
            "",
            "A failure to rediscover does not invalidate the incumbent as a local-search result.",
            "It means claims should describe it as incumbent-conditioned rather than generally",
            "discoverable.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


@click.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    default=_DEFAULT_CONFIG_PATH,
    show_default=True,
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
@click.option(
    "--markdown",
    type=click.Path(path_type=Path),
    default=_DEFAULT_MARKDOWN_PATH,
    show_default=True,
)
def main(
    config: Path,
    run_dirs: tuple[Path, ...],
    quick: bool,
    recovery_threshold: float,
    output: Path,
    markdown: Path,
) -> None:
    """Adjudicate weak-seed incumbent rediscovery runs."""
    if not math.isfinite(recovery_threshold):
        raise click.BadParameter("recovery threshold must be finite")
    payload = run_analysis(
        config,
        run_dirs=run_dirs,
        quick=quick,
        recovery_threshold=recovery_threshold,
    )
    write_json(output, payload)
    _write_markdown(markdown, payload)
    click.echo(output)
    click.echo(markdown)


if __name__ == "__main__":
    main()
