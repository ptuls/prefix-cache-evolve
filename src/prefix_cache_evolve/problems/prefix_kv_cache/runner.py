"""Run prefix KV-cache baseline reports or Levi evolution."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prefix_cache_evolve.evaluator_entry import load_candidate_factory, run_with_timeout
from prefix_cache_evolve.evaluators.baseline_suite import BASELINE_SUITE_EVALUATOR
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    BASELINES,
    REPORTING_BASELINES,
    EvaluationResult,
    EvaluatorConfig,
    PrefixKVCacheEvaluator,
    WorkloadRequest,
    scoring_fn_complexity,
)
from prefix_cache_evolve.workflow.configuration import (
    ConfigLoader,
    MinimalConfigProvider,
    YamlConfigProvider,
)
from prefix_cache_evolve.workflow.execution import LeviRunner
from prefix_cache_evolve.workflow.program import ProgramSource
from prefix_cache_evolve.workflow.reporting import EvolutionReporter

from . import reporting as baseline_reporting
from .configuration import (
    DEFAULT_CONFIG_PATH,
    load_evaluator_config,
    prefix_kv_config_environment,
)
from .pressure_aware_incumbent import build_candidate
from .trace_replay import calibrate_anonymized_trace, load_anonymized_trace

_DEFAULT_SEED_PATH = Path(__file__).parent / "pressure_aware_incumbent.py"
DEFAULT_SEED_SOURCE = ProgramSource(_DEFAULT_SEED_PATH.read_text(encoding="utf-8"))
_EVALUATOR_PATH = Path(__file__).parent / "evaluator.py"
_COMPACT_SEED_PATH = Path(__file__).parent / "compact_seed.py"
_DEFAULT_CONFIG_FILE = str(DEFAULT_CONFIG_PATH)
_CONFIG_LOADER = ConfigLoader()
_DEFAULT_CAPACITY_SWEEP_BLOCKS = (48, 96)
_QUICK_REPORT_WARNING = baseline_reporting.QUICK_REPORT_WARNING
_baseline_group = baseline_reporting.baseline_group
write_baseline_comparison_report = baseline_reporting.write_baseline_comparison_report
_SENSITIVITY_WEIGHTS = (
    "churn_weight",
    "underfill_weight",
    "wasted_admission_weight",
    "avoidable_eviction_weight",
    "fairness_weight",
)
_SENSITIVITY_FACTORS = (0.0, 0.5, 1.0, 1.5, 2.0)


def _build_runner() -> LeviRunner:
    import levi

    return LeviRunner(
        levi.evolve_code,
        _EVALUATOR_PATH,
        problem_description=(
            "Search for simple prefix KV-cache admission and eviction scoring "
            "heuristics that generalize across shifted LLM-serving workloads. "
            "PrefixBlockInfo is a frozen per-callback value object; use "
            "block.prefix_hash or block.block_id as the stable key, never "
            "id(block) or guessed fallback attributes. Documented block fields "
            "are block_id, prefix_hash, parent_hash, depth, start_token, "
            "end_token, token_count, tenant_id, created_at, last_accessed_at, "
            "hit_count, descendant_count, active_ref_count, "
            "estimated_recompute_cost, prev_last_accessed_at, last_access_gap, "
            "access_gap_mean, access_gap_var, subtree_hit_rate, "
            "subtree_active_ref_count, estimated_future_reuse, and "
            "estimated_next_reuse_distance. Future-reuse fields are None for "
            "deployable candidates. The only lifecycle callbacks that fire are "
            "on_request_start, on_cache_hit, and on_cache_miss. Do not add "
            "on_request_end, on_block_admitted, on_block_evicted, or state that "
            "depends on unsupported callbacks. session_id is request-only "
            "metadata; PrefixBlockInfo has tenant_id but no session_id. "
            "RequestInfo also exposes online recent_admission_pressure and "
            "recent_miss_rate. MultiTimescaleDecay and decay_vector are optional "
            "canonical primitives for bounded multi-timescale state. "
            "MultiTimescaleDecay.observe_vector applies distinct updates to "
            "different decay channels. Preserve or simplify canonical primitive "
            "state before replacing it with bespoke per-key decay dictionaries; "
            "canonical calls receive the bounded form-aware complexity subsidy. "
            "Explore recurrence-aware and regime-conditional control flow without "
            "hard-coding workload-family names or request_type values. The "
            "verifier rewards request-tail and worst-quarter service, and "
            "penalizes token-weighted wasted admissions and avoidable evictions. "
            "A small concave admission-utility reward measures saved tokens per "
            "admitted cache slot, so full and partial blocks are not treated "
            "identically. "
            "Priority is useful QoS metadata but does not imply future reuse."
        ),
        function_signature=(
            "def build_candidate(capacity_blocks: int, block_size_tokens: int, "
            "seed: int | None = None):"
        ),
    )


def _build_workflow(
    provider,
    *,
    program_source: ProgramSource = DEFAULT_SEED_SOURCE,
) -> object:
    from prefix_cache_evolve.workflow.workflow import EvolutionWorkflow

    return EvolutionWorkflow(
        program_source=program_source,
        config_provider=provider,
        runner=_build_runner(),
        reporter=EvolutionReporter(),
    )


def _load_seed_program_source(path: Path) -> ProgramSource:
    """Load an evolution seed from a candidate file or saved run directory."""

    candidate_path = _resolve_candidate_program(path)
    return ProgramSource(candidate_path.read_text(encoding="utf-8"))


def demo_run_evolution(
    iterations: int = 25,
    config_file: str = _DEFAULT_CONFIG_FILE,
    *,
    quick: bool = False,
    seed_program: Path | None = None,
    artifact_output: Path | None = Path("artifacts/prefix_kv_cache_runs"),
) -> object:
    provider = (
        MinimalConfigProvider() if quick else YamlConfigProvider(Path(config_file), _CONFIG_LOADER)
    )
    program_source = (
        _load_seed_program_source(seed_program) if seed_program is not None else DEFAULT_SEED_SOURCE
    )
    workflow = _build_workflow(provider, program_source=program_source)
    with prefix_kv_config_environment(Path(config_file), quick=quick):
        result = workflow.execute(iterations)
    if artifact_output is not None:
        artifact_dir = save_run_artifacts(
            result,
            artifact_output,
            iterations=iterations,
            config_label=provider.describe(),
            seed_label=str(seed_program or _DEFAULT_SEED_PATH),
            seed_source=program_source.text(),
            report_config=load_evaluator_config(Path(config_file)),
            report_config_file=config_file,
            config_snapshot=Path(config_file) if not quick else None,
        )
        print(f"saved_run_artifacts={artifact_dir}")
        print(f"baseline_comparison={artifact_dir / 'baseline_comparison.md'}")
    return result


def compare_baselines(
    *,
    quick: bool = False,
    capacity_blocks: int | None = None,
    capacity_sweep_blocks: tuple[int, ...] = (),
    block_size_tokens: int | None = None,
    candidate_program: Path | None = None,
    config_file: str = _DEFAULT_CONFIG_FILE,
) -> None:
    config = _config_from_args(
        quick=quick,
        capacity_blocks=capacity_blocks,
        capacity_sweep_blocks=capacity_sweep_blocks,
        block_size_tokens=block_size_tokens,
        config_file=config_file,
    )
    if quick:
        print(_QUICK_REPORT_WARNING)
    results = _evaluate_baselines(config, include_reporting=True)
    if candidate_program is not None:
        candidate_path = _resolve_candidate_program(candidate_program)
        results = {
            "candidate": _evaluate_candidate_program(config, candidate_path),
            **results,
        }
        report_path = candidate_path.parent / "baseline_comparison.md"
        write_baseline_comparison_report(
            report_path,
            results,
            candidate_path=candidate_path,
            command=_baseline_report_command(
                quick=quick,
                capacity_sweep_blocks=capacity_sweep_blocks,
                candidate_program=candidate_program,
                config_file=config_file,
            ),
            quick=quick,
            config=config,
        )
        print(f"baseline_comparison={report_path}")
    for name, result in results.items():
        print(f"{name}: combined_score={result.combined_score:.3f} [{_baseline_group(name)}]")
        for capacity, metrics in result.capacity_metrics.items():
            print(
                "  "
                f"{capacity}: token_hit_rate={metrics['token_hit_rate']:.3f}, "
                f"block_hit_rate={metrics['block_hit_rate']:.3f}, "
                f"churn_per_1k={metrics['cache_churn_per_1k']:.1f}"
            )
        for workload, metrics in result.workload_metrics.items():
            print(
                "  "
                f"{workload}: token_hit_rate={metrics['token_hit_rate']:.3f}, "
                f"block_hit_rate={metrics['block_hit_rate']:.3f}, "
                f"churn_per_1k={metrics['cache_churn_per_1k']:.1f}"
            )


def write_baseline_plots(
    output_dir: Path,
    *,
    quick: bool = False,
    capacity_blocks: int | None = None,
    capacity_sweep_blocks: tuple[int, ...] = (),
    block_size_tokens: int | None = None,
    config_file: str = _DEFAULT_CONFIG_FILE,
) -> tuple[Path, ...]:
    """Write lightweight SVG plots for baseline comparison and debugging."""

    config = _config_from_args(
        quick=quick,
        capacity_blocks=capacity_blocks,
        capacity_sweep_blocks=capacity_sweep_blocks,
        block_size_tokens=block_size_tokens,
        config_file=config_file,
    )
    results = _evaluate_baselines(config)
    return baseline_reporting.write_baseline_plot_files(output_dir, results)


def save_run_artifacts(
    result: object,
    output_root: Path,
    *,
    iterations: int,
    config_label: str,
    seed_label: str | None = None,
    seed_source: str | None = None,
    report_config: EvaluatorConfig | None = None,
    report_config_file: str = _DEFAULT_CONFIG_FILE,
    config_snapshot: Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Persist the best evolved program and evaluation metadata."""

    timestamp = timestamp or datetime.now(UTC)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_program = (
        getattr(result, "best_program", None)
        or getattr(result, "best_code", None)
        or getattr(result, "code", "")
        or ""
    )
    (run_dir / "best_program.py").write_text(str(best_program), encoding="utf-8")

    metrics = getattr(result, "metrics", {}) or {}
    artifacts = getattr(result, "artifacts", {}) or {}
    metadata = getattr(result, "metadata", {}) or {}
    config_snapshot_name = None
    if config_snapshot is not None and config_snapshot.is_file():
        config_snapshot_name = "config_snapshot.yaml"
        (run_dir / config_snapshot_name).write_text(
            config_snapshot.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    summary = {
        "run_id": run_id,
        "iterations": iterations,
        "config": config_label,
        "config_snapshot": config_snapshot_name,
        "seed_program": seed_label,
        "best_score": getattr(result, "best_score", None),
        "total_evaluations": getattr(result, "total_evaluations", None),
        "total_cost": getattr(result, "total_cost", None),
        "archive_size": getattr(result, "archive_size", None),
        "runtime_seconds": getattr(result, "runtime_seconds", None),
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "artifacts.json", artifacts)
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "run_summary.json", summary)

    _persist_paradigm_candidates(run_dir, metadata=metadata)
    _persist_best_generated_mutation(
        run_dir,
        metadata=metadata,
        seed_source=seed_source,
        config=report_config or _artifact_report_config(),
    )

    try:
        config = report_config or _artifact_report_config()
        candidate_path = run_dir / "best_program.py"
        report_results = {
            "candidate": _evaluate_candidate_program(config, candidate_path),
            **_evaluate_baselines(config, include_reporting=True),
        }
        write_baseline_comparison_report(
            run_dir / "baseline_comparison.md",
            report_results,
            candidate_path=candidate_path,
            command=_baseline_report_command(
                quick=False,
                capacity_sweep_blocks=config.effective_capacity_blocks(),
                candidate_program=run_dir,
                config_file=report_config_file,
            ),
            quick=False,
            config=config,
        )
    except Exception as exc:
        _write_json(
            run_dir / "baseline_comparison_error.json",
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    return run_dir


def _persist_paradigm_candidates(run_dir: Path, *, metadata: dict[str, Any]) -> None:
    """Copy all evaluated PE candidates into the final run artifacts."""

    source_value = metadata.get("levi_paradigm_candidates_dir")
    if not source_value:
        return
    source_dir = Path(str(source_value))
    if source_dir.is_dir():
        shutil.copytree(source_dir, run_dir / "paradigm_candidates", dirs_exist_ok=True)


def _persist_best_generated_mutation(
    run_dir: Path,
    *,
    metadata: dict[str, Any],
    seed_source: str | None,
    config: EvaluatorConfig,
) -> None:
    """Persist and decompose the strongest archived elite that differs from seed."""

    snapshot_value = metadata.get("levi_snapshot_path")
    if not seed_source or not snapshot_value:
        return
    snapshot_path = Path(str(snapshot_value))
    if not snapshot_path.is_file():
        return
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        elites = snapshot.get("elites", [])
        generated = [
            elite
            for elite in elites
            if _normalized_source(_elite_source(elite)) != _normalized_source(seed_source)
        ]
        if not generated:
            return
        strongest = max(
            generated,
            key=lambda elite: float(elite.get("primary_score", float("-inf"))),
        )
        generated_source = _elite_source(strongest)
        generated_path = run_dir / "best_generated_mutation.py"
        generated_path.write_text(generated_source, encoding="utf-8")
        seed_path = run_dir / "seed_program.py"
        seed_path.write_text(seed_source, encoding="utf-8")
        _write_json(
            run_dir / "best_generated_mutation_snapshot.json",
            {key: value for key, value in strongest.items() if key not in {"code", "content"}},
        )
        decomposition = {
            "schema": "prefix-kv-cache-generated-mutation-decomposition-v1",
            "snapshot": str(snapshot_path),
            "generated_program_id": strongest.get("program_id"),
            "snapshot_primary_score": strongest.get("primary_score"),
            "seed": _candidate_panel_decomposition(config, seed_path),
            "best_generated_mutation": _candidate_panel_decomposition(
                config,
                generated_path,
            ),
        }
        _write_json(run_dir / "best_generated_mutation_decomposition.json", decomposition)
        _write_generated_mutation_report(
            run_dir / "best_generated_mutation_decomposition.md",
            decomposition,
        )
    except Exception as exc:
        _write_json(
            run_dir / "best_generated_mutation_decomposition_error.json",
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "snapshot": str(snapshot_path),
            },
        )


def _elite_source(elite: dict[str, Any]) -> str:
    """Return source text from a Levi snapshot elite."""

    return str(elite.get("content") or elite.get("code") or "")


def _normalized_source(source: str) -> str:
    """Normalize source for exact seed-versus-generated identity checks."""

    return source.strip()


def _candidate_panel_decomposition(
    config: EvaluatorConfig,
    candidate_path: Path,
) -> dict[str, Any]:
    """Evaluate one candidate on selection, probe, and hidden panels."""

    source = candidate_path.read_text(encoding="utf-8")
    raw_complexity = scoring_fn_complexity(source)
    effective_complexity = scoring_fn_complexity(
        source,
        form_aware=config.form_aware_complexity,
    )
    selection = _evaluate_candidate_program(config, candidate_path)
    probe = _evaluate_candidate_program(config, candidate_path, splits=("probe",))
    hidden = _evaluate_candidate_program(config, candidate_path, splits=("hidden",))
    return {
        "candidate": str(candidate_path),
        "raw_complexity": raw_complexity,
        "effective_complexity": effective_complexity,
        "primitive_subsidy_nodes": raw_complexity - effective_complexity,
        "primitive_subsidy_exercised": effective_complexity < raw_complexity,
        "selection": _evaluation_result_summary(selection),
        "probe": _evaluation_result_summary(probe),
        "hidden": _evaluation_result_summary(hidden),
    }


def _write_generated_mutation_report(
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


def hidden_report(
    *,
    quick: bool = False,
    capacity_blocks: int | None = None,
    capacity_sweep_blocks: tuple[int, ...] = (),
    block_size_tokens: int | None = None,
    candidate_program: Path | None = None,
    config_file: str = _DEFAULT_CONFIG_FILE,
) -> None:
    config = _config_from_args(
        quick=quick,
        capacity_blocks=capacity_blocks,
        capacity_sweep_blocks=capacity_sweep_blocks,
        block_size_tokens=block_size_tokens,
        config_file=config_file,
    )
    if candidate_program is None:
        print("default_candidate:")
        champion = PrefixKVCacheEvaluator(config, splits=("hidden",))(build_candidate)
    else:
        candidate_path = _resolve_candidate_program(candidate_program)
        print(f"candidate={candidate_path}")
        champion = _evaluate_candidate_program(config, candidate_path, splits=("hidden",))
    print(f"  combined_score={champion.combined_score:.3f}")
    results = _evaluate_baselines(
        config,
        include_reporting=True,
        splits=("hidden",),
    )
    for name, result in results.items():
        print(f"{name}: combined_score={result.combined_score:.3f}")


def probe_report(
    *,
    output_path: Path,
    quick: bool = False,
    capacity_blocks: int | None = None,
    capacity_sweep_blocks: tuple[int, ...] = (),
    block_size_tokens: int | None = None,
    candidate_program: Path | None = None,
    config_file: str = _DEFAULT_CONFIG_FILE,
) -> dict[str, Any]:
    """Evaluate and report the quarantined structure-generalization probe."""

    config = _config_from_args(
        quick=quick,
        capacity_blocks=capacity_blocks,
        capacity_sweep_blocks=capacity_sweep_blocks,
        block_size_tokens=block_size_tokens,
        config_file=config_file,
    )
    candidate_path = _resolve_candidate_program(candidate_program or _COMPACT_SEED_PATH)
    results = {
        "candidate": _evaluate_candidate_program(
            config,
            candidate_path,
            splits=("probe",),
        ),
        **_evaluate_baselines(config, include_reporting=True, splits=("probe",)),
    }
    payload = {
        "schema": "prefix-kv-cache-structure-probe-v1",
        "candidate": str(candidate_path),
        "selection_score_excludes_probe": True,
        "results": {name: _evaluation_result_summary(result) for name, result in results.items()},
    }
    _write_json(output_path, payload)
    print(f"structure_probe={output_path}")
    for name, result in sorted(
        results.items(), key=lambda item: item[1].combined_score, reverse=True
    ):
        print(f"{name}: probe_combined_score={result.combined_score:.3f}")
        for workload, metrics in result.workload_metrics.items():
            print(
                f"  {workload}: token_hit_rate={metrics['token_hit_rate']:.3f}, "
                f"block_hit_rate={metrics['block_hit_rate']:.3f}, "
                f"churn_per_1k={metrics['cache_churn_per_1k']:.1f}"
            )
    return payload


def calibrate_trace_report(
    trace_path: Path,
    *,
    output_path: Path,
    arrival_bucket_ms: int,
    request_limit: int | None,
) -> dict[str, Any]:
    """Write production-trace calibration targets without loading prompt content."""

    calibration = calibrate_anonymized_trace(
        trace_path,
        arrival_bucket_ms=arrival_bucket_ms,
        request_limit=request_limit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, calibration)
    print(f"trace_calibration={output_path}")
    print(json.dumps(calibration, indent=2, sort_keys=True))
    return calibration


def replay_trace_report(
    trace_path: Path,
    *,
    output_path: Path,
    candidate_program: Path | None,
    arrival_bucket_ms: int,
    request_limit: int | None,
    config_file: str,
    capacity_blocks: int | None = None,
    capacity_sweep_blocks: tuple[int, ...] = (),
    block_size_tokens: int | None = None,
) -> dict[str, Any]:
    """Replay an anonymized metadata trace through deployable policies."""

    config = _config_from_args(
        quick=False,
        capacity_blocks=capacity_blocks,
        capacity_sweep_blocks=capacity_sweep_blocks,
        block_size_tokens=block_size_tokens,
        config_file=config_file,
    )
    requests = load_anonymized_trace(
        trace_path,
        block_size_tokens=config.block_size_tokens,
        arrival_bucket_ms=arrival_bucket_ms,
        request_limit=request_limit,
    )
    results = BASELINE_SUITE_EVALUATOR.evaluate_requests(config, BASELINES, requests)
    if candidate_program is not None:
        candidate_path = _resolve_candidate_program(candidate_program)
        results = {
            "candidate": _evaluate_replay_candidate_program(
                config,
                candidate_path,
                requests,
            ),
            **results,
        }
    payload = {
        "schema": "prefix-kv-cache-trace-replay-v1",
        "trace_path": str(trace_path),
        "request_count": len(requests),
        "arrival_bucket_ms": arrival_bucket_ms,
        "block_size_tokens": config.block_size_tokens,
        "capacity_blocks": list(config.effective_capacity_blocks()),
        "results": {name: _evaluation_result_summary(result) for name, result in results.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, payload)
    print(f"trace_replay={output_path}")
    for name, result in sorted(
        results.items(), key=lambda item: item[1].combined_score, reverse=True
    ):
        metrics = result.split_metrics["validation"]
        print(
            f"{name}: combined_score={result.combined_score:.3f}, "
            f"token_hit_rate={float(metrics['token_hit_rate']):.3f}, "
            f"churn_per_1k={float(metrics['cache_churn_per_1k']):.1f}"
        )
    return payload


def write_score_weight_sensitivity_report(
    output_path: Path,
    *,
    candidate_program: Path,
    config_file: str,
    capacity_blocks: int | None = None,
    capacity_sweep_blocks: tuple[int, ...] = (),
    block_size_tokens: int | None = None,
) -> Path:
    """Evaluate rank sensitivity to the verifier's principal penalty weights."""

    config = _config_from_args(
        quick=False,
        capacity_blocks=capacity_blocks,
        capacity_sweep_blocks=capacity_sweep_blocks,
        block_size_tokens=block_size_tokens,
        config_file=config_file,
    )
    candidate_path = _resolve_candidate_program(candidate_program)
    results = {
        "candidate": _evaluate_candidate_program(config, candidate_path),
        **_evaluate_baselines(config),
    }
    rows = _score_weight_sensitivity_rows(results, config)
    lines = [
        "# Prefix KV-Cache Score-Weight Sensitivity",
        "",
        f"Candidate: `{candidate_path}`",
        "",
        (
            "Each row rescales one score weight while holding all simulator trials "
            "and other weights fixed. This isolates objective sensitivity from "
            "workload randomness."
        ),
        "",
        "| Weight | Base | Factor | Candidate score | Candidate rank | Best policy |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['weight']}` | {row['base_value']:.4g} | "
            f"{row['factor']:.1f} | {row['candidate_score']:.3f} | "
            f"{row['candidate_rank']} | `{row['best_policy']}` |"
        )
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"score_weight_sensitivity={output_path}")
    return output_path


def _score_weight_sensitivity_rows(
    results: dict[str, EvaluationResult],
    config: EvaluatorConfig,
    *,
    weights: tuple[str, ...] = _SENSITIVITY_WEIGHTS,
    factors: tuple[float, ...] = _SENSITIVITY_FACTORS,
) -> list[dict[str, Any]]:
    """Rescore fixed trials over one-at-a-time score-weight perturbations."""

    rows = []
    for weight in weights:
        base_value = float(getattr(config, weight))
        for factor in factors:
            variant = replace(config, **{weight: base_value * factor})
            rescored = {}
            for name, result in results.items():
                complexity = int(result.candidate_metadata.get("scoring_fn_complexity", 0))
                rescored[name] = (
                    PrefixKVCacheEvaluator(variant)
                    .rescore_trials(
                        result.trials,
                        scoring_fn_complexity=complexity,
                    )
                    .combined_score
                )
            ranking = sorted(rescored, key=rescored.get, reverse=True)
            rows.append(
                {
                    "weight": weight,
                    "base_value": base_value,
                    "factor": factor,
                    "candidate_score": rescored.get("candidate", float("nan")),
                    "candidate_rank": (
                        ranking.index("candidate") + 1 if "candidate" in ranking else 0
                    ),
                    "best_policy": ranking[0],
                    "scores": rescored,
                }
            )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smoke-only single-seed slice; do not use it for ranking decisions.",
    )
    parser.add_argument(
        "--workload-preset",
        default="default",
        choices=("default", "small"),
    )
    parser.add_argument("--capacity-blocks", type=int, default=None)
    parser.add_argument(
        "--capacity-sweep-blocks",
        default="",
        help="Comma-separated capacities to evaluate, for example 48,96.",
    )
    parser.add_argument("--block-size-tokens", type=int, default=None)
    parser.add_argument("--baseline-report", action="store_true")
    parser.add_argument(
        "--candidate-program",
        type=Path,
        default=None,
        help="Candidate .py file or run directory to compare in --baseline-report.",
    )
    parser.add_argument("--hidden-report", action="store_true")
    parser.add_argument(
        "--probe-report",
        action="store_true",
        help="Evaluate the quarantined recurrence/structure-generalization probe.",
    )
    parser.add_argument(
        "--probe-output",
        type=Path,
        default=Path("artifacts/prefix_kv_cache_structure_probe.json"),
        help="JSON output for --probe-report.",
    )
    parser.add_argument(
        "--plot-report",
        action="store_true",
        help="Write SVG baseline plots without launching Levi.",
    )
    parser.add_argument(
        "--plot-output",
        default="artifacts/prefix_kv_cache_plots",
        help="Directory for --plot-report SVG files.",
    )
    parser.add_argument(
        "--artifact-output",
        default="artifacts/prefix_kv_cache_runs",
        help="Directory for saved evolution run artifacts.",
    )
    parser.add_argument(
        "--seed-program",
        type=Path,
        default=None,
        help=(
            "Candidate .py file or saved run directory to use as the evolution seed; "
            "defaults to the pressure-aware incumbent."
        ),
    )
    parser.add_argument(
        "--no-save-artifacts",
        action="store_true",
        help="Do not save best_program.py and run metadata after evolution.",
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG_FILE,
        help="Path to the Levi YAML config file.",
    )
    parser.add_argument(
        "--calibrate-trace",
        type=Path,
        default=None,
        help="Summarize an anonymized metadata-only JSONL production trace.",
    )
    parser.add_argument(
        "--replay-trace",
        type=Path,
        default=None,
        help="Replay an anonymized metadata-only JSONL production trace.",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=Path("artifacts/prefix_kv_cache_trace_report.json"),
        help="Output JSON for --calibrate-trace or --replay-trace.",
    )
    parser.add_argument(
        "--trace-arrival-bucket-ms",
        type=int,
        default=100,
        help="Convert trace timestamps to simulator arrival steps using this bucket.",
    )
    parser.add_argument(
        "--trace-request-limit",
        type=int,
        default=None,
        help="Optional prefix request count for trace calibration or replay.",
    )
    parser.add_argument(
        "--sensitivity-report",
        action="store_true",
        help="Rescore fixed full-panel trials under one-at-a-time weight changes.",
    )
    parser.add_argument(
        "--sensitivity-output",
        type=Path,
        default=Path("artifacts/prefix_kv_cache_weight_sensitivity.md"),
        help="Markdown output for --sensitivity-report.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    capacity_sweep_blocks = _parse_capacity_sweep(args.capacity_sweep_blocks)
    if args.calibrate_trace is not None:
        calibrate_trace_report(
            args.calibrate_trace,
            output_path=args.trace_output,
            arrival_bucket_ms=args.trace_arrival_bucket_ms,
            request_limit=args.trace_request_limit,
        )
        return
    if args.replay_trace is not None:
        replay_trace_report(
            args.replay_trace,
            output_path=args.trace_output,
            candidate_program=args.candidate_program,
            arrival_bucket_ms=args.trace_arrival_bucket_ms,
            request_limit=args.trace_request_limit,
            config_file=args.config,
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
        )
        return
    if args.sensitivity_report:
        if args.candidate_program is None:
            raise ValueError("--sensitivity-report requires --candidate-program")
        write_score_weight_sensitivity_report(
            args.sensitivity_output,
            candidate_program=args.candidate_program,
            config_file=args.config,
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
        )
        return
    if args.baseline_report:
        compare_baselines(
            quick=args.quick or args.workload_preset == "small",
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
            candidate_program=args.candidate_program,
            config_file=args.config,
        )
        return
    if args.hidden_report:
        hidden_report(
            quick=args.quick or args.workload_preset == "small",
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
            candidate_program=args.candidate_program,
            config_file=args.config,
        )
        return
    if args.probe_report:
        probe_report(
            output_path=args.probe_output,
            quick=args.quick or args.workload_preset == "small",
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
            candidate_program=args.candidate_program,
            config_file=args.config,
        )
        return
    if args.plot_report:
        paths = write_baseline_plots(
            Path(args.plot_output),
            quick=args.quick or args.workload_preset == "small",
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
            config_file=args.config,
        )
        for path in paths:
            print(path)
        return
    demo_run_evolution(
        iterations=args.iterations,
        config_file=args.config,
        quick=args.quick or args.workload_preset == "small",
        seed_program=args.seed_program,
        artifact_output=None if args.no_save_artifacts else Path(args.artifact_output),
    )


def _config_from_args(
    *,
    quick: bool,
    capacity_blocks: int | None,
    capacity_sweep_blocks: tuple[int, ...] = (),
    block_size_tokens: int | None,
    config_file: str = _DEFAULT_CONFIG_FILE,
) -> EvaluatorConfig:
    base = load_evaluator_config(Path(config_file))
    effective_capacity_sweep = capacity_sweep_blocks
    if not effective_capacity_sweep and capacity_blocks is None:
        effective_capacity_sweep = base.capacity_sweep_blocks or _DEFAULT_CAPACITY_SWEEP_BLOCKS
    config = replace(
        base,
        request_count=36 if quick else base.request_count,
        seeds=(3,) if quick else base.seeds,
        family_request_multipliers={} if quick else base.family_request_multipliers,
        capacity_blocks=capacity_blocks or base.capacity_blocks,
        capacity_sweep_blocks=effective_capacity_sweep,
        block_size_tokens=block_size_tokens or base.block_size_tokens,
    )
    return config


def _parse_capacity_sweep(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    capacities = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not capacities:
        return ()
    if any(capacity <= 0 for capacity in capacities):
        raise ValueError("--capacity-sweep-blocks values must be positive")
    return capacities


def _evaluate_baselines(
    config: EvaluatorConfig,
    *,
    include_reporting: bool = False,
    splits: tuple[str, ...] = ("train", "validation", "probe"),
) -> dict[str, EvaluationResult]:
    baselines = REPORTING_BASELINES if include_reporting else BASELINES
    return BASELINE_SUITE_EVALUATOR.evaluate(config, baselines, splits=splits)


def _artifact_report_config() -> EvaluatorConfig:
    return load_evaluator_config()


def _baseline_report_headline(
    ranked: list[tuple[str, EvaluationResult]],
) -> str:
    """Compatibility wrapper for the extracted report renderer."""

    return baseline_reporting.baseline_report_headline(ranked)


def _baseline_report_command(
    *,
    quick: bool,
    capacity_sweep_blocks: tuple[int, ...],
    candidate_program: Path,
    config_file: str = _DEFAULT_CONFIG_FILE,
) -> str:
    parts = [
        ".venv/bin/python -m prefix_cache_evolve.problems.prefix_kv_cache.runner",
        "--baseline-report",
    ]
    if quick:
        parts.append("--quick")
    if capacity_sweep_blocks:
        parts.append(
            "--capacity-sweep-blocks " + ",".join(str(value) for value in capacity_sweep_blocks)
        )
    parts.append(f"--candidate-program {candidate_program}")
    parts.append(f"--config {config_file}")
    return " ".join(parts)


def _evaluate_candidate_program(
    config: EvaluatorConfig,
    candidate_path: Path,
    *,
    splits: tuple[str, ...] = ("train", "validation", "probe"),
) -> EvaluationResult:
    source = candidate_path.read_text(encoding="utf-8")
    return run_with_timeout(
        _evaluate_candidate_program_in_worker,
        config,
        candidate_path,
        splits,
        scoring_fn_complexity(
            source,
            form_aware=config.form_aware_complexity,
        ),
        timeout_seconds=config.timeout_s,
    )


def _evaluate_candidate_program_in_worker(
    config: EvaluatorConfig,
    candidate_path: Path,
    splits: tuple[str, ...],
    complexity: int,
) -> EvaluationResult:
    candidate_factory = load_candidate_factory(str(candidate_path))
    return PrefixKVCacheEvaluator(config, splits=splits)(
        candidate_factory,
        scoring_fn_complexity=complexity,
    )


def _evaluate_replay_candidate_program(
    config: EvaluatorConfig,
    candidate_path: Path,
    requests: tuple[WorkloadRequest, ...],
) -> EvaluationResult:
    source = candidate_path.read_text(encoding="utf-8")
    return run_with_timeout(
        _evaluate_replay_candidate_program_in_worker,
        config,
        candidate_path,
        requests,
        scoring_fn_complexity(
            source,
            form_aware=config.form_aware_complexity,
        ),
        timeout_seconds=config.timeout_s,
    )


def _evaluate_replay_candidate_program_in_worker(
    config: EvaluatorConfig,
    candidate_path: Path,
    requests: tuple[WorkloadRequest, ...],
    complexity: int,
) -> EvaluationResult:
    candidate_factory = load_candidate_factory(str(candidate_path))
    return PrefixKVCacheEvaluator(config, splits=("validation",)).evaluate_requests(
        candidate_factory,
        requests,
        scoring_fn_complexity=complexity,
    )


def _resolve_candidate_program(path: Path) -> Path:
    if path.is_dir():
        path = path / "best_program.py"
    if not path.exists():
        raise FileNotFoundError(f"candidate program {path} does not exist")
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _evaluation_result_summary(result: EvaluationResult) -> dict[str, Any]:
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


if __name__ == "__main__":
    main()
