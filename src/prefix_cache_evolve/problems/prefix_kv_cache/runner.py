"""Run prefix KV-cache baseline reports or Levi evolution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click

from prefix_cache_evolve.evaluators.baseline_suite import BASELINE_SUITE_EVALUATOR
from prefix_cache_evolve.evaluators.baselines import BASELINES, REPORTING_BASELINES
from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.evaluators.prefix_kv_cache import PrefixKVCacheEvaluator
from prefix_cache_evolve.evaluators.results import EvaluationResult
from prefix_cache_evolve.evaluators.verifier import (
    require_single_score_identity,
    require_single_verifier_version,
)
from prefix_cache_evolve.workflow.config import (
    ConfigLoader,
    ConfigProvider,
    MinimalConfigProvider,
    YamlConfigProvider,
)
from prefix_cache_evolve.workflow.program import ProgramSource
from prefix_cache_evolve.workflow.reporting import EvolutionReporter

from . import reporting as baseline_reporting
from .candidate_panels import (
    HIDDEN_PANEL,
    PROBE_PANEL,
    SELECTION_PANEL,
    VALIDATION_PANEL,
    CandidatePanelBuilder,
)
from .candidate_panels import (
    evaluate_candidate_program as _evaluate_candidate_program,
)
from .candidate_panels import (
    evaluate_replay_candidate_program as _evaluate_replay_candidate_program,
)
from .configuration import (
    DEFAULT_CONFIG_PATH,
    load_evaluator_config,
    prefix_kv_config_environment,
)
from .incumbents import build_current_incumbent as build_candidate
from .incumbents.registry import current_incumbent, incumbent_record
from .reproducibility import (
    build_workload_manifest,
    file_sha256,
    request_stream_sha256,
    stable_workload_manifest_payload,
)
from .specialist import (
    compose_eviction_specialist_source,
)
from .trace_replay import calibrate_anonymized_trace, load_anonymized_trace
from .utilities import (
    capacity_blocks_for_token_tiers as _capacity_blocks_for_token_tiers,
)
from .utilities import (
    elite_source as _elite_source,
)
from .utilities import (
    evaluation_result_summary as _evaluation_result_summary,
)
from .utilities import (
    format_int_tuple as _format_int_tuple,
)
from .utilities import (
    normalized_source as _normalized_source,
)
from .utilities import (
    parse_positive_int_csv,
    parse_unique_positive_int_csv,
)
from .utilities import (
    promotion_check as _promotion_check,
)
from .utilities import (
    promotion_tripwire_check as _promotion_tripwire_check,
)
from .utilities import (
    raw_selection_improvement as _raw_selection_improvement,
)
from .utilities import (
    score_non_regression as _score_non_regression,
)
from .utilities import (
    split_metric_non_regression as _split_metric_non_regression,
)
from .utilities import (
    surrogate_probe_tripwire_suite as _surrogate_probe_tripwire_suite,
)
from .utilities import (
    workload_metric_non_regression as _workload_metric_non_regression,
)
from .utilities import (
    write_agentic_surrogate_probe_gate_report as _write_agentic_surrogate_probe_gate_report,
)
from .utilities import (
    write_generated_mutation_report as _write_generated_mutation_report,
)
from .utilities import (
    write_json as _write_json,
)
from .utilities import (
    write_specialist_promotion_adjudication_report as _write_specialist_report,
)
from .utilities import (
    write_surrogate_probe_tripwire_report as _write_surrogate_probe_tripwire_report,
)

if TYPE_CHECKING:
    from prefix_cache_evolve.workflow.execution import LeviRunner, LeviRunResult
    from prefix_cache_evolve.workflow.workflow import EvolutionWorkflow

_DEFAULT_SEED_PATH = current_incumbent("production").source_path
_EVICTION_SPECIALIST_SEED_PATH = Path(__file__).parent / "seeds" / "eviction_specialist.py"
DEFAULT_SEED_SOURCE = ProgramSource(_DEFAULT_SEED_PATH.read_text(encoding="utf-8"))
_EVALUATOR_PATH = Path(__file__).parent / "evaluator.py"
_COMPACT_SEED_PATH = incumbent_record("historical_compact_20260607").source_path
_DEFAULT_CONFIG_FILE = str(DEFAULT_CONFIG_PATH)
_CONFIG_LOADER = ConfigLoader()
_DEFAULT_CAPACITY_SWEEP_BLOCKS = (24, 48)
_DEFAULT_BLOCK_SIZE_SWEEP_TOKENS = (8, 16, 32)
_BLOCK_SIZE_ROBUSTNESS_BASELINES = ("vllm_apc", "tinylfu_lru", "lru")
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

    from prefix_cache_evolve.workflow.execution import LeviRunner

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
            "RequestInfo request_id is opaque, request_type is normalized to "
            "'request', and prompt_tokens is empty. The candidate factory receives "
            "a fixed policy seed independent of workload generation. RequestInfo "
            "also exposes online recent_admission_pressure and recent_miss_rate. "
            "MultiTimescaleDecay, decay_vector, and "
            "threshold_excess are optional canonical primitives for bounded "
            "multi-timescale state and compact threshold gates. "
            "MultiTimescaleDecay.observe_vector applies distinct updates to "
            "different decay channels. Preserve or simplify canonical primitive "
            "state before replacing it with bespoke per-key decay dictionaries; "
            "canonical calls receive the bounded form-aware complexity subsidy. "
            "Explore recurrence-aware and regime-conditional control flow without "
            "hard-coding workload-family names or using scrubbed request fields. The "
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
    provider: ConfigProvider,
    *,
    program_source: ProgramSource = DEFAULT_SEED_SOURCE,
) -> EvolutionWorkflow[LeviRunResult]:
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
    model: str | None = None,
    primary_model: str | None = None,
    secondary_model: str | None = None,
    search_seed: int | None = None,
    api_base: str | None = None,
    api_key_env: str | None = None,
) -> object:
    """Run one Levi evolution session and optionally persist its artifacts."""
    evaluator_config = load_evaluator_config(Path(config_file))
    base_workflow_config = _CONFIG_LOADER.load(Path(config_file))
    default_seed_path = (
        _EVICTION_SPECIALIST_SEED_PATH
        if evaluator_config.candidate_policy_surface == "eviction_only"
        else _DEFAULT_SEED_PATH
    )
    effective_seed_program = seed_program or default_seed_path
    if quick:
        quick_model = (
            model
            or primary_model
            or secondary_model
            or base_workflow_config.mutation_model
            or base_workflow_config.paradigm_model
        )
        provider: ConfigProvider = MinimalConfigProvider(
            model=quick_model,
            search_seed=(
                search_seed if search_seed is not None else base_workflow_config.search_seed
            ),
            api_base=api_base or base_workflow_config.api_base,
            api_key_env=api_key_env or base_workflow_config.api_key_env,
        )
    else:
        provider = YamlConfigProvider(
            Path(config_file),
            _CONFIG_LOADER,
            model=model,
            primary_model=primary_model,
            secondary_model=secondary_model,
            search_seed=search_seed,
            api_base=api_base,
            api_key_env=api_key_env,
        )
    program_source = _load_seed_program_source(effective_seed_program)
    workflow = _build_workflow(provider, program_source=program_source)
    with prefix_kv_config_environment(Path(config_file), quick=quick):
        result = workflow.execute(iterations)
    if artifact_output is not None:
        artifact_dir = save_run_artifacts(
            result,
            artifact_output,
            iterations=iterations,
            config_label=provider.describe(),
            seed_label=str(effective_seed_program),
            seed_source=program_source.text(),
            report_config=evaluator_config,
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
    """Evaluate and print the candidate and registered baselines."""
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
        results = _candidate_panel_builder().add_candidate(
            config,
            candidate_path,
            results,
        )
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
    print(
        "verifier_version="
        + require_single_score_identity(
            results.values(),
            context="baseline console report",
        ).verifier_version
    )
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
    resolved_report_config = report_config or _artifact_report_config()
    identity = require_single_score_identity(
        (metrics, artifacts),
        context="evolution run artifacts",
    )
    if identity.verifier_version != resolved_report_config.verifier_version:
        raise ValueError("run artifact verifier version does not match the operative report config")
    workload_metrics = artifacts.get("workload_metrics") if isinstance(artifacts, dict) else None
    tripwire_thresholds = dict(resolved_report_config.surrogate_probe_tripwire_thresholds)
    tripwire_suite = _surrogate_probe_tripwire_suite(
        workload_metrics,
        thresholds=tripwire_thresholds,
    )
    agentic_gate = tripwire_suite["channels"]["agentic_branching"]
    config_snapshot_name = None
    if config_snapshot is not None and config_snapshot.is_file():
        config_snapshot_name = "config_snapshot.yaml"
        shutil.copyfile(config_snapshot, run_dir / config_snapshot_name)
    summary = {
        "verifier_version": identity.verifier_version,
        "evaluation_context_sha256": identity.evaluation_context_sha256,
        "panel_sha256": identity.panel_sha256,
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
        "repository": _repository_state(),
        "agentic_surrogate_probe_gate": {
            key: agentic_gate[key]
            for key in (
                "status",
                "flagged",
                "flag_reason",
                "checked_metric_count",
                "failed_metric_count",
                "failed_metrics",
                "missing_metrics",
                "max_threshold_ratio",
            )
        },
        "surrogate_probe_tripwires": {
            key: tripwire_suite[key]
            for key in (
                "status",
                "flagged",
                "flagged_channels",
                "passed_channels",
                "max_threshold_ratio",
            )
        },
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "artifacts.json", artifacts)
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "agentic_surrogate_probe_gate.json", agentic_gate)
    _write_agentic_surrogate_probe_gate_report(
        run_dir / "agentic_surrogate_probe_gate.md",
        agentic_gate,
    )
    _write_json(run_dir / "surrogate_probe_tripwires.json", tripwire_suite)
    _write_surrogate_probe_tripwire_report(
        run_dir / "surrogate_probe_tripwires.md",
        tripwire_suite,
    )
    workload_manifest = build_workload_manifest(resolved_report_config)
    _write_json(run_dir / "workload_manifest.json", workload_manifest)
    summary["workload_manifest"] = {
        "path": "workload_manifest.json",
        "panel_sha256": workload_manifest["panel_sha256"],
        "evaluation_context_sha256": workload_manifest["evaluation_context_sha256"],
        "verifier_version": workload_manifest["verifier_version"],
    }
    _write_json(run_dir / "run_summary.json", summary)

    _persist_paradigm_candidates(run_dir, metadata=metadata)
    _persist_best_generated_mutation(
        run_dir,
        metadata=metadata,
        seed_source=seed_source,
        config=resolved_report_config,
    )
    promotion_adjudication = _persist_specialist_promotion_adjudication(
        run_dir,
        config=resolved_report_config,
    )
    if promotion_adjudication is not None:
        summary["promotion_adjudication"] = {
            key: promotion_adjudication[key]
            for key in ("status", "eligible", "promotion_complexity_limit")
        }
        _write_json(run_dir / "run_summary.json", summary)

    try:
        config = report_config or _artifact_report_config()
        candidate_path = run_dir / "best_program.py"
        report_results = _candidate_panel_builder().build_comparison(
            config,
            candidate_path,
            lambda: _evaluate_baselines(config, include_reporting=True),
        )
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


def _repository_state() -> dict[str, Any]:
    """Return the current Git revision and whether tracked files differ."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


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
        snapshot_identity = require_single_score_identity(
            (strongest,),
            context="best generated mutation snapshot",
        )
        if snapshot_identity.verifier_version != config.verifier_version:
            raise ValueError("generated mutation snapshot verifier version does not match config")
        decomposition = {
            "schema": "prefix-kv-cache-generated-mutation-decomposition-v1",
            "verifier_version": config.verifier_version,
            "snapshot": str(snapshot_path),
            "generated_program_id": strongest.get("program_id"),
            "snapshot_primary_score": strongest.get("primary_score"),
            "snapshot_primary_score_verifier_version": (snapshot_identity.verifier_version),
            "snapshot_primary_score_evaluation_context_sha256": (
                snapshot_identity.evaluation_context_sha256
            ),
            "snapshot_primary_score_panel_sha256": snapshot_identity.panel_sha256,
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


def _persist_specialist_promotion_adjudication(
    run_dir: Path,
    *,
    config: EvaluatorConfig,
) -> dict[str, Any] | None:
    """Re-evaluate a specialist winner as a complete policy before promotion."""
    if config.fixed_admission_policy is None:
        return None
    promotion_limit = (
        config.promotion_max_candidate_complexity
        if config.promotion_max_candidate_complexity is not None
        else config.max_candidate_complexity
    )
    promotion_config = config.with_updates(
        fixed_admission_policy=None,
        candidate_policy_surface="full",
        search_score_mode="combined",
        max_candidate_complexity=None,
        promotion_max_candidate_complexity=None,
    )
    try:
        candidate_path = run_dir / "best_program.py"
        if config.candidate_policy_surface == "eviction_only":
            candidate_path = run_dir / "promotion_candidate.py"
            candidate_path.write_text(
                compose_eviction_specialist_source(
                    (run_dir / "best_program.py").read_text(encoding="utf-8"),
                    _DEFAULT_SEED_PATH.read_text(encoding="utf-8"),
                ),
                encoding="utf-8",
            )
        candidate = _candidate_panel_decomposition(
            promotion_config,
            candidate_path,
        )
        incumbent = _candidate_panel_decomposition(
            promotion_config,
            _DEFAULT_SEED_PATH,
        )
        checks = {
            "complexity_within_promotion_limit": _promotion_check(
                candidate["effective_complexity"] <= promotion_limit
                if promotion_limit is not None
                else True,
                candidate["effective_complexity"],
                promotion_limit,
            ),
            "selection_non_regression": _score_non_regression(
                candidate,
                incumbent,
                panel="selection",
            ),
            "raw_selection_improvement": _raw_selection_improvement(
                candidate,
                incumbent,
            ),
            "validation_avoidable_eviction_non_regression": _split_metric_non_regression(
                candidate,
                incumbent,
                split="validation",
                metric="avoidable_eviction_rate",
                lower_is_better=True,
            ),
            "validation_short_reuse_after_eviction_non_regression": (
                _split_metric_non_regression(
                    candidate,
                    incumbent,
                    split="validation",
                    metric="short_reuse_after_eviction_missed_token_rate",
                    lower_is_better=True,
                )
            ),
            "aggregate_probe_non_regression": _score_non_regression(
                candidate,
                incumbent,
                panel="probe",
            ),
            "agent_trace_branching_non_regression": _workload_metric_non_regression(
                candidate,
                incumbent,
                panel="probe",
                workload="probe/agent_trace_branching",
                metric="token_hit_rate",
            ),
            "cyclic_working_set_pressure_non_regression": (
                _workload_metric_non_regression(
                    candidate,
                    incumbent,
                    panel="probe",
                    workload="probe/cyclic_working_set_pressure",
                    metric="token_hit_rate",
                )
            ),
            "hidden_non_regression": _score_non_regression(
                candidate,
                incumbent,
                panel="hidden",
            ),
            "surrogate_probe_tripwires": _promotion_tripwire_check(
                candidate,
                thresholds=dict(config.surrogate_probe_tripwire_thresholds),
            ),
        }
        eligible = all(check["passed"] for check in checks.values())
        payload = {
            "schema": "prefix-kv-cache-specialist-promotion-adjudication-v1",
            "verifier_version": config.verifier_version,
            "status": "pass" if eligible else "fail",
            "eligible": eligible,
            "specialist_fixed_admission_policy": config.fixed_admission_policy,
            "promotion_complexity_limit": promotion_limit,
            "candidate": candidate,
            "incumbent": incumbent,
            "checks": checks,
            "interpretation": (
                "The specialist winner is eligible for manual promotion as a complete policy."
                if eligible
                else (
                    "The specialist winner remains exploration-only and must not replace "
                    "the incumbent."
                )
            ),
        }
        _write_json(run_dir / "promotion_adjudication.json", payload)
        _write_specialist_report(
            run_dir / "promotion_adjudication.md",
            payload,
        )
        return payload
    except Exception as exc:
        payload = {
            "schema": "prefix-kv-cache-specialist-promotion-adjudication-v1",
            "verifier_version": config.verifier_version,
            "status": "fail",
            "eligible": False,
            "specialist_fixed_admission_policy": config.fixed_admission_policy,
            "promotion_complexity_limit": promotion_limit,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "interpretation": (
                "Promotion adjudication failed closed; the specialist winner must not "
                "replace the incumbent."
            ),
        }
        _write_json(run_dir / "promotion_adjudication.json", payload)
        _write_specialist_report(
            run_dir / "promotion_adjudication.md",
            payload,
        )
        return payload


def _candidate_panel_decomposition(
    config: EvaluatorConfig,
    candidate_path: Path,
) -> dict[str, Any]:
    """Evaluate one candidate on selection, probe, and hidden panels."""
    return _candidate_panel_builder().build_decomposition(config, candidate_path)


def hidden_report(
    *,
    quick: bool = False,
    capacity_blocks: int | None = None,
    capacity_sweep_blocks: tuple[int, ...] = (),
    block_size_tokens: int | None = None,
    candidate_program: Path | None = None,
    config_file: str = _DEFAULT_CONFIG_FILE,
) -> None:
    """Evaluate a candidate and baselines on the hidden split."""
    config = _config_from_args(
        quick=quick,
        capacity_blocks=capacity_blocks,
        capacity_sweep_blocks=capacity_sweep_blocks,
        block_size_tokens=block_size_tokens,
        config_file=config_file,
    )
    if candidate_program is None:
        print("default_candidate:")
        champion = PrefixKVCacheEvaluator(config, splits=HIDDEN_PANEL.splits)(build_candidate)
    else:
        candidate_path = _resolve_candidate_program(candidate_program)
        print(f"candidate={candidate_path}")
        champion = _candidate_panel_builder().evaluate(
            config,
            candidate_path,
            panel=HIDDEN_PANEL,
        )
    print(f"verifier_version={champion.verifier_version}")
    print(f"  combined_score={champion.combined_score:.3f}")
    results = _evaluate_baselines(
        config,
        include_reporting=True,
        splits=HIDDEN_PANEL.splits,
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
    results = _candidate_panel_builder().build_comparison(
        config,
        candidate_path,
        lambda: _evaluate_baselines(
            config,
            include_reporting=True,
            splits=PROBE_PANEL.splits,
        ),
        panel=PROBE_PANEL,
    )
    identity = require_single_score_identity(
        results.values(),
        context="structure probe report",
    )
    payload = {
        "schema": "prefix-kv-cache-structure-probe-v1",
        "verifier_version": identity.verifier_version,
        "evaluation_context_sha256": identity.evaluation_context_sha256,
        "panel_sha256": identity.panel_sha256,
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
    calibration = {
        **calibrate_anonymized_trace(
            trace_path,
            arrival_bucket_ms=arrival_bucket_ms,
            request_limit=request_limit,
        ),
        "trace_path": str(trace_path),
        "trace_sha256": file_sha256(trace_path),
        "request_limit": request_limit,
    }
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
    identity = require_single_score_identity(
        results.values(),
        context="trace replay report",
    )
    payload = {
        "schema": "prefix-kv-cache-trace-replay-v1",
        "verifier_version": identity.verifier_version,
        "evaluation_context_sha256": identity.evaluation_context_sha256,
        "panel_sha256": identity.panel_sha256,
        "trace_path": str(trace_path),
        "trace_sha256": file_sha256(trace_path),
        "request_stream_sha256": request_stream_sha256(requests),
        "request_count": len(requests),
        "request_limit": request_limit,
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


def write_workload_manifest_report(
    output_path: Path,
    *,
    reference_path: Path | None = None,
    quick: bool = False,
    capacity_blocks: int | None = None,
    capacity_sweep_blocks: tuple[int, ...] = (),
    block_size_tokens: int | None = None,
    config_file: str = _DEFAULT_CONFIG_FILE,
) -> dict[str, object]:
    """Write fingerprints and summaries for every generated synthetic stream."""
    config = _config_from_args(
        quick=quick,
        capacity_blocks=capacity_blocks,
        capacity_sweep_blocks=capacity_sweep_blocks,
        block_size_tokens=block_size_tokens,
        config_file=config_file,
    )
    payload = build_workload_manifest(config)
    _write_json(output_path, payload)
    print(f"workload_manifest={output_path}")
    print(f"panel_sha256={payload['panel_sha256']}")
    if reference_path is not None:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if stable_workload_manifest_payload(payload) != stable_workload_manifest_payload(reference):
            raise click.ClickException(
                f"workload manifest differs from stable reference fields in {reference_path}"
            )
        print(f"workload_manifest_reference=verified:{reference_path}")
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
    results = _candidate_panel_builder().build_comparison(
        config,
        candidate_path,
        lambda: _evaluate_baselines(config),
    )
    verifier_version = require_single_score_identity(
        results.values(),
        context="score-weight sensitivity report",
    ).verifier_version
    rows = _score_weight_sensitivity_rows(results, config)
    lines = [
        "# Prefix KV-Cache Score-Weight Sensitivity",
        "",
        f"Candidate: `{candidate_path}`",
        "",
        f"Verifier: `{verifier_version}`",
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


def write_block_size_robustness_report(
    output_path: Path,
    *,
    candidate_program: Path | None = None,
    quick: bool = False,
    config_file: str = _DEFAULT_CONFIG_FILE,
    block_sizes: tuple[int, ...] = _DEFAULT_BLOCK_SIZE_SWEEP_TOKENS,
) -> Path:
    """Compare the incumbent and credibility baselines at fixed token capacities."""
    if not block_sizes:
        raise ValueError("at least one block size is required")
    base = _config_from_args(
        quick=quick,
        capacity_blocks=None,
        block_size_tokens=None,
        config_file=config_file,
    )
    candidate_path = _resolve_candidate_program(candidate_program or _DEFAULT_SEED_PATH)
    capacity_tokens = base.effective_capacity_tokens()
    rows = []
    panel_builder = _candidate_panel_builder()
    for block_size_tokens in block_sizes:
        capacity_blocks = _capacity_blocks_for_token_tiers(
            capacity_tokens,
            block_size_tokens=block_size_tokens,
        )
        config = base.with_updates(
            capacity_blocks=capacity_blocks[0],
            capacity_sweep_blocks=capacity_blocks,
            block_size_tokens=block_size_tokens,
        )
        results = panel_builder.build_comparison(
            config,
            candidate_path,
            lambda: BASELINE_SUITE_EVALUATOR.evaluate(
                config,
                {name: BASELINES[name] for name in _BLOCK_SIZE_ROBUSTNESS_BASELINES},
                splits=VALIDATION_PANEL.splits,
            ),
            panel=VALIDATION_PANEL,
        )
        for name, result in results.items():
            validation = result.split_metrics["validation"]
            complexity_cost = float(result.score_breakdown.get("complexity_cost", 0.0))
            rows.append(
                {
                    "verifier_version": result.verifier_version,
                    "evaluation_context_sha256": result.evaluation_context_sha256,
                    "panel_sha256": result.panel_sha256,
                    "block_size_tokens": block_size_tokens,
                    "capacity_blocks": capacity_blocks,
                    "capacity_tokens": capacity_tokens,
                    "policy": name,
                    "combined_score": result.combined_score,
                    "raw_score_before_complexity": result.combined_score + complexity_cost,
                    "complexity_cost": complexity_cost,
                    "token_hit_rate": float(validation["token_hit_rate"]),
                    "block_hit_rate": float(validation["block_hit_rate"]),
                    "worst_quarter_token_hit_rate": float(
                        validation["worst_quarter_token_hit_rate"]
                    ),
                    "policy_underfill_rate": float(validation["policy_underfill_rate"]),
                    "cache_churn_per_1k": float(validation["cache_churn_per_1k"]),
                }
            )

    verifier_version = require_single_verifier_version(
        rows,
        context="block-size robustness report",
    )
    identities = {
        block_size: require_single_score_identity(
            (row for row in rows if row["block_size_tokens"] == block_size),
            context=f"block-size robustness {block_size}-token comparison",
        )
        for block_size in block_sizes
    }
    lines = [
        "# Prefix KV-Cache Block-Size Robustness",
        "",
        f"Candidate: `{candidate_path}`",
        "",
        f"Verifier: `{verifier_version}`",
        "",
        "Evaluation contexts: "
        + ", ".join(
            f"`{block_size}={identity.evaluation_context_sha256}`"
            for block_size, identity in identities.items()
        ),
        "",
        "Panels: "
        + ", ".join(
            f"`{block_size}={identity.panel_sha256}`" for block_size, identity in identities.items()
        ),
        "",
        (
            "Each block size replays identical synthetic token streams and preserves "
            "the same cache-capacity tiers in tokens. The production-oriented primary "
            f"setting is `{base.block_size_tokens}` tokens per block."
        ),
        "",
        (
            f"Canonical workload token granularity: "
            f"`{base.effective_workload_token_granularity()}`. "
            f"Capacity tiers: `{capacity_tokens}` tokens."
        ),
        "",
    ]
    if quick:
        lines.extend([f"> **{_QUICK_REPORT_WARNING}**", ""])
    lines.extend(
        [
            "## Candidate Summary",
            "",
            "| Block size | Candidate score | Raw before complexity | Complexity cost | "
            "Rank | Best policy | Gap to best | Validation token hit | Churn per 1k |",
            "|---:|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for block_size_tokens in block_sizes:
        block_rows = [row for row in rows if row["block_size_tokens"] == block_size_tokens]
        ranked = sorted(
            block_rows,
            key=lambda row: cast(float, row["combined_score"]),
            reverse=True,
        )
        candidate = next(row for row in block_rows if row["policy"] == "candidate")
        candidate_rank = next(
            rank for rank, row in enumerate(ranked, start=1) if row["policy"] == "candidate"
        )
        best = ranked[0]
        candidate_score = cast(float, candidate["combined_score"])
        best_score = cast(float, best["combined_score"])
        lines.append(
            f"| {block_size_tokens} | {candidate_score:.3f} | "
            f"{candidate['raw_score_before_complexity']:.3f} | "
            f"{candidate['complexity_cost']:.3f} | "
            f"{candidate_rank} / {len(ranked)} | `{best['policy']}` | "
            f"{candidate_score - best_score:.3f} | "
            f"{candidate['token_hit_rate']:.3f} | "
            f"{candidate['cache_churn_per_1k']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Detailed Results",
            "",
            "| Block size | Capacity blocks | Capacity tokens | Policy | Score | "
            "Raw before complexity | Complexity cost | Validation token hit | "
            "Validation block hit | Worst-quarter hit | Policy underfill | Churn per 1k |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['block_size_tokens']} | "
            f"{_format_int_tuple(cast(tuple[int, ...], row['capacity_blocks']))} | "
            f"{_format_int_tuple(cast(tuple[int, ...], row['capacity_tokens']))} | "
            f"`{row['policy']}` | {row['combined_score']:.3f} | "
            f"{row['raw_score_before_complexity']:.3f} | "
            f"{row['complexity_cost']:.3f} | "
            f"{row['token_hit_rate']:.3f} | {row['block_hit_rate']:.3f} | "
            f"{row['worst_quarter_token_hit_rate']:.3f} | "
            f"{row['policy_underfill_rate']:.3f} | "
            f"{row['cache_churn_per_1k']:.1f} |"
        )
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"block_size_robustness={output_path}")
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
            variant = config.with_updates(**{weight: base_value * factor})
            rescored_results = {}
            for name, result in results.items():
                complexity = int(result.candidate_metadata.get("scoring_fn_complexity", 0))
                rescored_results[name] = PrefixKVCacheEvaluator(variant).rescore_trials(
                    result.trials,
                    scoring_fn_complexity=complexity,
                )
            identity = require_single_score_identity(
                rescored_results.values(),
                context=f"score-weight sensitivity {weight} x {factor}",
            )
            rescored = {name: result.combined_score for name, result in rescored_results.items()}
            ranking = sorted(rescored, key=lambda name: rescored[name], reverse=True)
            rows.append(
                {
                    "verifier_version": identity.verifier_version,
                    "evaluation_context_sha256": identity.evaluation_context_sha256,
                    "panel_sha256": identity.panel_sha256,
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


@click.command()
@click.option("--iterations", type=click.IntRange(min=1), default=25, show_default=True)
@click.option(
    "--quick",
    is_flag=True,
    help="Use a smoke-only single-seed slice; do not use it for ranking decisions.",
)
@click.option(
    "--workload-preset",
    type=click.Choice(("default", "small")),
    default="default",
    show_default=True,
)
@click.option("--capacity-blocks", type=click.IntRange(min=1))
@click.option(
    "--capacity-sweep-blocks",
    default="",
    help="Comma-separated capacities to evaluate, for example 48,96.",
)
@click.option("--block-size-tokens", type=click.IntRange(min=1))
@click.option("--baseline-report", is_flag=True)
@click.option(
    "--candidate-program",
    type=click.Path(path_type=Path, exists=True, readable=True),
    help="Candidate .py file or run directory to compare in --baseline-report.",
)
@click.option("--hidden-report", is_flag=True)
@click.option(
    "--probe-report",
    is_flag=True,
    help="Evaluate the quarantined recurrence/structure-generalization probe.",
)
@click.option(
    "--probe-output",
    type=click.Path(path_type=Path),
    default=Path("artifacts/prefix_kv_cache_structure_probe.json"),
    show_default=True,
    help="JSON output for --probe-report.",
)
@click.option(
    "--plot-report",
    is_flag=True,
    help="Write SVG baseline plots without launching Levi.",
)
@click.option(
    "--plot-output",
    type=click.Path(path_type=Path),
    default=Path("artifacts/prefix_kv_cache_plots"),
    show_default=True,
    help="Directory for --plot-report SVG files.",
)
@click.option(
    "--artifact-output",
    type=click.Path(path_type=Path),
    default=Path("artifacts/prefix_kv_cache_runs"),
    show_default=True,
    help="Directory for saved evolution run artifacts.",
)
@click.option(
    "--seed-program",
    type=click.Path(path_type=Path, exists=True, readable=True),
    help=(
        "Candidate .py file or saved run directory to use as the evolution seed; "
        "defaults to the current production incumbent."
    ),
)
@click.option(
    "--no-save-artifacts",
    is_flag=True,
    help="Do not save best_program.py and run metadata after evolution.",
)
@click.option(
    "--config",
    type=click.Path(path_type=str, exists=True, dir_okay=False, readable=True),
    default=_DEFAULT_CONFIG_FILE,
    show_default=True,
    help="Path to the Levi YAML config file.",
)
@click.option(
    "--model",
    help=(
        "Use one LiteLLM model for all search calls, for example "
        "anthropic/<model>, gemini/<model>, ollama/<model>, or openai/<model>."
    ),
)
@click.option(
    "--primary-model",
    help="Override the mutation model with a provider-qualified LiteLLM model.",
)
@click.option(
    "--secondary-model",
    help="Override the paradigm-shift model with a provider-qualified LiteLLM model.",
)
@click.option(
    "--search-seed",
    type=click.IntRange(min=0),
    help="Override search.seed for Levi selection and supported model requests.",
)
@click.option(
    "--api-base",
    help="Override the model API base URL, useful for self-hosted OpenAI-compatible APIs.",
)
@click.option(
    "--api-key-env",
    help="Name of the environment variable containing the model API key.",
)
@click.option(
    "--show-config",
    is_flag=True,
    help="Print resolved evaluator, model, and seed settings without calling a model.",
)
@click.option(
    "--calibrate-trace",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    help="Summarize an anonymized metadata-only JSONL production trace.",
)
@click.option(
    "--replay-trace",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    help="Replay an anonymized metadata-only JSONL production trace.",
)
@click.option(
    "--trace-output",
    type=click.Path(path_type=Path),
    default=Path("artifacts/prefix_kv_cache_trace_report.json"),
    show_default=True,
    help="Output JSON for --calibrate-trace or --replay-trace.",
)
@click.option(
    "--trace-arrival-bucket-ms",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="Convert trace timestamps to simulator arrival steps using this bucket.",
)
@click.option(
    "--trace-request-limit",
    type=click.IntRange(min=1),
    help="Optional prefix request count for trace calibration or replay.",
)
@click.option(
    "--workload-manifest",
    is_flag=True,
    help="Write deterministic fingerprints and summaries for all synthetic streams.",
)
@click.option(
    "--workload-manifest-output",
    type=click.Path(path_type=Path),
    default=Path("artifacts/prefix_kv_cache_workload_manifest.json"),
    show_default=True,
    help="JSON output for --workload-manifest.",
)
@click.option(
    "--workload-manifest-reference",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help=(
        "Verify panel SHA, evaluation settings, and ordered streams against a committed "
        "manifest while ignoring environment metadata."
    ),
)
@click.option(
    "--sensitivity-report",
    is_flag=True,
    help="Rescore fixed full-panel trials under one-at-a-time weight changes.",
)
@click.option(
    "--sensitivity-output",
    type=click.Path(path_type=Path),
    default=Path("artifacts/prefix_kv_cache_weight_sensitivity.md"),
    show_default=True,
    help="Markdown output for --sensitivity-report.",
)
@click.option(
    "--block-size-report",
    is_flag=True,
    help="Compare block sizes over identical traffic and fixed token-capacity tiers.",
)
@click.option(
    "--block-size-sweep",
    default="8,16,32",
    show_default=True,
    help="Comma-separated cache block sizes for --block-size-report.",
)
@click.option(
    "--block-size-output",
    type=click.Path(path_type=Path),
    default=Path("artifacts/prefix_kv_cache_block_size_robustness.md"),
    show_default=True,
    help="Markdown output for --block-size-report.",
)
def main(**kwargs: Any) -> None:
    """Run reports, trace tools, or the Levi evolution workflow."""
    from .runner_commands import RunnerOptions, dispatch

    dispatch(RunnerOptions.from_mapping(kwargs))


def _show_resolved_config(
    *,
    iterations: int,
    config_file: str,
    quick: bool,
    model: str | None,
    primary_model: str | None,
    secondary_model: str | None,
    search_seed: int | None,
    api_base: str | None,
    api_key_env: str | None,
) -> None:
    """Print the effective workflow and evaluator configuration."""
    evaluator = load_evaluator_config(Path(config_file))
    base_workflow = _CONFIG_LOADER.load(Path(config_file))
    if quick:
        provider: ConfigProvider = MinimalConfigProvider(
            model=(
                model
                or primary_model
                or secondary_model
                or base_workflow.mutation_model
                or base_workflow.paradigm_model
            ),
            search_seed=search_seed if search_seed is not None else base_workflow.search_seed,
            api_base=api_base or base_workflow.api_base,
            api_key_env=api_key_env or base_workflow.api_key_env,
        )
    else:
        provider = YamlConfigProvider(
            Path(config_file),
            _CONFIG_LOADER,
            model=model,
            primary_model=primary_model,
            secondary_model=secondary_model,
            search_seed=search_seed,
            api_base=api_base,
            api_key_env=api_key_env,
        )
    workflow = provider.load(iterations)
    payload = {
        "config": str(Path(config_file)),
        "quick": quick,
        "iterations": iterations,
        "models": {
            "model": workflow.model,
            "mutation_model": workflow.mutation_model,
            "paradigm_model": workflow.paradigm_model,
        },
        "search_seed": workflow.search_seed,
        "api_base": workflow.api_base,
        "api_key_env": workflow.api_key_env,
        "api_key_env_set": bool(workflow.api_key_env and os.environ.get(workflow.api_key_env)),
        "search_reproducibility": {
            "python_random_seeded": True,
            "numpy_random_seeded": True,
            "model_request_seeds": True,
            "bit_exact_remote_search_guaranteed": False,
        },
        "pipeline": workflow.pipeline,
        "evaluator": {
            "workload_seeds": list(evaluator.seeds),
            "policy_seed": evaluator.policy_seed,
            "request_count": evaluator.request_count,
            "block_size_tokens": evaluator.block_size_tokens,
            "capacity_blocks": list(evaluator.effective_capacity_blocks()),
            "workload_token_granularity": evaluator.workload_token_granularity,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


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
    config = base.with_updates(
        request_count=36 if quick else base.request_count,
        seeds=(3,) if quick else base.seeds,
        family_request_multipliers={} if quick else base.family_request_multipliers,
        capacity_blocks=capacity_blocks or base.capacity_blocks,
        capacity_sweep_blocks=effective_capacity_sweep,
        block_size_tokens=block_size_tokens or base.block_size_tokens,
    )
    return config


def _parse_capacity_sweep(value: str) -> tuple[int, ...]:
    return parse_positive_int_csv(value, option_name="--capacity-sweep-blocks")


def _parse_block_size_sweep(value: str) -> tuple[int, ...]:
    block_sizes = parse_unique_positive_int_csv(value, option_name="--block-size-sweep")
    if not block_sizes:
        raise ValueError("--block-size-sweep values must be positive")
    return block_sizes


def _evaluate_baselines(
    config: EvaluatorConfig,
    *,
    include_reporting: bool = False,
    splits: tuple[str, ...] = SELECTION_PANEL.splits,
) -> dict[str, EvaluationResult]:
    baselines = REPORTING_BASELINES if include_reporting else BASELINES
    return BASELINE_SUITE_EVALUATOR.evaluate(config, baselines, splits=splits)


def _candidate_panel_builder() -> CandidatePanelBuilder:
    """Build panel orchestration from runner-level evaluation dependencies."""
    return CandidatePanelBuilder(
        evaluate_program=_evaluate_candidate_program,
        summarize_result=_evaluation_result_summary,
    )


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


def _resolve_candidate_program(path: Path) -> Path:
    if path.is_dir():
        path = path / "best_program.py"
    if not path.exists():
        raise FileNotFoundError(f"candidate program {path} does not exist")
    return path


if __name__ == "__main__":
    main()
