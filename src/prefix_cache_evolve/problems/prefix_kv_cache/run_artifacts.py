"""Persist evolution inputs, archived alternatives, and post-search reports."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml

from prefix_cache_evolve.evaluators.baseline_suite import select_baselines as _selected_baselines
from prefix_cache_evolve.evaluators.baselines import REPORTING_BASELINES
from prefix_cache_evolve.evaluators.complexity import scoring_fn_complexity
from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.evaluators.results import EvaluationResult
from prefix_cache_evolve.evaluators.verifier import require_single_score_identity
from prefix_cache_evolve.workflow.config import WorkflowFileConfig, load_yaml_document

from .candidate_panels import CandidatePanelBuilder
from .configuration import DEFAULT_CONFIG_PATH, load_evaluator_config
from .reporting import baseline_report_command as _baseline_report_command
from .reporting import write_baseline_comparison_report
from .reproducibility import build_workload_manifest, file_sha256
from .specialist import compose_eviction_specialist_source
from .utilities import elite_source as _elite_source
from .utilities import normalized_source as _normalized_source
from .utilities import promotion_check as _promotion_check
from .utilities import promotion_tripwire_check as _promotion_tripwire_check
from .utilities import raw_selection_improvement as _raw_selection_improvement
from .utilities import score_non_regression as _score_non_regression
from .utilities import split_metric_non_regression as _split_metric_non_regression
from .utilities import surrogate_probe_tripwire_suite as _surrogate_probe_tripwire_suite
from .utilities import workload_metric_non_regression as _workload_metric_non_regression
from .utilities import (
    write_agentic_surrogate_probe_gate_report as _write_agentic_surrogate_probe_gate_report,
)
from .utilities import write_generated_mutation_report as _write_generated_mutation_report
from .utilities import write_json as _write_json
from .utilities import write_specialist_promotion_adjudication_report as _write_specialist_report
from .utilities import (
    write_surrogate_probe_tripwire_report as _write_surrogate_probe_tripwire_report,
)

_DEFAULT_CONFIG_FILE = str(DEFAULT_CONFIG_PATH)


def _artifact_manifest_splits(config: EvaluatorConfig) -> tuple[str, ...]:
    """Keep trace artifacts scoped to search while preserving synthetic manifests."""
    return (
        ("train", "validation", "probe")
        if config.trace_workloads
        else ("train", "validation", "probe", "hidden")
    )


def _snapshot_trace_inputs(run_dir: Path, config: EvaluatorConfig, source_config: Path) -> Path:
    """Archive search inputs while leaving hidden data in its original panel."""
    document = WorkflowFileConfig.model_validate(load_yaml_document(source_config)).model_dump(
        exclude_none=True, exclude_unset=True
    )
    shutil.copyfile(source_config, run_dir / "source_config.yaml")
    trace_dir = run_dir / "traces"
    trace_dir.mkdir()
    traces = []
    for trace in config.trace_workloads:
        if trace.split == "hidden":
            # Preserve the declared holdout pin without opening its data. Final
            # hidden evaluation uses the retained original panel explicitly.
            traces.append(trace)
            continue
        target = trace_dir / f"{trace.split}-{trace.family}.jsonl"
        shutil.copyfile(trace.path, target)
        if file_sha256(target) != trace.sha256:
            raise ValueError(f"{trace.path}: trace SHA-256 changed before artifact snapshot")
        traces.append(trace.model_copy(update={"path": str(target.relative_to(run_dir))}))
    snapshot_config = config.with_updates(trace_workloads=traces)
    document.setdefault("problem", {})["settings"] = snapshot_config.model_dump(mode="json")
    snapshot = run_dir / "config_snapshot.yaml"
    snapshot.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return snapshot


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


def _persist_behavior_size_frontier(
    run_dir: Path,
    *,
    metadata: dict[str, Any],
    config: EvaluatorConfig,
) -> None:
    """Retain nondominated archived sources for a later simplification stage.

    This uses only successful visible-search measurements already in the Levi
    archive. It does not evaluate candidates, open holdouts, or claim promotion.
    """
    snapshot_path = Path(str(metadata.get("levi_snapshot_path", "")))
    if not snapshot_path.is_file():
        return
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        rows = []
        for elite in snapshot.get("elites", []):
            scores = elite.get("scores", {})
            raw = scores.get("selection_raw_score_before_complexity")
            source = _elite_source(elite)
            if not scores.get("success") or scores.get("invalid_fraction", 1.0) or not source:
                continue
            if not isinstance(raw, (int, float)) or not math.isfinite(raw):
                continue
            rows.append(
                {
                    "elite": elite,
                    "source": source,
                    "raw_before_complexity": raw,
                    "effective_ast_nodes": scoring_fn_complexity(
                        source, form_aware=config.form_aware_complexity
                    ),
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                }
            )
        if not rows:
            return
        identity = require_single_score_identity(
            (row["elite"] for row in rows), context="behavior/source-size archive"
        )
        summary_path = run_dir / "run_summary.json"
        if summary_path.is_file():
            require_single_score_identity(
                (identity, json.loads(summary_path.read_text(encoding="utf-8"))),
                context="archive and saved run",
            )
        if identity.verifier_version != config.verifier_version:
            raise ValueError("archive verifier differs from the run config")
        frontier = [
            row
            for row in rows
            if not any(
                other["raw_before_complexity"] >= row["raw_before_complexity"]
                and other["effective_ast_nodes"] <= row["effective_ast_nodes"]
                and (
                    other["raw_before_complexity"] > row["raw_before_complexity"]
                    or other["effective_ast_nodes"] < row["effective_ast_nodes"]
                )
                for other in rows
            )
        ]
        directory = run_dir / "behavior_size_frontier"
        directory.mkdir(exist_ok=True)
        for row in frontier:
            (directory / f"{row['source_sha256']}.py").write_text(row["source"], encoding="utf-8")
        _write_json(
            directory / "manifest.json",
            {
                "schema": "prefix-kv-cache-behavior-size-frontier-v1",
                "evaluation_context_sha256": identity.evaluation_context_sha256,
                "panel_sha256": identity.panel_sha256,
                "verifier_version": identity.verifier_version,
                "candidates": [
                    {key: value for key, value in row.items() if key not in {"source", "elite"}}
                    for row in frontier
                ],
            },
        )
    except (ValueError, OSError, TypeError) as exc:
        _write_json(
            run_dir / "behavior_size_frontier_error.json",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )


class ArtifactBaselineEvaluator(Protocol):
    """The baseline comparison capability needed when saving a run."""

    def __call__(
        self, config: EvaluatorConfig, *, baseline_names: tuple[str, ...]
    ) -> dict[str, EvaluationResult]:
        """Evaluate the requested reporting baselines."""
        ...


@dataclass(frozen=True)
class RunArtifactWriter:
    """Persist run evidence using explicitly supplied evaluation dependencies."""

    panels: CandidatePanelBuilder
    evaluate_baselines: ArtifactBaselineEvaluator
    incumbent_path: Path

    def save(
        self,
        result: object,
        output_root: Path,
        *,
        iterations: int,
        config_label: str,
        seed_label: str | None = None,
        seed_source: str | None = None,
        report_config: EvaluatorConfig,
        report_config_file: str = _DEFAULT_CONFIG_FILE,
        config_snapshot: Path | None = None,
        timestamp: datetime | None = None,
        baseline_names: tuple[str, ...] = (),
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
        if seed_source is not None:
            (run_dir / "seed_program.py").write_text(seed_source, encoding="utf-8")

        metrics = getattr(result, "metrics", {}) or {}
        artifacts = getattr(result, "artifacts", {}) or {}
        metadata = getattr(result, "metadata", {}) or {}
        resolved_report_config = report_config
        identity = require_single_score_identity(
            (metrics, artifacts),
            context="evolution run artifacts",
        )
        if identity.verifier_version != resolved_report_config.verifier_version:
            raise ValueError(
                "run artifact verifier version does not match the operative report config"
            )
        workload_metrics = (
            artifacts.get("workload_metrics") if isinstance(artifacts, dict) else None
        )
        tripwire_thresholds = dict(resolved_report_config.surrogate_probe_tripwire_thresholds)
        tripwire_suite = _surrogate_probe_tripwire_suite(
            workload_metrics,
            thresholds=tripwire_thresholds,
        )
        agentic_gate = tripwire_suite["channels"]["agentic_branching"]
        config_snapshot_name = None
        if resolved_report_config.trace_workloads:
            snapshot = _snapshot_trace_inputs(
                run_dir,
                resolved_report_config,
                config_snapshot or Path(report_config_file),
            )
            config_snapshot_name = snapshot.name
            resolved_report_config = load_evaluator_config(snapshot)
            report_config_file = str(snapshot)
        elif config_snapshot is not None and config_snapshot.is_file():
            config_snapshot_name = "config_snapshot.yaml"
            shutil.copyfile(config_snapshot, run_dir / config_snapshot_name)
        if config_snapshot_name is not None and seed_source is not None:
            # Both synthetic and trace snapshots must resolve the actual archived
            # seed, including a CLI override, after the original inputs are removed.
            snapshot = run_dir / config_snapshot_name
            source_config = run_dir / "source_config.yaml"
            if not source_config.exists():
                shutil.copyfile(snapshot, source_config)
            document = WorkflowFileConfig.model_validate(load_yaml_document(snapshot)).model_dump(
                exclude_none=True, exclude_unset=True
            )
            document.setdefault("search", {})["seed_program"] = "seed_program.py"
            snapshot.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
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
            "report_baselines": list(_selected_baselines(REPORTING_BASELINES, baseline_names)),
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
        workload_manifest = build_workload_manifest(
            resolved_report_config, splits=_artifact_manifest_splits(resolved_report_config)
        )
        _write_json(run_dir / "workload_manifest.json", workload_manifest)
        summary["workload_manifest"] = {
            "path": "workload_manifest.json",
            "panel_sha256": workload_manifest["panel_sha256"],
            "evaluation_context_sha256": workload_manifest["evaluation_context_sha256"],
            "verifier_version": workload_manifest["verifier_version"],
        }
        _write_json(run_dir / "run_summary.json", summary)

        _persist_paradigm_candidates(run_dir, metadata=metadata)
        _persist_behavior_size_frontier(run_dir, metadata=metadata, config=resolved_report_config)
        self.persist_generated_mutation(
            run_dir,
            metadata=metadata,
            seed_source=seed_source,
            config=resolved_report_config,
        )
        promotion_adjudication = self.adjudicate_specialist(
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
            config = resolved_report_config
            candidate_path = run_dir / "best_program.py"
            report_results = self.panels.build_comparison(
                config,
                candidate_path,
                lambda: self.evaluate_baselines(config, baseline_names=baseline_names),
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
                    baseline_names=baseline_names,
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

    def persist_generated_mutation(
        self,
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
                raise ValueError(
                    "generated mutation snapshot verifier version does not match config"
                )
            if config.trace_workloads:
                # Cross-panel decomposition includes hidden evaluation. Archive the
                # mutation above, but defer that evaluation to explicit adjudication.
                return
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
                "seed": self.panels.build_decomposition(config, seed_path),
                "best_generated_mutation": self.panels.build_decomposition(
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

    def adjudicate_specialist(
        self,
        run_dir: Path,
        *,
        config: EvaluatorConfig,
    ) -> dict[str, Any] | None:
        """Re-evaluate a specialist winner as a complete policy before promotion."""
        if config.fixed_admission_policy is None or config.trace_workloads:
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
                        self.incumbent_path.read_text(encoding="utf-8"),
                    ),
                    encoding="utf-8",
                )
            candidate = self.panels.build_decomposition(
                promotion_config,
                candidate_path,
                include_hidden=True,
            )
            incumbent = self.panels.build_decomposition(
                promotion_config,
                self.incumbent_path,
                include_hidden=True,
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
