"""Run prefix KV-cache baseline reports or Levi evolution."""

from __future__ import annotations

import json
import os
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click

from prefix_cache_evolve.evaluators.baseline_suite import BASELINE_SUITE_EVALUATOR
from prefix_cache_evolve.evaluators.baseline_suite import select_baselines as _selected_baselines
from prefix_cache_evolve.evaluators.baselines import BASELINES, REPORTING_BASELINES
from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.evaluators.evaluation import PrefixKVCacheEvaluator
from prefix_cache_evolve.evaluators.results import EvaluationResult
from prefix_cache_evolve.evaluators.trace_replay import (
    calibrate_anonymized_trace,
    load_anonymized_trace,
)
from prefix_cache_evolve.evaluators.verifier import (
    require_single_score_identity,
    require_single_verifier_version,
)
from prefix_cache_evolve.problems.prefix_kv_cache.reporting import (
    baseline_report_command as _baseline_report_command,
)
from prefix_cache_evolve.workflow.config import (
    ConfigLoader,
    ConfigProvider,
    MinimalConfigProvider,
    YamlConfigProvider,
)
from prefix_cache_evolve.workflow.power import prevent_idle_sleep
from prefix_cache_evolve.workflow.program import ProgramSource
from prefix_cache_evolve.workflow.reporting import EvolutionReporter

from . import reporting as baseline_reporting
from .candidate_panels import (
    HIDDEN_PANEL,
    PROBE_PANEL,
    SELECTION_PANEL,
    VALIDATION_PANEL,
    CandidatePanelBuilder,
    CandidatePolicyValidationError,
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
from .run_artifacts import RunArtifactWriter
from .run_artifacts import _artifact_manifest_splits as _artifact_manifest_splits
from .run_artifacts import _persist_behavior_size_frontier as _persist_behavior_size_frontier
from .sandbox import docker_evaluation_scope
from .utilities import (
    capacity_blocks_for_token_tiers as _capacity_blocks_for_token_tiers,
)
from .utilities import (
    evaluation_result_summary as _evaluation_result_summary,
)
from .utilities import (
    format_int_tuple as _format_int_tuple,
)
from .utilities import (
    parse_positive_int_csv,
    parse_unique_positive_int_csv,
)
from .utilities import (
    write_json as _write_json,
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


def _resolve_search_seed(config_file: str, override: Path | None = None) -> Path:
    """Resolve CLI, YAML-relative, then policy-surface default seed precedence."""
    if override is not None:
        return override
    workflow = _CONFIG_LOADER.load(Path(config_file))
    configured = workflow.raw.get("search", {}).get("seed_program")
    if configured:
        return (Path(config_file).parent / configured).resolve()
    evaluator = load_evaluator_config(Path(config_file))
    return (
        _EVICTION_SPECIALIST_SEED_PATH
        if evaluator.candidate_policy_surface == "eviction_only"
        else _DEFAULT_SEED_PATH
    )


@prevent_idle_sleep()
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
    baseline_names: tuple[str, ...] = (),
) -> object:
    """Run one Levi evolution session and optionally persist its artifacts."""
    evaluator_config = load_evaluator_config(Path(config_file))
    if not evaluator_config.workload_configs(("train", "validation")) and not any(
        trace.split in {"train", "validation"} for trace in evaluator_config.trace_workloads
    ):
        raise ValueError(
            "evolution requires a nonempty train or validation panel; prepare traces first"
        )
    _selected_baselines(REPORTING_BASELINES, baseline_names)
    if evaluator_config.trace_workloads:
        if quick:
            evaluator_config = evaluator_config.with_updates(
                request_count=36, seeds=(3,), family_request_multipliers={}
            )
        # Reject missing or changed search inputs before contacting a model.
        build_workload_manifest(evaluator_config, splits=("train", "validation", "probe"))
    base_workflow_config = _CONFIG_LOADER.load(Path(config_file))
    effective_seed_program = _resolve_search_seed(config_file, seed_program)
    if quick and not evaluator_config.sandbox_image:
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
    with (
        prefix_kv_config_environment(Path(config_file), quick=quick),
        docker_evaluation_scope(evaluator_config.sandbox_image),
    ):
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
            baseline_names=baseline_names,
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
    baseline_names: tuple[str, ...] = (),
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
    if candidate_program is None:
        results = _evaluate_baselines(config, include_reporting=True, baseline_names=baseline_names)
    else:
        candidate_path = _resolve_candidate_program(candidate_program)
        results = _candidate_panel_builder().build_comparison(
            config,
            candidate_path,
            lambda: _evaluate_baselines(
                config, include_reporting=True, baseline_names=baseline_names
            ),
        )
        report_path = _baseline_comparison_output_path(candidate_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_baseline_comparison_report(
            report_path,
            results,
            candidate_path=candidate_path,
            command=_baseline_report_command(
                quick=quick,
                capacity_sweep_blocks=capacity_sweep_blocks,
                candidate_program=candidate_program,
                config_file=config_file,
                baseline_names=baseline_names,
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


def _baseline_comparison_output_path(candidate_path: Path) -> Path:
    """Keep generated reports outside immutable promoted-incumbent bundles."""
    incumbent_root = _DEFAULT_SEED_PATH.parent.parent.resolve()
    if candidate_path.resolve().is_relative_to(incumbent_root):
        return (
            Path("artifacts")
            / "prefix_kv_cache_reports"
            / candidate_path.parent.name
            / "baseline_comparison.md"
        )
    return candidate_path.parent / "baseline_comparison.md"


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
    baseline_names: tuple[str, ...] = (),
) -> Path:
    """Persist one run using the runner's configured evaluation services."""
    return _artifact_writer().save(
        result,
        output_root,
        iterations=iterations,
        config_label=config_label,
        seed_label=seed_label,
        seed_source=seed_source,
        report_config=report_config or _artifact_report_config(),
        report_config_file=report_config_file,
        config_snapshot=config_snapshot,
        timestamp=timestamp,
        baseline_names=baseline_names,
    )


def _artifact_writer() -> RunArtifactWriter:
    """Compose artifact persistence with policy and baseline evaluation."""
    return RunArtifactWriter(
        panels=_candidate_panel_builder(),
        evaluate_baselines=partial(_evaluate_baselines, include_reporting=True),
        incumbent_path=_DEFAULT_SEED_PATH,
    )


def hidden_report(
    *,
    quick: bool = False,
    capacity_blocks: int | None = None,
    capacity_sweep_blocks: tuple[int, ...] = (),
    block_size_tokens: int | None = None,
    candidate_program: Path | None = None,
    config_file: str = _DEFAULT_CONFIG_FILE,
    baseline_names: tuple[str, ...] = (),
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
        baseline_names=baseline_names,
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
    baseline_names: tuple[str, ...] = (),
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
            baseline_names=baseline_names,
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
    baseline_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Replay an anonymized metadata trace through deployable policies."""
    selected_names = tuple(dict.fromkeys(baseline_names or tuple(BASELINES)))
    unknown = sorted(set(selected_names) - BASELINES.keys())
    if unknown:
        raise ValueError("unknown deployable trace baselines: " + ", ".join(unknown))
    config = _config_from_args(
        quick=False,
        capacity_blocks=capacity_blocks,
        capacity_sweep_blocks=capacity_sweep_blocks,
        block_size_tokens=block_size_tokens,
        config_file=config_file,
    )
    if candidate_program is not None and config.sandbox_image:
        raise ValueError(
            "sandboxed source evaluation requires a configured panel; "
            "prepare trace_workloads and use --baseline-report"
        )
    requests = load_anonymized_trace(
        trace_path,
        block_size_tokens=config.block_size_tokens,
        arrival_bucket_ms=arrival_bucket_ms,
        request_limit=request_limit,
    )
    results = {}
    for name in selected_names:
        print(f"replaying_policy={name} requests={len(requests)}", flush=True)
        results.update(
            BASELINE_SUITE_EVALUATOR.evaluate_requests(config, {name: BASELINES[name]}, requests)
        )
    if candidate_program is not None:
        candidate_path = _resolve_candidate_program(candidate_program)
        print(f"replaying_policy=candidate requests={len(requests)}", flush=True)
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
    """Write workload fingerprints, matching a reference's split set when supplied."""
    config = _config_from_args(
        quick=quick,
        capacity_blocks=capacity_blocks,
        capacity_sweep_blocks=capacity_sweep_blocks,
        block_size_tokens=block_size_tokens,
        config_file=config_file,
    )
    splits = _artifact_manifest_splits(config)
    stable_reference = None
    if reference_path is not None:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if not isinstance(reference, dict):
            raise click.ClickException("workload manifest reference must be a JSON object")
        stable_reference = stable_workload_manifest_payload(reference)
        reference_splits = stable_reference["evaluation"].get("splits")
        if (
            not isinstance(reference_splits, list)
            or any(
                split not in ("train", "validation", "probe", "hidden")
                for split in reference_splits
            )
            or len(set(reference_splits)) != len(reference_splits)
        ):
            raise click.ClickException(
                "workload manifest reference must declare distinct known splits"
            )
        splits = tuple(reference_splits)
    payload = build_workload_manifest(config, splits=splits)
    if stable_reference is not None:
        if stable_workload_manifest_payload(payload) != stable_reference:
            raise click.ClickException(
                f"workload manifest differs from stable reference fields in {reference_path}"
            )
    _write_json(output_path, payload)
    print(f"workload_manifest={output_path}")
    print(f"panel_sha256={payload['panel_sha256']}")
    if reference_path is not None:
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
    "--report-baseline",
    "report_baselines",
    type=click.Choice(tuple(REPORTING_BASELINES)),
    multiple=True,
    help="Select baselines for comparison, hidden/probe reports, or saved artifacts; repeatable.",
)
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
    help="Summarize an anonymized metadata-only JSONL trace.",
)
@click.option(
    "--replay-trace",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    help="Replay an anonymized metadata-only JSONL trace.",
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
    "--trace-baseline",
    "trace_baselines",
    type=click.Choice(tuple(BASELINES)),
    multiple=True,
    help="Deployable baseline for --replay-trace; repeat to select several. Default: all.",
)
@click.option(
    "--workload-manifest",
    is_flag=True,
    help="Write deterministic workload fingerprints; trace panels default to search splits.",
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
        "Verify the reference's split set, score identity, settings, and streams while "
        "ignoring environment metadata."
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

    try:
        dispatch(RunnerOptions.from_mapping(kwargs))
    except CandidatePolicyValidationError as exc:
        raise click.ClickException(str(exc)) from exc


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
    seed_program: Path | None = None,
) -> None:
    """Print the effective workflow and evaluator configuration."""
    evaluator = load_evaluator_config(Path(config_file))
    if quick:
        evaluator = evaluator.with_updates(
            request_count=36, seeds=(3,), family_request_multipliers={}
        )
    base_workflow = _CONFIG_LOADER.load(Path(config_file))
    if quick and not evaluator.sandbox_image:
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
        "budgets": {
            "evaluations": workflow.max_iterations,
            "dollars": workflow.budget_dollars,
            "seconds": workflow.budget_seconds,
        },
        "models": {
            "model": workflow.model,
            "mutation_model": workflow.mutation_model,
            "paradigm_model": workflow.paradigm_model,
        },
        "search_seed": workflow.search_seed,
        "seed_program": str(_resolve_search_seed(config_file, seed_program)),
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
            "search_score_mode": evaluator.search_score_mode,
            "max_candidate_complexity": evaluator.max_candidate_complexity,
            "promotion_max_candidate_complexity": evaluator.promotion_max_candidate_complexity,
            "workload_seeds": list(evaluator.seeds),
            "policy_seed": evaluator.policy_seed,
            "request_count": evaluator.request_count,
            "block_size_tokens": evaluator.block_size_tokens,
            "capacity_blocks": list(evaluator.effective_capacity_blocks()),
            "workload_token_granularity": evaluator.workload_token_granularity,
            "trace_workloads": [
                trace.model_dump(mode="json") for trace in evaluator.trace_workloads
            ],
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
    baseline_names: tuple[str, ...] = (),
) -> dict[str, EvaluationResult]:
    baselines = REPORTING_BASELINES if include_reporting else BASELINES
    selected = _selected_baselines(baselines, baseline_names)
    return BASELINE_SUITE_EVALUATOR.evaluate(config, selected, splits=splits)


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


def _resolve_candidate_program(path: Path) -> Path:
    if path.is_dir():
        path = path / "best_program.py"
    if not path.exists():
        raise FileNotFoundError(f"candidate program {path} does not exist")
    return path


if __name__ == "__main__":
    main()
