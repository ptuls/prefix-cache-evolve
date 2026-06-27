"""Levi evaluation entry point for prefix KV-cache scoring policies."""

from __future__ import annotations

import traceback
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Callable

from prefix_cache_evolve.evaluator_entry import (
    EvaluatorResult,
    load_candidate_factory,
    load_candidate_factory_from_source,
    run_with_timeout,
)
from prefix_cache_evolve.evaluators.complexity import scoring_fn_complexity
from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.evaluators.prefix_kv_cache import PrefixKVCacheEvaluator
from prefix_cache_evolve.evaluators.results import EvaluationResult as PrefixEvaluationResult
from prefix_cache_evolve.problems.prefix_kv_cache import candidate_validation
from prefix_cache_evolve.problems.prefix_kv_cache.candidate_contracts import (
    candidate_contract,
)
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import (
    active_evaluator_config,
)
from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import (
    build_workload_manifest,
)
from prefix_cache_evolve.problems.prefix_kv_cache.specialist import (
    candidate_evaluator,
    candidate_exported_names,
)

# Compatibility aliases for callers that imported the former private helpers.
_called_name = candidate_validation._called_name
_import_from_violations = candidate_validation._import_from_violations
_is_broad_exception_handler = candidate_validation._is_broad_exception_handler
_is_dunder_name = candidate_validation._is_dunder_name
_multi_timescale_decay_violations = candidate_validation._multi_timescale_decay_violations
_threshold_excess_violations = candidate_validation._threshold_excess_violations
_top_level_source_violations = candidate_validation._top_level_source_violations
_candidate_source_violations = candidate_validation.candidate_source_violations
_static_repair_feedback = candidate_validation.static_repair_feedback

DEFAULT_CONFIG = EvaluatorConfig(capacity_sweep_blocks=(24, 48))
_WORKLOAD_FEEDBACK_KEYS = (
    "token_hit_rate",
    "block_hit_rate",
    "worst_quarter_token_hit_rate",
    "request_token_hit_rate_p10",
    "wasted_admission_token_rate",
    "admission_token_utility",
    "avoidable_admission_rate",
    "avoidable_admission_regret_token_rate",
    "avoidable_rejection_rate",
    "avoidable_rejection_regret_token_rate",
    "avoidable_eviction_rate",
    "value_weighted_avoidable_eviction_rate",
    "value_weighted_avoidable_eviction_regret_token_rate",
    "short_reuse_after_eviction_missed_token_rate",
    "policy_underfill_rate",
    "cache_churn_per_1k",
)
_MAX_FEEDBACK_WORKLOADS = 5
_MUTATION_GUIDANCE_WORKLOADS = (
    "train/agentic_tool_workflows",
    "validation/stochastic_serving_mix",
)


def evaluate(program_path: str) -> EvaluatorResult:
    """Evaluate selection splits and the quarantined structure probe."""
    try:
        source = Path(program_path).read_text(encoding="utf-8")
    except Exception as exc:
        return _error_result(
            "failed to load candidate factory",
            _load_error_artifacts(exc),
        )
    complexity = _source_complexity(source)
    rejection = _static_rejection(source, complexity)
    if rejection is not None:
        return rejection
    return _evaluate_isolated(
        _evaluate_program_path,
        program_path,
        complexity,
    )


def evaluate_factory(factory: Callable) -> EvaluatorResult:
    """Evaluate an already-loaded candidate factory.

    Complexity is unavailable for opaque callables and is therefore set to 0.
    """
    return _evaluate_isolated(_evaluate_factory, factory, 0)


def evaluate_source(source: str) -> EvaluatorResult:
    """Evaluate candidate source and apply the formula-complexity penalty."""
    complexity = _source_complexity(source)
    rejection = _static_rejection(source, complexity)
    if rejection is not None:
        return rejection
    return _evaluate_isolated(
        _evaluate_source,
        source,
        complexity,
    )


def evaluate_hidden(factory: Callable) -> EvaluatorResult:
    """Evaluate the quarantined hidden split for final reporting only."""
    return _evaluate_isolated(
        _evaluate_factory,
        factory,
        0,
        splits=("hidden",),
        include_hidden=True,
    )


def _source_complexity(source: str) -> int:
    """Return effective source complexity under the active evaluator config."""
    config = active_evaluator_config(DEFAULT_CONFIG)
    return scoring_fn_complexity(
        source,
        form_aware=config.form_aware_complexity,
    )


def _static_rejection(source: str, complexity: int) -> EvaluatorResult | None:
    """Reject source patterns that are outside the deployable search contract."""
    config = active_evaluator_config(DEFAULT_CONFIG)
    validation = candidate_validation.validate_candidate_source(source, complexity, config)
    if validation.is_valid:
        return None
    return _error_result(
        f"Repair before retry: {validation.repair_summary} "
        f"Static policy violations: {validation.violation_summary}.",
        {
            "error_type": "StaticPolicyViolation",
            "error_message": validation.violation_summary,
            "violations": list(validation.violations),
            "repair_feedback": list(validation.repair_feedback),
            "suggestion": validation.repair_summary,
        },
    )


def _evaluate_isolated(
    worker: Callable[
        [object, int, tuple[str, ...]],
        tuple[PrefixEvaluationResult | None, dict | None],
    ],
    candidate: object,
    complexity: int,
    *,
    splits: tuple[str, ...] = ("train", "validation", "probe"),
    include_hidden: bool = False,
) -> EvaluatorResult:
    config = active_evaluator_config(DEFAULT_CONFIG)
    try:
        result, load_error = run_with_timeout(
            worker,
            candidate,
            complexity,
            splits,
            timeout_seconds=config.timeout_s,
            memory_limit_bytes=config.max_memory_bytes,
            cpu_limit_seconds=config.timeout_s,
        )
    except TimeoutError as exc:
        return _error_result(
            "evaluation timed out",
            {
                "error_type": "TimeoutError",
                "error_message": str(exc),
                "suggestion": "Inspect candidate scoring methods for long-running logic.",
            },
            splits=splits,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _error_result(
            "evaluation failed",
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "full_traceback": traceback.format_exc(),
                "suggestion": "Unexpected evaluator failure; inspect the traceback.",
            },
            splits=splits,
        )
    if load_error is not None:
        return _error_result(
            "failed to load candidate factory",
            load_error,
            splits=splits,
        )
    if result is None:  # pragma: no cover - defensive
        return _error_result(
            "evaluation failed",
            {
                "error_type": "RuntimeError",
                "error_message": "evaluation worker returned no result",
                "suggestion": "Unexpected evaluator failure; inspect the traceback.",
            },
            splits=splits,
        )
    return _success_result(result, include_hidden=include_hidden)


def _evaluate_program_path(
    program_path: object,
    complexity: int,
    splits: tuple[str, ...],
) -> tuple[PrefixEvaluationResult | None, dict | None]:
    try:
        config = active_evaluator_config(DEFAULT_CONFIG)
        factory = load_candidate_factory(
            str(program_path),
            exported_names=candidate_exported_names(config),
        )
    except Exception as exc:
        return None, _load_error_artifacts(exc)
    return _evaluate_factory(factory, complexity, splits)


def _evaluate_source(
    source: object,
    complexity: int,
    splits: tuple[str, ...],
) -> tuple[PrefixEvaluationResult | None, dict | None]:
    try:
        config = active_evaluator_config(DEFAULT_CONFIG)
        factory = load_candidate_factory_from_source(
            str(source),
            exported_names=candidate_exported_names(config),
        )
    except Exception as exc:
        return None, _load_error_artifacts(exc)
    return _evaluate_factory(factory, complexity, splits)


def _evaluate_factory(
    factory: object,
    complexity: int,
    splits: tuple[str, ...],
) -> tuple[PrefixEvaluationResult, None]:
    evaluator = candidate_evaluator(active_evaluator_config(DEFAULT_CONFIG), splits=splits)
    return evaluator(factory, scoring_fn_complexity=complexity), None  # type: ignore[arg-type]


def _load_error_artifacts(exc: Exception) -> dict:
    repair_feedback = _load_repair_feedback(exc)
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "full_traceback": traceback.format_exc(),
        "repair_feedback": list(repair_feedback),
        "suggestion": " ".join(repair_feedback),
    }


def _load_repair_feedback(exc: Exception) -> tuple[str, ...]:
    """Return a concise repair instruction for a candidate load failure."""
    if isinstance(exc, SyntaxError):
        line = f" at line {exc.lineno}" if exc.lineno is not None else ""
        return (f"Fix the syntax error{line}: {exc.msg}.",)
    if isinstance(exc, AttributeError) and "candidate module must expose" in str(exc):
        config = active_evaluator_config(DEFAULT_CONFIG)
        return (candidate_contract(config).load_repair_feedback,)
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return ("Use only available top-level standard-library or documented primitive imports.",)
    if isinstance(exc, NameError):
        return ("Add the missing top-level import or remove the undefined name.",)
    if isinstance(exc, TypeError):
        return ("Match the documented factory and primitive call signatures exactly.",)
    return (_load_suggestion(),)


def _success_result(
    prefix_result: PrefixEvaluationResult,
    *,
    include_hidden: bool = False,
) -> EvaluatorResult:
    config = active_evaluator_config(DEFAULT_CONFIG)
    complexity_cost = float(prefix_result.score_breakdown.get("complexity_cost", 0.0))
    raw_score = float(prefix_result.combined_score) + complexity_cost
    guidance_score = _search_guidance_score(prefix_result, config)
    if config.search_score_mode == "combined":
        search_score = float(prefix_result.combined_score)
    elif config.search_score_mode == "raw_before_complexity":
        search_score = raw_score
    elif config.search_score_mode == "robust_min":
        search_score = (
            min(float(prefix_result.combined_score), guidance_score)
            if guidance_score is not None
            else float(prefix_result.combined_score)
        )
    else:
        return _error_result(
            f"unknown search score mode {config.search_score_mode}",
            {
                "error_type": "ConfigurationError",
                "error_message": f"unknown search score mode {config.search_score_mode}",
            },
        )
    metrics = {
        "verifier_version": prefix_result.verifier_version,
        "evaluation_context_sha256": prefix_result.evaluation_context_sha256,
        "panel_sha256": prefix_result.panel_sha256,
        "combined_score": search_score,
        "charged_combined_score": prefix_result.combined_score,
        "raw_score_before_complexity": raw_score,
        "success": prefix_result.success,
        "invalid_fraction": prefix_result.invalid_fraction,
    }
    if guidance_score is not None:
        metrics["search_guidance_floor_score"] = guidance_score
    repair_feedback: tuple[str, ...] = ()
    if not prefix_result.success:
        invalid_reasons = tuple(
            dict.fromkeys(
                trial.invalid_reason for trial in prefix_result.trials if trial.invalid_reason
            )
        )
        metrics["error"] = "candidate failed runtime policy validation: " + (
            "; ".join(invalid_reasons) or "unknown invalid candidate result"
        )
        repair_feedback = _runtime_repair_feedback(invalid_reasons)
    for split, split_metrics in prefix_result.split_metrics.items():
        if split == "hidden" and not include_hidden:
            continue
        for key, value in split_metrics.items():
            if isinstance(value, (int, float, bool)):
                metrics[f"{split}_{key}"] = value
    metrics.update(_selection_feedback_metrics(prefix_result))
    per_example_scores, feedback_per_example = _workload_failure_feedback(prefix_result)
    promotion_feedback = _promotion_complexity_feedback(prefix_result)
    if promotion_feedback is not None:
        per_example_scores.append(0.0)
        feedback_per_example.append(promotion_feedback)
    metrics["per_example_scores"] = per_example_scores
    metrics["feedback_per_example"] = feedback_per_example
    metrics.update(_promotion_eligibility_metrics(prefix_result))

    artifacts = {
        "verifier_version": prefix_result.verifier_version,
        "evaluation_context_sha256": prefix_result.evaluation_context_sha256,
        "panel_sha256": prefix_result.panel_sha256,
        "split_metrics": {
            key: value
            for key, value in prefix_result.split_metrics.items()
            if include_hidden or key != "hidden"
        },
        "workload_metrics": {
            key: value
            for key, value in prefix_result.workload_metrics.items()
            if include_hidden or not key.startswith("hidden/")
        },
        "capacity_metrics": prefix_result.capacity_metrics,
        "candidate_metadata": prefix_result.candidate_metadata,
        "score_breakdown": prefix_result.score_breakdown,
    }
    if repair_feedback:
        artifacts["repair_feedback"] = list(repair_feedback)
        artifacts["suggestion"] = " ".join(repair_feedback)
    return EvaluatorResult(metrics=metrics, artifacts=artifacts)


def _search_guidance_score(
    prefix_result: PrefixEvaluationResult,
    config: EvaluatorConfig,
) -> float | None:
    """Rescore configured non-quarantined train families as a robust floor."""
    guidance_families = set(config.search_guidance_families)
    if not guidance_families:
        return None
    guidance_trials = [
        replace(trial, split="validation")
        for trial in prefix_result.trials
        if trial.split == "train" and trial.workload in guidance_families
    ]
    if not guidance_trials:
        return None
    scoring_config = config.with_updates(
        search_score_mode="combined",
        search_guidance_families=(),
    )
    complexity = int(prefix_result.candidate_metadata.get("scoring_fn_complexity", 0))
    return (
        PrefixKVCacheEvaluator(scoring_config)
        .rescore_trials(
            guidance_trials,
            scoring_fn_complexity=complexity,
        )
        .combined_score
    )


def _runtime_repair_feedback(invalid_reasons: tuple[str, ...]) -> tuple[str, ...]:
    """Translate runtime policy-contract failures into focused repair instructions."""
    repairs = []
    for reason in invalid_reasons:
        if reason.startswith("policy must implement "):
            method = reason.removeprefix("policy must implement ")
            repairs.append(f"Implement {method} with the documented signature.")
        elif reason.startswith("factory raised "):
            repairs.append(
                "Fix build_candidate so it accepts capacity_blocks, block_size_tokens, and seed "
                "and returns the policy object."
            )
        elif " returned non-numeric score" in reason or " returned non-finite score" in reason:
            method = reason.split(maxsplit=1)[0]
            repairs.append(f"Make {method} return one finite int or float on every path.")
        elif " raised " in reason:
            method = reason.split(maxsplit=1)[0]
            repairs.append(
                f"Repair {method} using only documented fields; remove guessed attributes and "
                "fallback logic."
            )
        elif reason.startswith("candidate used "):
            repairs.append("Delete or bound candidate state to stay within the memory limit.")
        else:
            repairs.append(f"Repair this runtime contract failure: {reason}.")
    return tuple(dict.fromkeys(repairs))


def _selection_feedback_metrics(
    prefix_result: PrefixEvaluationResult,
) -> dict[str, float]:
    """Flatten selection and targeted guidance diagnostics for Levi."""
    metrics = {
        f"selection_{key}": float(value)
        for key, value in prefix_result.score_breakdown.items()
        if isinstance(value, (int, float, bool))
    }
    complexity_cost = float(prefix_result.score_breakdown.get("complexity_cost", 0.0))
    metrics["selection_raw_score_before_complexity"] = (
        float(prefix_result.combined_score) + complexity_cost
    )

    validation_hits_by_capacity: dict[int, list[float]] = {}
    for trial in prefix_result.trials:
        if trial.split == "validation":
            validation_hits_by_capacity.setdefault(trial.capacity_blocks, []).append(
                float(trial.token_hit_rate)
            )
    capacity_hit_rates = [
        sum(values) / len(values) for values in validation_hits_by_capacity.values() if values
    ]
    if capacity_hit_rates:
        metrics["validation_capacity_token_hit_spread"] = max(capacity_hit_rates) - min(
            capacity_hit_rates
        )

    for workload, values in prefix_result.workload_metrics.items():
        if not (workload.startswith("validation/") or workload in _MUTATION_GUIDANCE_WORKLOADS):
            continue
        split, workload_name = workload.split("/", maxsplit=1)
        workload_name = workload_name.replace("-", "_")
        for key in _WORKLOAD_FEEDBACK_KEYS:
            value = values.get(key)
            if isinstance(value, (int, float, bool)):
                metrics[f"{split}_workload_{workload_name}_{key}"] = float(value)
    return metrics


def _promotion_eligibility_metrics(
    prefix_result: PrefixEvaluationResult,
) -> dict[str, float | bool]:
    """Report whether an exploratory candidate clears the final complexity gate."""
    config = active_evaluator_config(DEFAULT_CONFIG)
    limit = config.promotion_max_candidate_complexity
    complexity = int(prefix_result.candidate_metadata.get("scoring_fn_complexity", 0))
    if limit is None:
        return {"promotion_eligible": True, "promotion_complexity_excess": 0.0}
    excess = max(0, complexity - limit)
    return {
        "promotion_eligible": excess == 0,
        "promotion_complexity_excess": float(excess),
    }


def _promotion_complexity_feedback(prefix_result: PrefixEvaluationResult) -> str | None:
    """Ask exploratory over-cap candidates to preserve behavior while simplifying."""
    config = active_evaluator_config(DEFAULT_CONFIG)
    limit = config.promotion_max_candidate_complexity
    complexity = int(prefix_result.candidate_metadata.get("scoring_fn_complexity", 0))
    if limit is None or complexity <= limit:
        return None
    excess = complexity - limit
    return (
        f"Exploration-only candidate: behavioral evaluation completed at effective complexity "
        f"{complexity}, but final promotion requires at most {limit}. Delete or simplify at least "
        f"{excess} effective AST nodes in a dedicated simplification mutation while preserving "
        "the policy behavior and score gains. Do not add new behavior or replace the removed "
        "code with another subsystem."
    )


def _workload_failure_feedback(
    prefix_result: PrefixEvaluationResult,
) -> tuple[list[float], list[str]]:
    """Build focused non-quarantined diagnostics for Levi mutation prompts."""
    breakdown = prefix_result.score_breakdown
    summary = (
        "Selection diagnostics: "
        f"combined={prefix_result.combined_score:.3f}, "
        f"raw_before_complexity="
        f"{prefix_result.combined_score + breakdown.get('complexity_cost', 0.0):.3f}, "
        f"mean={breakdown.get('mean_workload_score', 0.0):.3f}, "
        f"weakest={breakdown.get('min_workload_score', 0.0):.3f}, "
        f"costs(churn={breakdown.get('churn_cost', 0.0):.3f}, "
        f"underfill={breakdown.get('underfill_cost', 0.0):.3f}, "
        f"fairness={breakdown.get('fairness_cost', 0.0):.3f}, "
        f"complexity={breakdown.get('complexity_cost', 0.0):.3f})."
    )
    diagnostics: dict[str, tuple[float, str]] = {}
    for workload, values in prefix_result.workload_metrics.items():
        is_target = workload in _MUTATION_GUIDANCE_WORKLOADS
        if not (workload.startswith("validation/") or is_target):
            continue
        token_hit = _metric(values, "token_hit_rate")
        worst_quarter = _metric(values, "worst_quarter_token_hit_rate")
        request_p10 = _metric(values, "request_token_hit_rate_p10")
        waste = _metric(values, "wasted_admission_token_rate")
        avoidable = _metric(values, "avoidable_eviction_rate")
        avoidable_admission = _metric(values, "avoidable_admission_rate")
        avoidable_admission_regret = _metric(
            values,
            "avoidable_admission_regret_token_rate",
        )
        avoidable_rejection = _metric(values, "avoidable_rejection_rate")
        avoidable_rejection_regret = _metric(
            values,
            "avoidable_rejection_regret_token_rate",
        )
        value_weighted_eviction = _metric(
            values,
            "value_weighted_avoidable_eviction_rate",
        )
        value_weighted_eviction_regret = _metric(
            values,
            "value_weighted_avoidable_eviction_regret_token_rate",
        )
        admission_regret = avoidable_admission_regret + avoidable_rejection_regret
        regret_signal = _regret_signal(
            admission_regret,
            value_weighted_eviction_regret,
        )
        churn = _metric(values, "cache_churn_per_1k")
        quality = max(
            0.0,
            min(
                0.999,
                0.55 * token_hit
                + 0.2 * worst_quarter
                + 0.1 * request_p10
                + 0.1 * (1.0 - waste)
                + 0.05 * (1.0 - avoidable)
                - min(0.2, churn / 10_000.0),
            ),
        )
        role = "Targeted mutation-guidance workload" if is_target else "Weak validation workload"
        diagnostics[workload] = (
            quality,
            (
                f"{summary} {role} {workload}: "
                f"token_hit={token_hit:.3f}, "
                f"block_hit={_metric(values, 'block_hit_rate'):.3f}, "
                f"worst_quarter={worst_quarter:.3f}, "
                f"request_p10={request_p10:.3f}, "
                f"recompute_cost={_metric(values, 'recompute_cost'):.1f}. "
                "Admission audit: "
                f"avoidable_accept_rate={avoidable_admission:.3f}, "
                f"avoidable_accept_regret_token_rate={avoidable_admission_regret:.4f}, "
                f"avoidable_reject_rate={avoidable_rejection:.3f}, "
                f"avoidable_reject_regret_token_rate={avoidable_rejection_regret:.4f}, "
                f"waste={waste:.3f}, "
                f"utility={_metric(values, 'admission_token_utility'):.3f}. "
                "Eviction audit: "
                f"avoidable_eviction={avoidable:.3f}, "
                f"value_weighted_avoidable_eviction_rate={value_weighted_eviction:.3f}, "
                f"value_weighted_eviction_regret_token_rate="
                f"{value_weighted_eviction_regret:.4f}, "
                f"short_reuse_after_eviction="
                f"{_metric(values, 'short_reuse_after_eviction_missed_token_rate'):.3f}. "
                "Cache economics: "
                f"churn_per_1k={churn:.1f}, "
                f"underfill={_metric(values, 'policy_underfill_rate'):.3f}, "
                f"admission_regret_token_rate={admission_regret:.4f}, "
                f"eviction_regret_token_rate={value_weighted_eviction_regret:.4f}, "
                f"dominant_regret={regret_signal}. "
                "Make one focused change that improves this workload without "
                "worsening cache economics or complexity."
            ),
        )

    selected = [
        diagnostics[workload]
        for workload in _MUTATION_GUIDANCE_WORKLOADS
        if workload in diagnostics
    ]
    remaining = [
        diagnostic
        for workload, diagnostic in diagnostics.items()
        if workload not in _MUTATION_GUIDANCE_WORKLOADS
    ]
    selected.extend(
        sorted(remaining, key=lambda item: item[0])[: _MAX_FEEDBACK_WORKLOADS - len(selected)]
    )
    return (
        [quality for quality, _ in selected],
        [feedback for _, feedback in selected],
    )


def _regret_signal(admission_regret: float, eviction_regret: float) -> str:
    """Describe which measured decision surface contributes more regret."""
    tolerance = 1e-12
    if admission_regret <= tolerance and eviction_regret <= tolerance:
        return "none_measured"
    if admission_regret > eviction_regret + tolerance:
        return "admission"
    if eviction_regret > admission_regret + tolerance:
        return "eviction"
    return "balanced"


def _metric(values: dict, key: str) -> float:
    value = values.get(key, 0.0)
    return float(value) if isinstance(value, (int, float, bool)) else 0.0


def _error_result(
    message: str,
    artifacts: dict,
    *,
    splits: tuple[str, ...] = ("train", "validation", "probe"),
) -> EvaluatorResult:
    config = active_evaluator_config(DEFAULT_CONFIG)
    context_sha, panel_sha = _error_score_identity(
        config.model_dump_json(),
        splits,
    )
    return EvaluatorResult(
        metrics={
            "verifier_version": config.verifier_version,
            "evaluation_context_sha256": context_sha,
            "panel_sha256": panel_sha,
            "combined_score": config.v_min - 1.0 - config.invalid_surcharge,
            "success": False,
            "invalid_fraction": 1.0,
            "error": message,
        },
        artifacts={
            "verifier_version": config.verifier_version,
            "evaluation_context_sha256": context_sha,
            "panel_sha256": panel_sha,
            **artifacts,
        },
    )


@lru_cache(maxsize=16)
def _error_score_identity(
    config_json: str,
    splits: tuple[str, ...],
) -> tuple[str, str]:
    """Return a cached identity for evaluator failures on one panel."""
    config = EvaluatorConfig.model_validate_json(config_json)
    manifest = build_workload_manifest(config, splits=splits)
    return (
        str(manifest["evaluation_context_sha256"]),
        str(manifest["panel_sha256"]),
    )


def _load_suggestion() -> str:
    config = active_evaluator_config(DEFAULT_CONFIG)
    if config.candidate_policy_surface == "eviction_only":
        return (
            "Define `score_eviction(block, now, frequency, priority)` as a top-level "
            "function and return one finite numeric eviction rank."
        )
    return (
        "Ensure the module defines `candidate_factory(capacity_blocks, "
        "block_size_tokens, seed=None)` or `build_candidate(...)` and returns an "
        "object implementing the prefix KV-cache scoring interface."
    )
