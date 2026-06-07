"""Levi evaluation entry point for prefix KV-cache scoring policies."""

from __future__ import annotations

import ast
import traceback
from pathlib import Path
from typing import Callable

from prefix_cache_evolve.evaluator_entry import (
    EvaluatorResult,
    load_candidate_factory,
    load_candidate_factory_from_source,
    run_with_timeout,
)
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluationResult as PrefixEvaluationResult,
)
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluatorConfig,
    PrefixKVCacheEvaluator,
    scoring_fn_complexity,
)
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import (
    active_evaluator_config,
)

DEFAULT_CONFIG = EvaluatorConfig(capacity_sweep_blocks=(48, 96))
_UNSUPPORTED_CALLBACKS = {
    "on_request_end",
    "on_block_admitted",
    "on_block_evicted",
}
_FUTURE_REUSE_FIELDS = {
    "estimated_future_reuse",
    "estimated_next_reuse_distance",
}
_WORKLOAD_FEEDBACK_KEYS = (
    "token_hit_rate",
    "block_hit_rate",
    "worst_quarter_token_hit_rate",
    "request_token_hit_rate_p10",
    "wasted_admission_token_rate",
    "admission_token_utility",
    "avoidable_eviction_rate",
    "policy_underfill_rate",
    "cache_churn_per_1k",
)
_MAX_FEEDBACK_WORKLOADS = 5


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
    violations = _candidate_source_violations(source, complexity, config)
    if not violations:
        return None
    summary = "; ".join(violations)
    return _error_result(
        f"candidate rejected by static policy checks: {summary}",
        {
            "error_type": "StaticPolicyViolation",
            "error_message": summary,
            "violations": list(violations),
            "suggestion": (
                "Make one compact deployable change; remove defensive, unsupported, "
                "future-knowledge, and unused code."
            ),
        },
    )


def _candidate_source_violations(
    source: str,
    complexity: int,
    config: EvaluatorConfig,
) -> tuple[str, ...]:
    """Return deterministic static violations for one candidate source."""

    violations = []
    if config.max_candidate_complexity is not None and complexity > config.max_candidate_complexity:
        violations.append(
            f"effective complexity {complexity} exceeds limit {config.max_candidate_complexity}"
        )
    if not config.reject_unsupported_source_patterns:
        return tuple(violations)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return tuple(violations)

    imported_names: dict[str, str] = {}
    used_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                imported_names[name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            for alias in node.names:
                name = alias.asname or alias.name
                imported_names[name] = alias.name

    for name, imported_from in imported_names.items():
        if name == "*":
            violations.append("star imports are not allowed")
        elif name not in used_names:
            violations.append(f"unused import {imported_from}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _UNSUPPORTED_CALLBACKS:
                violations.append(f"unsupported callback {node.name}")
        elif isinstance(node, ast.Attribute) and node.attr in _FUTURE_REUSE_FIELDS:
            violations.append(f"future-knowledge field {node.attr} is not deployable")
        elif isinstance(node, ast.ExceptHandler) and _is_broad_exception_handler(node):
            violations.append("broad exception handlers are not allowed")
        elif isinstance(node, ast.Call):
            called_name = _called_name(node.func)
            if called_name in {"getattr", "id"}:
                violations.append(f"{called_name}() is not allowed in candidate code")
            elif called_name == "MultiTimescaleDecay":
                violations.extend(_multi_timescale_decay_violations(node))

    return tuple(dict.fromkeys(violations))


def _is_broad_exception_handler(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name):
        return handler.type.id in {"BaseException", "Exception"}
    return isinstance(handler.type, ast.Attribute) and handler.type.attr in {
        "BaseException",
        "Exception",
    }


def _called_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _multi_timescale_decay_violations(node: ast.Call) -> tuple[str, ...]:
    violations = []
    if len(node.args) > 1:
        violations.append("MultiTimescaleDecay accepts only one positional argument")
    half_lives = (
        node.args[0]
        if node.args
        else next(
            (keyword.value for keyword in node.keywords if keyword.arg == "half_lives"),
            None,
        )
    )
    if half_lives is None:
        violations.append("MultiTimescaleDecay requires a half-life sequence")
    elif isinstance(half_lives, ast.Constant):
        violations.append("MultiTimescaleDecay half-lives must be a sequence")
    elif isinstance(half_lives, (ast.List, ast.Tuple)):
        width = len(half_lives.elts)
        if not 1 <= width <= 8:
            violations.append("MultiTimescaleDecay requires one to eight half-lives")
    return tuple(violations)


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
        )
    except TimeoutError as exc:
        return _error_result(
            "evaluation timed out",
            {
                "error_type": "TimeoutError",
                "error_message": str(exc),
                "suggestion": "Inspect candidate scoring methods for long-running logic.",
            },
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
        )
    if load_error is not None:
        return _error_result("failed to load candidate factory", load_error)
    if result is None:  # pragma: no cover - defensive
        return _error_result(
            "evaluation failed",
            {
                "error_type": "RuntimeError",
                "error_message": "evaluation worker returned no result",
                "suggestion": "Unexpected evaluator failure; inspect the traceback.",
            },
        )
    return _success_result(result, include_hidden=include_hidden)


def _evaluate_program_path(
    program_path: object,
    complexity: int,
    splits: tuple[str, ...],
) -> tuple[PrefixEvaluationResult | None, dict | None]:
    try:
        factory = load_candidate_factory(str(program_path))
    except Exception as exc:
        return None, _load_error_artifacts(exc)
    return _evaluate_factory(factory, complexity, splits)


def _evaluate_source(
    source: object,
    complexity: int,
    splits: tuple[str, ...],
) -> tuple[PrefixEvaluationResult | None, dict | None]:
    try:
        factory = load_candidate_factory_from_source(str(source))
    except Exception as exc:
        return None, _load_error_artifacts(exc)
    return _evaluate_factory(factory, complexity, splits)


def _evaluate_factory(
    factory: object,
    complexity: int,
    splits: tuple[str, ...],
) -> tuple[PrefixEvaluationResult, None]:
    evaluator = PrefixKVCacheEvaluator(
        active_evaluator_config(DEFAULT_CONFIG),
        splits=splits,
    )
    return evaluator(factory, scoring_fn_complexity=complexity), None  # type: ignore[arg-type]


def _load_error_artifacts(exc: Exception) -> dict:
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "full_traceback": traceback.format_exc(),
        "suggestion": _load_suggestion(),
    }


def _success_result(
    prefix_result: PrefixEvaluationResult,
    *,
    include_hidden: bool = False,
) -> EvaluatorResult:
    metrics = {
        "combined_score": prefix_result.combined_score,
        "success": prefix_result.success,
        "invalid_fraction": prefix_result.invalid_fraction,
    }
    for split, split_metrics in prefix_result.split_metrics.items():
        if split == "hidden" and not include_hidden:
            continue
        for key, value in split_metrics.items():
            if isinstance(value, (int, float, bool)):
                metrics[f"{split}_{key}"] = value
    metrics.update(_selection_feedback_metrics(prefix_result))
    per_example_scores, feedback_per_example = _workload_failure_feedback(prefix_result)
    metrics["per_example_scores"] = per_example_scores
    metrics["feedback_per_example"] = feedback_per_example

    artifacts = {
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
    return EvaluatorResult(metrics=metrics, artifacts=artifacts)


def _selection_feedback_metrics(
    prefix_result: PrefixEvaluationResult,
) -> dict[str, float]:
    """Flatten selection diagnostics so Levi can retain and expose them."""

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
        if not workload.startswith("validation/"):
            continue
        workload_name = workload.removeprefix("validation/").replace("-", "_")
        for key in _WORKLOAD_FEEDBACK_KEYS:
            value = values.get(key)
            if isinstance(value, (int, float, bool)):
                metrics[f"validation_workload_{workload_name}_{key}"] = float(value)
    return metrics


def _workload_failure_feedback(
    prefix_result: PrefixEvaluationResult,
) -> tuple[list[float], list[str]]:
    """Build focused validation-only diagnostics for Levi mutation prompts."""

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
    diagnostics = []
    for workload, values in prefix_result.workload_metrics.items():
        if not workload.startswith("validation/"):
            continue
        token_hit = _metric(values, "token_hit_rate")
        worst_quarter = _metric(values, "worst_quarter_token_hit_rate")
        request_p10 = _metric(values, "request_token_hit_rate_p10")
        waste = _metric(values, "wasted_admission_token_rate")
        avoidable = _metric(values, "avoidable_eviction_rate")
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
        diagnostics.append(
            (
                quality,
                (
                    f"{summary} Weak validation workload {workload}: "
                    f"token_hit={token_hit:.3f}, "
                    f"block_hit={_metric(values, 'block_hit_rate'):.3f}, "
                    f"worst_quarter={worst_quarter:.3f}, "
                    f"request_p10={request_p10:.3f}, "
                    f"waste={waste:.3f}, "
                    f"utility={_metric(values, 'admission_token_utility'):.3f}, "
                    f"avoidable_eviction={avoidable:.3f}, "
                    f"churn_per_1k={churn:.1f}, "
                    f"underfill={_metric(values, 'policy_underfill_rate'):.3f}. "
                    "Make one focused change that improves this workload without "
                    "worsening cache economics or complexity."
                ),
            )
        )

    weakest = sorted(diagnostics, key=lambda item: item[0])[:_MAX_FEEDBACK_WORKLOADS]
    return (
        [quality for quality, _ in weakest],
        [feedback for _, feedback in weakest],
    )


def _metric(values: dict, key: str) -> float:
    value = values.get(key, 0.0)
    return float(value) if isinstance(value, (int, float, bool)) else 0.0


def _error_result(message: str, artifacts: dict) -> EvaluatorResult:
    config = active_evaluator_config(DEFAULT_CONFIG)
    return EvaluatorResult(
        metrics={
            "combined_score": config.v_min - 1.0 - config.invalid_surcharge,
            "success": False,
            "invalid_fraction": 1.0,
            "error": message,
        },
        artifacts=artifacts,
    )


def _load_suggestion() -> str:
    return (
        "Ensure the module defines `candidate_factory(capacity_blocks, "
        "block_size_tokens, seed=None)` or `build_candidate(...)` and returns an "
        "object implementing the prefix KV-cache scoring interface."
    )
