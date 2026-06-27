"""Adapters that execute candidate evolution through Levi."""

import importlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from prefix_cache_evolve.artifacts import write_json
from prefix_cache_evolve.evaluator_entry import (
    EvaluatorResult,
    _exec_registered_module,
    load_candidate_factory_from_source,
)
from prefix_cache_evolve.workflow.config import LeviRunConfig
from prefix_cache_evolve.workflow.levi_compat import (
    LeviRuntimeSettings,
    activate_levi_runtime,
)

_SCORE_IDENTITY_KEYS = (
    "verifier_version",
    "evaluation_context_sha256",
    "panel_sha256",
)


@dataclass
class LeviRunResult:
    """Stable result shape consumed by the repo's portfolio/reporting code."""

    best_program: str
    best_score: float
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
    total_evaluations: int = 0
    total_cost: float = 0.0
    archive_size: int = 0
    runtime_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def code(self) -> str:
        """Return the best evolved program."""
        return self.best_program

    @property
    def best_code(self) -> str:
        """Return the best evolved program for compatibility callers."""
        return self.best_program


class LeviScoreFunction:
    """Picklable score function adapter used by Levi worker processes."""

    def __init__(
        self,
        evaluate_factory: Callable[[Callable[..., object]], EvaluatorResult],
        evaluate_source: Callable[[str], EvaluatorResult] | None = None,
        score_metric: str = "combined_score",
    ) -> None:
        self._evaluate_factory = evaluate_factory
        self._evaluate_source = evaluate_source
        self._score_metric = score_metric

    def __call__(self, factory: Callable[..., object], inputs=None) -> dict[str, Any]:
        """Evaluate one candidate and return Levi-compatible scalar metrics."""
        source = _candidate_source(factory, inputs)
        if self._evaluate_source is not None:
            if source is None:
                return {
                    "error": ("candidate source unavailable; source-aware evaluation is required"),
                }
            result = self._evaluate_source(source)
        else:
            result = self._evaluate_factory(factory)
        metrics = result.metrics or {}
        success = metrics.get("success")
        if success is not None and not bool(success):
            return {"error": _failure_message(result)}
        score = metrics.get(self._score_metric, 0.0)
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            score = 0.0

        levi_metrics: dict[str, Any] = {"score": float(score)}
        for key in _SCORE_IDENTITY_KEYS:
            value = metrics.get(key)
            if isinstance(value, str):
                levi_metrics[key] = value
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                levi_metrics[key] = float(value)
        per_example_scores = metrics.get("per_example_scores")
        feedback_per_example = metrics.get("feedback_per_example")
        if (
            isinstance(per_example_scores, list)
            and isinstance(feedback_per_example, list)
            and len(per_example_scores) == len(feedback_per_example)
            and all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in per_example_scores
            )
            and all(isinstance(value, str) for value in feedback_per_example)
        ):
            levi_metrics["per_example_scores"] = [float(value) for value in per_example_scores]
            levi_metrics["feedback_per_example"] = list(feedback_per_example)
        return levi_metrics


def _failure_message(result: EvaluatorResult) -> str:
    """Return evaluator failure text with any structured repair guidance."""
    metrics = result.metrics or {}
    message = str(metrics.get("error") or "candidate failed evaluator validation")
    artifacts = result.artifacts or {}
    repairs = artifacts.get("repair_feedback")
    if isinstance(repairs, list):
        repair_lines = [repair.strip() for repair in repairs if isinstance(repair, str)]
        if repair_lines and "Repair before retry:" not in message:
            message = f"Repair before retry: {' '.join(repair_lines)} Failure: {message}."
    return message


class LeviRunner:
    """Coordinates execution through Levi's evolve_code API."""

    def __init__(
        self,
        evolve_code: Callable[..., Any],
        evaluator_path: Path,
        problem_description: str,
        function_signature: str,
    ) -> None:
        self._evolve_code = evolve_code
        self._evaluator_path = evaluator_path
        self._problem_description = problem_description
        self._function_signature = function_signature
        self._evaluate_factory = self._load_evaluate_factory(evaluator_path)
        self._evaluate_source = self._load_evaluate_source(evaluator_path)

    def run(self, program_path: Path, config: LeviRunConfig) -> LeviRunResult:
        """Run Levi from a seed program and resolved configuration."""
        seed_program = program_path.read_text(encoding="utf-8")
        score_fn = LeviScoreFunction(self._evaluate_factory, self._evaluate_source)
        kwargs = config.evolve_kwargs()
        kwargs["function_signature"] = (
            getattr(config, "function_signature", "") or self._function_signature
        )
        kwargs["seed_program"] = seed_program
        kwargs["score_fn"] = score_fn
        output_dir = kwargs.get("output_dir")
        if not output_dir:
            output_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            kwargs["output_dir"] = output_dir

        paradigm_max_tokens = _configured_paradigm_max_tokens(config)
        paradigm_model_names = _configured_paradigm_model_names(config)
        search_seed = int(getattr(config, "search_seed", 0))
        runtime_settings = LeviRuntimeSettings(
            search_seed=search_seed,
            paradigm_max_tokens=paradigm_max_tokens,
            paradigm_model_names=paradigm_model_names,
            api_base=getattr(config, "api_base", None),
            api_key_env=getattr(config, "api_key_env", None),
            paradigm_candidate_output_dir=Path(output_dir) / "paradigm_candidates",
        )
        with activate_levi_runtime(runtime_settings):
            result = self._evolve_code(
                getattr(config, "problem_description", None) or self._problem_description,
                **kwargs,
            )

        best_program = getattr(result, "best_program", "")
        evaluation = self._evaluate_best_program(best_program)
        metrics = evaluation.metrics
        artifacts = evaluation.artifacts
        score_identity = {
            key: value
            for key in _SCORE_IDENTITY_KEYS
            if isinstance((value := metrics.get(key)), str)
        }
        snapshot_path = Path(output_dir) / "snapshot.json"
        if score_identity:
            _stamp_levi_snapshot_score_identity(snapshot_path, score_identity)
        metadata = {
            **score_identity,
            "levi_total_cost": getattr(result, "total_cost", 0.0),
            "levi_runtime_seconds": getattr(result, "runtime_seconds", 0.0),
            "levi_output_dir": str(output_dir),
            "levi_snapshot_path": str(snapshot_path),
            "levi_paradigm_candidates_dir": str(Path(output_dir) / "paradigm_candidates"),
            "search_seed": search_seed,
            "model": getattr(config, "model", None),
            "paradigm_model": getattr(config, "paradigm_model", None),
            "paradigm_max_tokens": paradigm_max_tokens,
            "mutation_model": getattr(config, "mutation_model", None),
            "api_base": getattr(config, "api_base", None),
            "api_key_env": getattr(config, "api_key_env", None),
            "pipeline": dict(getattr(config, "pipeline", {}) or {}),
            "runtime": {
                "python": sys.version,
                "packages": {
                    package: _package_version(package)
                    for package in ("levi", "litellm", "numpy", "pydantic", "pyyaml")
                },
            },
            "reproducibility_limits": (
                "Python and NumPy selection plus supported model requests are seeded. "
                "Remote provider behavior and asynchronous worker ordering may still vary."
            ),
        }

        return LeviRunResult(
            best_program=best_program,
            best_score=float(getattr(result, "best_score", metrics.get("combined_score", 0.0))),
            metrics=metrics,
            artifacts=artifacts,
            total_evaluations=int(getattr(result, "total_evaluations", 0) or 0),
            total_cost=float(getattr(result, "total_cost", 0.0) or 0.0),
            archive_size=int(getattr(result, "archive_size", 0) or 0),
            runtime_seconds=float(getattr(result, "runtime_seconds", 0.0) or 0.0),
            metadata=metadata,
        )

    def _evaluate_best_program(self, source: str) -> EvaluatorResult:
        if self._evaluate_source is not None:
            return self._evaluate_source(source)
        try:
            factory = load_candidate_factory_from_source(source)
        except Exception as exc:
            return EvaluatorResult(
                metrics={
                    "combined_score": 0.0,
                    "error": "failed to load Levi best program",
                },
                artifacts={"error_type": type(exc).__name__, "error_message": str(exc)},
            )
        return self._evaluate_factory(factory)

    def _load_evaluate_factory(
        self, evaluator_path: Path
    ) -> Callable[[Callable[..., object]], EvaluatorResult]:
        module_name = _module_name_from_package_path(evaluator_path)
        if module_name is not None:
            module = importlib.import_module(module_name)
            evaluate_factory = getattr(module, "evaluate_factory", None)
            if callable(evaluate_factory):
                return evaluate_factory

        spec = importlib.util.spec_from_file_location(
            f"prefix_cache_evolve_levi_evaluator_{abs(hash(evaluator_path.resolve()))}",
            evaluator_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to load evaluator from {evaluator_path}")
        loader = spec.loader
        module = importlib.util.module_from_spec(spec)
        _exec_registered_module(module, lambda: loader.exec_module(module))
        evaluate_factory = getattr(module, "evaluate_factory", None)
        if not callable(evaluate_factory):
            raise AttributeError(f"{evaluator_path} must expose evaluate_factory(factory)")
        return evaluate_factory

    def _load_evaluate_source(
        self, evaluator_path: Path
    ) -> Callable[[str], EvaluatorResult] | None:
        module_name = _module_name_from_package_path(evaluator_path)
        if module_name is not None:
            module = importlib.import_module(module_name)
            evaluate_source = getattr(module, "evaluate_source", None)
            return evaluate_source if callable(evaluate_source) else None

        spec = importlib.util.spec_from_file_location(
            f"prefix_cache_evolve_levi_evaluator_{abs(hash(evaluator_path.resolve()))}",
            evaluator_path,
        )
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        module = importlib.util.module_from_spec(spec)
        _exec_registered_module(module, lambda: loader.exec_module(module))
        evaluate_source = getattr(module, "evaluate_source", None)
        return evaluate_source if callable(evaluate_source) else None


def _package_version(package: str) -> str | None:
    """Return an installed package version without failing the run."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _candidate_source(factory: Callable[..., object], inputs: Any) -> str | None:
    """Recover the candidate source Levi attaches while evaluating code."""
    if isinstance(inputs, str):
        return inputs
    if isinstance(inputs, dict):
        for key in ("source", "code", "program", "candidate_source"):
            value = inputs.get(key)
            if isinstance(value, str):
                return value

    for attr in ("__source__", "source", "code"):
        value = getattr(factory, attr, None)
        if isinstance(value, str):
            return value

    globals_dict = getattr(factory, "__globals__", {})
    if isinstance(globals_dict, dict):
        value = globals_dict.get("__source_code__")
        if isinstance(value, str):
            return value
    return None


def _configured_paradigm_max_tokens(config: Any) -> int | None:
    """Return the paradigm output budget, falling back to the pipeline budget."""
    punctuated_equilibrium = getattr(config, "punctuated_equilibrium", {}) or {}
    pipeline = getattr(config, "pipeline", {}) or {}
    raw_value = punctuated_equilibrium.get("max_tokens")
    if raw_value is None:
        raw_value = pipeline.get("max_tokens")
    if raw_value is None:
        return None
    try:
        max_tokens = int(raw_value)
    except (TypeError, ValueError):
        return None
    return max_tokens if max_tokens > 0 else None


def _configured_paradigm_model_names(config: Any) -> frozenset[str]:
    """Return model names that Levi may use for paradigm-shift generation."""
    from levi.clients.base import client_name

    punctuated_equilibrium = getattr(config, "punctuated_equilibrium", {}) or {}
    configured_models = punctuated_equilibrium.get("heavy_models")
    if not configured_models:
        configured_models = (
            getattr(config, "paradigm_model", None)
            or getattr(config, "model", None)
            or getattr(config, "mutation_model", None),
        )
    elif not isinstance(configured_models, (list, tuple)):
        configured_models = (configured_models,)

    return frozenset(
        client_name(model)
        for model in configured_models
        if model is not None and client_name(model)
    )


def _stamp_levi_snapshot_score_identity(
    path: Path,
    score_identity: dict[str, str],
) -> None:
    """Stamp every score-bearing Levi snapshot record with its score identity."""
    if not path.is_file():
        return
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot.update(score_identity)
    for key in ("metadata", "run_state"):
        record = snapshot.get(key)
        if isinstance(record, dict):
            record.update(score_identity)
    for key in ("elites", "score_history"):
        records = snapshot.get(key)
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    record.update(score_identity)
    write_json(path, snapshot)


def _module_name_from_package_path(path: Path) -> str | None:
    """Return an importable module name for a file inside a Python package."""
    resolved = path.resolve()
    if resolved.suffix != ".py" or not resolved.exists():
        return None

    parts = [resolved.stem]
    parent = resolved.parent
    while (parent / "__init__.py").exists():
        parts.append(parent.name)
        parent = parent.parent

    if len(parts) == 1:
        return None
    return ".".join(reversed(parts))
