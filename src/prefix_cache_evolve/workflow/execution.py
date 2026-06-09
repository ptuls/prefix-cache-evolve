"""Adapters that execute candidate evolution through Levi."""

import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import numpy as np

from prefix_cache_evolve.evaluator_entry import (
    EvaluatorResult,
    _exec_registered_module,
    load_candidate_factory_from_source,
)

_levi_reasoning_max_tokens: int | None = None
_levi_paradigm_candidate_output_dir: Path | None = None
_levi_search_seed = 0
_levi_completion_index = 0
_levi_api_base: str | None = None
_levi_api_key_env: str | None = None
_levi_failure_feedback_path: Path | None = None
_levi_global_failure_feedback_path: Path | None = None
_levi_failure_feedback_scope: str | None = None
_levi_global_failure_feedback_max_events = 1_000
_MAX_RECURRING_FAILURES = 3
_MIN_RECURRING_FAILURE_COUNT = 2
_FAILURE_MEMORY_SCOPE_VERSION = 1


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

    def run(self, program_path: Path, config) -> LeviRunResult:
        """Run Levi from a seed program and resolved configuration."""
        seed_program = program_path.read_text(encoding="utf-8")
        kwargs = config.evolve_kwargs()
        kwargs["function_signature"] = (
            getattr(config, "function_signature", "") or self._function_signature
        )
        kwargs["seed_program"] = seed_program
        output_dir = kwargs.get("output_dir")
        if not output_dir:
            output_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            kwargs["output_dir"] = output_dir
        failure_feedback_path = Path(output_dir) / "failure_feedback.jsonl"
        (
            failure_memory_mode,
            global_failure_feedback_path,
            failure_feedback_scope,
            max_global_failure_events,
        ) = _configured_levi_failure_memory(
            config,
            self._evaluator_path,
            str(kwargs["function_signature"]),
        )
        kwargs["score_fn"] = LeviScoreFunction(self._evaluate_factory, self._evaluate_source)

        _enable_levi_failure_memory_support(
            failure_feedback_path,
            global_path=global_failure_feedback_path,
            scope=failure_feedback_scope,
            max_global_events=max_global_failure_events,
        )
        _enable_levi_code_feedback_support()
        _enable_levi_degenerate_centroid_fallback()
        _enable_levi_reasoning_completion_support(_configured_reasoning_max_tokens(config))
        search_seed = int(getattr(config, "search_seed", 0))
        _enable_levi_reproducibility_support(
            search_seed,
            api_base=getattr(config, "api_base", None),
            api_key_env=getattr(config, "api_key_env", None),
        )
        _enable_levi_paradigm_candidate_persistence(Path(output_dir) / "paradigm_candidates")
        result = self._evolve_code(
            getattr(config, "problem_description", None) or self._problem_description,
            **kwargs,
        )

        best_program = getattr(result, "best_program", "")
        evaluation = self._evaluate_best_program(best_program)
        metrics = evaluation.metrics
        artifacts = evaluation.artifacts
        metadata = {
            "levi_total_cost": getattr(result, "total_cost", 0.0),
            "levi_runtime_seconds": getattr(result, "runtime_seconds", 0.0),
            "levi_output_dir": str(output_dir),
            "levi_snapshot_path": str(Path(output_dir) / "snapshot.json"),
            "levi_paradigm_candidates_dir": str(Path(output_dir) / "paradigm_candidates"),
            "levi_failure_feedback_path": str(failure_feedback_path),
            "levi_failure_memory_mode": failure_memory_mode,
            "levi_global_failure_feedback_path": (
                str(global_failure_feedback_path)
                if global_failure_feedback_path is not None
                else None
            ),
            "levi_failure_feedback_scope": failure_feedback_scope,
            "levi_global_failure_feedback_max_events": max_global_failure_events,
            "search_seed": search_seed,
            "model": getattr(config, "model", None),
            "paradigm_model": getattr(config, "paradigm_model", None),
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
        module = importlib.util.module_from_spec(spec)
        _exec_registered_module(module, lambda: spec.loader.exec_module(module))  # type: ignore[call-arg]
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
        module = importlib.util.module_from_spec(spec)
        _exec_registered_module(module, lambda: spec.loader.exec_module(module))  # type: ignore[call-arg]
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


def _enable_levi_code_feedback_support() -> None:
    """Inject evaluator and repeated-failure feedback into Levi code prompts."""
    from levi.artifacts.code import CodeAdapter

    original = CodeAdapter.build_mutation_prompt
    if not getattr(original, "_prefix_cache_evolve_repair_feedback_patch", False):
        accepts_feedback = "feedback" in inspect.signature(original).parameters

        def build_mutation_prompt_with_feedback(
            self,
            parents,
            *,
            feedback: list[str] | None = None,
            **kwargs,
        ) -> str:
            if accepts_feedback:
                prompt = original(self, parents, feedback=feedback, **kwargs)
            else:
                prompt = original(self, parents, **kwargs)
            return _inject_levi_feedback_sections(prompt, feedback=feedback)

        build_mutation_prompt_with_feedback._prefix_cache_evolve_repair_feedback_patch = True
        CodeAdapter.build_mutation_prompt = build_mutation_prompt_with_feedback

    for method_name in (
        "build_diversity_prompt",
        "build_init_variant_prompt",
        "build_paradigm_shift_prompt",
        "build_variant_prompt",
    ):
        _patch_levi_prompt_with_failure_memory(CodeAdapter, method_name)


def _patch_levi_prompt_with_failure_memory(adapter_class: type, method_name: str) -> None:
    """Add deterministic recurring-failure constraints to one Levi prompt method."""
    original = getattr(adapter_class, method_name)
    if getattr(original, "_prefix_cache_evolve_failure_memory_patch", False):
        return

    def prompt_with_failure_memory(self, *args, **kwargs) -> str:
        return _inject_levi_feedback_sections(original(self, *args, **kwargs))

    prompt_with_failure_memory._prefix_cache_evolve_failure_memory_patch = True
    setattr(adapter_class, method_name, prompt_with_failure_memory)


def _inject_levi_feedback_sections(
    prompt: str,
    *,
    feedback: list[str] | None = None,
) -> str:
    """Insert selected-parent diagnostics and scoped global failure memory."""
    sections = []
    feedback_lines = [line.strip() for line in feedback or [] if line.strip()]
    if feedback_lines:
        sections.append(
            "## Evaluator Feedback\n"
            "These diagnostics describe the selected parent. Make one focused mutation "
            "against one primary diagnosis and preserve metrics not identified as weak.\n"
            + "\n".join(f"- {line}" for line in feedback_lines)
        )
    recurring_failures = _recurring_levi_failure_feedback()
    if recurring_failures:
        source = (
            "this and prior compatible runs"
            if _levi_global_failure_feedback_path is not None
            else "this Levi run"
        )
        sections.append(
            "## Failure Memory\n"
            f"Treat these recurring evaluator failures from {source} as hard constraints for "
            "every candidate.\n"
            + "\n".join(f"- ({count}x) {guidance}" for count, guidance in recurring_failures)
        )
    sections.append(
        "## Preflight Repair Checks\n"
        "- Keep the documented entry point and produce complete valid Python.\n"
        "- Never reference request_type or prompt_tokens; both fields are deliberately "
        "scrubbed before candidate callbacks.\n"
        "- Use only documented fields and callbacks; remove guessed fallbacks and broad "
        "exception handlers.\n"
        "- Keep imports top-level and delete unused imports.\n"
        "- For an exploration-only over-cap parent, perform the requested dedicated "
        "simplification mutation: preserve behavior, add nothing, and delete at least the "
        "reported excess.\n"
        "- When near a complexity cap, make a net deletion before adding behavior."
    )
    section = "\n\n".join(sections)
    output_marker = "\n\n## Output\n"
    if output_marker in prompt:
        return prompt.replace(output_marker, f"\n\n{section}{output_marker}", 1)
    return f"{prompt}\n\n{section}"


def _levi_failure_scope(evaluator_path: Path, function_signature: str) -> str:
    """Return a stable scope for failures sharing one evaluator contract."""
    module_name = _module_name_from_package_path(evaluator_path) or str(evaluator_path.resolve())
    contract = f"v{_FAILURE_MEMORY_SCOPE_VERSION}\n{module_name}\n{function_signature.strip()}"
    digest = hashlib.sha256(contract.encode("utf-8")).hexdigest()[:16]
    return f"{evaluator_path.stem}:{digest}"


def _configured_levi_failure_memory(
    config: Any,
    evaluator_path: Path,
    function_signature: str,
) -> tuple[str, Path | None, str | None, int]:
    """Resolve independent-run or opt-in cross-run failure memory."""
    settings = getattr(config, "failure_memory", None)
    if isinstance(settings, dict):
        mode = str(settings.get("mode", "run_only"))
        global_path_value = settings.get("global_path", "runs/global_failure_feedback.jsonl")
        max_global_events = int(settings.get("max_global_events", 1_000))
    else:
        mode = str(getattr(settings, "mode", "run_only"))
        global_path_value = getattr(settings, "global_path", "runs/global_failure_feedback.jsonl")
        max_global_events = int(getattr(settings, "max_global_events", 1_000))
    if mode == "run_only":
        return mode, None, None, max_global_events
    if mode != "global":
        raise ValueError(f"unknown failure memory mode {mode!r}")
    return (
        mode,
        Path(str(global_path_value)),
        _levi_failure_scope(evaluator_path, function_signature),
        max_global_events,
    )


def _enable_levi_failure_memory_support(
    path: Path,
    *,
    global_path: Path | None = None,
    scope: str | None = None,
    max_global_events: int = 1_000,
) -> None:
    """Persist every Levi evaluation error for deterministic later prompt feedback."""
    global _levi_failure_feedback_path
    global _levi_failure_feedback_scope
    global _levi_global_failure_feedback_max_events
    global _levi_global_failure_feedback_path

    _levi_failure_feedback_path = path
    _levi_global_failure_feedback_path = global_path
    _levi_failure_feedback_scope = scope
    _levi_global_failure_feedback_max_events = max_global_events
    _reset_levi_run_failure_feedback(path)
    from levi.pipeline.state import PipelineState

    original = PipelineState.record_error
    if getattr(original, "_prefix_cache_evolve_failure_memory_patch", False):
        return

    def record_error_with_failure_memory(self, error: str) -> None:
        original(self, error)
        _record_levi_failure_feedback(error)

    record_error_with_failure_memory._prefix_cache_evolve_failure_memory_patch = True
    PipelineState.record_error = record_error_with_failure_memory


def _reset_levi_run_failure_feedback(path: Path) -> None:
    """Start each Levi run with an empty per-run failure audit."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def _record_levi_failure_feedback(error: str) -> None:
    """Append deterministic repair categories for one failed evaluation."""
    path = _levi_failure_feedback_path
    entries = _levi_failure_feedback_entries(error)
    if not entries:
        return
    if path is not None:
        _append_levi_failure_feedback(path, entries)
    global_path = _levi_global_failure_feedback_path
    scope = _levi_failure_feedback_scope
    global_entries = tuple(
        (key, guidance) for key, guidance in entries if _is_global_failure_feedback_key(key)
    )
    if global_path is not None and scope is not None and global_entries:
        _append_levi_failure_feedback(global_path, global_entries, scope=scope)


def _append_levi_failure_feedback(
    path: Path,
    entries: tuple[tuple[str, str], ...],
    *,
    scope: str | None = None,
) -> None:
    """Append failure entries without allowing memory persistence to fail a run."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            for key, guidance in entries:
                entry = {"key": key, "guidance": guidance}
                if scope is not None:
                    entry["scope"] = scope
                stream.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError:
        return


def _levi_failure_feedback_entries(error: str) -> tuple[tuple[str, str], ...]:
    """Classify one evaluator error into stable, actionable repair categories."""
    message = " ".join(str(error).split())
    lower = message.lower()
    patterns = (
        (
            "sanitized_request_type",
            "request_type",
            "Remove request_type entirely; it is not available to candidate policies.",
        ),
        (
            "sanitized_prompt_tokens",
            "prompt_tokens",
            "Remove prompt_tokens entirely; it is not available to candidate policies.",
        ),
        (
            "future_knowledge",
            "future-knowledge field",
            "Remove future-knowledge fields and use only observed recurrence, subtree, gap, or "
            "pressure signals.",
        ),
        (
            "getattr",
            "getattr()",
            "Replace getattr() fallbacks with direct access to one documented field.",
        ),
        (
            "broad_exception",
            "broad exception handlers",
            "Remove broad try/except fallback logic and use the documented contract directly.",
        ),
        (
            "unused_import",
            "unused import",
            "Delete every unused import before returning the candidate.",
        ),
        (
            "unsupported_callback",
            "unsupported callback",
            "Delete unsupported callbacks and implement only the documented policy surface.",
        ),
        (
            "complexity_cap",
            "effective complexity",
            "Simplify the current design with a net deletion; do not add a replacement subsystem.",
        ),
        (
            "syntax_error",
            "syntax error",
            "Return one complete syntactically valid Python module before changing behavior.",
        ),
        (
            "missing_entry_point",
            "not found",
            "Keep the documented candidate entry point with its exact signature.",
        ),
        (
            "runtime_score_admission",
            "score_admission raised",
            "Repair score_admission using only documented fields and return a finite number on "
            "every path.",
        ),
        (
            "runtime_score_eviction",
            "score_eviction raised",
            "Repair score_eviction using only documented fields and return a finite number on "
            "every path.",
        ),
        (
            "timeout",
            "timeout",
            "Remove unbounded or expensive candidate logic so evaluation completes within the "
            "timeout.",
        ),
    )
    entries = [(key, guidance) for key, marker, guidance in patterns if marker in lower]
    if entries:
        return tuple(entries)

    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", lower)
    normalized = re.sub(r"0x[0-9a-f]+", "<address>", normalized)
    return ((f"other:{normalized[:160]}", message[:300]),)


def _is_global_failure_feedback_key(key: str) -> bool:
    """Return whether a failure category is stable enough to reuse across runs."""
    return key != "timeout" and not key.startswith("other:")


def _recurring_levi_failure_feedback() -> list[tuple[int, str]]:
    """Return common deterministic failures from this run and prior scoped runs."""
    counts: dict[str, int] = {}
    guidance_by_key: dict[str, str] = {}
    run_counts, run_guidance = _read_levi_failure_feedback(_levi_failure_feedback_path)
    global_counts, global_guidance = _read_levi_failure_feedback(
        _levi_global_failure_feedback_path,
        scope=_levi_failure_feedback_scope,
        max_events=_levi_global_failure_feedback_max_events,
    )
    for key in run_counts.keys() | global_counts.keys():
        # Global memory includes the current run, so max avoids counting it twice.
        counts[key] = max(run_counts.get(key, 0), global_counts.get(key, 0))
    guidance_by_key.update(global_guidance)
    guidance_by_key.update(run_guidance)
    recurring = [
        (count, guidance_by_key[key])
        for key, count in counts.items()
        if count >= _MIN_RECURRING_FAILURE_COUNT
    ]
    return sorted(recurring, key=lambda item: (-item[0], item[1]))[:_MAX_RECURRING_FAILURES]


def _read_levi_failure_feedback(
    path: Path | None,
    *,
    scope: str | None = None,
    max_events: int | None = None,
) -> tuple[dict[str, int], dict[str, str]]:
    """Read counts from one best-effort JSONL failure store."""
    if path is None or not path.exists():
        return {}, {}
    counts: dict[str, int] = {}
    guidance_by_key: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}, {}
    if max_events is not None:
        lines = list(reversed(lines))
    accepted_events = 0
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if scope is not None and entry.get("scope") != scope:
            continue
        key = entry.get("key")
        guidance = entry.get("guidance")
        if not isinstance(key, str) or not isinstance(guidance, str):
            continue
        counts[key] = counts.get(key, 0) + 1
        guidance_by_key[key] = guidance
        accepted_events += 1
        if max_events is not None and accepted_events >= max_events:
            break
    return counts, guidance_by_key


def _enable_levi_degenerate_centroid_fallback() -> None:
    """Keep a usable CVT archive when valid initialization behaviors are duplicates."""
    from levi.pool.cvt_map_elites import CVTMAPElitesPool

    original = CVTMAPElitesPool.set_centroids_from_data
    if getattr(original, "_prefix_cache_evolve_degenerate_centroid_patch", False):
        return

    def set_centroids_with_fallback(self, behavior_vectors, n_centroids=50):
        data = np.asarray(behavior_vectors, dtype=float)
        actual_n_centroids = min(n_centroids, len(data))
        if not len(data) or len(np.unique(data, axis=0)) >= actual_n_centroids:
            return original(self, behavior_vectors, n_centroids)

        self._n_centroids = n_centroids
        self._centroids = self._init_cvt_centroids()
        self._mins = np.zeros(self._n_dims)
        self._maxs = np.ones(self._n_dims)
        self._ranges = np.ones(self._n_dims)
        distances = np.sum(
            (data[:, np.newaxis, :] - self._centroids[np.newaxis, :, :]) ** 2, axis=2
        )
        return self._n_centroids, np.argmin(distances, axis=1)

    set_centroids_with_fallback._prefix_cache_evolve_degenerate_centroid_patch = True
    CVTMAPElitesPool.set_centroids_from_data = set_centroids_with_fallback


def _configured_reasoning_max_tokens(config: Any) -> int | None:
    """Return the explicit paradigm reasoning budget, falling back to the pipeline budget."""
    punctuated_equilibrium = getattr(config, "punctuated_equilibrium", {}) or {}
    pipeline = getattr(config, "pipeline", {}) or {}
    raw_value = punctuated_equilibrium.get("max_tokens")
    if raw_value is None:
        raw_value = pipeline.get("max_tokens")
    try:
        max_tokens = int(raw_value)
    except (TypeError, ValueError):
        return None
    return max_tokens if max_tokens > 0 else None


def _enable_levi_reasoning_completion_support(max_tokens: int | None) -> None:
    """Prevent Levi's 4096-token PE cap from exhausting reasoning-model output."""
    global _levi_reasoning_max_tokens
    _levi_reasoning_max_tokens = max_tokens
    if max_tokens is None:
        return

    from levi.pipeline.state import PipelineState

    original = PipelineState.acompletion
    if getattr(original, "_prefix_cache_evolve_reasoning_budget_patch", False):
        return

    async def acompletion_with_reasoning_budget(
        self,
        client_spec,
        *,
        prompt,
        temperature=None,
        max_tokens=None,
        timeout=None,
        **extras,
    ):
        reasoning_effort = extras.get("reasoning_effort")
        if reasoning_effort and reasoning_effort != "disabled":
            configured_max = _levi_reasoning_max_tokens
            if configured_max is not None:
                max_tokens = max(configured_max, max_tokens or 0)
        return await original(
            self,
            client_spec,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **extras,
        )

    acompletion_with_reasoning_budget._prefix_cache_evolve_reasoning_budget_patch = True
    acompletion_with_reasoning_budget._prefix_cache_evolve_seed_patch = getattr(
        original,
        "_prefix_cache_evolve_seed_patch",
        False,
    )
    PipelineState.acompletion = acompletion_with_reasoning_budget


def _enable_levi_reproducibility_support(
    seed: int,
    *,
    api_base: str | None = None,
    api_key_env: str | None = None,
) -> None:
    """Seed Levi and apply secret-safe model request defaults."""
    global _levi_api_base
    global _levi_api_key_env
    global _levi_completion_index
    global _levi_search_seed

    _levi_search_seed = int(seed)
    _levi_completion_index = 0
    _levi_api_base = api_base
    _levi_api_key_env = api_key_env
    random.seed(_levi_search_seed)
    np.random.seed(_levi_search_seed)

    from levi.pipeline.state import PipelineState

    original = PipelineState.acompletion
    if getattr(original, "_prefix_cache_evolve_seed_patch", False):
        return

    async def acompletion_with_seed(
        self,
        client_spec,
        *,
        prompt,
        temperature=None,
        max_tokens=None,
        timeout=None,
        **extras,
    ):
        global _levi_completion_index

        request_seed = _levi_search_seed + _levi_completion_index
        _levi_completion_index += 1
        extras.setdefault("seed", request_seed)
        extras.setdefault("drop_params", True)
        if _levi_api_base:
            extras.setdefault("api_base", _levi_api_base)
        if _levi_api_key_env:
            api_key = os.environ.get(_levi_api_key_env)
            if not api_key:
                raise ValueError(
                    f"configured LLM API key environment variable {_levi_api_key_env!r} is not set"
                )
            extras.setdefault("api_key", api_key)
        return await original(
            self,
            client_spec,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **extras,
        )

    acompletion_with_seed._prefix_cache_evolve_seed_patch = True
    acompletion_with_seed._prefix_cache_evolve_reasoning_budget_patch = getattr(
        original,
        "_prefix_cache_evolve_reasoning_budget_patch",
        False,
    )
    PipelineState.acompletion = acompletion_with_seed


def _enable_levi_paradigm_candidate_persistence(output_dir: Path) -> None:
    """Persist every evaluated PE candidate, including archive rejections."""
    global _levi_paradigm_candidate_output_dir
    _levi_paradigm_candidate_output_dir = output_dir

    from levi.equilibrium.equilibrium import PunctuatedEquilibrium

    original_trigger = PunctuatedEquilibrium.trigger
    if getattr(original_trigger, "_prefix_cache_evolve_candidate_persistence_patch", False):
        return
    original_evaluate = PunctuatedEquilibrium._evaluate

    async def evaluate_with_capture(self, code):
        result = await original_evaluate(self, code)
        capture = getattr(self, "_prefix_cache_evolve_candidate_capture", None)
        if isinstance(capture, dict):
            capture.setdefault("candidates", []).append(
                {
                    "code": code,
                    "result": result,
                }
            )
        return result

    async def trigger_with_candidate_persistence(
        self,
        n_evaluations: int,
        budget_progress: float = 0.0,
    ):
        capture = {
            "trigger_evaluation": n_evaluations,
            "budget_progress": budget_progress,
            "candidates": [],
        }
        self._prefix_cache_evolve_candidate_capture = capture
        stats = None
        try:
            stats = await original_trigger(self, n_evaluations, budget_progress)
            return stats
        finally:
            _persist_levi_paradigm_candidate_capture(capture, stats)
            self._prefix_cache_evolve_candidate_capture = None

    trigger_with_candidate_persistence._prefix_cache_evolve_candidate_persistence_patch = True
    PunctuatedEquilibrium._evaluate = evaluate_with_capture
    PunctuatedEquilibrium.trigger = trigger_with_candidate_persistence


def _persist_levi_paradigm_candidate_capture(capture: dict[str, Any], stats: Any) -> None:
    """Write one PE event's evaluated source files and results."""
    output_dir = _levi_paradigm_candidate_output_dir
    if output_dir is None or not capture.get("candidates"):
        return

    event_dir = output_dir / f"eval_{int(capture['trigger_evaluation']):04d}"
    event_dir.mkdir(parents=True, exist_ok=True)
    manifest_candidates = []
    for index, candidate in enumerate(capture["candidates"]):
        candidate_type = "paradigm_shift" if index == 0 else "variant"
        stem = f"{index:02d}_{candidate_type}"
        source_name = f"{stem}.py"
        result_name = f"{stem}_result.json"
        (event_dir / source_name).write_text(str(candidate["code"]), encoding="utf-8")
        _write_levi_json(event_dir / result_name, candidate["result"])
        result = candidate["result"] if isinstance(candidate["result"], dict) else {}
        manifest_candidates.append(
            {
                "candidate_type": candidate_type,
                "source": source_name,
                "result": result_name,
                "score": result.get("score"),
                "error": result.get("error"),
            }
        )

    _write_levi_json(
        event_dir / "manifest.json",
        {
            "trigger_evaluation": capture["trigger_evaluation"],
            "budget_progress": capture["budget_progress"],
            "stats": stats,
            "candidates": manifest_candidates,
        },
    )


def _write_levi_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


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
