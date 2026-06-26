"""Scoped compatibility adapters for the pinned Levi dependency."""

from __future__ import annotations

import inspect
import os
import random
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any

from prefix_cache_evolve.artifacts import write_json

_LEGACY_PARADIGM_MAX_TOKENS = 4_096


@dataclass(frozen=True)
class LeviRuntimeSettings:
    """Per-run settings consumed by installed Levi compatibility adapters."""

    search_seed: int
    paradigm_max_tokens: int | None = None
    paradigm_model_names: frozenset[str] = frozenset()
    api_base: str | None = None
    api_key_env: str | None = None
    paradigm_candidate_output_dir: Path | None = None


@dataclass
class _LeviRuntimeState:
    """Mutable request state isolated to one active Levi run."""

    settings: LeviRuntimeSettings
    completion_indexes: Iterator[int] = field(default_factory=count)

    def next_request_seed(self) -> int:
        return self.settings.search_seed + next(self.completion_indexes)


_ACTIVE_RUNTIME: ContextVar[_LeviRuntimeState | None] = ContextVar(
    "prefix_cache_evolve_levi_runtime",
    default=None,
)
_RUNTIME_LOCK = threading.RLock()


@contextmanager
def activate_levi_runtime(settings: LeviRuntimeSettings) -> Iterator[None]:
    """Install compatibility adapters and activate isolated settings for one run."""
    import numpy as np

    _install_levi_compatibility()
    with _RUNTIME_LOCK:
        python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        random.seed(settings.search_seed)
        np.random.seed(settings.search_seed)
        token = _ACTIVE_RUNTIME.set(_LeviRuntimeState(settings))
        try:
            yield
        finally:
            _ACTIVE_RUNTIME.reset(token)
            random.setstate(python_random_state)
            np.random.set_state(numpy_random_state)


def persist_paradigm_candidate_capture(
    capture: dict[str, Any],
    stats: Any,
    *,
    output_dir: Path,
) -> None:
    """Write one punctuated-equilibrium event's candidate sources and results."""
    if not capture.get("candidates"):
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
        write_json(event_dir / result_name, candidate["result"])
        result = candidate["result"] if isinstance(candidate["result"], dict) else {}
        manifest_candidates.append(
            {
                "candidate_type": candidate_type,
                "source": source_name,
                "result": result_name,
                "score": result.get("score"),
                "verifier_version": result.get("verifier_version"),
                "evaluation_context_sha256": result.get("evaluation_context_sha256"),
                "panel_sha256": result.get("panel_sha256"),
                "error": result.get("error"),
            }
        )

    write_json(
        event_dir / "manifest.json",
        {
            "trigger_evaluation": capture["trigger_evaluation"],
            "budget_progress": capture["budget_progress"],
            "stats": stats,
            "candidates": manifest_candidates,
        },
    )


def _install_levi_compatibility() -> None:
    _install_code_feedback_adapter()
    _install_degenerate_centroid_adapter()
    _install_completion_adapter()
    _install_paradigm_candidate_adapter()


def _install_code_feedback_adapter() -> None:
    """Make older Levi code adapters accept evaluator failure feedback."""
    from levi.artifacts.code import CodeAdapter

    original = CodeAdapter.build_mutation_prompt
    if getattr(original, "_prefix_cache_evolve_repair_feedback_patch", False):
        return
    accepts_feedback = "feedback" in inspect.signature(original).parameters

    def build_mutation_prompt_with_feedback(
        self,
        parents,
        *,
        feedback: list[str] | None = None,
        **kwargs,
    ) -> str:
        if _ACTIVE_RUNTIME.get() is None:
            if accepts_feedback:
                return original(self, parents, feedback=feedback, **kwargs)
            return original(self, parents, **kwargs)
        if accepts_feedback:
            prompt = original(self, parents, feedback=feedback, **kwargs)
        else:
            prompt = original(self, parents, **kwargs)
        sections = []
        feedback_lines = [line.strip() for line in feedback or [] if line.strip()]
        if feedback_lines:
            sections.append(
                "## Evaluator Feedback\n"
                "### Failure Cases With Measurements\n"
                "Each item is an aggregate from one evaluated workload, not an isolated "
                "anecdote. Use its outcome, admission-regret, eviction-regret, and cache-"
                "economics measurements to select one causal change. Do not optimize every "
                "metric at once or infer access to quarantined probe results.\n"
                + "\n".join(f"- {line}" for line in feedback_lines)
            )
        sections.append(
            "## Preflight Repair Checks\n"
            "- Keep the documented entry point and produce complete valid Python.\n"
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

    setattr(
        build_mutation_prompt_with_feedback,
        "_prefix_cache_evolve_repair_feedback_patch",
        True,
    )
    CodeAdapter.build_mutation_prompt = build_mutation_prompt_with_feedback


def _install_degenerate_centroid_adapter() -> None:
    """Keep a usable CVT archive when initialization behaviors are duplicates."""
    import numpy as np
    from levi.pool.cvt_map_elites import CVTMAPElitesPool

    original = CVTMAPElitesPool.set_centroids_from_data
    if getattr(original, "_prefix_cache_evolve_degenerate_centroid_patch", False):
        return

    def set_centroids_with_fallback(self, behavior_vectors, n_centroids=50):
        if _ACTIVE_RUNTIME.get() is None:
            return original(self, behavior_vectors, n_centroids)
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
            (data[:, np.newaxis, :] - self._centroids[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        return self._n_centroids, np.argmin(distances, axis=1)

    setattr(
        set_centroids_with_fallback,
        "_prefix_cache_evolve_degenerate_centroid_patch",
        True,
    )
    CVTMAPElitesPool.set_centroids_from_data = set_centroids_with_fallback


def _install_completion_adapter() -> None:
    """Apply active run settings to Levi model requests."""
    from levi.clients.base import client_name
    from levi.pipeline.state import PipelineState

    original = PipelineState.acompletion
    if getattr(original, "_prefix_cache_evolve_runtime_patch", False):
        return

    async def acompletion_with_runtime(
        self,
        client_spec,
        *,
        prompt,
        temperature=None,
        max_tokens=None,
        timeout=None,
        **extras,
    ):
        runtime = _ACTIVE_RUNTIME.get()
        if runtime is not None:
            settings = runtime.settings
            is_paradigm_model = client_name(client_spec) in settings.paradigm_model_names
            if (
                max_tokens == _LEGACY_PARADIGM_MAX_TOKENS
                and is_paradigm_model
                and settings.paradigm_max_tokens is not None
            ):
                max_tokens = settings.paradigm_max_tokens
            extras.setdefault("seed", runtime.next_request_seed())
            extras.setdefault("drop_params", True)
            if settings.api_base:
                extras.setdefault("api_base", settings.api_base)
            if settings.api_key_env:
                api_key = os.environ.get(settings.api_key_env)
                if not api_key:
                    raise ValueError(
                        "configured LLM API key environment variable "
                        f"{settings.api_key_env!r} is not set"
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

    setattr(acompletion_with_runtime, "_prefix_cache_evolve_runtime_patch", True)
    PipelineState.acompletion = acompletion_with_runtime


def _install_paradigm_candidate_adapter() -> None:
    """Persist every evaluated paradigm candidate, including archive rejections."""
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
            runtime = _ACTIVE_RUNTIME.get()
            output_dir = (
                runtime.settings.paradigm_candidate_output_dir if runtime is not None else None
            )
            if output_dir is not None:
                persist_paradigm_candidate_capture(
                    capture,
                    stats,
                    output_dir=output_dir,
                )
            self._prefix_cache_evolve_candidate_capture = None

    setattr(
        trigger_with_candidate_persistence,
        "_prefix_cache_evolve_candidate_persistence_patch",
        True,
    )
    PunctuatedEquilibrium._evaluate = evaluate_with_capture
    PunctuatedEquilibrium.trigger = trigger_with_candidate_persistence
