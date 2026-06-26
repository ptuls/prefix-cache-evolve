"""Build and evaluate candidate policy panels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from prefix_cache_evolve.evaluator_entry import load_candidate_factory, run_with_timeout
from prefix_cache_evolve.evaluators.complexity import scoring_fn_complexity
from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.evaluators.results import EvaluationResult
from prefix_cache_evolve.evaluators.workloads import WorkloadRequest

from .specialist import candidate_evaluator, candidate_exported_names


@dataclass(frozen=True)
class CandidatePanel:
    """A named evaluator split selection."""

    name: str
    splits: tuple[str, ...]


SELECTION_PANEL = CandidatePanel(
    name="selection",
    splits=("train", "validation", "probe"),
)
VALIDATION_PANEL = CandidatePanel(name="validation", splits=("validation",))
PROBE_PANEL = CandidatePanel(name="probe", splits=("probe",))
HIDDEN_PANEL = CandidatePanel(name="hidden", splits=("hidden",))


class CandidateProgramEvaluator(Protocol):
    """Evaluate one candidate program on a requested panel."""

    def __call__(
        self,
        config: EvaluatorConfig,
        candidate_path: Path,
        *,
        splits: tuple[str, ...],
    ) -> EvaluationResult:
        """Evaluate the candidate for the supplied splits."""
        ...


class BaselinePanelBuilder(Protocol):
    """Build baseline results for the current panel."""

    def __call__(self) -> Mapping[str, EvaluationResult]:
        """Return baseline results in report order."""
        ...


class ResultSummarizer(Protocol):
    """Convert an evaluation result to its artifact representation."""

    def __call__(self, result: EvaluationResult) -> dict[str, Any]:
        """Return the stable artifact summary for a result."""
        ...


class ComplexityEvaluator(Protocol):
    """Measure candidate source complexity."""

    def __call__(self, source: str, *, form_aware: bool = False) -> int:
        """Return complexity for candidate source."""
        ...


@dataclass(frozen=True)
class CandidatePanelBuilder:
    """Orchestrate candidate evaluation and panel artifact construction."""

    evaluate_program: CandidateProgramEvaluator
    summarize_result: ResultSummarizer
    evaluate_complexity: ComplexityEvaluator = scoring_fn_complexity

    def evaluate(
        self,
        config: EvaluatorConfig,
        candidate_path: Path,
        *,
        panel: CandidatePanel = SELECTION_PANEL,
    ) -> EvaluationResult:
        """Evaluate a candidate on one named panel."""
        return self.evaluate_program(
            config,
            candidate_path,
            splits=panel.splits,
        )

    def add_candidate(
        self,
        config: EvaluatorConfig,
        candidate_path: Path,
        baseline_results: Mapping[str, EvaluationResult],
        *,
        panel: CandidatePanel = SELECTION_PANEL,
    ) -> dict[str, EvaluationResult]:
        """Prepend a candidate result to baseline results already evaluated."""
        return {
            "candidate": self.evaluate(config, candidate_path, panel=panel),
            **baseline_results,
        }

    def build_comparison(
        self,
        config: EvaluatorConfig,
        candidate_path: Path,
        baseline_builder: BaselinePanelBuilder,
        *,
        panel: CandidatePanel = SELECTION_PANEL,
    ) -> dict[str, EvaluationResult]:
        """Evaluate a candidate, then build and append baseline results."""
        return {
            "candidate": self.evaluate(config, candidate_path, panel=panel),
            **baseline_builder(),
        }

    def build_decomposition(
        self,
        config: EvaluatorConfig,
        candidate_path: Path,
    ) -> dict[str, Any]:
        """Build the selection, probe, and hidden artifact for one candidate."""
        source = candidate_path.read_text(encoding="utf-8")
        raw_complexity = self.evaluate_complexity(source)
        effective_complexity = self.evaluate_complexity(
            source,
            form_aware=config.form_aware_complexity,
        )
        panels = (
            SELECTION_PANEL,
            PROBE_PANEL,
            HIDDEN_PANEL,
        )
        return {
            "candidate": str(candidate_path),
            "raw_complexity": raw_complexity,
            "effective_complexity": effective_complexity,
            "primitive_subsidy_nodes": raw_complexity - effective_complexity,
            "primitive_subsidy_exercised": effective_complexity < raw_complexity,
            **{
                panel.name: self.summarize_result(
                    self.evaluate(config, candidate_path, panel=panel)
                )
                for panel in panels
            },
        }


def evaluate_candidate_program(
    config: EvaluatorConfig,
    candidate_path: Path,
    *,
    splits: tuple[str, ...] = SELECTION_PANEL.splits,
) -> EvaluationResult:
    """Evaluate a candidate program in an isolated worker."""
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
        memory_limit_bytes=config.max_memory_bytes,
        cpu_limit_seconds=config.timeout_s,
    )


def _evaluate_candidate_program_in_worker(
    config: EvaluatorConfig,
    candidate_path: Path,
    splits: tuple[str, ...],
    complexity: int,
) -> EvaluationResult:
    """Load and evaluate a candidate inside the isolated worker."""
    candidate_factory = cast(
        Any,
        load_candidate_factory(
            str(candidate_path),
            exported_names=candidate_exported_names(config),
        ),
    )
    return candidate_evaluator(config, splits=splits)(
        candidate_factory,
        scoring_fn_complexity=complexity,
    )


def evaluate_replay_candidate_program(
    config: EvaluatorConfig,
    candidate_path: Path,
    requests: tuple[WorkloadRequest, ...],
) -> EvaluationResult:
    """Evaluate a candidate against a fixed replay request panel."""
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
        memory_limit_bytes=config.max_memory_bytes,
        cpu_limit_seconds=config.timeout_s,
    )


def _evaluate_replay_candidate_program_in_worker(
    config: EvaluatorConfig,
    candidate_path: Path,
    requests: tuple[WorkloadRequest, ...],
    complexity: int,
) -> EvaluationResult:
    """Load and evaluate a candidate against fixed requests in a worker."""
    candidate_factory = cast(
        Any,
        load_candidate_factory(
            str(candidate_path),
            exported_names=candidate_exported_names(config),
        ),
    )
    return candidate_evaluator(config, splits=VALIDATION_PANEL.splits).evaluate_requests(
        candidate_factory,
        requests,
        scoring_fn_complexity=complexity,
    )
