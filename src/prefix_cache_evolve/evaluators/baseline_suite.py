"""Evaluation orchestration for registered baseline suites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

from prefix_cache_evolve.evaluators.baselines import (
    BASELINE_REGISTRY,
    PolicyFactory,
)
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluationResult,
    EvaluatorConfig,
    PrefixKVCacheEvaluator,
    WorkloadRequest,
)


class BaselineCapabilities(Protocol):
    """Capabilities needed to configure one baseline evaluation."""

    def requires_future_reuse(self, name: str) -> bool:
        """Return whether the named baseline requires future-reuse metadata."""
        ...


@dataclass(frozen=True)
class BaselineSuiteEvaluator:
    """Evaluates baseline mappings without coupling callers to simulator setup."""

    capabilities: BaselineCapabilities = BASELINE_REGISTRY
    evaluator_factory: Callable[..., PrefixKVCacheEvaluator] = PrefixKVCacheEvaluator

    def evaluate(
        self,
        config: EvaluatorConfig,
        baselines: Mapping[str, PolicyFactory],
        *,
        splits: tuple[str, ...],
    ) -> dict[str, EvaluationResult]:
        """Evaluate each baseline on generated workloads."""
        return {
            name: self._evaluator(config, name, splits=splits)(factory)
            for name, factory in baselines.items()
        }

    def evaluate_requests(
        self,
        config: EvaluatorConfig,
        baselines: Mapping[str, PolicyFactory],
        requests: tuple[WorkloadRequest, ...],
        *,
        workload: str = "trace_replay",
        split: str = "validation",
        seed: int = 0,
    ) -> dict[str, EvaluationResult]:
        """Evaluate each baseline on a fixed request sequence."""
        return {
            name: self._evaluator(config, name, splits=(split,)).evaluate_requests(
                factory,
                requests,
                workload=workload,
                split=split,
                seed=seed,
            )
            for name, factory in baselines.items()
        }

    def _evaluator(
        self,
        config: EvaluatorConfig,
        name: str,
        *,
        splits: tuple[str, ...],
    ) -> PrefixKVCacheEvaluator:
        return self.evaluator_factory(
            config,
            splits=splits,
            expose_future_reuse=self.capabilities.requires_future_reuse(name),
        )


BASELINE_SUITE_EVALUATOR = BaselineSuiteEvaluator()
