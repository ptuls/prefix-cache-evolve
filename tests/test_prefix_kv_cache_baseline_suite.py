"""Tests for baseline-suite evaluation orchestration."""

from __future__ import annotations

from types import SimpleNamespace

from prefix_cache_evolve.evaluators.baseline_suite import BaselineSuiteEvaluator
from prefix_cache_evolve.evaluators.baselines import baseline_lru_blocks
from prefix_cache_evolve.evaluators.prefix_kv_cache import EvaluatorConfig


def test_baseline_suite_configures_each_evaluator_from_capabilities() -> None:
    calls = []

    class Capabilities:
        def requires_future_reuse(self, name):
            return name == "oracle"

    class Evaluator:
        def __init__(self, config, *, splits, expose_future_reuse):
            calls.append((splits, expose_future_reuse))

        def __call__(self, factory):
            return SimpleNamespace(factory=factory)

    suite = BaselineSuiteEvaluator(
        capabilities=Capabilities(),
        evaluator_factory=Evaluator,
    )

    results = suite.evaluate(
        EvaluatorConfig(),
        {"lru": baseline_lru_blocks, "oracle": baseline_lru_blocks},
        splits=("validation",),
    )

    assert set(results) == {"lru", "oracle"}
    assert calls == [(("validation",), False), (("validation",), True)]
