"""Tests for prefix KV-cache baseline infrastructure."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from prefix_cache_evolve.evaluators.baseline_suite import BaselineSuiteEvaluator
from prefix_cache_evolve.evaluators.baselines import (
    BASELINE_REGISTRY,
    BaselineRegistry,
    BaselineSpec,
    baseline_lru_blocks,
    baseline_sglang_radix_attention,
)
from prefix_cache_evolve.evaluators.contracts import PrefixBlockInfo
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluatorConfig,
)
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    PrefixBlockInfo as CompatiblePrefixBlockInfo,
)


def test_baseline_registry_separates_deployable_and_reporting_policies() -> None:
    deployable = BASELINE_REGISTRY.factories()
    reporting = BASELINE_REGISTRY.factories(include_reporting=True)
    comparison = BASELINE_REGISTRY.factories(comparison_only=True)

    assert "lru" in deployable
    assert "sglang_radix_attention" in deployable
    assert "sglang_radix_attention" not in comparison
    assert "oracle_future_reuse" not in deployable
    assert "oracle_future_reuse" in reporting
    assert BASELINE_REGISTRY.group("oracle_future_reuse") == ("reporting-only/future-knowledge")
    assert BASELINE_REGISTRY.requires_future_reuse("oracle_future_reuse") is True
    assert BASELINE_REGISTRY.group("candidate") == "deployable"


def test_baseline_registry_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate baseline"):
        BaselineRegistry(
            (
                BaselineSpec("lru", baseline_lru_blocks),
                BaselineSpec("lru", baseline_lru_blocks),
            )
        )


def test_baseline_registry_rejects_deployable_future_knowledge() -> None:
    with pytest.raises(
        ValueError, match="deployable baseline 'oracle' cannot require future reuse"
    ):
        BaselineRegistry(
            (
                BaselineSpec(
                    "oracle",
                    baseline_lru_blocks,
                    deployable=True,
                    requires_future_reuse=True,
                ),
            )
        )


def test_sglang_radix_attention_matches_leaf_lru_contract() -> None:
    radix = baseline_sglang_radix_attention(8, 4)
    lru = baseline_lru_blocks(8, 4)
    block = PrefixBlockInfo(
        block_id=1,
        prefix_hash=1,
        parent_hash=None,
        depth=1,
        start_token=0,
        end_token=3,
        token_count=3,
        tenant_id=0,
        created_at=0,
        last_accessed_at=2,
        hit_count=0,
        descendant_count=0,
        active_ref_count=0,
        estimated_recompute_cost=3.0,
    )

    assert radix.score_admission(block, now=10) > 0.0
    assert radix.score_admission(block, now=10) == lru.score_admission(block, now=10)
    assert radix.score_eviction(block, now=10) == lru.score_eviction(block, now=10)


def test_evaluator_reexports_candidate_visible_contracts() -> None:
    assert CompatiblePrefixBlockInfo is PrefixBlockInfo


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
