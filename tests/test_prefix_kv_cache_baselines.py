"""Tests for prefix KV-cache baseline registration."""

from __future__ import annotations

import pytest

from prefix_cache_evolve.evaluators.baselines import (
    BASELINE_REGISTRY,
    BaselineRegistry,
    BaselineSpec,
    baseline_lru_blocks,
)
from prefix_cache_evolve.evaluators.contracts import PrefixBlockInfo
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    PrefixBlockInfo as CompatiblePrefixBlockInfo,
)


def test_baseline_registry_separates_deployable_and_reporting_policies() -> None:
    deployable = BASELINE_REGISTRY.factories()
    reporting = BASELINE_REGISTRY.factories(include_reporting=True)

    assert "lru" in deployable
    assert "oracle_future_reuse" not in deployable
    assert "oracle_future_reuse" in reporting
    assert BASELINE_REGISTRY.group("oracle_future_reuse") == (
        "reporting-only/future-knowledge"
    )
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


def test_evaluator_reexports_candidate_visible_contracts() -> None:
    assert CompatiblePrefixBlockInfo is PrefixBlockInfo
