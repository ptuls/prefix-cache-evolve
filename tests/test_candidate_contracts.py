"""Tests for candidate module contracts."""

import pytest

from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.problems.prefix_kv_cache.candidate_contracts import (
    EVICTION_ONLY_CONTRACT,
    FULL_POLICY_CONTRACT,
    candidate_contract,
)
from prefix_cache_evolve.problems.prefix_kv_cache.specialist import (
    candidate_exported_names,
)


@pytest.mark.parametrize(
    ("surface", "expected"),
    (
        ("full", FULL_POLICY_CONTRACT),
        ("eviction_only", EVICTION_ONLY_CONTRACT),
    ),
)
def test_candidate_contract_is_the_export_source_of_truth(surface, expected) -> None:
    config = EvaluatorConfig(candidate_policy_surface=surface)

    assert candidate_contract(config) is expected
    assert candidate_exported_names(config) == expected.exported_names


def test_candidate_contract_rejects_unknown_surface() -> None:
    config = EvaluatorConfig.model_construct(candidate_policy_surface="unknown")

    with pytest.raises(ValueError, match="unknown candidate_policy_surface"):
        candidate_contract(config)
