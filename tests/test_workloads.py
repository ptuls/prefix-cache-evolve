"""Tests for explicit synthetic workload metadata."""

from prefix_cache_evolve.evaluators.utilities import prefix_role
from prefix_cache_evolve.evaluators.workloads import build_workload


def test_workload_requests_carry_aligned_prompt_role_metadata() -> None:
    request = build_workload(
        "shared_system_prompt",
        request_count=1,
        block_size_tokens=8,
        seed=1,
    )[0]

    assert len(request.prompt_token_roles) == len(request.prompt_tokens)
    assert prefix_role(request.prompt_token_roles[:8]) == "system"
    assert prefix_role(request.prompt_token_roles[16:24]) == "developer"
    assert prefix_role(request.prompt_token_roles[24:]) == "user"


def test_workload_role_metadata_is_owned_by_each_request() -> None:
    first = build_workload(
        "shared_system_prompt",
        request_count=2,
        block_size_tokens=8,
        seed=1,
    )
    build_workload(
        "adversarial_unique_prompts",
        request_count=4,
        block_size_tokens=8,
        seed=2,
    )

    assert (
        first[0].prompt_token_roles
        == build_workload(
            "shared_system_prompt",
            request_count=2,
            block_size_tokens=8,
            seed=1,
        )[0].prompt_token_roles
    )
