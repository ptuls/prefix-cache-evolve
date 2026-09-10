"""Compatibility checks for incremental hashing of long cache prefixes."""

from __future__ import annotations

import random

import pytest

from prefix_cache_evolve.evaluators.contracts import RequestInfo
from prefix_cache_evolve.evaluators.utilities import request_prefix_hashes, stable_hash
from prefix_cache_evolve.evaluators.workloads import WorkloadRequest


@pytest.mark.parametrize("block_size", [1, 2, 8, 16, 512])
@pytest.mark.parametrize("tenant_id", [0, -17, 2**96 + 7])
@pytest.mark.parametrize("info_tokens", [False, True])
def test_incremental_prefix_hashes_match_historical_tuple_representation(
    block_size, tenant_id, info_tokens
):
    random_source = random.Random(71)
    token_pool = tuple(
        random_source.getrandbits(96) * (-1 if index % 3 == 0 else 1)
        for index in range(2 * block_size + 1)
    )
    for count in sorted({0, 1, block_size - 1, block_size, block_size + 1, len(token_pool)}):
        tokens = token_pool[:count]
        request = WorkloadRequest(
            info=RequestInfo(
                request_id=0,
                tenant_id=tenant_id,
                session_id=1,
                prompt_length=count,
                priority=0,
                request_type="hash_compatibility",
                prompt_tokens=tokens if info_tokens else (),
            ),
            true_output_length=0,
            prompt_tokens=() if info_tokens else tokens,
        )
        expected = [
            stable_hash((tenant_id, tokens[: start + block_size]))
            for start in range(0, count, block_size)
        ]
        assert request_prefix_hashes(request, block_size) == expected
