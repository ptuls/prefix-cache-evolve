"""Construct supported policy factories and compose admission with eviction."""

from __future__ import annotations

import inspect
from typing import Callable

from prefix_cache_evolve.evaluators.contracts import PrefixBlockInfo, PrefixKVPolicy, RequestInfo


def build_policy(
    factory: Callable[..., PrefixKVPolicy],
    capacity_blocks: int,
    block_size_tokens: int,
    seed: int,
) -> PrefixKVPolicy:
    """Call a supported factory signature without swallowing factory exceptions."""
    argument_options = (
        (capacity_blocks, block_size_tokens, seed),
        (capacity_blocks, block_size_tokens),
        (),
    )
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(*argument_options[0])

    for args in argument_options:
        try:
            signature.bind(*args)
        except TypeError:
            continue
        return factory(*args)
    raise TypeError(
        "candidate factory must accept (capacity_blocks, block_size_tokens, seed), "
        "(capacity_blocks, block_size_tokens), or no arguments"
    )


class FixedAdmissionPolicy:
    """Uses one policy for admission and another policy for eviction."""

    _REQUIRED_METHODS = (
        "on_request_start",
        "score_admission",
        "score_eviction",
        "on_cache_hit",
        "on_cache_miss",
    )

    def __init__(
        self,
        *,
        admission_policy: PrefixKVPolicy,
        eviction_policy: PrefixKVPolicy,
    ) -> None:
        for policy in (admission_policy, eviction_policy):
            for method_name in self._REQUIRED_METHODS:
                if not callable(getattr(policy, method_name, None)):
                    raise TypeError(f"policy must implement {method_name}()")
        self._admission_policy = admission_policy
        self._eviction_policy = eviction_policy

    def on_request_start(self, request: RequestInfo, now: int) -> None:
        """Notify both policies of the request."""
        self._admission_policy.on_request_start(request, now)
        self._eviction_policy.on_request_start(request, now)

    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        """Use the fixed admission policy to rank admission."""
        return self._admission_policy.score_admission(block, now)

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        """Use the evolving eviction policy to rank eviction."""
        return self._eviction_policy.score_eviction(block, now)

    def on_cache_hit(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        """Notify both policies of a hit."""
        self._admission_policy.on_cache_hit(block, request, now)
        self._eviction_policy.on_cache_hit(block, request, now)

    def on_cache_miss(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        """Notify both policies of a miss."""
        self._admission_policy.on_cache_miss(block, request, now)
        self._eviction_policy.on_cache_miss(block, request, now)
