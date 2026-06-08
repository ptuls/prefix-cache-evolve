"""Baseline policies and their reporting metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from prefix_cache_evolve.evaluators.contracts import (
    PolicyFactory,
    PrefixBlockInfo,
    PrefixKVPolicy,
    RequestInfo,
)


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    """Defines one baseline and the capabilities needed to evaluate it."""

    name: str
    factory: PolicyFactory
    deployable: bool = True
    requires_future_reuse: bool = False
    include_in_comparison: bool = True

    @property
    def group(self) -> str:
        if self.deployable:
            return "deployable"
        return "reporting-only/future-knowledge"


class BaselineRegistry:
    """Provides one source of truth for baseline factories and capabilities."""

    def __init__(self, specifications: Iterable[BaselineSpec]) -> None:
        self._specifications: dict[str, BaselineSpec] = {}
        for specification in specifications:
            if specification.name in self._specifications:
                raise ValueError(f"duplicate baseline {specification.name!r}")
            if specification.deployable and specification.requires_future_reuse:
                raise ValueError(
                    f"deployable baseline {specification.name!r} cannot require future reuse"
                )
            self._specifications[specification.name] = specification

    def factories(
        self,
        *,
        include_reporting: bool = False,
        comparison_only: bool = False,
    ) -> dict[str, PolicyFactory]:
        """Return baseline factories eligible for the requested evaluation."""

        return {
            name: specification.factory
            for name, specification in self._specifications.items()
            if (include_reporting or specification.deployable)
            and (not comparison_only or specification.include_in_comparison)
        }

    def group(self, name: str) -> str:
        """Return the reporting group for a baseline or deployable candidate."""

        specification = self._specifications.get(name)
        return specification.group if specification is not None else "deployable"

    def requires_future_reuse(self, name: str) -> bool:
        """Return whether a baseline needs simulator-provided future knowledge."""

        specification = self._specifications.get(name)
        return bool(specification and specification.requires_future_reuse)


class _BasePolicy:
    def on_request_start(self, request: RequestInfo, now: int) -> None:
        return None

    def on_cache_hit(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        return None

    def on_cache_miss(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        return None


class _NoCachePolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return -1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return 0.0


class _LRUPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return float(now - block.last_accessed_at)


class _SGLangRadixAttentionPolicy(_BasePolicy):
    """Models SGLang RadixAttention's default radix-cache replacement policy."""

    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        del block, now
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return float(now - block.last_accessed_at)


class _VLLMAPCPolicy(_BasePolicy):
    """Models vLLM automatic prefix caching within the simulator contract."""

    def __init__(self, block_size_tokens: int) -> None:
        self._block_size_tokens = block_size_tokens

    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        del now
        return 1.0 if block.token_count == self._block_size_tokens else -1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        age = max(0, now - block.last_accessed_at)
        depth_tiebreak = float(block.depth) / (float(block.depth) + 1.0)
        return float(age) + depth_tiebreak


def _recency_tiebreak(block: PrefixBlockInfo, now: int) -> float:
    """Returns an LRU tie-break score that cannot cross an integer priority."""

    age = max(0, now - block.last_accessed_at)
    return float(age) / (float(age) + 1.0)


class _LFUPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return float(-block.hit_count) + _recency_tiebreak(block, now)


class _DepthPreferShallowPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return float(block.depth) + _recency_tiebreak(block, now)


class _RecomputeGreedyPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return block.estimated_recompute_cost

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return -block.estimated_recompute_cost


class _CostAwareLRUPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        age = max(0, now - block.last_accessed_at)
        return float(age) / max(1.0, block.estimated_recompute_cost)


class _PrefixFanoutPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return float(-block.descendant_count) + _recency_tiebreak(block, now)


class _PrefixAnchorPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        del now
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        age = max(0.0, float(now - block.last_accessed_at))
        descendant_protection = 3.0 * math.log1p(max(0, block.descendant_count))
        depth_protection = 1.5 / max(1.0, float(block.depth))
        return age - descendant_protection - depth_protection


class _TinyLFULRUPolicy(_BasePolicy):
    def __init__(self) -> None:
        self._frequency: dict[int, int] = {}

    def on_cache_hit(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        del request, now
        self._frequency[block.prefix_hash] = self._frequency.get(block.prefix_hash, 0) + 1

    def on_cache_miss(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        del request, now
        self._frequency[block.prefix_hash] = self._frequency.get(block.prefix_hash, 0) + 1

    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        del now
        frequency = self._frequency.get(block.prefix_hash, 0)
        if block.depth <= 2:
            return 1.0
        return 1.0 if frequency >= 2 else -1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return float(now - block.last_accessed_at)


class _TenantFairLRUPolicy(_BasePolicy):
    def __init__(self) -> None:
        self._tenant_hit_tokens: dict[int, int] = {}
        self._tenant_seen_tokens: dict[int, int] = {}

    def on_cache_hit(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        del request, now
        self._record_observation(block, hit=True)

    def on_cache_miss(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        del request, now
        self._record_observation(block, hit=False)

    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        tenant_hit_rate = self._tenant_hit_rate(block.tenant_id)
        return float(now - block.last_accessed_at) + 8.0 * tenant_hit_rate

    def _record_observation(self, block: PrefixBlockInfo, *, hit: bool) -> None:
        tenant_id = block.tenant_id
        self._tenant_seen_tokens[tenant_id] = (
            self._tenant_seen_tokens.get(tenant_id, 0) + block.token_count
        )
        if hit:
            self._tenant_hit_tokens[tenant_id] = (
                self._tenant_hit_tokens.get(tenant_id, 0) + block.token_count
            )

    def _tenant_hit_rate(self, tenant_id: int) -> float:
        hit_tokens = self._tenant_hit_tokens.get(tenant_id, 0)
        seen_tokens = self._tenant_seen_tokens.get(tenant_id, 0)
        return float(hit_tokens) / max(1.0, float(seen_tokens))


class _FutureReuseHeuristicPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        del now
        future_reuse = block.estimated_future_reuse
        next_distance = block.estimated_next_reuse_distance
        if future_reuse is None or next_distance is None:
            return 0.0
        if future_reuse <= 0.0 or math.isinf(next_distance):
            return 1_000_000.0
        return float(next_distance / (1.0 + future_reuse))


class _OracleFutureReusePolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        del now
        future_reuse = block.estimated_future_reuse
        if future_reuse is None:
            return 1.0
        return 1.0 if future_reuse > 0.0 else -1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        del now
        future_reuse = block.estimated_future_reuse
        next_distance = block.estimated_next_reuse_distance
        if future_reuse is None or next_distance is None:
            return 0.0
        if future_reuse <= 0.0 or math.isinf(next_distance):
            return 1_000_000.0
        return float(next_distance)


def baseline_no_cache(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _NoCachePolicy()


def baseline_lru_blocks(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _LRUPolicy()


def baseline_sglang_radix_attention(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    """Return SGLang's admit-all, zero-reference leaf-LRU radix-cache policy."""

    return _SGLangRadixAttentionPolicy()


def baseline_vllm_apc(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _VLLMAPCPolicy(block_size_tokens)


def baseline_lfu_blocks(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _LFUPolicy()


def baseline_depth_prefer_shallow(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _DepthPreferShallowPolicy()


def baseline_recompute_cost_greedy(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _RecomputeGreedyPolicy()


def baseline_cost_aware_lru(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _CostAwareLRUPolicy()


def baseline_prefix_fanout(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _PrefixFanoutPolicy()


def baseline_prefix_anchor(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _PrefixAnchorPolicy()


def baseline_tinylfu_lru(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _TinyLFULRUPolicy()


def baseline_tenant_fair_lru(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _TenantFairLRUPolicy()


def baseline_future_reuse_heuristic(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _FutureReuseHeuristicPolicy()


def baseline_oracle_future_reuse(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _OracleFutureReusePolicy()


BASELINE_REGISTRY = BaselineRegistry(
    (
        BaselineSpec("no_cache", baseline_no_cache),
        BaselineSpec("lru", baseline_lru_blocks),
        BaselineSpec(
            "sglang_radix_attention",
            baseline_sglang_radix_attention,
            include_in_comparison=False,
        ),
        BaselineSpec("vllm_apc", baseline_vllm_apc),
        BaselineSpec("lfu", baseline_lfu_blocks),
        BaselineSpec("depth_prefer_shallow", baseline_depth_prefer_shallow),
        BaselineSpec("recompute_greedy", baseline_recompute_cost_greedy),
        BaselineSpec("cost_aware_lru", baseline_cost_aware_lru),
        BaselineSpec("prefix_fanout", baseline_prefix_fanout),
        BaselineSpec("prefix_anchor", baseline_prefix_anchor),
        BaselineSpec("tinylfu_lru", baseline_tinylfu_lru),
        BaselineSpec("tenant_fair_lru", baseline_tenant_fair_lru),
        BaselineSpec(
            "future_reuse_heuristic",
            baseline_future_reuse_heuristic,
            deployable=False,
            requires_future_reuse=True,
        ),
        BaselineSpec(
            "oracle_future_reuse",
            baseline_oracle_future_reuse,
            deployable=False,
            requires_future_reuse=True,
        ),
    )
)

ALL_BASELINES = BASELINE_REGISTRY.factories()
ALL_REPORTING_BASELINES = BASELINE_REGISTRY.factories(include_reporting=True)
BASELINES = BASELINE_REGISTRY.factories(comparison_only=True)
REPORTING_BASELINES = BASELINE_REGISTRY.factories(
    include_reporting=True,
    comparison_only=True,
)
