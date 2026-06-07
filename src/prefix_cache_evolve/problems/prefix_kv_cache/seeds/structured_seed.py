"""Selected compact structured prefix KV-cache evolution seed."""

import math

from prefix_cache_evolve.problems.prefix_kv_cache.primitives import MultiTimescaleDecay


class StructuredSeedPolicy:
    """Uses bounded reuse and miss decay with subtree and regime context."""

    def __init__(self, capacity_blocks, block_size_tokens, seed=None):
        self._block_size_tokens = max(1, block_size_tokens)
        self._state = MultiTimescaleDecay(
            (4.0, 20.0, 12.0),
            max_keys=max(64, 16 * capacity_blocks),
        )
        self._priority = 0.0
        self._pressure = 0.0
        self._miss_rate = 0.0

    def on_request_start(self, request, now):
        self._priority = max(0.0, float(request.priority))
        self._pressure = float(request.recent_admission_pressure or 0.0)
        self._miss_rate = float(request.recent_miss_rate or 0.0)

    def on_cache_hit(self, block, request, now):
        self._state.observe_vector(block.prefix_hash, (2.0, 1.0, 0.0), now)

    def on_cache_miss(self, block, request, now):
        self._state.observe_vector(block.prefix_hash, (0.7, 0.4, 0.6), now)

    def score_admission(self, block, now):
        fast, slow, _ = self._state.values(block.prefix_hash, now)
        structure = 0.15 * math.log1p(block.descendant_count)
        structure += 0.08 * math.log1p(block.subtree_active_ref_count + block.active_ref_count)
        return (
            0.52 * math.log1p(fast + 0.6 * slow)
            + 0.18 * math.log1p(block.estimated_recompute_cost / 64.0)
            + 0.12 * math.log1p(1.0 + block.subtree_hit_rate)
            + structure
            + 0.12 * self._priority
            - 0.1 * block.depth
            - 0.08 * math.log1p(max(1.0, block.token_count) / self._block_size_tokens)
            - 0.2 * self._pressure
            - 0.12 * self._miss_rate
            - 0.12
        )

    def score_eviction(self, block, now):
        fast, slow, misses = self._state.values(block.prefix_hash, now)
        return (
            0.92 * math.log1p(max(0, now - block.last_accessed_at))
            - math.log1p(fast + 0.6 * slow)
            - 0.18
            * math.log1p(
                block.descendant_count + block.subtree_active_ref_count + block.active_ref_count
            )
            - 0.28 * math.log1p(block.estimated_recompute_cost / 64.0)
            + 0.24 * misses
            - 0.08 * block.depth
        )


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return StructuredSeedPolicy(capacity_blocks, block_size_tokens, seed)


candidate_factory = build_candidate
