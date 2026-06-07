"""Compact structured prefix KV-cache policy using canonical decay state."""

import math

from prefix_cache_evolve.problems.prefix_kv_cache.primitives import MultiTimescaleDecay


class StructuredRecurrencePolicy:
    """Combines bounded recurrence, regime, subtree, and decay signals."""

    def __init__(self, capacity_blocks, block_size_tokens, seed=None):
        self._block_size_tokens = max(1, block_size_tokens)
        self._state = MultiTimescaleDecay(
            (4.0, 20.0, 8.0, 12.0),
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
        self._state.observe_vector(
            block.prefix_hash,
            (2.0, 1.0, 0.2 * self._priority, 0.0),
            now,
        )

    def on_cache_miss(self, block, request, now):
        self._state.observe_vector(
            block.prefix_hash,
            (0.7, 0.4, 0.05 * self._priority, 0.6),
            now,
        )

    def score_admission(self, block, now):
        fast, slow, priority, _ = self._state.values(block.prefix_hash, now)
        reuse = math.log1p(fast + 0.6 * slow)
        structure = 0.15 * math.log1p(block.descendant_count)
        structure += 0.08 * math.log1p(
            block.subtree_active_ref_count + block.active_ref_count
        )
        recurrence = 0.1 * math.log1p(1.0 + (block.access_gap_mean or 0.0))
        recurrence -= 0.05 * math.log1p(1.0 + (block.access_gap_var or 0.0))
        return (
            0.52 * reuse
            + 0.1 * priority
            + 0.18 * math.log1p(block.estimated_recompute_cost / 64.0)
            + 0.12 * math.log1p(1.0 + block.subtree_hit_rate)
            + structure
            + recurrence
            + 0.12 * self._priority
            - 0.1 * block.depth
            - 0.08 * math.log1p(max(1.0, block.token_count) / self._block_size_tokens)
            - 0.2 * self._pressure
            - 0.12 * self._miss_rate
            - 0.12
        )

    def score_eviction(self, block, now):
        fast, slow, priority, misses = self._state.values(block.prefix_hash, now)
        reuse = math.log1p(fast + 0.6 * slow)
        recurrence = 0.04 * math.log1p(1.0 + (block.access_gap_mean or 0.0))
        recurrence -= 0.02 * math.log1p(1.0 + (block.access_gap_var or 0.0))
        return (
            0.92 * math.log1p(max(0, now - block.last_accessed_at))
            - reuse
            - 0.18
            * math.log1p(
                block.descendant_count
                + block.subtree_active_ref_count
                + block.active_ref_count
            )
            - 0.28 * math.log1p(block.estimated_recompute_cost / 64.0)
            - 0.18 * priority
            + 0.24 * misses
            - 0.08 * block.depth
            + recurrence
        )


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return StructuredRecurrencePolicy(capacity_blocks, block_size_tokens, seed)


candidate_factory = build_candidate
