import math

from prefix_cache_evolve.problems.prefix_kv_cache.primitives import (
    MultiTimescaleDecay,
    threshold_excess,
)


class CompactReusePolicy:
    def __init__(self, capacity_blocks, block_size_tokens, seed=None):
        self._state = MultiTimescaleDecay((12.0, 1.5, 3.5))
        self._admission_pressure = 0.0

    def on_request_start(self, request, now):
        pressure = max(0.0, request.recent_admission_pressure)
        misses = max(0.0, request.recent_miss_rate)
        priority = max(0.0, request.priority)
        self._admission_pressure = min(
            1.5,
            self._admission_pressure * 2.0 ** (-1.0 / 4.0)
            + pressure * (0.75 + 0.5 * min(1.0, misses))
            + 0.12 * priority
            - 0.08 * min(1.0, misses),
        )

    def on_cache_hit(self, block, request, now):
        self._state.observe_vector(
            block.prefix_hash,
            (2.5, max(0.0, request.priority), 1.0),
            now,
        )

    def on_cache_miss(self, block, request, now):
        self._state.observe_vector(block.prefix_hash, (1.0, 0.0, 0.35), now)

    def score_admission(self, block, now):
        frequency, priority, recency = self._state.values(block.prefix_hash, now)
        reuse = math.log1p(frequency) + 0.22 * math.log1p(recency)
        if block.last_access_gap is not None:
            reuse += 0.18 * math.log1p(max(0.0, block.last_access_gap))
        value = (
            0.95 * reuse
            + 0.28 * priority
            + 0.35 * math.log1p(block.descendant_count)
            + 1.5 * math.log1p(block.estimated_recompute_cost / 96.0)
        )
        pressure_penalty = (
            self._admission_pressure
            * (0.55 + 0.12 * max(0, block.depth - 1))
            / (1.0 + reuse + 0.35 * priority)
        )
        return (
            value
            - 0.7
            - (0.2 * max(0, block.depth - 4) + 0.24 * max(0, block.depth - 2))
            / max(1.0, 0.75 + 0.25 * frequency)
            - pressure_penalty
            - 0.22 * max(0.0, self._admission_pressure - 0.8)
            - 0.18 * max(0.0, 1.0 - priority) * max(0.0, self._admission_pressure - 0.25)
            - 0.12 * max(0.0, 1.0 - priority) * max(0.0, block.descendant_count - 1)
            + 0.12 * threshold_excess(priority, 0.5) * threshold_excess(frequency, 1.0)
        )

    def score_eviction(self, block, now):
        frequency, priority, recency = self._state.values(block.prefix_hash, now)
        return (
            0.82 * math.log1p(max(0, now - block.last_accessed_at))
            - 1.65 * math.log1p(frequency + block.hit_count)
            - 0.18 * math.log1p(block.descendant_count)
            - 0.5 * priority
            - 0.24 * math.log1p(recency)
        )


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return CompactReusePolicy(capacity_blocks, block_size_tokens, seed)


candidate_factory = build_candidate
