"""Pressure-aware prefix KV-cache policy with recurrence-gated depth relief."""

import math

DEFAULT_FREQUENCY_HALF_LIFE = 12.0
DEFAULT_PRIORITY_HALF_LIFE = 1.5
DEFAULT_PRESSURE_HALF_LIFE = 4.0


class CompactReusePolicy:
    """Tracks time-decayed observed reuse, priority, and admission pressure."""

    def __init__(
        self,
        capacity_blocks,
        block_size_tokens,
        seed=None,
    ):
        self._state = {}
        self._admission_pressure = 0.0

    def on_request_start(self, request, now):
        pressure = max(0.0, request.recent_admission_pressure)
        misses = max(0.0, request.recent_miss_rate)
        burst = pressure * (0.75 + 0.5 * min(1.0, misses))
        self._admission_pressure = min(
            1.5,
            self._admission_pressure * 2.0 ** (-1.0 / DEFAULT_PRESSURE_HALF_LIFE) + burst,
        )

    def on_cache_hit(self, block, request, now):
        self._observe(block, 2.5, max(0, request.priority), now)

    def on_cache_miss(self, block, request, now):
        self._observe(block, 1.0, 0.0, now)

    def score_admission(self, block, now):
        frequency, priority = self._values(block.prefix_hash, now)
        structure = 0.35 * math.log1p(block.descendant_count)
        reuse = math.log1p(frequency)
        if block.last_access_gap is not None:
            reuse += 0.18 * math.log1p(max(0.0, block.last_access_gap))
        pressure_penalty = (
            self._admission_pressure
            * (0.55 + 0.12 * max(0, block.depth - 1))
            / (1.0 + reuse + 0.35 * priority)
        )
        value = (
            0.95 * reuse
            + 0.28 * priority
            + structure
            + 1.5 * math.log1p(block.estimated_recompute_cost / 96.0)
        )
        depth_penalty = 0.2 * max(0, block.depth - 4) + 0.24 * max(0, block.depth - 2)
        persistent_pressure_penalty = 0.22 * max(0.0, self._admission_pressure - 0.8)
        return (
            value
            - 0.7
            - depth_penalty / max(1.0, 0.75 + 0.25 * frequency)
            - pressure_penalty
            - persistent_pressure_penalty
        )

    def score_eviction(self, block, now):
        frequency, priority = self._values(block.prefix_hash, now)
        return (
            0.85 * math.log1p(max(0, now - block.last_accessed_at))
            - 1.8 * math.log1p(frequency + block.hit_count)
            - 0.2 * math.log1p(block.descendant_count)
            - 0.55 * priority
        )

    def _observe(self, block, weight, priority_value, now):
        key = block.prefix_hash
        frequency, priority = self._values(key, now)
        self._state[key] = (
            frequency + weight,
            max(priority, priority_value),
            now,
        )

    def _values(self, key, now):
        frequency, priority, observed_at = self._state.get(key, (0.0, 0.0, now))
        elapsed = max(0, now - observed_at)
        frequency *= 2.0 ** (-elapsed / DEFAULT_FREQUENCY_HALF_LIFE)
        priority *= 2.0 ** (-elapsed / DEFAULT_PRIORITY_HALF_LIFE)
        self._state[key] = (frequency, priority, now)
        return frequency, priority


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return CompactReusePolicy(capacity_blocks, block_size_tokens, seed)


candidate_factory = build_candidate
