import math

from prefix_cache_evolve.problems.prefix_kv_cache.primitives import (
    MultiTimescaleDecay,
    threshold_excess,
)


class CompactReusePolicy:
    def __init__(self, capacity_blocks, block_size_tokens, seed=None):
        self._sessions = MultiTimescaleDecay((6.0, 48.0))
        self._block_size_tokens = max(1.0, float(block_size_tokens))
        self._pressure = 0.0

    def on_request_start(self, request, now):
        pressure = request.recent_admission_pressure
        misses = request.recent_miss_rate
        self._pressure = min(
            2.0,
            0.78 * self._pressure + 0.70 * min(1.0, pressure) + 0.25 * min(1.0, misses),
        )
        self._priority = request.priority

        if request.session_id is None:
            self._session_reuse = 0.0
        else:
            key = (request.tenant_id, request.session_id)
            fast, slow = self._sessions.values(key, now)
            self._session_reuse = min(
                1.5,
                math.log1p(0.7 * fast + 0.3 * slow),
            )
            self._sessions.observe(key, 1.0, now)

    def _on_cache_event(self, block, request, now):
        pass

    on_cache_hit = _on_cache_event
    on_cache_miss = _on_cache_event

    def _block_features(self, block):
        return (
            min(1.0, block.token_count / self._block_size_tokens),
            min(
                3.0,
                block.estimated_recompute_cost / self._block_size_tokens,
            ),
        )

    def score_admission(self, block, now):
        fullness, recompute = self._block_features(block)
        block_reuse = math.log1p(block.hit_count)
        gap_evidence = 0.0

        if block.last_access_gap is not None:
            gap_evidence = 0.48 + 0.20 * min(
                2.0,
                math.log1p(max(0.0, block.last_access_gap)),
            )

        recurrence = block_reuse + self._session_reuse + gap_evidence
        pressure_penalty = (
            self._pressure * (0.85 + 0.08 * math.log1p(block.depth)) / (1.0 + 0.75 * recurrence)
        )
        priority_bonus = 0.18 * threshold_excess(self._priority, 0.5) * min(1.5, recurrence)

        return (
            -0.65
            + 0.55 * fullness
            + 0.60 * math.log1p(recompute)
            + 0.48 * block_reuse
            + 1.25 * self._session_reuse
            + gap_evidence
            + 0.24 * math.log1p(block.descendant_count)
            - 0.16 * math.log1p(block.depth)
            - pressure_penalty
            - 0.16 * threshold_excess(self._pressure, 0.9)
            + priority_bonus
        )

    def score_eviction(self, block, now):
        age = now - block.last_accessed_at
        fullness, recompute = self._block_features(block)
        gap_protection = 0.0

        if block.last_access_gap is not None:
            gap_protection = 0.25 * min(
                2.0,
                math.log1p(max(0.0, block.last_access_gap)),
            )

        return (
            0.90 * math.log1p(age)
            + 0.14 * math.log1p(block.depth)
            - 1.45 * math.log1p(block.hit_count)
            - 0.28 * math.log1p(block.descendant_count)
            - 0.45 * max(0.0, block.subtree_hit_rate)
            - 0.22 * fullness
            - 0.30 * math.log1p(recompute)
            - gap_protection
        )


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return CompactReusePolicy(capacity_blocks, block_size_tokens, seed)


candidate_factory = build_candidate
