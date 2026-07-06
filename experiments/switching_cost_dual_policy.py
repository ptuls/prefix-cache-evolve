"""Production incumbent: explicit shadow-price dual with a switching cost.

A single retention index scores both admission and eviction.  A scalar dual variable
(the admission shadow price) is calibrated online toward the marginal displaced
victim---the cheapest legal eviction candidate in each round---rather than a slow
congestion signal.  Admission accepts a block only when its value clears the shadow
price by a switching margin, which governs cache churn; eviction removes the leaf
furthest below the price.  The formulation is a restless-bandit index with an explicit
switching cost.
"""

import math

from prefix_cache_evolve.problems.prefix_kv_cache.primitives import MultiTimescaleDecay

_RECOMPUTE_SCALE = 96.0
_PRICE_CALIBRATION = 0.5
_PRICE_RELAXATION = 0.97
_SWITCHING_MARGIN = 0.9


class SwitchingCostDualPolicy:
    def __init__(self, capacity_blocks, block_size_tokens, seed=None):
        self._evidence = MultiTimescaleDecay((12.0, 1.5, 3.5))
        self._price = 0.0
        self._round_now = None
        self._round_min = math.inf

    def on_request_start(self, request, now):
        self._price *= _PRICE_RELAXATION
        floor = 0.6 * max(0.0, request.recent_admission_pressure)
        if self._price < floor:
            self._price = floor

    def on_cache_hit(self, block, request, now):
        self._evidence.observe_vector(
            block.prefix_hash,
            (2.0, max(0.0, request.priority), 1.0),
            now,
        )

    def on_cache_miss(self, block, request, now):
        self._evidence.observe_vector(block.prefix_hash, (1.0, 0.0, 0.3), now)

    def _value(self, block, now):
        frequency, priority, recency = self._evidence.values(block.prefix_hash, now)
        reuse = math.log1p(frequency) + 0.2 * math.log1p(recency)
        return (
            reuse
            + 0.28 * priority
            + 0.35 * math.log1p(block.descendant_count)
            + 1.2 * math.log1p(block.estimated_recompute_cost / _RECOMPUTE_SCALE)
        )

    def score_admission(self, block, now):
        return self._value(block, now) - self._price - _SWITCHING_MARGIN

    def score_eviction(self, block, now):
        value = self._value(block, now)
        if now != self._round_now:
            if self._round_min < math.inf:
                self._price += _PRICE_CALIBRATION * (self._round_min - self._price)
                if self._price < 0.0:
                    self._price = 0.0
            self._round_now = now
            self._round_min = value
        elif value < self._round_min:
            self._round_min = value
        return self._price - value


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return SwitchingCostDualPolicy(capacity_blocks, block_size_tokens, seed)


candidate_factory = build_candidate
