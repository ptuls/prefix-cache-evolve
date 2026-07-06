"""Explicit primal-dual (shadow-price) prefix KV-cache policy prototype.

A single retention-value index ``V(block)`` is compared against one scalar dual
variable, the shadow price (water level). Admission accepts a block iff its value
exceeds the price; eviction removes the resident leaf furthest below the price.

Unlike the production incumbent, the price is not a load EMA. It is calibrated
online, in the same value units as ``V``, toward the marginal displaced victim:
the cheapest legal eviction candidate offered in each eviction round. That is the
empirical quantity the oracle admission shadow price tracks (mu/w, the value of
the cheapest legal displacement), so the dual moves with the fast per-decision
price changes the incumbent's smoothed congestion signal cannot follow.

This is a mechanism prototype. Coefficients are reasonable but untuned; the point
is to test whether value-calibrated dual tracking converts to lower shadow-price
tracking error and better binding-regime score, not to win a tuned bake-off.
"""

import math

from prefix_cache_evolve.problems.prefix_kv_cache.primitives import MultiTimescaleDecay

_RECOMPUTE_SCALE = 96.0
_PRICE_CALIBRATION = 0.5
_PRICE_RELAXATION = 0.97


class ExplicitDualPolicy:
    def __init__(self, capacity_blocks, block_size_tokens, seed=None):
        self._evidence = MultiTimescaleDecay((12.0, 1.5, 3.5))
        self._price = 0.0
        self._round_now = None
        self._round_min = math.inf

    def on_request_start(self, request, now):
        # Relax the price toward zero between binding events so a transient burst
        # does not pin the water level high once pressure subsides, and floor it
        # at the reported admission pressure so a slack cache prices at ~zero.
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
        # Admit iff the block's value clears the current water level.
        return self._value(block, now) - self._price

    def score_eviction(self, block, now):
        # Eviction removes the maximum score, i.e. the lowest-value leaf. Each
        # eviction round (one arrival step) contributes its cheapest candidate as
        # a sample of the marginal displacement price; pull the dual toward it.
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
    return ExplicitDualPolicy(capacity_blocks, block_size_tokens, seed)


candidate_factory = build_candidate
