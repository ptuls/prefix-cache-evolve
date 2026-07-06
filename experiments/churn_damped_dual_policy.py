"""Churn-damped explicit-dual prefix KV-cache policy prototype.

Extends the explicit primal-dual policy (one value index versus one scalar
shadow price calibrated toward the marginal evicted victim) with two churn
controls, because the undamped dual tracked the oracle price well but evicted
~5x too often:

  * A hysteresis dead-band on the dual update: the price only moves when the
    marginal-victim signal departs from it by more than a band, so it ignores
    the fast micro-movements that flip borderline admission decisions and thrash
    the cache. The price acts on large oracle moves, not small ones.
  * A recent-eviction guard: the block displaced in each eviction round is
    recorded in a fast-decaying channel, and a block that was just evicted must
    clear a higher admission bar to return. This suppresses the
    admit/evict/re-admit cycle without blocking genuine reuse.

The goal is to keep the binding-regime shadow-price tracking gains while pulling
churn back toward the incumbent's level. Coefficients are reasonable but untuned.
"""

import math

from prefix_cache_evolve.problems.prefix_kv_cache.primitives import MultiTimescaleDecay

_RECOMPUTE_SCALE = 96.0
_PRICE_CALIBRATION = 0.5
_PRICE_DEADBAND = 0.25
_PRICE_RELAXATION = 0.97
_READMIT_PENALTY = 1.5


class ChurnDampedDualPolicy:
    def __init__(self, capacity_blocks, block_size_tokens, seed=None):
        self._evidence = MultiTimescaleDecay((12.0, 1.5, 3.5))
        self._evicted = MultiTimescaleDecay((2.0,))
        self._price = 0.0
        self._round_now = None
        self._round_min = math.inf
        self._round_victim = None

    def on_request_start(self, request, now):
        # Relax toward zero between binding events and floor at reported pressure
        # so a slack cache prices at ~zero.
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
        # Admit iff the block clears the water level, with an extra bar for a
        # block we displaced very recently so it cannot thrash straight back in.
        recently_evicted = self._evicted.values(block.prefix_hash, now)[0]
        return self._value(block, now) - self._price - _READMIT_PENALTY * recently_evicted

    def _finalize_round(self):
        if self._round_min == math.inf:
            return
        # Dead-band: only act on a marginal-victim signal that departs from the
        # current price by more than the band. Small moves are ignored.
        if abs(self._round_min - self._price) > _PRICE_DEADBAND:
            self._price += _PRICE_CALIBRATION * (self._round_min - self._price)
            if self._price < 0.0:
                self._price = 0.0
        self._evicted.observe(self._round_victim, 1.0, self._round_now)

    def score_eviction(self, block, now):
        # Eviction removes the maximum score, i.e. the lowest-value leaf, which is
        # the marginal displaced victim used to calibrate the dual.
        value = self._value(block, now)
        if now != self._round_now:
            self._finalize_round()
            self._round_now = now
            self._round_min = value
            self._round_victim = block.prefix_hash
        elif value < self._round_min:
            self._round_min = value
            self._round_victim = block.prefix_hash
        return self._price - value


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return ChurnDampedDualPolicy(capacity_blocks, block_size_tokens, seed)


candidate_factory = build_candidate
