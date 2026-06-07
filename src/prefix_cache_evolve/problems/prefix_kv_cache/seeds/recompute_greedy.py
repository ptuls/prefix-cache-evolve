"""Recompute-cost-greedy prefix KV-cache seed."""

from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    baseline_recompute_cost_greedy,
)

candidate_factory = baseline_recompute_cost_greedy
build_candidate = baseline_recompute_cost_greedy
