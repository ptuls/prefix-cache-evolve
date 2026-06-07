"""Depth-preferring prefix KV-cache seed."""

from prefix_cache_evolve.evaluators.prefix_kv_cache import baseline_depth_prefer_shallow

candidate_factory = baseline_depth_prefer_shallow
build_candidate = baseline_depth_prefer_shallow
