"""LRU prefix KV-cache seed."""

from prefix_cache_evolve.evaluators.prefix_kv_cache import baseline_lru_blocks

candidate_factory = baseline_lru_blocks
build_candidate = baseline_lru_blocks
