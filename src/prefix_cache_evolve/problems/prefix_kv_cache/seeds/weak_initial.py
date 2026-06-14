class WeakHybridPolicy:
    def on_request_start(self, request, now):
        return None

    def on_cache_hit(self, block, request, now):
        return None

    def on_cache_miss(self, block, request, now):
        return None

    def score_admission(self, block, now):
        age = max(0, now - block.created_at)
        shallow_bonus = 1.0 / (1.0 + block.depth)
        reuse_hint = min(4.0, float(block.hit_count + block.descendant_count))
        return 0.35 + shallow_bonus + 0.08 * reuse_hint - 0.002 * age

    def score_eviction(self, block, now):
        age_since_access = max(0, now - block.last_accessed_at)
        reuse_pressure = 2.0 * block.hit_count + 0.75 * block.descendant_count
        recompute_pressure = 0.15 * block.estimated_recompute_cost
        depth_penalty = 0.2 * block.depth
        return age_since_access + depth_penalty - reuse_pressure - recompute_pressure


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return WeakHybridPolicy()


candidate_factory = build_candidate
