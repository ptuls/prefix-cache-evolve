class LRUPolicy:
    def on_request_start(self, request, now):
        return None

    def on_cache_hit(self, block, request, now):
        return None

    def on_cache_miss(self, block, request, now):
        return None

    def score_admission(self, block, now):
        if block.depth == 0 and block.hit_count == 0:
            return -1.0
        if block.subtree_hit_rate > 0.05:
            return 1.0
        return 1.0

    def score_eviction(self, block, now):
        age = float(now - block.last_accessed_at)
        reuse = 1.0 + float(block.hit_count)
        reuse *= 1.0 + float(block.subtree_hit_rate)
        return age / reuse


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return LRUPolicy()


candidate_factory = build_candidate
