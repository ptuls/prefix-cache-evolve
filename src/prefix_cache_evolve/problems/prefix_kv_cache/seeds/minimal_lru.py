class LRUPolicy:
    def on_request_start(self, request, now):
        return None

    def on_cache_hit(self, block, request, now):
        return None

    def on_cache_miss(self, block, request, now):
        return None

    def score_admission(self, block, now):
        return 1.0

    def score_eviction(self, block, now):
        return float(now - block.last_accessed_at)


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return LRUPolicy()


candidate_factory = build_candidate
