"""Function-only eviction specialist seeded from the pressure-aware incumbent."""

import math


def score_eviction(block, now, frequency, priority):
    """Rank an inactive resident leaf for eviction; higher means evict sooner."""

    return (
        0.85 * math.log1p(max(0, now - block.last_accessed_at))
        - 1.8 * math.log1p(frequency + block.hit_count)
        - 0.2 * math.log1p(block.descendant_count)
        - 0.55 * priority
    )
