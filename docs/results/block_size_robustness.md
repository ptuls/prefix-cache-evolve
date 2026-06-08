# Prefix KV-Cache Block-Size Robustness

Candidate: `src/prefix_cache_evolve/problems/prefix_kv_cache/pressure_aware_incumbent.py`

Each block size replays identical synthetic token streams and preserves the same cache-capacity tiers in tokens. The production-oriented primary setting is `16` tokens per block.

Canonical workload token granularity: `8`. Capacity tiers: `(384, 768)` tokens.

## Candidate Summary

| Block size | Candidate score | Raw before complexity | Complexity cost | Rank | Best policy | Gap to best | Validation token hit | Churn per 1k |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 8 | 77.230 | 85.462 | 8.232 | 1 / 4 | `candidate` | 0.000 | 0.669 | 92.7 |
| 16 | 62.757 | 70.989 | 8.232 | 2 / 4 | `tinylfu_lru` | -0.791 | 0.598 | 168.1 |
| 32 | -20.982 | -12.750 | 8.232 | 1 / 4 | `candidate` | 0.000 | 0.382 | 152.8 |

## Detailed Results

| Block size | Capacity blocks | Capacity tokens | Policy | Score | Raw before complexity | Complexity cost | Validation token hit | Validation block hit | Worst-quarter hit | Policy underfill | Churn per 1k |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 48 / 96 | 384 / 768 | `candidate` | 77.230 | 85.462 | 8.232 | 0.669 | 0.625 | 0.521 | 0.070 | 92.7 |
| 8 | 48 / 96 | 384 / 768 | `vllm_apc` | 60.178 | 60.178 | 0.000 | 0.637 | 0.573 | 0.496 | 0.056 | 807.9 |
| 8 | 48 / 96 | 384 / 768 | `tinylfu_lru` | 70.362 | 70.362 | 0.000 | 0.644 | 0.603 | 0.457 | 0.112 | 161.1 |
| 8 | 48 / 96 | 384 / 768 | `lru` | 51.186 | 51.186 | 0.000 | 0.662 | 0.623 | 0.503 | 0.000 | 1385.0 |
| 16 | 24 / 48 | 384 / 768 | `candidate` | 62.757 | 70.989 | 8.232 | 0.598 | 0.557 | 0.463 | 0.027 | 168.1 |
| 16 | 24 / 48 | 384 / 768 | `vllm_apc` | 46.427 | 46.427 | 0.000 | 0.529 | 0.449 | 0.431 | 0.173 | 313.1 |
| 16 | 24 / 48 | 384 / 768 | `tinylfu_lru` | 63.548 | 63.548 | 0.000 | 0.580 | 0.535 | 0.432 | 0.027 | 499.0 |
| 16 | 24 / 48 | 384 / 768 | `lru` | 49.499 | 49.499 | 0.000 | 0.569 | 0.526 | 0.431 | 0.000 | 999.2 |
| 32 | 12 / 24 | 384 / 768 | `candidate` | -20.982 | -12.750 | 8.232 | 0.382 | 0.353 | 0.216 | 0.029 | 152.8 |
| 32 | 12 / 24 | 384 / 768 | `vllm_apc` | -25.095 | -25.095 | 0.000 | 0.257 | 0.173 | 0.175 | 0.449 | 160.9 |
| 32 | 12 / 24 | 384 / 768 | `tinylfu_lru` | -25.044 | -25.044 | 0.000 | 0.324 | 0.301 | 0.170 | 0.002 | 814.9 |
| 32 | 12 / 24 | 384 / 768 | `lru` | -26.667 | -26.667 | 0.000 | 0.320 | 0.298 | 0.165 | 0.000 | 881.7 |
