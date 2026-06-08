# Reasoning Decode-KV Robustness

This panel keeps the operative prefix-only verifier unchanged and replays existing
algorithms with an opt-in shared-capacity model. In shared mode, generated decode KV
grows over logical time, is non-evictable, and forces inactive prefix-leaf eviction.
When pinned prefixes plus decode KV exhaust capacity, the simulator records failed
decode-block allocations.

Panel: capacities `[24, 48]`, block size `16`, `96` requests per workload,
seeds `[11, 23, 37]`, and workloads `['concurrent_long_generation', 'reasoning_burst', 'reasoning_burst_shifted', 'stochastic_serving_mix']`.

Decode allocation failure is reported but is not yet a score term. Raw-score ranking
therefore measures prefix-policy quality under pressure, not end-to-end serving
feasibility.

## Prefix Only

| Rank | Policy | Raw score | Token hit | Request p10 | Churn/1k | Prefix KV | Decode KV | Decode fail | Decode-pressure evictions |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `oracle_future_reuse` | 128.258 | 0.7466 | 0.3210 | 45.6 | 23.6 | 0.0 | 0.0% | 0.0 |
| 2 | `future_reuse_heuristic` | 115.063 | 0.7387 | 0.3210 | 336.4 | 28.1 | 0.0 | 0.0% | 0.0 |
| 3 | `incumbent` | 113.719 | 0.7145 | 0.3284 | 140.2 | 28.1 | 0.0 | 0.0% | 0.0 |
| 4 | `recompute_greedy` | 109.078 | 0.7300 | 0.3233 | 363.7 | 28.1 | 0.0 | 0.0% | 0.0 |
| 5 | `prefix_fanout` | 106.641 | 0.7124 | 0.3210 | 400.2 | 28.1 | 0.0 | 0.0% | 0.0 |
| 6 | `cost_aware_lru` | 104.848 | 0.7160 | 0.3233 | 399.7 | 28.1 | 0.0 | 0.0% | 0.0 |
| 7 | `lfu` | 102.803 | 0.7098 | 0.3210 | 408.4 | 28.1 | 0.0 | 0.0% | 0.0 |
| 8 | `prefix_anchor` | 100.428 | 0.7028 | 0.3210 | 427.1 | 28.1 | 0.0 | 0.0% | 0.0 |
| 9 | `vllm_apc` | 98.451 | 0.6058 | 0.3261 | 43.8 | 10.5 | 0.0 | 0.0% | 0.0 |
| 10 | `lru` | 97.799 | 0.6991 | 0.3210 | 436.2 | 28.1 | 0.0 | 0.0% | 0.0 |
| 11 | `tenant_fair_lru` | 97.799 | 0.6991 | 0.3210 | 436.2 | 28.1 | 0.0 | 0.0% | 0.0 |
| 12 | `depth_prefer_shallow` | 97.502 | 0.6946 | 0.3210 | 448.4 | 28.1 | 0.0 | 0.0% | 0.0 |
| 13 | `tinylfu_lru` | 96.131 | 0.6718 | 0.3284 | 332.5 | 24.5 | 0.0 | 0.0% | 0.0 |
| 14 | `no_cache` | -58.232 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 | 0.0% | 0.0 |

## Shared

| Rank | Policy | Raw score | Token hit | Request p10 | Churn/1k | Prefix KV | Decode KV | Decode fail | Decode-pressure evictions |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `oracle_future_reuse` | 61.103 | 0.4749 | 0.3035 | 95.9 | 5.8 | 26.6 | 77.3% | 9.0 |
| 2 | `incumbent` | 57.496 | 0.4716 | 0.3035 | 155.4 | 6.6 | 26.6 | 77.5% | 14.1 |
| 3 | `vllm_apc` | 57.438 | 0.4668 | 0.3045 | 119.8 | 4.8 | 27.8 | 77.5% | 10.6 |
| 4 | `future_reuse_heuristic` | 56.380 | 0.4724 | 0.3035 | 300.3 | 7.5 | 26.3 | 78.3% | 25.0 |
| 5 | `prefix_fanout` | 55.862 | 0.4719 | 0.3035 | 301.6 | 7.5 | 26.3 | 78.3% | 25.0 |
| 6 | `recompute_greedy` | 55.696 | 0.4717 | 0.3035 | 302.1 | 7.5 | 26.3 | 78.3% | 25.0 |
| 7 | `lfu` | 55.585 | 0.4716 | 0.3035 | 302.1 | 7.5 | 26.3 | 78.3% | 25.1 |
| 8 | `tinylfu_lru` | 55.378 | 0.4642 | 0.3045 | 246.1 | 6.0 | 27.5 | 77.9% | 20.7 |
| 9 | `cost_aware_lru` | 55.348 | 0.4711 | 0.3035 | 303.4 | 7.5 | 26.3 | 78.3% | 25.0 |
| 10 | `prefix_anchor` | 55.216 | 0.4707 | 0.3035 | 304.3 | 7.5 | 26.3 | 78.3% | 25.2 |
| 11 | `depth_prefer_shallow` | 55.041 | 0.4705 | 0.3035 | 304.7 | 7.5 | 26.3 | 78.3% | 25.2 |
| 12 | `lru` | 54.855 | 0.4697 | 0.3035 | 306.4 | 7.5 | 26.3 | 78.3% | 25.3 |
| 13 | `tenant_fair_lru` | 54.855 | 0.4697 | 0.3035 | 306.4 | 7.5 | 26.3 | 78.3% | 25.3 |
| 14 | `no_cache` | -58.232 | 0.0000 | 0.0000 | 0.0 | 0.0 | 31.3 | 75.9% | 0.0 |

## Rank and Metric Shift

| Policy | Prefix rank | Shared rank | Raw delta | Hit delta | Churn delta |
|---|---:|---:|---:|---:|---:|
| `oracle_future_reuse` | 1 | 1 | -67.155 | -0.2718 | +50.3 |
| `incumbent` | 3 | 2 | -56.223 | -0.2429 | +15.2 |
| `vllm_apc` | 9 | 3 | -41.013 | -0.1390 | +76.0 |
| `future_reuse_heuristic` | 2 | 4 | -58.683 | -0.2663 | -36.0 |
| `prefix_fanout` | 5 | 5 | -50.780 | -0.2405 | -98.5 |
| `recompute_greedy` | 4 | 6 | -53.382 | -0.2584 | -61.6 |
| `lfu` | 7 | 7 | -47.218 | -0.2382 | -106.3 |
| `tinylfu_lru` | 13 | 8 | -40.753 | -0.2076 | -86.4 |
| `cost_aware_lru` | 6 | 9 | -49.500 | -0.2449 | -96.4 |
| `prefix_anchor` | 8 | 10 | -45.212 | -0.2320 | -122.8 |
| `depth_prefer_shallow` | 12 | 11 | -42.461 | -0.2241 | -143.7 |
| `lru` | 10 | 12 | -42.944 | -0.2293 | -129.8 |
| `tenant_fair_lru` | 11 | 13 | -42.944 | -0.2293 | -129.8 |
| `no_cache` | 14 | 14 | +0.000 | +0.0000 | +0.0 |

## Interpretation

`oracle_future_reuse` has the strongest raw prefix-policy score in shared mode;
`incumbent` is the strongest deployable policy by raw behavior. The
incumbent ranks `2` with token hit `0.4716` and decode allocation failure `77.5%`.
Its charged score is `49.264` after the incumbent-only
complexity charge; baseline implementations are not complexity-charged in this
report, so raw behavior is the meaningful policy comparison.

The failure rate is the main systems result: these prefix-cache-sized capacities
cannot sustain the synthetic reasoning bursts when prompt and decode KV share the
same pool. Eviction-policy differences still change which reusable prefixes survive,
but no prefix policy can recover capacity occupied by active decode state. A production
extension should add scheduler actions such as admission control, preemption, or
separate prefix/decode budgets before using this mode as an optimization objective.
