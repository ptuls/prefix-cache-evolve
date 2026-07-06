# Prefix KV-Cache Best Program Baseline Comparison

Candidate: `experiments/switching_cost_dual_policy.py`

Verifier: `1.0.0`

Evaluation context: `e0948efc40c7f0a5c93c6b3022a3c3cdf8066672fac192bc08ee26611bc00416`

Panel: `9431a5bd792de803d68e99d6c98d273f30d5d3e757ea83939760d0cfec60c354`

Command:

```bash
.venv/bin/python -m prefix_cache_evolve.problems.prefix_kv_cache.runner --baseline-report --candidate-program experiments/switching_cost_dual_policy.py --config /Users/ptuls/repos/prefix-cache-evolve/configs/prefix_kv_cache.yaml
```

## Headline

The candidate clears the deployable credibility baselines in this capacity sweep. It trails `oracle_future_reuse`. It beats `future_reuse_heuristic`.

| Rank | Policy | Group | Combined score | Capacity 24 token hit | Capacity 48 token hit | Worst-quarter hit | Request p10 hit | Token-wtd admission waste | Admission token utility | Avoidable eviction | Priority-burst weighted hit | Priority-noise token hit | Policy underfill | Churn per 1k |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `oracle_future_reuse` | reporting-only/future-knowledge | 83.213 | 0.601 | 0.668 | 0.475 | 0.262 | 0.022 | 13.777 | 0.000 | 0.780 | 0.524 | 0.186 | 18.9 |
| 2 | `candidate` | deployable | 66.145 | 0.613 | 0.666 | 0.462 | 0.256 | 0.597 | 3.401 | 0.183 | 0.772 | 0.521 | 0.029 | 136.7 |
| 3 | `tinylfu_lru` | deployable | 63.548 | 0.539 | 0.627 | 0.432 | 0.259 | 0.697 | 2.949 | 0.163 | 0.734 | 0.518 | 0.027 | 499.0 |
| 4 | `lfu` | deployable | 61.971 | 0.554 | 0.639 | 0.460 | 0.257 | 0.832 | 1.340 | 0.112 | 0.774 | 0.521 | 0.000 | 922.6 |
| 5 | `future_reuse_heuristic` | reporting-only/future-knowledge | 60.989 | 0.584 | 0.662 | 0.468 | 0.260 | 0.777 | 1.514 | 0.003 | 0.775 | 0.524 | 0.000 | 855.4 |
| 6 | `cost_aware_lru` | deployable | 50.672 | 0.531 | 0.628 | 0.437 | 0.251 | 0.799 | 1.269 | 0.170 | 0.736 | 0.481 | 0.000 | 989.9 |
| 7 | `prefix_anchor` | deployable | 50.525 | 0.532 | 0.631 | 0.436 | 0.254 | 0.807 | 1.260 | 0.168 | 0.737 | 0.481 | 0.000 | 984.3 |
| 8 | `lru` | deployable | 49.499 | 0.525 | 0.628 | 0.431 | 0.253 | 0.804 | 1.246 | 0.176 | 0.737 | 0.481 | 0.000 | 999.2 |
| 9 | `tenant_fair_lru` | deployable | 49.450 | 0.525 | 0.628 | 0.430 | 0.253 | 0.804 | 1.245 | 0.176 | 0.737 | 0.481 | 0.000 | 1000.1 |
| 10 | `vllm_apc` | deployable | 46.427 | 0.522 | 0.590 | 0.431 | 0.257 | 0.484 | 19.568 | 0.085 | 0.650 | 0.465 | 0.173 | 313.1 |
| 11 | `prefix_fanout` | deployable | 37.942 | 0.534 | 0.632 | 0.433 | 0.257 | 0.822 | 1.226 | 0.199 | 0.725 | 0.417 | 0.000 | 1035.5 |
| 12 | `depth_prefer_shallow` | deployable | 35.974 | 0.522 | 0.612 | 0.425 | 0.258 | 0.826 | 1.202 | 0.214 | 0.724 | 0.417 | 0.000 | 1050.2 |
| 13 | `recompute_greedy` | deployable | 19.818 | 0.526 | 0.622 | 0.401 | 0.251 | 0.803 | 1.246 | 0.205 | 0.723 | 0.240 | 0.000 | 1109.4 |
| 14 | `no_cache` | deployable | -61.310 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |

## Validation Workload Detail

| Policy | phase_shift_prompts token hit | multi_tenant_skew token hit | hotset_cold_scan token hit | concurrent_long_generation token hit | stochastic_serving_mix token hit | rolling_template_versions token hit | heavy_tailed_prefix_lengths token hit | priority_burst_recovery token hit | priority_one_off_noise token hit | tenant_phase_shift_cycles token hit | Validation block hit | Validation churn per 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `oracle_future_reuse` | 0.549 | 0.624 | 0.644 | 0.866 | 0.530 | 0.821 | 0.625 | 0.519 | 0.524 | 0.559 | 0.587 | 18.9 |
| `candidate` | 0.546 | 0.610 | 0.644 | 0.881 | 0.515 | 0.810 | 0.563 | 0.510 | 0.521 | 0.505 | 0.569 | 136.7 |
| `tinylfu_lru` | 0.549 | 0.559 | 0.620 | 0.768 | 0.400 | 0.821 | 0.553 | 0.490 | 0.518 | 0.523 | 0.535 | 499.0 |
| `lfu` | 0.549 | 0.559 | 0.644 | 0.777 | 0.470 | 0.821 | 0.535 | 0.515 | 0.521 | 0.529 | 0.550 | 922.6 |
| `future_reuse_heuristic` | 0.549 | 0.624 | 0.644 | 0.859 | 0.509 | 0.821 | 0.581 | 0.516 | 0.524 | 0.535 | 0.577 | 855.4 |
| `cost_aware_lru` | 0.549 | 0.559 | 0.620 | 0.800 | 0.470 | 0.798 | 0.476 | 0.490 | 0.481 | 0.503 | 0.530 | 989.9 |
| `prefix_anchor` | 0.549 | 0.559 | 0.620 | 0.768 | 0.445 | 0.800 | 0.514 | 0.491 | 0.481 | 0.509 | 0.530 | 984.3 |
| `lru` | 0.549 | 0.559 | 0.620 | 0.768 | 0.434 | 0.796 | 0.488 | 0.491 | 0.481 | 0.506 | 0.526 | 999.2 |
| `tenant_fair_lru` | 0.549 | 0.559 | 0.620 | 0.768 | 0.434 | 0.796 | 0.486 | 0.491 | 0.481 | 0.505 | 0.526 | 1000.1 |
| `vllm_apc` | 0.549 | 0.559 | 0.370 | 0.571 | 0.498 | 0.821 | 0.553 | 0.431 | 0.465 | 0.476 | 0.449 | 313.1 |
| `prefix_fanout` | 0.549 | 0.559 | 0.563 | 0.768 | 0.493 | 0.821 | 0.550 | 0.480 | 0.417 | 0.481 | 0.510 | 1035.5 |
| `depth_prefer_shallow` | 0.549 | 0.559 | 0.560 | 0.768 | 0.395 | 0.821 | 0.528 | 0.482 | 0.417 | 0.516 | 0.505 | 1050.2 |
| `recompute_greedy` | 0.549 | 0.581 | 0.612 | 0.852 | 0.480 | 0.776 | 0.481 | 0.478 | 0.240 | 0.413 | 0.495 | 1109.4 |
| `no_cache` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |

## Held-Out Structure-Generalization Probe

These recurrence-heavy families are evaluated and reported but excluded from the candidate-selection combined score.

| Policy | agent_trace_branching token hit | cyclic_working_set_pressure token hit | Probe block hit | Probe churn per 1k |
|---|---:|---:|---:|---:|
| `oracle_future_reuse` | 0.384 | 0.873 | 0.606 | 1579.9 |
| `candidate` | 0.472 | 0.883 | 0.652 | 227.4 |
| `tinylfu_lru` | 0.364 | 0.797 | 0.538 | 1710.9 |
| `lfu` | 0.367 | 0.863 | 0.592 | 1954.0 |
| `future_reuse_heuristic` | 0.371 | 0.871 | 0.599 | 1901.9 |
| `cost_aware_lru` | 0.365 | 0.808 | 0.557 | 2084.2 |
| `prefix_anchor` | 0.366 | 0.807 | 0.556 | 2072.0 |
| `lru` | 0.365 | 0.807 | 0.556 | 2085.9 |
| `tenant_fair_lru` | 0.365 | 0.807 | 0.556 | 2085.9 |
| `vllm_apc` | 0.374 | 0.802 | 0.482 | 1656.2 |
| `prefix_fanout` | 0.367 | 0.879 | 0.584 | 1982.6 |
| `depth_prefer_shallow` | 0.363 | 0.879 | 0.582 | 2016.5 |
| `recompute_greedy` | 0.367 | 0.867 | 0.593 | 1954.0 |
| `no_cache` | 0.000 | 0.000 | 0.000 | 0.0 |

## Notes

- Candidate `scoring_fn_complexity` in this report is `372`; the combined score includes that penalty.
- Candidate score breakdown: mean workload `62.625`, minimum-workload contribution `19.843`, churn cost `2.050`, underfill cost `0.343`, fairness cost `8.424`, and complexity cost `5.506`.
- `policy_underfill_rate` is policy bypass multiplied by unused mean capacity. It penalizes deliberate bypass while cache space remains idle, without charging natural underfill when the policy admits every miss.
- `future_reuse_heuristic` and `oracle_future_reuse` use simulator-provided future knowledge and are not deployable. The former is count-weighted; the latter is a Belady-style next-use oracle constrained by the simulator's leaf-only eviction model.
- `tinylfu_lru` admits only shallow or repeated blocks, so it often trades lower hit rate for lower churn.
- `vllm_apc` behaviorally emulates the core vLLM APC cache policy: exact-prefix reuse of full blocks, active-reference protection, and LRU eviction of reusable unreferenced blocks. It does not reproduce vLLM's internal data structures or additional serving optimizations, including scheduling, allocation, continuous batching, offload, and kernels.
- `sglang_radix_attention` models SGLang RadixAttention's default radix-cache replacement behavior: retain prefixes at cache-page boundaries and recursively evict the least-recently-used zero-reference leaf. The simulator treats every modeled block-tree node as a cacheable radix unit, making it behaviorally equivalent to `lru`; capacity remains fixed-block-counted rather than token/page-counted, and cache-aware scheduling and attention kernels are out of scope. It remains registered as a selectable reference but is excluded from default comparisons. See https://arxiv.org/html/2312.07104v1 and the pinned SGLang source at https://github.com/sgl-project/sglang/tree/52f221cce088abc998fa9d3812416a45ee0e2e25/python/sglang/srt/mem_cache.
- `prefix_anchor` is a deployable structural anchor baseline; `prefix_fanout` is a simpler descendant-count protection baseline.
- Priority-burst weighted hit is reported from `priority_burst_recovery`; priority-noise token hit checks the opposite failure mode, where high priority does not imply reuse.
- Request p10, worst-quarter hit, token-weighted admission waste, admission token utility, and avoidable eviction are aggregated across the validation panel.
- This report uses `request_count=96`, seeds `(11, 23, 37)`, block size `16`, block-capacity sweep `(24, 48)`, token-capacity sweep `(384, 768)`, and canonical synthetic workload token granularity `8`.
