# Prefix KV-Cache Best Program Baseline Comparison

Candidate: `src/prefix_cache_evolve/problems/prefix_kv_cache/incumbents/production_16tok_20260609/policy.py`

Verifier: `1.0.0`

Evaluation context: `3c1b20a27b647154abaceb54e279acd1a3982387a9a01fa8ad7fdb80456516c0`

Panel: `2db03322c52ea3e612f94cbddde199caaa65953716d7d24e4c14ff84cb21918e`

Command:

```bash
.venv/bin/python -m prefix_cache_evolve.problems.prefix_kv_cache.runner --baseline-report --candidate-program src/prefix_cache_evolve/problems/prefix_kv_cache/incumbents/production_16tok_20260609/policy.py --config /Users/ptuls/repos/prefix-cache-evolve/configs/prefix_kv_cache.yaml
```

## Headline

The candidate clears the deployable credibility baselines in this capacity sweep. It trails `oracle_future_reuse` and `future_reuse_heuristic`.

| Rank | Policy | Group | Combined score | Capacity 24 token hit | Capacity 48 token hit | Worst-quarter hit | Request p10 hit | Token-wtd admission waste | Admission token utility | Avoidable eviction | Priority-burst weighted hit | Priority-noise token hit | Policy underfill | Churn per 1k |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `oracle_future_reuse` | reporting-only/future-knowledge | 46.964 | 0.619 | 0.680 | 0.400 | 0.062 | 0.025 | 9.572 | 0.000 | 0.750 | 0.507 | 0.176 | 26.7 |
| 2 | `future_reuse_heuristic` | reporting-only/future-knowledge | 21.898 | 0.592 | 0.676 | 0.394 | 0.062 | 0.722 | 0.894 | 0.003 | 0.750 | 0.507 | 0.000 | 795.9 |
| 3 | `candidate` | deployable | 14.379 | 0.599 | 0.654 | 0.380 | 0.062 | 0.585 | 1.751 | 0.112 | 0.737 | 0.507 | 0.025 | 186.1 |
| 4 | `vllm_apc` | deployable | 6.516 | 0.536 | 0.611 | 0.339 | 0.062 | 0.467 | 7.331 | 0.100 | 0.476 | 0.336 | 0.106 | 220.6 |
| 5 | `depth_prefer_shallow` | deployable | 6.402 | 0.547 | 0.638 | 0.357 | 0.062 | 0.737 | 0.772 | 0.160 | 0.673 | 0.445 | 0.000 | 922.0 |
| 6 | `prefix_fanout` | deployable | 2.961 | 0.559 | 0.654 | 0.357 | 0.062 | 0.732 | 0.782 | 0.155 | 0.673 | 0.445 | 0.000 | 920.1 |
| 7 | `lfu` | deployable | 2.715 | 0.565 | 0.659 | 0.388 | 0.062 | 0.768 | 0.839 | 0.083 | 0.750 | 0.507 | 0.000 | 852.7 |
| 8 | `tinylfu_lru` | deployable | -0.755 | 0.545 | 0.643 | 0.347 | 0.062 | 0.728 | 0.862 | 0.143 | 0.685 | 0.507 | 0.005 | 732.3 |
| 9 | `prefix_anchor` | deployable | -3.151 | 0.541 | 0.647 | 0.348 | 0.062 | 0.743 | 0.788 | 0.133 | 0.690 | 0.507 | 0.000 | 899.0 |
| 10 | `lru` | deployable | -4.018 | 0.531 | 0.643 | 0.343 | 0.062 | 0.741 | 0.779 | 0.140 | 0.685 | 0.507 | 0.000 | 910.9 |
| 11 | `tenant_fair_lru` | deployable | -4.850 | 0.524 | 0.642 | 0.341 | 0.062 | 0.733 | 0.772 | 0.144 | 0.685 | 0.507 | 0.000 | 918.3 |
| 12 | `cost_aware_lru` | deployable | -4.997 | 0.523 | 0.644 | 0.347 | 0.062 | 0.730 | 0.761 | 0.151 | 0.685 | 0.450 | 0.000 | 921.1 |
| 13 | `recompute_greedy` | deployable | -32.456 | 0.496 | 0.614 | 0.276 | 0.062 | 0.771 | 0.647 | 0.237 | 0.543 | 0.198 | 0.000 | 1050.1 |
| 14 | `no_cache` | deployable | -61.440 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |

## Validation Workload Detail

| Policy | phase_shift_prompts token hit | multi_tenant_skew token hit | hotset_cold_scan token hit | concurrent_long_generation token hit | stochastic_serving_mix token hit | rolling_template_versions token hit | heavy_tailed_prefix_lengths token hit | priority_burst_recovery token hit | priority_one_off_noise token hit | tenant_phase_shift_cycles token hit | Validation block hit | Validation churn per 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `oracle_future_reuse` | 0.736 | 0.745 | 0.627 | 0.819 | 0.496 | 0.651 | 0.539 | 0.493 | 0.507 | 0.365 | 0.497 | 26.7 |
| `future_reuse_heuristic` | 0.736 | 0.743 | 0.627 | 0.812 | 0.443 | 0.651 | 0.510 | 0.493 | 0.507 | 0.330 | 0.483 | 795.9 |
| `candidate` | 0.710 | 0.703 | 0.627 | 0.815 | 0.424 | 0.651 | 0.487 | 0.477 | 0.507 | 0.318 | 0.470 | 186.1 |
| `vllm_apc` | 0.736 | 0.730 | 0.515 | 0.757 | 0.395 | 0.651 | 0.485 | 0.314 | 0.336 | 0.183 | 0.346 | 220.6 |
| `depth_prefer_shallow` | 0.736 | 0.730 | 0.581 | 0.807 | 0.304 | 0.651 | 0.457 | 0.442 | 0.445 | 0.278 | 0.427 | 922.0 |
| `prefix_fanout` | 0.736 | 0.730 | 0.581 | 0.807 | 0.404 | 0.651 | 0.481 | 0.442 | 0.445 | 0.200 | 0.427 | 920.1 |
| `lfu` | 0.736 | 0.631 | 0.627 | 0.807 | 0.359 | 0.651 | 0.462 | 0.493 | 0.507 | 0.316 | 0.459 | 852.7 |
| `tinylfu_lru` | 0.736 | 0.631 | 0.585 | 0.807 | 0.290 | 0.651 | 0.446 | 0.451 | 0.507 | 0.275 | 0.440 | 732.3 |
| `prefix_anchor` | 0.736 | 0.631 | 0.585 | 0.807 | 0.316 | 0.651 | 0.421 | 0.455 | 0.507 | 0.274 | 0.440 | 899.0 |
| `lru` | 0.736 | 0.631 | 0.585 | 0.807 | 0.297 | 0.651 | 0.390 | 0.451 | 0.507 | 0.274 | 0.436 | 910.9 |
| `tenant_fair_lru` | 0.736 | 0.572 | 0.585 | 0.807 | 0.296 | 0.651 | 0.388 | 0.451 | 0.507 | 0.274 | 0.432 | 918.3 |
| `cost_aware_lru` | 0.736 | 0.613 | 0.585 | 0.807 | 0.346 | 0.651 | 0.374 | 0.451 | 0.450 | 0.271 | 0.430 | 921.1 |
| `recompute_greedy` | 0.653 | 0.613 | 0.544 | 0.811 | 0.370 | 0.649 | 0.367 | 0.351 | 0.198 | 0.113 | 0.369 | 1050.1 |
| `no_cache` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |

## Held-Out Structure-Generalization Probe

These recurrence-heavy families are evaluated and reported but excluded from the candidate-selection combined score.

| Policy | agent_trace_branching token hit | cyclic_working_set_pressure token hit | Probe block hit | Probe churn per 1k |
|---|---:|---:|---:|---:|
| `oracle_future_reuse` | 0.505 | 0.780 | 0.631 | 1099.0 |
| `future_reuse_heuristic` | 0.477 | 0.775 | 0.615 | 1568.6 |
| `candidate` | 0.526 | 0.761 | 0.631 | 186.6 |
| `vllm_apc` | 0.491 | 0.546 | 0.442 | 1191.0 |
| `depth_prefer_shallow` | 0.463 | 0.773 | 0.594 | 1682.3 |
| `prefix_fanout` | 0.471 | 0.773 | 0.598 | 1632.8 |
| `lfu` | 0.471 | 0.761 | 0.604 | 1623.3 |
| `tinylfu_lru` | 0.480 | 0.666 | 0.561 | 1225.7 |
| `prefix_anchor` | 0.470 | 0.666 | 0.557 | 1718.8 |
| `lru` | 0.469 | 0.666 | 0.556 | 1728.3 |
| `tenant_fair_lru` | 0.469 | 0.666 | 0.556 | 1728.3 |
| `cost_aware_lru` | 0.469 | 0.666 | 0.556 | 1728.3 |
| `recompute_greedy` | 0.469 | 0.769 | 0.607 | 1624.1 |
| `no_cache` | 0.000 | 0.000 | 0.000 | 0.0 |

## Notes

- Candidate `scoring_fn_complexity` in this report is `572`; the combined score includes that penalty.
- Candidate score breakdown: mean workload `46.970`, minimum-workload contribution `3.425`, churn cost `2.791`, underfill cost `0.302`, fairness cost `25.320`, and complexity cost `7.603`.
- `policy_underfill_rate` is policy bypass multiplied by unused mean capacity. It penalizes deliberate bypass while cache space remains idle, without charging natural underfill when the policy admits every miss.
- `future_reuse_heuristic` and `oracle_future_reuse` use simulator-provided future knowledge and are not deployable. The former is count-weighted; the latter is a Belady-style next-use oracle constrained by the simulator's leaf-only eviction model.
- `tinylfu_lru` admits only shallow or repeated blocks, so it often trades lower hit rate for lower churn.
- `vllm_apc` behaviorally emulates the core vLLM APC cache policy: exact-prefix reuse of full blocks, active-reference protection, and LRU eviction of reusable unreferenced blocks. It does not reproduce vLLM's internal data structures or additional serving optimizations, including scheduling, allocation, continuous batching, offload, and kernels.
- `sglang_radix_attention` models SGLang RadixAttention's default radix-cache replacement behavior: retain prefixes at cache-page boundaries and recursively evict the least-recently-used zero-reference leaf. The simulator treats every modeled block-tree node as a cacheable radix unit, making it behaviorally equivalent to `lru`; capacity remains fixed-block-counted rather than token/page-counted, and cache-aware scheduling and attention kernels are out of scope. It remains registered as a selectable reference but is excluded from default comparisons. See https://arxiv.org/html/2312.07104v1 and the pinned SGLang source at https://github.com/sgl-project/sglang/tree/52f221cce088abc998fa9d3812416a45ee0e2e25/python/sglang/srt/mem_cache.
- `prefix_anchor` is a deployable structural anchor baseline; `prefix_fanout` is a simpler descendant-count protection baseline.
- Priority-burst weighted hit is reported from `priority_burst_recovery`; priority-noise token hit checks the opposite failure mode, where high priority does not imply reuse.
- Request p10, worst-quarter hit, token-weighted admission waste, admission token utility, and avoidable eviction are aggregated across the validation panel.
- This report uses `request_count=96`, seeds `(11, 23, 37)`, block size `24`, block-capacity sweep `(24, 48)`, token-capacity sweep `(576, 1152)`, and canonical synthetic workload token granularity `8`.
