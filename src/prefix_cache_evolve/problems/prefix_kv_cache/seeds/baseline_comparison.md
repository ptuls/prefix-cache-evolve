# Prefix KV-Cache Best Program Baseline Comparison

Candidate: `src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/joint_mooncake_synthetic_20260907.py`

Verifier: `1.0.0`

Evaluation context: `1fcac0b85b8b001746195a5206d9733d8ae80a89e31d158eda22f50cc1be5d66`

Panel: `6d38c40899eb5500ae61339f7d91c83eac04dd2d5f74775b771b351ff730e16a`

Command:

```bash
.venv/bin/python -m prefix_cache_evolve.problems.prefix_kv_cache.runner --baseline-report --candidate-program src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/joint_mooncake_synthetic_20260907.py --config configs/prefix_kv_cache_mooncake_synthetic_wildchat_lmcache.yaml --report-baseline lru --report-baseline vllm_apc --report-baseline tinylfu_lru
```

## Headline

The candidate ranking is shown against deployable and reporting-only baselines.

| Rank | Policy | Group | Combined score | Capacity 24 token hit | Capacity 48 token hit | Worst-quarter hit | Request p10 hit | Token-wtd admission waste | Admission token utility | Avoidable eviction | Priority-burst weighted hit | Priority-noise token hit | Policy underfill | Churn per 1k |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `tinylfu_lru` | deployable | 62.491 | 0.000 | 0.000 | 0.422 | 0.244 | 0.668 | 4.095 | 0.142 | 0.734 | 0.518 | 0.054 | 469.1 |
| 2 | `lru` | deployable | 47.321 | 0.000 | 0.000 | 0.423 | 0.245 | 0.779 | 1.819 | 0.155 | 0.737 | 0.481 | 0.000 | 1599.2 |
| 3 | `vllm_apc` | deployable | 36.183 | 0.000 | 0.000 | 0.423 | 0.248 | 0.499 | 17.655 | 0.077 | 0.650 | 0.465 | 0.147 | 924.6 |
| 4 | `candidate` | deployable | 25.854 | 0.000 | 0.000 | 0.449 | 0.237 | 0.646 | 5.885 | 0.129 | 0.767 | 0.524 | 0.086 | 152.1 |

## Validation Workload Detail

| Policy | phase_shift_prompts token hit | multi_tenant_skew token hit | hotset_cold_scan token hit | concurrent_long_generation token hit | stochastic_serving_mix token hit | rolling_template_versions token hit | heavy_tailed_prefix_lengths token hit | priority_burst_recovery token hit | priority_one_off_noise token hit | tenant_phase_shift_cycles token hit | mooncake_toolagent_1 token hit | mooncake_toolagent_2 token hit | wildchat token hit | lmcache_agentic token hit | Validation block hit | Validation churn per 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tinylfu_lru` | 0.549 | 0.559 | 0.620 | 0.768 | 0.400 | 0.821 | 0.553 | 0.490 | 0.518 | 0.523 | 0.366 | 0.359 | 0.428 | 0.910 | 0.532 | 469.1 |
| `lru` | 0.549 | 0.559 | 0.620 | 0.768 | 0.434 | 0.796 | 0.488 | 0.491 | 0.481 | 0.506 | 0.357 | 0.346 | 0.578 | 0.947 | 0.529 | 1599.2 |
| `vllm_apc` | 0.549 | 0.559 | 0.370 | 0.571 | 0.498 | 0.821 | 0.553 | 0.431 | 0.465 | 0.476 | 0.359 | 0.348 | 0.578 | 0.947 | 0.464 | 924.6 |
| `candidate` | 0.546 | 0.586 | 0.644 | 0.780 | 0.473 | 0.803 | 0.585 | 0.507 | 0.524 | 0.534 | 0.336 | 0.359 | 0.040 | 0.867 | 0.536 | 152.1 |

## Held-Out Structure-Generalization Probe

These probe workloads are evaluated and reported but excluded from the candidate-selection combined score.

| Policy | agent_trace_branching token hit | cyclic_working_set_pressure token hit | Probe block hit | Probe churn per 1k |
|---|---:|---:|---:|---:|
| `tinylfu_lru` | 0.364 | 0.797 | 0.538 | 1710.9 |
| `lru` | 0.365 | 0.807 | 0.556 | 2085.9 |
| `vllm_apc` | 0.374 | 0.802 | 0.482 | 1656.2 |
| `candidate` | 0.443 | 0.864 | 0.629 | 502.6 |

## Notes

- Candidate `scoring_fn_complexity` in this report is `613`; the combined score includes that penalty.
- Candidate score breakdown: mean workload `58.878`, minimum-workload contribution `-15.799`, churn cost `1.846`, underfill cost `1.896`, fairness cost `5.475`, and complexity cost `8.008`.
- `policy_underfill_rate` is policy bypass multiplied by unused mean capacity. It penalizes deliberate bypass while cache space remains idle, without charging natural underfill when the policy admits every miss.
- `future_reuse_heuristic` and `oracle_future_reuse` use simulator-provided future knowledge and are not deployable. The former is count-weighted; the latter is a Belady-style next-use oracle constrained by the simulator's leaf-only eviction model.
- `tinylfu_lru` admits only shallow or repeated blocks, so it often trades lower hit rate for lower churn.
- `vllm_apc` behaviorally emulates the core vLLM APC cache policy: exact-prefix reuse of full blocks, active-reference protection, and LRU eviction of reusable unreferenced blocks. It does not reproduce vLLM's internal data structures or additional serving optimizations, including scheduling, allocation, continuous batching, offload, and kernels.
- `sglang_radix_attention` models SGLang RadixAttention's default radix-cache replacement behavior: retain prefixes at cache-page boundaries and recursively evict the least-recently-used zero-reference leaf. The simulator treats every modeled block-tree node as a cacheable radix unit, making it behaviorally equivalent to `lru`; capacity remains fixed-block-counted rather than token/page-counted, and cache-aware scheduling and attention kernels are out of scope. It remains registered as a selectable reference but is excluded from default comparisons. See https://arxiv.org/html/2312.07104v1 and the pinned SGLang source at https://github.com/sgl-project/sglang/tree/52f221cce088abc998fa9d3812416a45ee0e2e25/python/sglang/srt/mem_cache.
- `prefix_anchor` is a deployable structural anchor baseline; `prefix_fanout` is a simpler descendant-count protection baseline.
- Priority-burst weighted hit is reported from `priority_burst_recovery`; priority-noise token hit checks the opposite failure mode, where high priority does not imply reuse. `n/a` means that workload is absent from this panel.
- Request p10, worst-quarter hit, token-weighted admission waste, admission token utility, and avoidable eviction are aggregated across the validation panel.
- Synthetic workloads use `request_count=96`, seeds `(11, 23, 37)`, block size `16`, block-capacity sweep `(24, 48)`, token-capacity sweep `(384, 768)`, and canonical synthetic workload token granularity `8`.
- Fixed trace streams: `train/mooncake_toolagent_1` (483 requests), `validation/mooncake_toolagent_1` (600 requests), `validation/mooncake_toolagent_2` (659 requests), `train/wildchat` (35 requests), `validation/wildchat` (11 requests), `train/lmcache_agentic` (133 requests), `validation/lmcache_agentic` (99 requests). Each is evaluated once per capacity, independently of synthetic seeds. Trace timing is a replay proxy, not serving latency.
