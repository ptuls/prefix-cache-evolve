# Prefix KV-Cache Best Program Baseline Comparison

Candidate: `src/prefix_cache_evolve/problems/prefix_kv_cache/pressure_aware_incumbent.py`

> Historical discovery-verifier result: 8-token blocks and 48/96-block
> capacity tiers. The operative production-oriented verifier now uses
> 16-token blocks and 24/48-block tiers; see
> [`block_size_robustness.md`](block_size_robustness.md).

Command:

```bash
.venv/bin/python -m prefix_cache_evolve.problems.prefix_kv_cache.runner --baseline-report --candidate-program src/prefix_cache_evolve/problems/prefix_kv_cache/pressure_aware_incumbent.py --config configs/prefix_kv_cache_discovery.yaml
```

The synthetic access streams are summarized and fingerprinted in
[`discovery_workload_manifest.json`](discovery_workload_manifest.json). The
panel SHA-256 is
`4607782d231560f5d51c5f0347a789b7b82a7e8ff4d78ec5f1adb576c68d2c8f`.

## Headline

The candidate clears the deployable credibility baselines in this capacity sweep. It trails `oracle_future_reuse`. It beats `future_reuse_heuristic`.

| Rank | Policy | Group | Combined score | Capacity 48 token hit | Capacity 96 token hit | Worst-quarter hit | Request p10 hit | Token-wtd admission waste | Admission token utility | Avoidable eviction | Priority-burst weighted hit | Priority-noise token hit | Policy underfill | Churn per 1k |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `oracle_future_reuse` | reporting-only/future-knowledge | 97.074 | 0.683 | 0.745 | 0.548 | 0.326 | 0.017 | 13.164 | 0.000 | 0.780 | 0.602 | 0.141 | 43.9 |
| 2 | `candidate` | deployable | 77.113 | 0.674 | 0.687 | 0.521 | 0.324 | 0.407 | 8.143 | 0.074 | 0.766 | 0.596 | 0.070 | 92.7 |
| 3 | `tinylfu_lru` | deployable | 70.362 | 0.618 | 0.690 | 0.457 | 0.299 | 0.357 | 8.062 | 0.108 | 0.722 | 0.575 | 0.112 | 161.1 |
| 4 | `future_reuse_heuristic` | reporting-only/future-knowledge | 69.857 | 0.669 | 0.740 | 0.542 | 0.323 | 0.706 | 2.440 | 0.005 | 0.776 | 0.602 | 0.000 | 1197.3 |
| 5 | `prefix_anchor` | deployable | 60.199 | 0.639 | 0.724 | 0.507 | 0.321 | 0.705 | 2.298 | 0.110 | 0.735 | 0.578 | 0.000 | 1363.3 |
| 6 | `vllm_apc` | deployable | 60.178 | 0.621 | 0.696 | 0.496 | 0.321 | 0.486 | 7.973 | 0.128 | 0.649 | 0.543 | 0.056 | 807.9 |
| 7 | `lfu` | deployable | 58.541 | 0.647 | 0.733 | 0.533 | 0.319 | 0.740 | 2.346 | 0.070 | 0.775 | 0.602 | 0.000 | 1281.6 |
| 8 | `cost_aware_lru` | deployable | 52.305 | 0.627 | 0.721 | 0.507 | 0.316 | 0.707 | 2.255 | 0.127 | 0.735 | 0.565 | 0.000 | 1401.1 |
| 9 | `tenant_fair_lru` | deployable | 51.219 | 0.627 | 0.721 | 0.502 | 0.317 | 0.716 | 2.271 | 0.119 | 0.737 | 0.602 | 0.000 | 1384.9 |
| 10 | `lru` | deployable | 51.186 | 0.627 | 0.721 | 0.503 | 0.318 | 0.718 | 2.268 | 0.119 | 0.737 | 0.602 | 0.000 | 1385.0 |
| 11 | `depth_prefer_shallow` | deployable | 50.409 | 0.628 | 0.715 | 0.497 | 0.323 | 0.724 | 2.214 | 0.167 | 0.723 | 0.521 | 0.000 | 1471.5 |
| 12 | `prefix_fanout` | deployable | 47.619 | 0.628 | 0.719 | 0.496 | 0.320 | 0.723 | 2.215 | 0.167 | 0.709 | 0.472 | 0.000 | 1479.5 |
| 13 | `recompute_greedy` | deployable | 33.772 | 0.624 | 0.703 | 0.480 | 0.304 | 0.720 | 2.158 | 0.181 | 0.701 | 0.364 | 0.000 | 1583.0 |
| 14 | `no_cache` | deployable | -61.304 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |

## Validation Workload Detail

| Policy | phase_shift_prompts token hit | multi_tenant_skew token hit | hotset_cold_scan token hit | concurrent_long_generation token hit | stochastic_serving_mix token hit | rolling_template_versions token hit | heavy_tailed_prefix_lengths token hit | priority_burst_recovery token hit | priority_one_off_noise token hit | tenant_phase_shift_cycles token hit | Validation block hit | Validation churn per 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `oracle_future_reuse` | 0.794 | 0.822 | 0.644 | 0.924 | 0.600 | 0.847 | 0.697 | 0.519 | 0.602 | 0.561 | 0.663 | 43.9 |
| `candidate` | 0.794 | 0.802 | 0.638 | 0.890 | 0.467 | 0.847 | 0.631 | 0.506 | 0.596 | 0.519 | 0.625 | 92.7 |
| `tinylfu_lru` | 0.759 | 0.757 | 0.614 | 0.854 | 0.474 | 0.814 | 0.587 | 0.478 | 0.575 | 0.529 | 0.603 | 161.1 |
| `future_reuse_heuristic` | 0.794 | 0.822 | 0.644 | 0.924 | 0.574 | 0.847 | 0.656 | 0.516 | 0.602 | 0.543 | 0.654 | 1197.3 |
| `prefix_anchor` | 0.794 | 0.812 | 0.620 | 0.924 | 0.495 | 0.847 | 0.593 | 0.490 | 0.578 | 0.509 | 0.627 | 1363.3 |
| `vllm_apc` | 0.794 | 0.802 | 0.532 | 0.823 | 0.511 | 0.847 | 0.611 | 0.433 | 0.543 | 0.476 | 0.573 | 807.9 |
| `lfu` | 0.794 | 0.779 | 0.644 | 0.924 | 0.520 | 0.847 | 0.618 | 0.516 | 0.602 | 0.538 | 0.640 | 1281.6 |
| `cost_aware_lru` | 0.794 | 0.785 | 0.620 | 0.924 | 0.514 | 0.845 | 0.553 | 0.490 | 0.565 | 0.506 | 0.621 | 1401.1 |
| `tenant_fair_lru` | 0.794 | 0.784 | 0.620 | 0.924 | 0.479 | 0.845 | 0.570 | 0.491 | 0.602 | 0.508 | 0.624 | 1384.9 |
| `lru` | 0.794 | 0.779 | 0.620 | 0.924 | 0.480 | 0.845 | 0.570 | 0.491 | 0.602 | 0.510 | 0.623 | 1385.0 |
| `depth_prefer_shallow` | 0.794 | 0.812 | 0.539 | 0.924 | 0.450 | 0.847 | 0.607 | 0.481 | 0.521 | 0.510 | 0.604 | 1471.5 |
| `prefix_fanout` | 0.794 | 0.812 | 0.541 | 0.924 | 0.546 | 0.847 | 0.613 | 0.471 | 0.472 | 0.462 | 0.602 | 1479.5 |
| `recompute_greedy` | 0.793 | 0.801 | 0.612 | 0.924 | 0.532 | 0.845 | 0.546 | 0.465 | 0.364 | 0.428 | 0.588 | 1583.0 |
| `no_cache` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |

## Held-Out Structure-Generalization Probe

These recurrence-heavy families are evaluated and reported but excluded from the candidate-selection combined score.

| Policy | agent_trace_branching token hit | cyclic_working_set_pressure token hit | Probe block hit | Probe churn per 1k |
|---|---:|---:|---:|---:|
| `oracle_future_reuse` | 0.402 | 0.894 | 0.640 | 2967.0 |
| `candidate` | 0.376 | 0.881 | 0.612 | 27.8 |
| `tinylfu_lru` | 0.382 | 0.764 | 0.558 | 3107.6 |
| `future_reuse_heuristic` | 0.394 | 0.892 | 0.634 | 3375.9 |
| `prefix_anchor` | 0.389 | 0.847 | 0.605 | 3602.4 |
| `vllm_apc` | 0.393 | 0.802 | 0.557 | 3205.7 |
| `lfu` | 0.391 | 0.865 | 0.617 | 3515.6 |
| `cost_aware_lru` | 0.388 | 0.824 | 0.594 | 3678.8 |
| `tenant_fair_lru` | 0.388 | 0.811 | 0.588 | 3702.3 |
| `lru` | 0.388 | 0.811 | 0.588 | 3702.3 |
| `depth_prefer_shallow` | 0.386 | 0.883 | 0.619 | 3577.3 |
| `prefix_fanout` | 0.390 | 0.883 | 0.621 | 3506.9 |
| `recompute_greedy` | 0.391 | 0.888 | 0.631 | 3447.0 |
| `no_cache` | 0.000 | 0.000 | 0.000 | 0.0 |

## Notes

- Candidate `scoring_fn_complexity` in this report is `648`; the combined score includes that penalty.
- Candidate score breakdown: mean workload `75.009`, minimum-workload contribution `20.341`, churn cost `1.391`, underfill cost `0.835`, fairness cost `7.663`, and complexity cost `8.348`.
- The policy behavior is unchanged from the original `77.230` report. The `0.116` score reduction comes entirely from charging module constants that the previous counter ignored.
- `policy_underfill_rate` is policy bypass multiplied by unused mean capacity. It penalizes deliberate bypass while cache space remains idle, without charging natural underfill when the policy admits every miss.
- `future_reuse_heuristic` and `oracle_future_reuse` use simulator-provided future knowledge and are not deployable. The former is count-weighted; the latter is a Belady-style next-use oracle constrained by the simulator's leaf-only eviction model.
- `tinylfu_lru` admits only shallow or repeated blocks, so it often trades lower hit rate for lower churn.
- `vllm_apc` models vLLM automatic prefix caching: it admits only full blocks and uses LRU eviction with deepest-prefix tie-breaking. The simulator supplies active-reference pinning and legal leaf filtering.
- `sglang_radix_attention` models SGLang RadixAttention's default radix-cache replacement behavior: retain prefixes at cache-page boundaries and recursively evict the least-recently-used zero-reference leaf. The simulator treats every modeled block-tree node as a cacheable radix unit, making it behaviorally equivalent to `lru`; capacity remains fixed-block-counted rather than token/page-counted, and cache-aware scheduling and attention kernels are out of scope. It remains registered as a selectable reference but is excluded from default comparisons. See https://arxiv.org/html/2312.07104v1 and the pinned SGLang source at https://github.com/sgl-project/sglang/tree/52f221cce088abc998fa9d3812416a45ee0e2e25/python/sglang/srt/mem_cache.
- `prefix_anchor` is a deployable structural anchor baseline; `prefix_fanout` is a simpler descendant-count protection baseline.
- Priority-burst weighted hit is reported from `priority_burst_recovery`; priority-noise token hit checks the opposite failure mode, where high priority does not imply reuse.
- Request p10, worst-quarter hit, token-weighted admission waste, admission token utility, and avoidable eviction are aggregated across the validation panel.
- This report uses `request_count=96`, seeds `(11, 23, 37)`, block size `8`, block-capacity sweep `(48, 96)`, token-capacity sweep `(384, 768)`, and canonical synthetic workload token granularity `8`.
