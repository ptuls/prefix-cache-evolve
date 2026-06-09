# Block-Granularity Sweep

This sweep tests whether the promoted 16-token incumbent continues improving
with larger physical cache blocks. It compares the new and old incumbents at
`16`, `24`, `32`, `48`, and `64` tokens per block while holding cache capacity
fixed at `384` and `768` tokens.

Every block size divides both capacity tiers exactly:

| Tokens/block | Capacity blocks |
|---:|---:|
| 16 | 24, 48 |
| 24 | 16, 32 |
| 32 | 12, 24 |
| 48 | 8, 16 |
| 64 | 6, 12 |

The full train and validation panels use identical committed seeds and
canonical token streams. Scores include source-complexity cost.

## Results

| Tokens/block | Old score | New score | Delta | Old raw | New raw | Old validation hit | New validation hit | Old request p10 | New request p10 | Old churn/1k | New churn/1k |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 62.757 | 65.649 | +2.892 | 70.989 | 73.252 | 0.5982 | 0.5964 | 0.2564 | 0.2562 | 168.1 | 163.9 |
| 24 | 2.683 | 3.006 | +0.322 | 10.915 | 10.608 | 0.5470 | 0.5457 | 0.0615 | 0.0615 | 228.7 | 239.6 |
| 32 | -20.982 | -17.602 | +3.380 | -12.750 | -9.999 | 0.3823 | 0.3800 | 0.0000 | 0.0000 | 152.8 | 174.0 |
| 48 | -40.210 | -40.346 | -0.136 | -31.978 | -32.743 | 0.2465 | 0.2440 | 0.0000 | 0.0000 | 143.1 | 212.8 |
| 64 | -37.851 | -37.107 | +0.745 | -29.619 | -29.504 | 0.2285 | 0.2239 | 0.0000 | 0.0000 | 135.1 | 150.2 |

## Interpretation

The simple hypothesis that larger blocks favor the new policy is false.
Sixteen tokens per block is the clear optimum among the tested values for both
incumbents.

The principal failure is not policy complexity or eviction choice:

- validation request p10 falls from about `0.256` at 16 tokens to `0.062` at
  24 tokens;
- request p10 reaches zero at 32 tokens and remains zero;
- validation token hit falls from about `0.60` to `0.38` at 32 tokens and
  about `0.23` at 64 tokens;
- raw scores become negative from 32 tokens onward, so the result is not caused
  by the complexity charge.

Coarse blocks destroy reusable-prefix resolution. A mismatch late in a large
block prevents reuse of all tokens in that block, while fixed token capacity
also leaves fewer independently retainable entries. The slight score advantage
of one incumbent over the other after 32 tokens only identifies which policy is
less poor in a failed geometry.

The actionable range is therefore below 24 tokens. A follow-up crossover sweep
should test `8`, `10`, `12`, `14`, `16`, `18`, `20`, and `22` tokens per block,
using token capacities divisible by all tested sizes or evaluating each size at
explicitly reported near-equal capacities.

## Larger block capacities

A follow-up sweep fixed block count at `24`, `48`, `96`, and `128` for each
coarse block size. This increases total cache capacity rather than holding it
constant.

### New incumbent

| Tokens/block | 24-block hit / p10 | 48-block hit / p10 | 96-block hit / p10 | 128-block hit / p10 | Joint score |
|---:|---:|---:|---:|---:|---:|
| 24 | 0.6027 / 0.2111 | 0.6463 / 0.2517 | 0.6666 / 0.2621 | 0.6671 / 0.2621 | 21.653 |
| 32 | 0.4150 / 0.0571 | 0.4310 / 0.0743 | 0.4760 / 0.0743 | 0.4778 / 0.0743 | -14.205 |
| 48 | 0.3319 / 0.0204 | 0.3402 / 0.0299 | 0.3807 / 0.0299 | 0.3816 / 0.0299 | -29.740 |
| 64 | 0.2975 / 0.0228 | 0.3023 / 0.0268 | 0.3428 / 0.0268 | 0.3428 / 0.0268 | -31.251 |

Capacity helps, but every curve largely saturates by 96 blocks. At 24
tokens/block, aggregate hit and request p10 recover to roughly the 16-token
level at high capacities, yet the joint score remains only `21.653` versus
`66.710` at 16 tokens. The remaining deficit is workload imbalance:
`tenant_phase_shift_cycles` falls to `0.330` token hit, the minimum workload
score falls from `42.456` to `6.849`, and the fairness cost rises from `7.749`
to `21.996`.

At 32 tokens and above, capacity cannot recover reuse boundaries. For example,
`phase_shift_prompts` has zero token hit at 32, 48, and 64 tokens/block even in
the four-capacity aggregate. Additional blocks can retain more coarse entries,
but cannot reuse a prefix ending inside a mismatching block.

Thus larger block capacity partially fixes cache pressure, but not granularity
loss. Beyond 128 blocks, the measured hit and p10 plateaus make a qualitative
reversal unlikely.

## TinyLFU-LRU comparison

TinyLFU-LRU was evaluated on the same `24`, `48`, `96`, and `128` block grid.
Registered baselines receive no source-complexity charge; the evolved incumbent
pays `7.603` points.

| Tokens/block | New charged | TinyLFU-LRU | Charged delta | New before complexity | Behavioral delta |
|---:|---:|---:|---:|---:|---:|
| 16 | 66.710 | 66.559 | +0.150 | 74.312 | +7.753 |
| 24 | 21.653 | 9.766 | +11.887 | 29.255 | +19.490 |
| 32 | -14.205 | -10.808 | -3.397 | -6.602 | +4.206 |
| 48 | -29.740 | -24.187 | -5.553 | -22.137 | +2.049 |
| 64 | -31.251 | -24.824 | -6.428 | -23.649 | +1.175 |

The evolved policy has higher token hit in all 20 block-size/capacity cells.
Representative comparisons are:

| Tokens/block | Blocks | New hit | TinyLFU-LRU hit | New churn/1k | TinyLFU-LRU churn/1k |
|---:|---:|---:|---:|---:|---:|
| 16 | 24 | 0.5772 | 0.5457 | 280.4 | 951.2 |
| 16 | 128 | 0.6559 | 0.6539 | 16.2 | 12.0 |
| 24 | 24 | 0.6027 | 0.5607 | 318.3 | 987.3 |
| 24 | 128 | 0.6671 | 0.6571 | 27.3 | 53.2 |
| 32 | 128 | 0.4778 | 0.4670 | 11.3 | 48.1 |
| 48 | 128 | 0.3816 | 0.3741 | 30.3 | 42.6 |
| 64 | 128 | 0.3428 | 0.3387 | 1.9 | 7.4 |

Consequently, TinyLFU-LRU's charged wins at 32 tokens and above are complexity
wins, not behavioral wins. This distinction does not make coarse geometries
deployable: both policies still have near-zero request-tail service and
negative raw scores. At the viable 16-token geometry, the evolved incumbent
wins both charged and behavioral comparisons. At 24 tokens it also dominates
TinyLFU-LRU, but both remain far below their 16-token results.

## vLLM APC comparison

The benchmark's deployable vLLM APC approximation was evaluated on the same
grid. It receives no source-complexity charge.

| Tokens/block | New incumbent | vLLM APC | Delta |
|---:|---:|---:|---:|
| 16 | 66.710 | 49.846 | +16.864 |
| 24 | 21.653 | 9.601 | +12.052 |
| 32 | -14.205 | -20.372 | +6.168 |
| 48 | -29.740 | -54.794 | +25.054 |
| 64 | -31.251 | -55.551 | +24.300 |

The incumbent has higher token hit in all 20 geometry cells. Representative
comparisons are:

| Tokens/block | Blocks | New hit | vLLM APC hit | New churn/1k | vLLM APC churn/1k | vLLM underfill |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 24 | 0.5772 | 0.5196 | 280.4 | 733.0 | 0.168 |
| 16 | 128 | 0.6559 | 0.6056 | 16.2 | 43.5 | 0.228 |
| 24 | 24 | 0.6027 | 0.5484 | 318.3 | 492.2 | 0.085 |
| 24 | 128 | 0.6671 | 0.6222 | 27.3 | 42.6 | 0.160 |
| 32 | 128 | 0.4778 | 0.3337 | 11.3 | 7.4 | 0.524 |
| 48 | 128 | 0.3816 | 0.1579 | 30.3 | 3.7 | 0.744 |
| 64 | 128 | 0.3428 | 0.1183 | 1.9 | 0.0 | 0.844 |

APC's churn falls at coarse granularities because it increasingly fails to fill
the available cache with reusable complete blocks. Its underfill rises from
roughly `0.17--0.23` at 16 tokens to `0.81--0.84` at 64 tokens, while
avoidable-rejection regret also grows sharply. Low churn in this regime is not
efficient replacement; it is evidence that coarse full-block admission cannot
represent the workload's reusable prefix boundaries.

## Full registered-baseline sweep

The remaining registered baselines were evaluated on the same four-capacity
grid. The command was:

```bash
.venv/bin/python scripts/sweep_prefix_kv_baselines.py --workers 4
```

The complete machine-readable results are in
[`baseline_geometry_sweep.json`](baseline_geometry_sweep.json). Scores below
include the incumbent's `7.603`-point source-complexity charge; registered
baselines receive no corresponding charge.

| Policy | Group | 16 | 24 | 32 | 48 | 64 |
|---|---|---:|---:|---:|---:|---:|
| `evolved_incumbent` | deployable | 66.710 | 21.653 | -14.205 | -29.740 | -31.251 |
| `lfu` | deployable | 66.807 | 12.327 | -8.744 | -22.698 | -23.753 |
| `tinylfu_lru` | deployable | 66.559 | 9.766 | -10.808 | -24.187 | -24.824 |
| `depth_prefer_shallow` | deployable | 44.643 | 16.426 | -11.940 | -24.072 | -24.963 |
| `prefix_fanout` | deployable | 45.501 | 12.011 | -10.540 | -24.033 | -24.036 |
| `cost_aware_lru` | deployable | 57.449 | 8.598 | -11.366 | -23.765 | -24.511 |
| `prefix_anchor` | deployable | 57.334 | 8.513 | -10.605 | -23.978 | -24.716 |
| `lru` | deployable | 56.823 | 8.110 | -11.218 | -24.326 | -24.934 |
| `sglang_radix_attention` | deployable | 56.823 | 8.110 | -11.218 | -24.326 | -24.934 |
| `tenant_fair_lru` | deployable | 56.798 | 11.205 | -11.214 | -24.350 | -24.934 |
| `vllm_apc` | deployable | 49.846 | 9.601 | -20.372 | -54.794 | -55.551 |
| `recompute_greedy` | deployable | 29.511 | -16.086 | -20.424 | -29.779 | -30.340 |
| `no_cache` | deployable | -61.310 | -61.440 | -61.423 | -61.804 | -61.456 |
| `future_reuse_heuristic` | reporting-only | 66.806 | 27.812 | -14.340 | -28.490 | -29.877 |
| `oracle_future_reuse` | reporting-only | 83.279 | 47.251 | -4.736 | -21.537 | -24.361 |

Several distinctions matter:

- At 16 tokens, LFU narrowly leads the charged ranking by `0.097`. The
  incumbent's score before complexity is `74.312`, however, which is `7.506`
  points above LFU. Their validation token hit is nearly identical (`0.6109`
  versus `0.6107`), while the incumbent's churn is `105.2` versus LFU's
  `595.4` per 1,000 requests. This is a complexity-accounting win for LFU, not
  evidence of better operational behavior.
- At 24 tokens, the incumbent is the strongest deployable policy by `5.227`
  charged points over depth-prefer-shallow and `9.326` over LFU. This is the
  geometry where selective admission contributes the clearest ranking gain.
- At 32, 48, and 64 tokens, LFU has the best charged deployable score. The
  incumbent still leads LFU before complexity by `2.142`, `0.561`, and `0.104`
  points, respectively. These shrinking differences show policy choice
  becoming irrelevant as block geometry removes reusable boundaries.
- Even the future-reuse oracle becomes negative at 32 tokens and above. The
  geometry failure therefore persists despite perfect reuse information inside
  the simulator's policy interface.
- SGLang RadixAttention and LRU are exactly equal here because the benchmark
  models both as admit-all inactive-leaf LRU. This result does not compare
  production scheduling, radix-tree lookup, or implementation overhead.

The all-baseline result refines the admission-versus-eviction claim. Admission
is not universally dominant: at the viable 16-token geometry, LFU eviction is
already strong enough to tie the incumbent's hit rate. Selective admission's
measured value is lower churn and stronger behavior before complexity, and its
largest charged advantage appears at 24 tokens. Beyond 32 tokens, neither
admission nor eviction strategy repairs the dominant loss of prefix resolution.
