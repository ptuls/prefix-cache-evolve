# Priority-Aware Pressure Term Ablation

Run `artifacts/prefix_kv_cache_runs/20260608T001616Z` provides an exact controlled
ablation of the discovered fourth admission term. Its saved seed and winner differ by
only:

```python
- 0.18 * max(0.0, 1.0 - priority) * max(0.0, self._admission_pressure - 0.25)
```

## Verdict

The term is load-bearing, not incidental.

| Metric | Without term | With term | Delta |
|---|---:|---:|---:|
| Charged selection | 76.630 | 77.230 | +0.600 |
| Raw selection before complexity | 84.598 | 85.462 | +0.863 |
| Validation churn per 1k | 148.7 | 92.7 | -56.0 (-37.6%) |
| Validation token-weighted admission waste | 0.421 | 0.407 | -0.014 |
| Hidden score | 9.691 | 11.158 | +1.467 |
| Effective complexity | 609 | 636 | +27 |

The term starts throttling low-priority admissions once decayed pressure exceeds
`0.25`, earlier than the existing all-block persistent-pressure threshold at `0.8`.
It preserves priority-backed blocks while reducing noisy admission and turnover.

## Probe Decomposition

The charged aggregate probe score falls from `75.002` to `74.785`, but no behavioral
probe sub-metric causes that decline. Raw probe score before complexity improves by
`0.047`; the entire charged decline comes from the complexity-cost increase of
`0.264`.

| Probe component | Delta with term |
|---|---:|
| Charged aggregate probe | -0.217 |
| Raw probe before complexity | +0.047 |
| Mean workload score | +0.012 |
| Minimum-workload contribution | +0.024 |
| Churn cost | -0.013 |
| Underfill cost | +0.002 |
| Complexity cost | +0.264 |
| Agent-trace token hit | +0.000046 |
| Agent-trace churn per 1k | -1.736 |
| Cyclic token hit | 0.000 |
| Cyclic churn per 1k | 0.000 |

The agentic surrogate-to-probe tripwire also passes with an absolute token-hit gap of
`0.1025`, below the `0.12` threshold. This rules out a quiet held-out behavioral
regression and supports promotion of the new policy.
