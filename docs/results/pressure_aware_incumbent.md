# Pressure-Aware Incumbent Promotion

> The promotion score below was measured on the historical 8-token,
> 48/96-block discovery verifier. The operative production-oriented verifier
> now uses 16-token blocks and 24/48-block tiers; see
> [`block_size_robustness.md`](block_size_robustness.md).

The promoted deployable incumbent is the exact winner from the retained historical
discovery search:

- Stable source:
  `src/prefix_cache_evolve/problems/prefix_kv_cache/pressure_aware_incumbent.py`
- Evolution run: `runs/20260608_093834_121786`
- Retained run artifact: `artifacts/prefix_kv_cache_runs/20260608T001616Z`
- Reported model cost: `$4.558`
- Selection score: `77.230`
- Raw selection before complexity: `85.462`
- Charged aggregate probe score: `74.785`
- Hidden score: `11.158`
- Agentic surrogate-to-probe tripwire: pass, `0.1025 < 0.12`

## Discovered Change

The winner makes exactly one semantic change to the prior `76.630` incumbent:

```python
- 0.18 * max(0.0, 1.0 - priority) * max(0.0, self._admission_pressure - 0.25)
```

The term begins throttling low-priority admissions under moderate persistent pressure,
earlier than the existing all-block threshold at pressure `0.8`. It preserves
priority-backed blocks while reducing noisy admission and cache turnover.

## Exact Ablation

The saved seed and winner differ only by the discovered fourth admission term, making
the run's seed-versus-winner decomposition an exact controlled ablation.

| Candidate | Selection | Raw before cx | Churn cost | Validation churn/1k | Cx | Probe | Hidden |
|---|---:|---:|---:|---:|---:|---:|---:|
| Without fourth term | 76.630 | 84.598 | 2.230 | 148.7 | 609 | 75.002 | 9.691 |
| Promoted incumbent | 77.230 | 85.462 | 1.391 | 92.7 | 636 | 74.785 | 11.158 |

Removing the term loses `0.600` charged selection and `0.863` raw selection, forfeits
a `37.6%` reduction in validation churn, and loses `1.467` hidden points. The term is
load-bearing, not incidental.

## Probe Adjudication

The `-0.217` charged aggregate-probe delta is not a quiet behavioral regression.
Raw probe score improves by `0.047`; the entire charged decline comes from the
`0.264` larger source-complexity cost.

| Probe component | Delta with term |
|---|---:|
| Raw probe before complexity | +0.047 |
| Mean workload score | +0.012 |
| Minimum-workload contribution | +0.024 |
| Churn cost | -0.013 |
| Underfill cost | +0.002 |
| Agent-trace token hit | +0.000046 |
| Agent-trace churn per 1k | -1.736 |
| Cyclic token hit | 0.000 |
| Cyclic churn per 1k | 0.000 |
| Complexity cost | +0.264 |

The tripwire also passes, so promotion does not weaken held-out probe discipline.
Full details are retained in
[`priority_aware_pressure_ablation.md`](priority_aware_pressure_ablation.md).

## Methodological Result

The important outcome is not only the `77.230` score. Earlier seed-locked searches
optimized the combined scalar from one archive region and could not structurally
retain specialist steps for the two remaining weak workloads. The successful run
changed the search process without leaking the held-out probe:

- mutation feedback always included the non-quarantined `agentic_tool_workflows`
  surrogate and selectable `stochastic_serving_mix`;
- agentic hit and underfill plus stochastic-mix hit became specialist archive
  dimensions;
- the held-out `agent_trace_branching` probe remained reporting-only;
- an automatic surrogate-to-probe divergence tripwire guarded against proxy drift.

That search discovered one compact pressure term that improves both targeted
weaknesses operationally:

| Workload | Churn/1k before | Churn/1k after | Waste before | Waste after |
|---|---:|---:|---:|---:|
| Agentic tool workflows | 142.4 | 74.7 | 0.104 | 0.044 |
| Stochastic serving mix | 191.0 | 164.9 | 0.573 | 0.556 |

The result supports held-out-faithful diagnostics and specialist archive dimensions as
a practical way to escape seed-locked scalar-search basins while preserving final
probe quarantine.
