# Eviction Decision and Distillation Analysis

This analysis compares eviction rankers on the incumbent's exact cache states, then
runs each ranker end to end with admission and lifecycle callbacks frozen.

## Same-State Decision Summary

- Eviction decisions observed: `5402`.
- Decisions with multiple legal victims: `77.5%` (mean `10.64`, max `47`).
- Full specialist changed the incumbent victim on `21.5%` of decisions.
- On changed decisions, it selected a later-reused victim `31.4%` of the time and an earlier-reused victim `19.1%` of the time.
- It corrected `281` avoidable choices and introduced `150`, a net reduction of `131`. Short-reuse corrections were net `47`.

| Alternative | Changed | Better on changed | Worse on changed | Avoidable rate delta | Short-reuse rate delta |
|---|---:|---:|---:|---:|---:|
| `descendant_reweight` | 15.0% | 24.9% | 17.4% | -0.722% | +0.370% |
| `age_descendant_reweight` | 10.9% | 20.7% | 25.3% | +0.333% | +0.648% |
| `one_term_reuse_interaction` | 1.6% | 51.2% | 28.6% | -0.407% | -0.130% |
| `two_term_reuse_interactions` | 0.8% | 53.3% | 22.2% | -0.315% | -0.111% |
| `guarded_support` | 13.4% | 29.1% | 17.2% | -1.648% | -1.074% |
| `guarded_support_descendant` | 21.8% | 29.8% | 20.8% | -1.740% | -0.611% |
| `full_specialist` | 21.5% | 31.4% | 19.1% | -2.425% | -0.870% |

### Full Specialist by Split

| Split | Decisions | Multiple legal | Mean legal | Changed | Avoidable rate delta | Short-reuse rate delta |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 1561 | 66.6% | 9.60 | 7.1% | -2.434% | +0.961% |
| `validation` | 1169 | 99.7% | 15.22 | 23.9% | -3.593% | -0.513% |
| `probe` | 811 | 16.0% | 1.80 | 4.7% | -1.480% | +0.123% |
| `hidden` | 1861 | 99.7% | 12.48 | 39.5% | -2.096% | -3.063% |

## End-to-End Adjudication

| Variant | Selection raw | Delta | Validation hit | Avoidable | Short reuse | Churn/1k | Probe raw | Hidden raw | Function cx | Composed cx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `incumbent` | 70.989 | +0.000 | 0.5982 | 0.1530 | 0.0378 | 168.1 | 63.568 | 10.456 | 66 | 636 |
| `descendant_reweight` | 71.157 | +0.168 | 0.5986 | 0.1460 | 0.0378 | 166.7 | 63.690 | 10.450 | 66 | 636 |
| `age_descendant_reweight` | 71.033 | +0.044 | 0.5983 | 0.1513 | 0.0379 | 167.9 | 63.697 | 10.297 | 66 | 636 |
| `one_term_reuse_interaction` | 70.997 | +0.007 | 0.5982 | 0.1526 | 0.0378 | 167.9 | 63.608 | 10.470 | 121 | 691 |
| `two_term_reuse_interactions` | 71.000 | +0.011 | 0.5982 | 0.1527 | 0.0377 | 167.9 | 63.608 | 10.506 | 133 | 703 |
| `guarded_support` | 71.086 | +0.097 | 0.5985 | 0.1499 | 0.0376 | 167.1 | 63.988 | 10.736 | 178 | 748 |
| `guarded_support_descendant` | 72.069 | +1.080 | 0.5992 | 0.1358 | 0.0372 | 165.0 | 63.930 | 10.445 | 205 | 775 |
| `full_specialist` | 72.091 | +1.101 | 0.5993 | 0.1350 | 0.0370 | 164.8 | 63.911 | 10.584 | 441 | 1011 |

## Interpretation

The specialist is making consequential choices, not merely breaking ties. Same-state
regret improves overall, especially on validation, but it is not uniformly better:
short-reuse regret increases on train and probe. End-to-end replay remains the
promotion criterion because changed victims alter later cache states.

The guarded-support-plus-descendant variant captures `98.0%`
of the full specialist's selection gain, identifying descendant-aware guarded age as
the main useful design. It still composes to `775` nodes. The one- and
two-interaction additions are too weak and also exceed the `650`-node cap.

A single coefficient distillation is useful: increasing descendant protection from
`0.2` to `0.6` gains `+0.168` raw selection at
unchanged composed complexity `636`. It is not a
promotion candidate under the fail-closed rule because hidden score changes
by `-0.006`.
