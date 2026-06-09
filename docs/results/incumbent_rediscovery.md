# Incumbent Rediscovery Experiment

**Verdict: `not_supported`.** The supplied runs did not independently recover a deployable incumbent-family policy across all panels.

This experiment tests whether neutral evolution from a weaker base independently
recovers the retained incumbent's behavior or design family. Probe and hidden
panels are used only after search.

## Reference Starting Conditions

| Reference | Selection charged / raw | Probe charged / raw | Hidden charged / raw | Effective complexity | Design-family signals |
|---|---:|---:|---:|---:|---:|
| `weak_initial` | 53.366 / 57.514 | 33.611 / 37.758 | -20.332 / -16.184 | 255 | 1 |
| `intermediate_compact` | 57.151 / 63.744 | 55.676 / 62.268 | -14.431 / -7.838 | 473 | 4 |
| `incumbent` | 65.649 / 73.252 | 74.899 / 82.501 | 3.064 / 10.667 | 572 | 6 |

## Required 80% Gap-Recovery Scores

| Starting seed | Selection charged / raw | Probe charged / raw | Hidden charged / raw |
|---|---:|---:|---:|
| `weak_initial` | 63.192 / 70.104 | 66.641 / 73.553 | -1.615 / 5.297 |
| `intermediate_compact` | 63.949 / 71.350 | 71.054 / 78.455 | -0.435 / 6.966 |

## Run Adjudication

| Run | Seed | Selection charged / raw recovery | Probe charged / raw recovery | Hidden charged / raw recovery | Agentic gate | Complexity | Family | Verdict |
|---|---|---:|---:|---:|---|---:|---:|---|
| `20260609T081715Z` | `weak_initial` | 49.2% / 97.0% | -21.5% / 0.7% | 45.5% / 74.0% | `fail` | 1214 | 5 | `no` |

## Run Interpretation

- `20260609T081715Z` used search seed `101`, completed `98` evaluations, and reported cost `$2.859`.
- The run independently assembled `5` of `6` incumbent-family signals, but this is mechanism emergence rather than behavioral rediscovery.
- Its strongest generated policy has effective complexity `1214`, above the `650` deployability limit.
- It misses the all-panel charged recovery rule; notably, probe recovery is `-21.5%`.
- It fails the agentic surrogate gate on `wasted_admission_token_rate`.

## Decision Rule

- A behavioral rediscovery must recover at least `80%` of the charged seed-to-incumbent gap on selection, probe, and hidden panels while remaining valid, deployable,
  and within every agentic surrogate-gate limit.
- A mechanism rediscovery must also reproduce the incumbent design family under
  coarse AST-derived signals. This is a diagnostic classification, not proof of
  semantic equivalence.
- The overall claim is supported only after at least two independent search seeds
  rediscover an incumbent-family policy from the weakest initial seed.

A failure to rediscover does not invalidate the incumbent as a local-search result.
It means claims should describe it as incumbent-conditioned rather than generally
discoverable.
