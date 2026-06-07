# Pressure-Aware Incumbent Promotion

The original 300-evaluation compact-policy search completed 298 recorded evaluations
and produced the pressure-aware policy. Focused follow-up refinements then promoted
the current deployable incumbent:

- Stable source:
  `src/prefix_cache_evolve/problems/prefix_kv_cache/pressure_aware_incumbent.py`
- Latest 300-budget run: `runs/20260607_231959_613687`
- Retained latest-run artifact: `artifacts/prefix_kv_cache_runs/20260607T135914Z`
- Reported latest-run model cost: `$3.608`
- Selection score: `76.630`, up from the compact seed's `74.321`
- Aggregate probe score: `75.002`
- Hidden score: `9.691`
- Strongest latest-run mutation: `76.121`

## Policy Change

The promoted policy preserves the compact seed's decayed frequency, priority,
structural-value, recompute-cost, and eviction terms. It adds:

- a decayed global admission-pressure signal derived from recent cache pressure and
  miss rate;
- stronger admission resistance for deep, low-evidence blocks during pressure bursts;
- last-access-gap reuse evidence;
- priority credit only on observed cache hits rather than all misses from a
  high-priority request.
- recurrence-gated depth relief: a first deep miss retains the full depth penalty,
  while repeated observed accesses progressively relax it.
- a small extra admission throttle once decayed pressure exceeds `0.8`.

The winning mutation refined the strongest GPT-5.5 paradigm by accumulating admission
pressure across requests with a four-step half-life and reducing the pressure penalty
for blocks that already have reuse or priority evidence. The follow-up diagnosis
separated agent-trace underfill from stochastic-mix noise: the former needed a narrow
escape from categorical deep rejection, while the latter still needed persistent
pressure throttling. The latest 300-budget run independently refined the sustained
pressure coefficient to `0.22`; composing it with recurrence-gated depth relief
produced the strongest current policy.

## Adjudication

| Candidate | Selection | Raw before cx | Mean | Min contrib. | Churn cost | Underfill cost | Cx | Probe | Agent hit | Cyclic hit | Hidden |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Compact seed | 74.321 | 80.914 | 73.991 | 21.293 | 5.925 | 0.781 | 473 | 47.425 | 0.2295 | 0.8272 | -1.951 |
| Original pressure-aware policy | 76.069 | 84.185 | 74.870 | 20.251 | 2.464 | 0.810 | 624 | 54.543 | 0.2476 | 0.8812 | 6.527 |
| Recurrence-aware refinement | 76.480 | 84.241 | 75.066 | 20.244 | 2.615 | 0.791 | 588 | 74.602 | 0.3756 | 0.8812 | 7.266 |
| Current composed incumbent | 76.630 | 84.598 | 75.044 | 20.251 | 2.230 | 0.804 | 609 | 75.002 | 0.3760 | 0.8812 | 9.691 |

The current policy improves selection, aggregate probe, agent-trace behavior, and
hidden score. Relative to the original pressure-aware policy, capacity-48/96 token
hit rises from `0.654`/`0.658` to `0.674`/`0.687`; validation churn falls from
`164.2` to `148.7` per 1,000 requests. Simplifying fixed decay parameters offsets
the added focused terms, keeping effective complexity at `609` versus `624`.

The latest diversified run completed 298 recorded evaluations with one occupied
archive cell, 254 errors, and two accepted improvements at evaluations 84 and 235.
Its best-score curve rose from `76.069` to `76.115` and finally `76.121`. Because
data-driven initialization retained only duplicate valid seed variants, the archive
never reached the three occupied cells required by punctuated equilibrium.

## Remaining Weakness

Agent-trace performance improves but remains behind robust replacement baselines:

| Policy | Agent trace token hit | Cyclic pressure token hit |
|---|---:|---:|
| Pressure-aware incumbent | 0.376 | 0.881 |
| TinyLFU-LRU | 0.382 | 0.764 |
| vLLM APC | 0.393 | 0.802 |
| Oracle future reuse | 0.402 | 0.894 |

The incumbent now approaches robust replacement baselines on irregular branching
agent traces while using far less churn: `57.3` per 1,000 agent-trace requests versus
thousands for the replacement baselines. Stochastic serving mixes remain the clearest
selection weakness, but sustained-pressure throttling reduces their churn from
`277.8` to `191.0` per 1,000 requests and token-weighted admission waste from `0.647`
to `0.573`.
