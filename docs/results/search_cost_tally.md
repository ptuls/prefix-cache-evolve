# Search Cost Tally

The current production-oriented policy was not discovered in a single run costing less
than $5. The retained logs support three different cost scopes:

| Scope | Included stages | Evaluations | Recorded API cost | Summed run time |
|---|---|---:|---:|---:|
| Final production stage | 16-token production search and simplification | 397 | $5.070 | 65.0 min |
| Retained lineage | Six discovery-verifier runs, production search, and simplification | 1,505 | $21.981 | 221.7 min |
| Documented pre-promotion research | Early old-verifier runs, structured searches, retained lineage, and both eviction-specialist stages | 2,989 | at least $44.845 | at least 389.6 min |

## Component Costs

| Stage | Evaluations | Cost |
|---|---:|---:|
| Early old-verifier iterations | 394 | $6.473 |
| Three structured-policy searches | 294 | $7.682 |
| Six retained discovery-verifier runs | 1,108 | $16.911 |
| Full-policy eviction-specialist run | 298 | $4.376 |
| Function-only eviction-specialist run | 498 | $4.334 |
| Production-oriented search | 299 | $4.133 |
| Dedicated simplification search | 98 | $0.937 |
| **Documented pre-promotion total** | **2,989** | **$44.845** |

The total is a lower bound on research cost. It excludes deterministic coefficient
sweeps, simulator CPU time, report generation, manual analysis and composition, and
failed or unretained experiments without reliable cost metadata. The later weak-seed
rediscovery run is also excluded because it occurred after the current policy had
already been promoted.

## Retained Sources

- Discovery-verifier lineage:
  `artifacts/prefix_kv_cache_runs/*/run_summary.json`
- Production search:
  `artifacts/prefix_kv_cache_production_runs/20260609T045230Z/run_summary.json`
- Simplification:
  `artifacts/prefix_kv_cache_simplification_runs/20260609T051303Z/run_summary.json`
- Full-policy eviction specialist:
  `artifacts/prefix_kv_cache_eviction_specialist_runs/20260608T040632Z/run_summary.json`
- Function-only eviction specialist:
  `runs/20260608_150431_280387/snapshot.json`
- Early and structured stages whose artifact directories are no longer retained:
  the cost tables in `docs/research_log.tex`

The primary productivity result is iteration velocity, not one-shot cheapness.
Fine-grained diagnostics, retained artifacts, specialist experiments, explicit
simplification, and fail-closed adjudication made it possible to turn each result into
the next falsifiable experiment quickly and to preserve useful lessons from failed
candidates.
