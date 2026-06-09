# Incumbent Rediscovery Protocol

The normal full-policy search is designed for local improvement: it starts from the
pressure-aware incumbent and its prompt explicitly preserves known useful mechanisms.
That workflow cannot establish that evolution would independently discover the same
policy family.

The rediscovery experiment separates local refinement from discoverability. It uses
[`configs/prefix_kv_cache_rediscovery.yaml`](../configs/prefix_kv_cache_rediscovery.yaml),
which keeps the operative evaluator unchanged but removes:

- the incumbent source as the default experimental seed;
- incumbent scores, baseline scores, and coefficient values;
- instructions to preserve pressure-aware admission or canonical state;
- instructions to default to a local mutation of the supplied parent.

The neutral prompt still discloses documented online fields, including recent admission
pressure and miss rate, because those fields are part of the candidate API. It does not
name or recommend the retained incumbent's algorithm or canonical primitives. Its
`failure_memory.mode` is `run_only`: failures can guide later iterations within one
trial, but no trial reads or writes lessons from another trial.

## Starting Conditions

Run two seed tiers:

| Tier | Seed | Purpose |
|---|---|---|
| Weak | `initial_program.py` | Tests independent discovery from a small mostly stateless hybrid. |
| Intermediate | `compact_seed.py` | Tests whether decayed observed-reuse state is a necessary stepping stone. |

Use at least three independent search seeds per tier. A 100-evaluation pilot is the
minimum useful run because broad initialization consumes a substantial fraction of a
smaller budget. Confirm any positive pilot with the normal 300-evaluation budget.

```bash
uv run prefix-cache-evolve \
  --iterations 100 \
  --config configs/prefix_kv_cache_rediscovery.yaml \
  --seed-program src/prefix_cache_evolve/problems/prefix_kv_cache/initial_program.py \
  --search-seed 101 \
  --artifact-output artifacts/prefix_kv_cache_rediscovery_runs/weak_initial

uv run prefix-cache-evolve \
  --iterations 100 \
  --config configs/prefix_kv_cache_rediscovery.yaml \
  --seed-program src/prefix_cache_evolve/problems/prefix_kv_cache/compact_seed.py \
  --search-seed 101 \
  --artifact-output artifacts/prefix_kv_cache_rediscovery_runs/intermediate_compact
```

Repeat both commands with at least two additional search seeds. Saved runs now always
persist the exact `seed_program.py`, even when the seed remains the winner.

## Adjudication

The adjudicator prefers `best_generated_mutation.py` over a retained seed winner, then
re-evaluates the generated source, its exact seed, and the incumbent on selection,
quarantined probe, and hidden panels:

```bash
uv run prefix-cache-analyze-rediscovery \
  --run artifacts/prefix_kv_cache_rediscovery_runs/weak_initial/<run-a> \
  --run artifacts/prefix_kv_cache_rediscovery_runs/weak_initial/<run-b> \
  --run artifacts/prefix_kv_cache_rediscovery_runs/weak_initial/<run-c>
```

For each panel, gap recovery is:

```text
(generated score - seed score) / (incumbent score - seed score)
```

A **behavioral rediscovery** must be generated, valid, deployable, pass the
multi-metric agentic surrogate gate, and recover at least 80% of the charged
seed-to-incumbent gap on all three panels. A **mechanism rediscovery** must
additionally reproduce the incumbent design family under coarse source-level
signals: pressure-conditioned admission, observed-reuse state, structural admission,
and at least four retained family signals.

The overall discoverability claim passes only after at least two distinct search seeds
produce mechanism rediscoveries from the weakest initial seed. One success is
preliminary. Success only from the compact seed indicates path dependence. No success
means the retained result should be described as incumbent-conditioned local evolution,
not as a generally discoverable solution.

The source-family check is deliberately secondary. A generated policy that reaches
incumbent-level behavior through a different mechanism remains an important result,
but it is evidence that the incumbent is non-unique rather than independently
rediscovered.
