# Prefix Cache Evolve

`prefix-cache-evolve` is a standalone research benchmark for designing and
evaluating admission and eviction policies for LLM prefix KV caches. It combines
a deterministic prefix-tree simulator, a multi-workload verifier, deployable and
future-knowledge baselines, trace replay, and a Levi-driven code-evolution loop.

The repository was extracted from `randomize-evolve` so the prefix-cache work can
develop independently. It does not require the original repository at runtime.

## What Is Included

- A deterministic prefix-cache simulator with root-contiguous hits, leaf-only
  eviction, active decode pins, forced bypass, and partial blocks.
- Online candidate metadata for recurrence, subtree value, admission pressure,
  miss rate, priority, tenant, and session behavior.
- Deployable baselines including LRU, LFU, TinyLFU-LRU, recompute-aware,
  prefix-fanout, and tenant-fair policies.
- Reporting-only future-knowledge baselines, including a constrained next-use
  oracle.
- Fine-grained verifier metrics for token and block hits, request tails,
  admission utility and waste, avoidable evictions, churn, fairness, and
  complexity.
- Synthetic train, validation, quarantined recurrence probe, and hidden workload
  panels.
- An anonymized metadata-only trace calibration and replay path.
- Compact and structured policy seeds, a deterministic coefficient tuner, and a
  structured ablation harness.
- A Levi adapter that persists the winning program and automatically decomposes
  the strongest generated non-seed mutation.

## Candidate Contract

A candidate module exports either `build_candidate(...)` or
`candidate_factory(...)`:

```python
def build_candidate(
    capacity_blocks: int,
    block_size_tokens: int,
    seed: int | None = None,
):
    ...
```

The returned policy implements:

```python
def on_request_start(request, now): ...
def on_cache_hit(block, request, now): ...
def on_cache_miss(block, request, now): ...
def score_admission(block, now) -> float: ...
def score_eviction(block, now) -> float: ...
```

Only the three documented lifecycle callbacks fire. Admission occurs when
`score_admission(...) > 0`; the simulator evicts the legal inactive resident leaf
with the highest eviction score. Candidate code cannot mutate simulator-owned
cache state or access future reuse.

## Quick Start

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'

.venv/bin/pytest -q
.venv/bin/ruff check .

# Fast smoke report. Do not use --quick for ranking decisions.
.venv/bin/prefix-cache-evolve --baseline-report --quick

# Full validation comparison for the compact incumbent.
.venv/bin/prefix-cache-evolve \
  --baseline-report \
  --candidate-program src/prefix_cache_evolve/problems/prefix_kv_cache/compact_seed.py
```

Install Levi explicitly before launching evolution. It is intentionally not a
normal project dependency because it is currently installed from Git, which
would otherwise make baseline-only development and offline synchronization
resolve a network dependency.

```bash
uv pip install 'levi @ git+https://github.com/ttanv/levi.git'
.venv/bin/prefix-cache-evolve \
  --iterations 100 \
  --seed-program src/prefix_cache_evolve/problems/prefix_kv_cache/compact_seed.py
```

The main configuration is [`configs/prefix_kv_cache.yaml`](configs/prefix_kv_cache.yaml).
The runner packages a fallback copy so the installed console command can still
load its default configuration outside a source checkout.

## Reports And Analysis

```bash
# Quarantined recurrence/structure probe.
.venv/bin/prefix-cache-evolve \
  --probe-report \
  --candidate-program src/prefix_cache_evolve/problems/prefix_kv_cache/compact_seed.py

# Hidden panel for final adjudication only.
.venv/bin/prefix-cache-evolve \
  --hidden-report \
  --candidate-program src/prefix_cache_evolve/problems/prefix_kv_cache/compact_seed.py

# Score-weight sensitivity.
.venv/bin/prefix-cache-evolve \
  --sensitivity-report \
  --candidate-program src/prefix_cache_evolve/problems/prefix_kv_cache/compact_seed.py

# Structured policy ablation and compact coefficient tuning.
.venv/bin/prefix-cache-ablate-structured
.venv/bin/prefix-cache-tune-compact --samples 180 --full-top 12
```

Trace replay consumes anonymized request metadata while preserving hidden prompt
content. See [`configs/prefix_kv_trace_schema.json`](configs/prefix_kv_trace_schema.json).

```bash
.venv/bin/prefix-cache-evolve --calibrate-trace trace.jsonl
.venv/bin/prefix-cache-evolve \
  --replay-trace trace.jsonl \
  --candidate-program src/prefix_cache_evolve/problems/prefix_kv_cache/compact_seed.py
```

## Current Result

The compact incumbent remains the recommended seed. The latest structured
searches fixed canonical `MultiTimescaleDecay.observe_vector` adoption, but the
best generated mutations still accumulated enough bespoke control flow to lose
after complexity and to regress aggregate probe or hidden behavior.

Key retained findings:

- Explicit recurrence terms increased churn and were harmful in the structured
  ablation.
- Priority-decay state was inert.
- Subtree and online regime context carried useful behavior.
- One generated mutation improved raw validation and both recurrence-family hit
  rates, but its 1,322-node effective implementation failed final adjudication.

See:

- [`docs/technical_report.tex`](docs/technical_report.tex)
- [`docs/results/baseline_comparison.md`](docs/results/baseline_comparison.md)
- [`docs/results/structured_ablation.md`](docs/results/structured_ablation.md)
- [`docs/results/three_run_adjudication.md`](docs/results/three_run_adjudication.md)

## Repository Layout

```text
configs/                       Operative Levi/evaluator config and trace schema
docs/                          Technical report and retained result summaries
scripts/                       Deterministic tuner and structured ablation tools
src/prefix_cache_evolve/
  evaluator_entry.py           Candidate loading and isolated evaluation helpers
  evaluators/prefix_kv_cache.py
                                Simulator, workloads, baselines, and verifier
  problems/prefix_kv_cache/    Runner, evaluator entry point, seeds, and replay
  workflow/                    Small Levi configuration/execution adapter
tests/                         Functional PyTest coverage
```

## Trust Boundary

Candidate-visible fields are online-computable at or before the current request.
Future-use information is quarantined to reporting-only baselines and verifier
audits. Probe families are reported but excluded from normal candidate selection;
hidden families are reserved for final adjudication.
