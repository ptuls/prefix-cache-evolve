# Prefix Cache Evolve

`prefix-cache-evolve` is a standalone research benchmark for designing and
evaluating admission and eviction policies for LLM prefix KV caches. It combines
a deterministic prefix-tree simulator, a multi-workload verifier, deployable and
future-knowledge baselines, trace replay, and a Levi-driven code-evolution loop.

## What Is Included

- A deterministic prefix-cache simulator with root-contiguous hits, leaf-only
  eviction, active decode pins, forced bypass, and partial blocks.
- Online candidate metadata for recurrence, subtree value, admission pressure,
  miss rate, priority, tenant, and session behavior.
- Deployable baselines including vLLM APC, SGLang RadixAttention, LRU, LFU,
  TinyLFU-LRU, recompute-aware, prefix-fanout, and tenant-fair policies.
- Reporting-only future-knowledge baselines, including a constrained next-use
  oracle.
- Fine-grained verifier metrics for token and block hits, request tails,
  admission utility and waste, avoidable evictions, policy-caused underfill,
  churn, fairness, and complexity.
- Synthetic train, validation, quarantined recurrence probe, and hidden workload
  panels, including irregular agentic tool workflows with forks and replans.
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

# Launch the interactive policy comparison lab.
.venv/bin/prefix-cache-lab

# Fast smoke report. Do not use --quick for ranking decisions.
.venv/bin/prefix-cache-evolve --baseline-report --quick

# Full validation comparison for the pressure-aware incumbent.
.venv/bin/prefix-cache-evolve \
  --baseline-report \
  --candidate-program \
  src/prefix_cache_evolve/problems/prefix_kv_cache/pressure_aware_incumbent.py
```

Install Levi explicitly before launching evolution. It is intentionally not a
normal project dependency because it is currently installed from Git, which
would otherwise make baseline-only development and offline synchronization
resolve a network dependency. Evolution defaults to the pressure-aware incumbent
as its seed; use `--seed-program` to override it.

```bash
uv pip install 'levi @ git+https://github.com/ttanv/levi.git'
.venv/bin/prefix-cache-evolve \
  --iterations 100
```

The main configuration is [`configs/prefix_kv_cache.yaml`](configs/prefix_kv_cache.yaml).
The runner packages a fallback copy so the installed console command can still
load its default configuration outside a source checkout.

Full evolution runs target a 20-program initial population: the pressure-aware
incumbent, four GPT-5.5-generated algorithmically diverse seeds, and three mixed
variants per seed. Failed seed retries can consume additional evaluations. The
data-driven archive retains both performance tradeoffs and structural policy
differences before normal mutation begins.

## Interactive Lab

The Prefix Cache Lab runs selected policies over the same deterministic
synthetic request stream, then provides request-by-request playback in a local
browser UI. It compares final rankings, metric trajectories, admissions,
evictions, latency, and the resident prefix-block state after every request.

```bash
.venv/bin/prefix-cache-lab --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. Deployable baselines, the pressure-aware incumbent,
and clearly labeled reporting-only future-knowledge policies are available.
Hidden workloads remain excluded from the UI. The frontend consumes a
source-agnostic request snapshot contract so a live-traffic adapter can be added
without changing the visualization model.

## Baseline Sources

The `sglang_radix_attention` baseline models the default replacement behavior of
SGLang's radix cache: retain prefixes at cache-page boundaries, protect nodes
referenced by running requests, and recursively evict the least-recently-used
unreferenced leaf. The benchmark treats every modeled block-tree node as a
cacheable radix unit, making the mapped policy behaviorally equivalent to the
generic admit-all `lru` baseline. It does not model SGLang's cache-aware
scheduler or attention kernels. It is a block-tree approximation: the benchmark
charges capacity in fixed simulator blocks rather than SGLang's token/page-
It remains selectable in the interactive lab but is excluded from default
comparisons because it duplicates `lru` under this model.

- Paper: [Efficiently Programming Large Language Models using SGLang](https://arxiv.org/html/2312.07104v1)
- Pinned SGLang implementation:
  [`radix_attention.py`](https://github.com/sgl-project/sglang/blob/52f221cce088abc998fa9d3812416a45ee0e2e25/python/sglang/srt/layers/radix_attention.py),
  [`radix_cache.py`](https://github.com/sgl-project/sglang/blob/52f221cce088abc998fa9d3812416a45ee0e2e25/python/sglang/srt/mem_cache/radix_cache.py),
  [`cache_init_params.py`](https://github.com/sgl-project/sglang/blob/52f221cce088abc998fa9d3812416a45ee0e2e25/python/sglang/srt/mem_cache/cache_init_params.py),
  and [`evict_policy.py`](https://github.com/sgl-project/sglang/blob/52f221cce088abc998fa9d3812416a45ee0e2e25/python/sglang/srt/mem_cache/evict_policy.py).

## Reports And Analysis

```bash
# Quarantined recurrence/structure probe.
.venv/bin/prefix-cache-evolve \
  --probe-report \
  --candidate-program \
  src/prefix_cache_evolve/problems/prefix_kv_cache/pressure_aware_incumbent.py

# Hidden panel for final adjudication only.
.venv/bin/prefix-cache-evolve \
  --hidden-report \
  --candidate-program \
  src/prefix_cache_evolve/problems/prefix_kv_cache/pressure_aware_incumbent.py

# Score-weight sensitivity.
.venv/bin/prefix-cache-evolve \
  --sensitivity-report \
  --candidate-program \
  src/prefix_cache_evolve/problems/prefix_kv_cache/pressure_aware_incumbent.py

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
  --candidate-program \
  src/prefix_cache_evolve/problems/prefix_kv_cache/pressure_aware_incumbent.py
```

## Current Result

The pressure-aware incumbent is the recommended deployable candidate and the
strongest current search parent. A 300-evaluation compact-policy search improved
selection from `74.321` to `76.069`; a focused recurrence-gated depth refinement
then raised selection to `76.480`. Composing that refinement with the persistent
pressure throttle found by the latest 300-budget run raises selection to `76.630`,
aggregate probe to `75.002`, and hidden score to `9.691`.

Key retained findings:

- Decayed admission pressure sharply reduces churn while preserving useful
  reused branches.
- A small extra throttle above sustained admission pressure `0.8` reduces noisy
  churn without suppressing recurrence-backed deep admissions.
- Recurrence-gated depth relief raises agent-trace hit rate from `0.230` to
  `0.376` while holding agent-trace churn to `57.3` per 1,000 requests.
- Broad explicit recurrence terms increased churn in the structured ablation;
  recurrence evidence is useful when it only relaxes an otherwise categorical
  depth penalty.
- Priority-decay state was inert.
- Subtree and online regime context carried useful behavior.
- One generated mutation improved raw validation and both recurrence-family hit
  rates, but its 1,322-node effective implementation failed final adjudication.

See:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/technical_report.tex`](docs/technical_report.tex)
- [`docs/results/baseline_comparison.md`](docs/results/baseline_comparison.md)
- [`docs/results/pressure_aware_incumbent.md`](docs/results/pressure_aware_incumbent.md)
- [`docs/results/structured_ablation.md`](docs/results/structured_ablation.md)
- [`docs/results/three_run_adjudication.md`](docs/results/three_run_adjudication.md)

## Repository Layout

```text
configs/                       Operative Levi/evaluator config and trace schema
docs/                          Technical report and retained result summaries
scripts/                       Deterministic tuner and structured ablation tools
src/prefix_cache_evolve/
  evaluator_entry.py           Candidate loading and isolated evaluation helpers
  evaluators/contracts.py      Candidate-visible policy interface
  evaluators/baselines.py      Baseline policies and capability registry
  evaluators/baseline_suite.py Baseline evaluation orchestration
  evaluators/complexity.py     Candidate source-complexity analysis
  evaluators/prefix_kv_cache.py
                                Simulator, workloads, metrics, and verifier
  problems/prefix_kv_cache/    Runner, reporting, evaluator entry point, seeds,
                                and replay
  workflow/                    Small Levi configuration/execution adapter
tests/                         Functional PyTest coverage
```

## Trust Boundary

Candidate-visible fields are online-computable at or before the current request.
Future-use information is quarantined to reporting-only baselines and verifier
audits. Probe families are reported but excluded from normal candidate selection;
hidden families are reserved for final adjudication.
