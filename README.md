# Prefix Cache Evolve

## Motivation

The KV-cache is a central object of importance in LLM inference. In particular,
prefix KV-caches avoid repeated prefill work when requests share prompt
prefixes. However, any cache policy must be robust on a variety of simulated
traffic such as agentic systems, tool use, multi-turn sessions, and templated
serving workloads. Limited capacity makes cache admission and eviction a
consequential online decision. A policy must preserve reusable prefixes without
causing excessive churn, unfairness, or unused capacity.

Modern KV-caches are typically hand-designed to handle a variety of workflow
loads, such as vLLM's
[Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/).
With the advent of more powerful LLMs and, in particular, coding agents, we ask:

> Can LLM-guided program evolution discover better online prefix KV-cache
> admission and eviction heuristics than hand-written baselines?

We attempt to evolve an optimal KV-cache policy. Broadly speaking, we set up a
strong verifier and scoring system and use LLMs as mutation operators via
[Levi](https://ttanv.github.io/levi/docs#why-levi) to evolve candidate policies
on a wide set of workload scenarios. The goal is to evolve a policy better than
any hand-designed KV-cache policy.

Our repo is a reproducible benchmark and policy-search harness for
studying that question. It combines a deterministic prefix-tree simulator,
deployable and future-knowledge baselines, diverse synthetic workloads, trace
replay, and an LLM-guided code-evolution loop.

This is a research benchmark, not a drop-in replacement for the cache manager in
vLLM, SGLang, TensorRT-LLM, or another serving stack.

## Research Questions

1. Can program evolution produce deployable online policies that beat strong
   hand-written baselines?
2. Which online signals are genuinely useful: recurrence, subtree value,
   admission pressure, tenant or session metadata, or priority?
3. Which workload families and diagnostics expose overfitting or quiet
   regressions?
4. Can evolved policies transfer across cache geometries and from synthetic
   workloads to production trace replay?

The repo tests these questions with fixed multi-seed workload panels, strong
deployable baselines, held-out probes, hidden final-adjudication workloads,
cache-geometry sweeps, trace replay, and controlled ablations.

## Run It

Python 3.11 or newer is required.

```bash
# Evaluator, simulator, reports, and interactive lab.
make setup

# Launch the policy comparison lab.
uv run prefix-cache-lab

# Run a fast baseline smoke comparison.
uv run prefix-cache-evolve --baseline-report --quick
```

Evaluate the production incumbent on the full validation panel:

```bash
uv run prefix-cache-evolve \
  --baseline-report \
  --candidate-program \
  src/prefix_cache_evolve/problems/prefix_kv_cache/incumbents/production_16tok_20260609/policy.py
```

For development:

```bash
make setup-dev
make check
```

For optional LLM-guided evolution:

```bash
make setup-evolution

# Inspect the resolved configuration without contacting a model provider.
uv run prefix-cache-evolve --show-config

# Start an incumbent-seeded search.
uv run prefix-cache-evolve --iterations 100
```

Evolution uses external model services and may incur cost. Read the
[reproducibility and model-provider guide](docs/reproducibility.md) before a
paid run.

## Interactive Lab

![Prefix Cache Lab demo](assets/prefix_lab.gif)

The Prefix Cache Lab replays selected policies over the same deterministic
request stream. It provides request-by-request inspection of policy rankings,
metric trajectories, admissions, evictions, latency, and resident prefix state.

```bash
uv run prefix-cache-lab --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. Hidden workloads remain excluded from the UI.

## System

![System diagram](assets/system_overview.png)

1. A candidate implements online admission, eviction, and lifecycle callbacks.
2. Static checks and isolated execution enforce the candidate contract.
3. The deterministic simulator evaluates the candidate across workloads, seeds,
   capacities, and cache geometries.
4. The verifier scores hit rate, service tails, churn, underfill, admission
   waste, avoidable eviction, fairness, and source complexity.
5. Fine-grained failures and workload diagnostics guide subsequent mutations.
6. Final promotion requires deployable complexity and fail-closed cross-panel
   adjudication.

Promoted policies are stored as immutable source-and-manifest bundles under
`src/prefix_cache_evolve/problems/prefix_kv_cache/incumbents`. Validate source,
provenance, and benchmark pins with `uv run prefix-cache-tools incumbents validate`.

Candidate-visible fields are online-computable at or before the current request.
Future-use information is quarantined to reporting-only baselines and verifier
audits. Held-out probes and hidden workloads remain outside normal selection.

Detailed documentation:

- [Project overview and current results](docs/project_overview.md)
- [Technical report](docs/technical_report.tex)
- [Research log](docs/research_log.tex)
- [Reproducibility and model providers](docs/reproducibility.md)
- [Analysis and report tools](src/prefix_cache_evolve/tools/README.md)
