# Analysis And Report Tools

This directory owns the consolidated `prefix-cache-tools` command tree for
diagnostic analysis, causal experiments, controlled ablations, and deterministic
tuning. Run commands from the repository root after installing the development
environment:

```bash
make setup-dev
.venv/bin/prefix-cache-tools --help
```

Use JSON artifacts as the machine-readable record. Markdown outputs are
convenience summaries, while the interpretation and retained lessons belong in
the research log. Do not compare scores with different `verifier_version`,
`panel_sha256`, or `evaluation_context_sha256` values.

## Command Tree

```text
prefix-cache-tools
├── analyze
│   ├── eviction
│   ├── reasoning-kv
│   ├── rediscovery
│   └── regret
├── ablate
│   └── structured
├── incumbents
│   ├── list
│   └── validate
└── tune
    └── compact
```

Run any command with `--help` for its complete option list:

```bash
.venv/bin/prefix-cache-tools analyze regret --help
```

## Incumbent Registry

List the immutable promoted-policy bundles and validate their source,
provenance, complexity, and benchmark pins:

```bash
.venv/bin/prefix-cache-tools incumbents list
.venv/bin/prefix-cache-tools incumbents validate
```

Promotion creates a new bundle and changes the current role in
`incumbents/registry.json`; existing bundle source is never overwritten.

## Analyses

### Eviction Specialist Analysis

Compares eviction choices, measures same-state avoidable choices, evaluates
specialist variants, and reports compact distillations.

```bash
.venv/bin/prefix-cache-tools analyze eviction
```

Default outputs:

- `artifacts/prefix_kv_cache_eviction_analysis.json`
- `artifacts/prefix_kv_cache_eviction_analysis.md`

This analysis can identify useful eviction mechanisms, but it does not promote a
candidate. Promotion still requires complete-policy composition and fail-closed
cross-panel adjudication.

### Admission And Eviction Regret

The default mode audits avoidable admission, avoidable rejection, and
value-weighted avoidable eviction by workload-capacity-seed group.

```bash
.venv/bin/prefix-cache-tools analyze regret
```

Default outputs:

- `artifacts/prefix_kv_cache_admission_eviction_regret_audit.json`
- `artifacts/prefix_kv_cache_admission_eviction_regret_audit.md`

The same command exposes three mutually exclusive mechanism experiments.

Measure oracle and policy-implied admission shadow-price trajectories:

```bash
.venv/bin/prefix-cache-tools analyze regret \
  --shadow-price \
  --splits validation \
  --capacity-blocks 8 --capacity-blocks 16 \
  --capacity-blocks 24 --capacity-blocks 48
```

Default output:
`artifacts/prefix_kv_cache_shadow_price_tracking.json`.

Run the crossed incumbent/oracle admission-by-eviction causal factorial:

```bash
.venv/bin/prefix-cache-tools analyze regret --causal-components
```

Default output:
`artifacts/prefix_kv_cache_causal_component_factorial.json`.

Cross every distinct built-in admission policy with representative eviction
rules:

```bash
.venv/bin/prefix-cache-tools analyze regret --all-admission-policies
```

Default outputs:

- `artifacts/prefix_kv_cache_admission_eviction_policy_matrix.json`
- `artifacts/prefix_kv_cache_admission_eviction_policy_matrix.md`

Use repeated `--splits`, `--workloads`, `--seeds`, and, where supported,
`--capacity-blocks` options to select a smaller panel. These narrower runs are
diagnostics unless the scope is explicitly part of a declared experiment.

### Shared Reasoning-KV Pressure

Replays registered policies with active decode KV charged against the same
capacity as reusable prefixes.

```bash
.venv/bin/prefix-cache-tools analyze reasoning-kv
```

Default outputs:

- `artifacts/prefix_kv_cache_reasoning_kv_analysis.json`
- `artifacts/prefix_kv_cache_reasoning_kv_analysis.md`

This is a robustness analysis. It demonstrates when prefix eviction alone is
insufficient; it is not a scheduler benchmark.

### Weak-Seed Rediscovery

Adjudicates saved weak-seed evolution runs against the unchanged canonical
selection, probe, and hidden panels.

```bash
.venv/bin/prefix-cache-tools analyze rediscovery \
  --run artifacts/prefix_kv_cache_rediscovery_runs/weak_initial/<run-a> \
  --run artifacts/prefix_kv_cache_rediscovery_runs/weak_initial/<run-b>
```

Default output:
`artifacts/prefix_kv_cache_rediscovery_analysis.json`.

A generated candidate is behaviorally close only when it is valid, deployable,
passes the agentic gate, and recovers the configured weak-seed-to-incumbent
charged-score gap on every canonical panel. Two distinct successful search seeds
are required to support the discoverability claim. `--quick` is smoke-only and
must not be used for the final verdict.

## Ablation And Tuning

### Structured Policy Ablation

Disables structured policy terms one at a time to measure which mechanisms
carry behavior.

```bash
.venv/bin/prefix-cache-tools ablate structured
```

Default outputs:

- `artifacts/prefix_kv_cache_structured_ablation.json`
- `artifacts/prefix_kv_cache_structured_ablation.md`

### Compact Policy Tuning

Samples compact-policy coefficients on a quick panel, then evaluates the top
samples on the full panel. Results are emitted as JSON lines on standard output.

```bash
.venv/bin/prefix-cache-tools tune compact --samples 180 --full-top 12
```

Use `--decay-ablation` to evaluate explicit frequency and priority half-lives:

```bash
.venv/bin/prefix-cache-tools tune compact \
  --decay-ablation \
  --frequency-half-life 12 \
  --priority-half-life 1.5
```

Tuning proposes parameter sets; it does not edit or promote a policy source.

## Evaluator-Owned Reports

The main `prefix-cache-evolve` runner owns reports that directly evaluate a
candidate or replay a trace. Inspect all available report modes with:

```bash
.venv/bin/prefix-cache-evolve --help
```

Common reports include:

```bash
# Baseline comparison.
.venv/bin/prefix-cache-evolve \
  --baseline-report \
  --candidate-program src/prefix_cache_evolve/problems/prefix_kv_cache/incumbents/production_16tok_20260609/policy.py

# Quarantined probe and hidden final-adjudication panels.
.venv/bin/prefix-cache-evolve \
  --probe-report \
  --candidate-program src/prefix_cache_evolve/problems/prefix_kv_cache/incumbents/production_16tok_20260609/policy.py
.venv/bin/prefix-cache-evolve \
  --hidden-report \
  --candidate-program src/prefix_cache_evolve/problems/prefix_kv_cache/incumbents/production_16tok_20260609/policy.py

# Block-size robustness and score-weight sensitivity.
.venv/bin/prefix-cache-evolve \
  --block-size-report \
  --candidate-program src/prefix_cache_evolve/problems/prefix_kv_cache/incumbents/production_16tok_20260609/policy.py
.venv/bin/prefix-cache-evolve \
  --sensitivity-report \
  --candidate-program src/prefix_cache_evolve/problems/prefix_kv_cache/incumbents/production_16tok_20260609/policy.py
```

Trace calibration and replay consume user-supplied anonymized metadata:

```bash
.venv/bin/prefix-cache-evolve --calibrate-trace trace.jsonl
.venv/bin/prefix-cache-evolve \
  --replay-trace trace.jsonl \
  --candidate-program src/prefix_cache_evolve/problems/prefix_kv_cache/incumbents/production_16tok_20260609/policy.py
```

See `configs/prefix_kv_trace_schema.json` and `docs/reproducibility.md` before
publishing trace-derived results.

## Standalone Report Scripts

Two report-specific scripts remain outside the consolidated CLI:

```bash
# Regenerate the retained-search trajectory TikZ figure.
.venv/bin/python scripts/plot_prefix_kv_eval_trajectory.py

# Sweep the incumbent and all registered baselines across cache geometries.
.venv/bin/python scripts/sweep_prefix_kv_baselines.py
```

Default outputs:

- `docs/figures/incumbent_eval_trajectory.tex`
- `docs/results/baseline_geometry_sweep.json`

The trajectory script requires the retained Levi snapshots named in its source.
The geometry sweep spans intentionally different evaluation contexts by block
size and records one score identity per geometry.

## Result Discipline

- Treat future-aware oracle and constrained-next-use policies as reporting-only.
- Keep probe and hidden panels outside normal search selection.
- Treat `--quick`, reduced request counts, narrowed workloads, and reduced seed
  sets as diagnostics unless the experiment explicitly declares that scope.
- Do not promote from a scalar score alone. Inspect complexity, tripwires,
  aggregate probe, hidden performance, and the exact candidate source.
- Record supported conclusions in `docs/technical_report.tex` and chronology,
  failed runs, and detailed lessons in `docs/research_log.tex`.
