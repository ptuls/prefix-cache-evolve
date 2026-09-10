# Analysis And Report Tools

`prefix-cache-tools analyze policy-costs` compares raw behavior, implementation
AST, callback wall/CPU distributions, eviction scans and sampled retained state
in the same pinned Docker image. Repeat `--candidate` and `--baseline` to compare
policies; `--reference` checks exact visible-behavior simplifications. See the
[two-stage search and measurement guide](../../../docs/policy_costs.md).

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
├── datasets
│   ├── lmcache-agentic
│   ├── mooncake
│   ├── trace-panel
│   ├── temporal-trace-panel
│   └── wildchat
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

### WildChat Conversion

Install the optional dataset and tokenizer dependencies, then stream the
official [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)
dataset or convert a local JSON, JSONL, or Parquet export:

```bash
make setup-wildchat
export PREFIX_CACHE_TRACE_HASH_KEY="$(openssl rand -hex 32)"

.venv/bin/prefix-cache-tools datasets wildchat \
  --conversation-limit 10000 \
  --minimum-requests-per-conversation 2
```

Default outputs:

- `artifacts/traces/wildchat.jsonl`
- `artifacts/traces/wildchat.jsonl.manifest.json`

The manifest records the resolved Hugging Face commit, tokenizer, HMAC-key
fingerprint, conversion parameters, source limitations, and output SHA-256.
The converter does not write raw messages or identifiers to the trace,
manifest, or temporary sorting database. Hugging Face and local source caching
remain governed by the source loader. The same private HMAC key is required to
reproduce identical prefix hashes.

WildChat supplies assistant response-completion timestamps and a timestamp for
the last response in each conversation. By default these recorded message times
serve as replay proxies. Missing times use `--turn-spacing-ms`, bounded by
neighboring recorded responses and anchored at the last response when needed.
`--timestamp-mode synthetic` ignores message timestamps and spaces every turn.
The manifest counts recorded and synthetic timestamps. Turn IDs distinguish
sessions with identical content and deduplicate overlapping snapshots. Recorded
times override inferred duplicates; inferred times are adjusted when necessary
to preserve conversation order across snapshots.
The canonical chat serialization cannot reproduce the provider's private system
prompt or exact serving template, and completion times are not request arrivals.
Report this as conversation-derived replay, not production-trace replay.

### LMCache Agentic Conversion

The LMCache Agentic Traces already provide cumulative agent-session inputs,
output lengths, and measured gaps between tool iterations. Convert a pinned
Hugging Face revision or a local JSON, JSONL, or Parquet file:

```bash
export PREFIX_CACHE_TRACE_HASH_KEY="$(openssl rand -hex 32)"

.venv/bin/prefix-cache-tools datasets lmcache-agentic \
  --dataset-revision 6e043b9e89865df3aec19fd5679286b683bfd70e \
  --output artifacts/traces/lmcache-agentic.jsonl
```

The converter validates strict cumulative growth within each session and
writes content-free HMAC prefix paths. It retains measured `pre_gap` values and
adds a deterministic proxy for the preceding decode duration. Because the
source has no absolute cross-session arrival times, sessions are assigned to
deterministic closed-loop concurrency lanes. Tokenization and canonical prompt
serialization approximate each provider's private template. Treat the result
as collected-agent replay rather than production serving telemetry.

### Mooncake Conversion

Convert the native public Mooncake trace format while preserving its 512-token
cumulative prefix hashes and request-arrival timestamps:

```bash
.venv/bin/prefix-cache-tools datasets mooncake \
  --input .cache/mooncake/toolagent_trace.jsonl \
  --expected-sha256 48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71 \
  --output artifacts/traces/mooncake-toolagent.jsonl
```

The output path must be new. `OUTPUT.manifest.json` records checksums, geometry,
counts, and unavailable metadata. Session and tenant identifiers use one unknown
placeholder, and priority is zero. The converter validates cumulative hash
relationships and retains the native granularity; it cannot recover session
holdouts or finer-grained sharing. Use `--replay-trace` with
`--block-size-tokens 512`, and size capacity in those blocks. The full pinned
download and replay recipe is in the
[production replay guide](../../../docs/reproducibility.md#agentic-production-replay-mooncake).

For production sources without session identities, use `datasets temporal-trace-panel`
with repeated `--window SPLIT START_MS END_MS` arguments. This preserves every
request within each disjoint time window and keeps identity placeholders unchanged.
It checks chronological split ordering and writes hash-pinned traces plus a
manifest and evolution config. Temporal separation does not imply independent
sessions; each window starts cold. See the [production evolution recipe](
../../../docs/reproducibility.md#production-trace-evolution-with-chronological-holdouts)
for the native-block configuration, Docker isolation, budgets, and final holdout.
For a focused comparison, repeat `--trace-baseline` on `prefix-cache-evolve`,
for example `--trace-baseline lru --trace-baseline vllm_apc`. The default runs
all deployable baselines. Size `evaluator.timeout` and the evaluator memory
limit for the trace; these settings remain recorded in its evaluation context.

### Trace Panels

For Qwen-Bailian and AgentX imports, pinned downloads, and reserving AgentX as
an independent hidden validation source, see the
[serving dataset guide](../../../docs/serving_datasets.md). The commands are
`datasets qwen`, `datasets agentx`, and `datasets attach-holdout`.
`trace-panel --capacity-tokens` sets trace-specific capacities in token units.

Create a new bundle with train, validation, and hidden traces plus an evolution
configuration:

```bash
.venv/bin/prefix-cache-tools datasets trace-panel \
  --trace artifacts/traces/wildchat.jsonl \
  --family wildchat \
  --output-dir artifacts/traces/wildchat-panel
.venv/bin/prefix-cache-evolve \
  --config artifacts/traces/wildchat-panel/evolution.yaml --baseline-report
```

The default 60%/20%/20% partition keeps tenants and their complete sessions in
one split. `--group-by session` groups by session instead. Validation is visible
during search; hidden is reserved for final evaluation. Synthetic probes are
retained; `--keep-synthetic` also retains synthetic selection and hidden families.
File hashes, request counts, block geometry, and actual partitions are recorded.
Trace streams are evaluated once per capacity, independently of synthetic seeds.

Size cache capacity and evaluator limits in the generated YAML before a larger
run. `--quick` leaves traces complete. See
[Trace Panels For Evolution](../../../docs/reproducibility.md#trace-panels-for-evolution)
for preparation, model-powered search, and saved-run replay.

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
