# Reproducibility and Model Providers

This document separates deterministic policy evaluation from LLM-guided search,
which depends on external model services and asynchronous scheduling.

For independent production and coding-agent datasets, see
[Qwen search and AgentX held-out validation](serving_datasets.md). AgentX is
reserved in `hidden`; the split named `validation` remains visible to search.

## Installation

Use the committed `uv.lock` with Python 3.11, 3.12, or 3.13. Published
benchmark artifacts use Python 3.11 unless they state otherwise:

```bash
# Evaluator, simulator, reports, and lab only.
make setup

# Evolution support through Levi.
make setup-evolution

# Tests, formatter, and type checker; no Git-hosted Levi dependency.
make setup-dev
make check
```

Equivalent commands are `uv sync --frozen --no-default-groups`, `uv sync
--frozen --no-default-groups --extra evolution`, and `uv sync --frozen --group
dev`. To run the complete suite including Levi adapter tests, combine the last
two as `uv sync --frozen --group dev --extra evolution`.

## Unattended Local Runs

On macOS, evolution automatically holds a `caffeinate -i` assertion throughout
initialization, search, and automatic artifact reporting. This prevents idle
system sleep on battery or AC power while allowing the display to sleep. The
assertion is released on completion or failure, and is tied to the runner PID
so it also expires if the process is killed. Other platforms are unchanged.
This does not override deliberate sleep or closing the laptop lid; keep the lid
open for unattended local runs. Search wall-clock budgets continue to include
time spent suspended.

## Seeded Components

The main configuration has three distinct seed controls:

```yaml
search:
  seed: 20260609

problem:
  settings:
    verifier_version: "1.0.0"
    seeds: [11, 23, 37]
    policy_seed: 0
```

- `verifier_version` must match the verifier implemented by the checked-out
  source.
- `problem.settings.seeds` deterministically generates synthetic workloads.
- `policy_seed` is passed to candidate factories independently of workload
  generation.
- `search.seed` seeds Python and NumPy selection inside the Levi process and
  supplies monotonically derived request seeds to model providers that support
  seeded generation.

Inspect all resolved values without making a model request:

```bash
uv run prefix-cache-evolve --show-config
uv run prefix-cache-evolve --show-config --search-seed 17
```

Policy evaluation is deterministic for the same source, config, Python
environment, and deterministic policy. LLM search is not guaranteed to be
bit-for-bit reproducible: remote providers may ignore seeds or change serving
implementations, and concurrent workers can complete in a different order.
For the strongest practical repeatability, set `temperature: 0`,
`pipeline.n_llm_workers: 1`, and `evaluator.parallel_evaluations: 1` in a copied
configuration.

New archive initialization preserves `cvt.n_centroids` even when few candidate
programs pass evaluation. If there are fewer distinct behavior vectors than
requested cells, it uses Levi's tiling of normalized behavior space instead of
shrinking the archive to the surviving sample count. This tiling uses the seeded
NumPy RNG. With sufficient distinct observations, data-driven clustering remains
unchanged. Only valid candidates populate cells; empty cells remain available for
later mutations, including alternatives with lower scores than the global best.
Paradigm search still requires enough occupied cells for its configured clusters.
Resumed historical snapshots retain their saved tiling; start a new run to apply
the revised initialization behavior.

Saved evolution runs contain the configuration snapshot, workload manifest,
resolved model identifiers, search seed, package versions, Git commit and dirty
state, Levi snapshot, model cost, and candidate source. New score records and
snapshot history entries carry a three-part identity:

- `verifier_version`: the semantic verifier contract.
- `panel_sha256`: ordered request streams plus panel geometry.
- `evaluation_context_sha256`: verifier version, normalized evaluator config,
  and panel SHA-256.

Reports reject missing or mixed identities inside an exact comparison.
Intentional cross-context reports, such as geometry sweeps, record and validate
one identity per geometry. Historical unstamped runs remain legacy records and
cannot be tabulated with current versioned results.

## Incumbent Registry

Promoted policies are committed as immutable bundles under
`src/prefix_cache_evolve/problems/prefix_kv_cache/incumbents/`. Each bundle
contains the exact candidate `policy.py` and a `manifest.json` recording:

- source SHA-256, import target, and effective complexity;
- verifier, panel, evaluation-context, score, and headline metric pins;
- originating run, source artifact, recorded evaluations, and API cost;
- lineage and any distinction between original promotion accounting and the
  current replay.

`registry.json` preserves historical incumbents and assigns the current
production and retained discovery roles. Promoting a new policy means adding a
new bundle and changing the registry; do not overwrite an existing bundle.
Validate all stored identities with:

```bash
uv run prefix-cache-tools incumbents validate
uv run prefix-cache-tools incumbents list
```

CI runs the same validation and fails on unregistered bundles, source drift,
complexity drift, stale import targets, or mismatched source-artifact hashes.

## Weak-Seed Rediscovery

Deterministic replay of the incumbent is different from independently finding a
similar policy. The normal search remains incumbent-seeded because it is the
productive optimization lane. Rediscovery uses
`configs/prefix_kv_cache_rediscovery.yaml`, the candidate-valid
`seeds/weak_initial.py` seed, and a neutral prompt with no incumbent source,
score, coefficients, or mechanism-preservation instructions. Search ranks the
minimum of ordinary selection score and a non-quarantined agentic-workflow
guidance score. Final rediscovery adjudication always re-evaluates generated
source with the unchanged canonical `configs/prefix_kv_cache.yaml`; probe and
hidden panels remain unavailable during search.

Ordinary run artifacts include selection and probe results but never evaluate
or disclose the hidden panel. Hidden results are opened only by an explicit
final-adjudication command, a requested hidden report, or specialist promotion
adjudication. Keep those results separate from subsequent search decisions.

Run at least three independent search seeds at the normal 300-evaluation budget:

```bash
uv run prefix-cache-evolve \
  --iterations 300 \
  --config configs/prefix_kv_cache_rediscovery.yaml \
  --seed-program src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/weak_initial.py \
  --search-seed 101 \
  --artifact-output artifacts/prefix_kv_cache_rediscovery_runs/weak_initial
```

Repeat with additional search seeds, then adjudicate every saved run together:

```bash
uv run prefix-cache-tools analyze rediscovery \
  --run artifacts/prefix_kv_cache_rediscovery_runs/weak_initial/<run-a> \
  --run artifacts/prefix_kv_cache_rediscovery_runs/weak_initial/<run-b> \
  --run artifacts/prefix_kv_cache_rediscovery_runs/weak_initial/<run-c>
```

The primary criterion is behavioral rather than source similarity. A generated
policy must be valid, remain at or below 650 effective AST nodes, pass the
agentic surrogate gate, and recover at least 80% of the charged
weak-seed-to-incumbent gap on selection, quarantined probe, and hidden panels.
The discoverability claim is supported only after at least two distinct search
seeds pass.

The current result is negative. Three staged 298-evaluation runs on June 14,
2026 used search seeds 211, 307, and 401, totaling 894 evaluations, `$21.5381`,
and 8,705 seconds. None passed the behavioral criterion. The ordinary-score run
overfit selection and failed probe plus the agentic gate. Adding the robust
guidance floor improved probe and passed the gate but selected a compact,
high-churn policy that failed selection and hidden transfer. After documenting
generic stateful primitives and enabling inspirations, the final corrected run
did not beat the weak seed; its strongest generated mutation was 708 nodes,
failed the agentic gate, and scored below the weak seed on every canonical
charged panel. Exact incumbent replay is reproducible, but independent
weak-seed rediscovery is not yet established.

## Bring Your Own Model

Models use [LiteLLM](https://docs.litellm.ai/) provider-qualified identifiers.
The simplest override uses one model for every search role:

```bash
uv run prefix-cache-evolve --show-config --model openai/<model-id>
uv run prefix-cache-evolve --show-config --model anthropic/<model-id>
uv run prefix-cache-evolve --show-config --model gemini/<model-id>
uv run prefix-cache-evolve --show-config --model ollama/<model-id>
```

Use separate mutation and paradigm-shift models when desired:

```bash
uv run prefix-cache-evolve \
  --primary-model openai/<mutation-model-id> \
  --secondary-model anthropic/<paradigm-model-id> \
  --iterations 100
```

Standard provider environment variables are:

| Provider | Model prefix | Credential |
|---|---|---|
| OpenAI | `openai/` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini/` | `GEMINI_API_KEY` |
| Ollama | `ollama/` | Usually none |

Never place API keys in YAML. For a self-hosted OpenAI-compatible endpoint:

```bash
export LOCAL_MODEL_API_KEY=local-no-key-required

uv run prefix-cache-evolve \
  --model openai/<served-model-name> \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key-env LOCAL_MODEL_API_KEY \
  --search-seed 17 \
  --iterations 100
```

The equivalent YAML is:

```yaml
llm:
  default_provider: openai
  primary_model: <served-model-name>
  secondary_model: <served-model-name>
  api_base: http://127.0.0.1:8000/v1
  api_key_env: LOCAL_MODEL_API_KEY
  temperature: 0.3
  max_tokens: 6000

search:
  seed: 17

punctuated_equilibrium:
  reasoning_effort: medium
  max_tokens: 12000
```

Set `punctuated_equilibrium.max_tokens` separately for reasoning-capable
paradigm models. The repository compatibility layer overrides Levi's historical
4,096-token paradigm-generation default only for the configured paradigm model.
Mutation calls keep their own pipeline budget, and Levi versions that natively
forward the configured paradigm budget bypass the compatibility override.

Run `--show-config` before a paid search. It validates the YAML and prints the
resolved model, endpoint, search seed, workload seeds, policy seed, capacities,
and worker settings without contacting the provider.

## WildChat Replay

[WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) can supply
real multi-turn conversation structure for prefix-cache replay without
retaining prompt text in the benchmark artifact:

```bash
make setup-wildchat
export PREFIX_CACHE_TRACE_HASH_KEY="$(openssl rand -hex 32)"

uv run prefix-cache-tools datasets wildchat \
  --conversation-limit 10000 \
  --minimum-requests-per-conversation 2
uv run prefix-cache-evolve \
  --calibrate-trace artifacts/traces/wildchat.jsonl
```

The converter resolves `allenai/WildChat-1M` to an exact Hugging Face commit and
records it in `wildchat.jsonl.manifest.json`. Local JSON, JSONL, and Parquet
exports are also supported with `--input`; their SHA-256 is recorded instead.
Keep the HMAC key private and stable for a published experiment. The manifest
contains only its SHA-256 fingerprint.

Each distinct assistant turn becomes one request. Its prompt is the deterministic
cumulative conversation prefix, tokenized with the selected tiktoken encoding.
Repeated turn identifiers in overlapping conversation snapshots are deduplicated;
conflicting content or recorded timestamps are rejected. Recorded times override
inferred estimates for the same turn. Inferred times retain the first estimate
unless conversation order requires an adjustment, including across branches.
Duplicate validation keeps a reply HMAC in the temporary database, so replies
with equal token lengths still distinguish conflicting content without retaining
their text.
Session identity comes from the
first available turn identifier, since `conversation_hash` identifies content
and is not unique. Exports without turn IDs fall back to content hash, tenant,
and last-response timestamp; deduplication is approximate for those exports.
Tenant IDs, session IDs, and every token block are HMAC-SHA256 identifiers.
Conversion v3 includes the model in each prefix namespace. Missing models use
a session-local namespace, preventing unsupported cross-model or unknown-model
KV sharing. Canonical chat v2 JSON-escapes message content so literal role
markers cannot collide with message boundaries. Existing v2 traces must be
reconverted from source to obtain these corrections.
The converter does not write message text or raw identifiers to the trace,
manifest, or temporary sorting database. Hugging Face and local source caching
remain governed by the source loader.

This evidence has two explicit limitations:

1. WildChat's [dataset card](https://huggingface.co/datasets/allenai/WildChat-1M/blob/7d6490e462285cf85d91eabea0f9a954fbddcd1f/README.md)
   defines assistant timestamps as response-completion times, and the row
   timestamp as the last response. The default `--timestamp-mode message` uses
   recorded response times as replay proxies. Missing times use
   `--turn-spacing-ms`, bounded by neighboring recorded responses and anchored
   at the last response when needed; spacing contracts to fit those bounds. Use
   `--timestamp-mode synthetic` to force that spacing for all requests. Neither
   mode recovers request-arrival times or measured serving latency.
2. The canonical serialization does not reproduce the original provider's
   private system prompt or exact chat template.

Accordingly, describe results as **WildChat conversation-derived replay**, not
as production serving-trace replay. Comply with the dataset's ODC-BY license and
do not commit or redistribute the source conversations through this repository.

The v2 conversion manifest records timing-source counts and skipped duplicates.
Regenerate traces and panels together when adopting v2; its session identifiers
and timing differ from the earlier converter. The multi-turn filter in the
example excludes single-turn traffic. Use the default minimum of one request
when the experiment should retain those cold requests, and record the filter
and sample limit with any result.

## Agentic Production Replay (Mooncake)

Mooncake's public
[tool-and-agent trace](https://github.com/kvcache-ai/Mooncake/blob/3cca71daccf2a7afb8fe3f0295358f70e3a69fdb/FAST25-release/traces/toolagent_trace.jsonl)
contains 23,608 requests sampled from a Kimi production cluster. The
[authors' paper, section 7.2 and appendix A](https://madsys.cs.tsinghua.edu.cn/publication/mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving/ToS2025-Qin.pdf)
describes a one-hour sample with arrival timestamps, input/output lengths, and
cumulative prefix hashes at 512-token granularity. The release also contains
conversation and synthetic traces; the recipe below pins the tool-and-agent file.

```bash
mkdir -p .cache/mooncake
curl --fail --location \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/3cca71daccf2a7afb8fe3f0295358f70e3a69fdb/FAST25-release/traces/toolagent_trace.jsonl \
  --output .cache/mooncake/toolagent_trace.jsonl

uv run prefix-cache-tools datasets mooncake \
  --input .cache/mooncake/toolagent_trace.jsonl \
  --expected-sha256 48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71 \
  --output artifacts/traces/mooncake-toolagent.jsonl

# Give the larger replay an explicit resource budget.
uv run python - <<'PY'
from pathlib import Path

import yaml

document = yaml.safe_load(Path("configs/prefix_kv_cache.yaml").read_text())
document["evaluator"]["timeout"] = 1200
document["problem"]["settings"]["max_memory_bytes"] = 2_147_483_648
Path("artifacts/traces/mooncake-replay.yaml").write_text(
    yaml.safe_dump(document, sort_keys=False)
)
PY

uv run prefix-cache-evolve \
  --config artifacts/traces/mooncake-replay.yaml \
  --replay-trace artifacts/traces/mooncake-toolagent.jsonl \
  --candidate-program \
  src/prefix_cache_evolve/problems/prefix_kv_cache/incumbents/production_16tok_20260609/policy.py \
  --block-size-tokens 512 \
  --capacity-blocks 512 \
  --capacity-sweep-blocks 128,512,2048 \
  --trace-baseline lru \
  --trace-baseline tinylfu_lru \
  --trace-baseline vllm_apc \
  --trace-arrival-bucket-ms 100 \
  --trace-request-limit 1024 \
  --trace-output artifacts/mooncake-toolagent/replay.json
```

The converter requires a new output path and writes `OUTPUT.manifest.json`.
It verifies the full source checksum, timestamp ordering, block geometry, and
consistent parent/length identities for repeated prefix hashes. Conversion
needs no tokenizer or dataset dependency: the source is already opaque metadata.
The recipe replays the first 1,024 requests in their original order; record any
change to that limit or the cache capacities when comparing results.
`--trace-baseline` selects deployable comparisons and can be repeated; omitting
it runs every deployable baseline. The candidate still runs in an isolated
worker. The YAML's top-level `evaluator.timeout` controls its time limit; the
example raises it to 1,200 seconds and sets a 2 GiB memory limit for this larger
sample. Those limits are part of the recorded evaluation context.

The native hash resolution cannot reveal sharing within a 512-token block.
This is a separate geometry-transfer experiment from the 16-token headline
benchmark. Session, tenant, and priority fields are absent: the adapter uses
one explicit unknown identity and priority zero, and records those omissions.
Its session and tenant counts are unavailable. Tenant-fair baseline results
therefore do not measure fairness. Group-based `trace-panel` preparation rejects
the single unknown group; genuine session-disjoint evolution splits require a
source that supplies session identity. No random per-request split is substituted.

Recorded arrivals are preserved, then bucketed for the simulator. Latency and
decode behavior still come from its configured model; this replay evaluates
cache policies rather than a live serving deployment. See the
[initial check](results/mooncake_toolagent_20260905.md) for measurements and scope.

For more recent coding-agent behavior, [TraceLab](https://github.com/uw-syfi/TraceLab)
releases 357,161 model calls from 43 developers' day-to-day Claude Code and Codex
usage, with session and tool timing metadata. Its documented `prefix_tokens`
field is an observed cache-hit counter, so it alone cannot identify which blocks
would hit under a different policy. It is useful for calibrating timing and
context growth. [LMCache's agentic traces](https://huggingface.co/datasets/sammshen/lmcache-agentic-traces)
provide cumulative prompts from agent task runs including SWE-bench and GAIA;
their provenance is primarily benchmark execution. Keep that distinction when
making claims about production traffic.

LMCache conversion v2 preserves measured session identities but uses one
unknown tenant accounting pool: the source does not identify tenants. With
`--skip-invalid`, an invalid row discards its entire session, including buffered
and subsequent rows, because skipping one relative `pre_gap` would fabricate
timing and splice cumulative histories. A missing session identity fails the
conversion even in skip mode, because the affected timeline cannot be identified.
Decode duration has a minimum of one
arrival bucket, matching the simulator even for zero output tokens. Models
use disjoint prefix namespaces within an abstract shared capacity pool; this
is not a measured single-model GPU serving workload.

## Production Trace Evolution With Chronological Holdouts

Mooncake's absent session IDs prevent session-disjoint splitting, but allow an
explicitly chronological experiment. `temporal-trace-panel` retains every request
in each half-open time window, in source order, with unchanged prefix and identity
fields. It verifies the complete input, rejects overlapping or reversed splits,
and pins each window's source, bounds, contents, and count. Evaluation rechecks the
bounds and hashes. Recurring sessions and prefixes may cross time windows; this
does not establish independent sessions or transfer to a different capture.
Absent sessions are represented by JSON `null` and policy-visible
`request.session_id is None`. The exact legacy Mooncake placeholder
`mooncake:unknown` is normalized to `None` on load. Missing identities never
provide session recurrence evidence and cannot establish session-disjoint
splits. The unknown tenant placeholder remains one accounting pool, not
measured tenant metadata.
The container reloads a full policy's source module for each trial, so mutable
module or class state cannot carry observations into another capacity or window.

The production search template uses native 512-token blocks, capacities
128/512/2048, and the existing verifier 1.0.0 score weights. It removes historical
16-token score targets from the prompt. Synthetic regression panels run
separately at their original geometry. Each temporal window starts with a fresh
cache and policy, so the experiment measures cold-start replay. Omitted gaps and
length-based pin durations are explicit modeling limits.

After converting Mooncake using the recipe above:

```bash
docker build --tag prefix-cache-evolve-sandbox --file docker/sandbox/Dockerfile .

uv run prefix-cache-tools datasets temporal-trace-panel \
  --trace artifacts/traces/mooncake-toolagent.jsonl \
  --config configs/prefix_kv_cache_mooncake.yaml \
  --output-dir artifacts/mooncake-evolution/panel \
  --family mooncake_toolagent \
  --window train 0 90000 \
  --window validation 1200000 1290000 \
  --window validation 2100000 2190000 \
  --window hidden 3000000 3180000 \
  --window hidden 3300000 3480000

uv run prefix-cache-evolve \
  --config artifacts/mooncake-evolution/panel/evolution.yaml \
  --iterations 32 --show-config
```

This panel contains 483 train requests, 600 and 659 validation requests, and
1,296 requests in each final window. The train window was already inspected in
the initial replay. The later windows are reserved before search and must not
feed further mutation after final evaluation. The workflow has a $20 accounted
search budget, 32 evaluations, and a 12-hour deadline. In-flight provider calls
can exceed a dollar threshold; offline preflight, review, and final reports are
outside these search limits. The CLI iteration count overrides `max_iterations`,
so pass `--iterations 32` explicitly. `--show-config` reports the resolved budgets.
The active template uses explicit `gpt-5.6-luna` for ordinary mutations and
`gpt-5.6-sol` at medium reasoning effort for diversity and punctuated-equilibrium
paradigm shifts. Freeze those resolved model roles with each run; earlier result
bundles retain their historical model configuration.

Before search, run the parent and selected baselines on the search panel, verify
the actual worker/container path, and freeze the config plus Docker image ID.
Set `problem.settings.sandbox_image` to the `sha256:...` ID returned by
`docker image inspect --format '{{.Id}}' prefix-cache-evolve-sandbox`; retain the
template and pinned config as provenance. The image must contain the audited
checkout. Search then uses the ordinary evolution runner:

```bash
uv run prefix-cache-evolve \
  --config artifacts/mooncake-evolution/panel/evolution.yaml \
  --iterations 32 \
  --report-baseline lru \
  --report-baseline tinylfu_lru \
  --report-baseline vllm_apc \
  --artifact-output artifacts/mooncake-evolution/runs
```

`--report-baseline` also selects comparisons for baseline, hidden, and probe
reports. Omission retains the complete existing suite. It limits offline report
cost without changing the candidate score or request panel.

With `sandbox_image` configured, source candidates run as a non-root user with
network disabled, a read-only filesystem, dropped capabilities, memory/CPU/PID
limits, and a container deadline. Only selected trace inputs and the candidate
are mounted; hidden inputs and host credentials are not supplied. There is no
fallback to host source execution. The parent worker gets 30 seconds for shutdown
and cleanup beyond the container evaluation deadline. Search-scoped cleanup
removes containers whose Levi workers were cancelled. The build context is
restricted by `.dockerignore` to source, config, and dependency files.

Levi's generic code loader executes modules before invoking a score function.
The repository adapter routes source-aware evaluation around that loader so
static policy checks run first. These checks enforce the policy contract; Docker
provides the OS isolation. Source and context identities must agree between the
preflight, worker feedback, and saved run.
This backend accepts configured panels. Ad hoc `--replay-trace` source evaluation
with `sandbox_image` set is rejected; prepare `trace_workloads` and use the panel
report or evolution commands so the sandbox can stage and quarantine inputs.

Choose one eligible candidate from validation before opening hidden windows.
Compare it, the unchanged parent, and the three deployable baselines on the same
final windows and capacities. Report charged score, raw behavior before complexity,
token hits, churn, waste, and underfill. The churn penalty saturates at 25, so a
score improvement alone does not establish an improvement in cache behavior.
Baseline source complexity is zero, whereas candidate source is charged. Run the
original synthetic panel separately and record any transfer regression before
considering an immutable incumbent promotion.

The completed September 6 run and its exact identities are recorded in the
[production agentic evolution report](results/mooncake_evolution_20260906.md).
Its one-hour bound allowed three evaluator attempts, retained the unchanged
parent, and therefore produced no new incumbent bundle.

## Trace Panels For Evolution

Split a converted trace into complete, disjoint groups and write an evolution
configuration with pinned file hashes:

```bash
uv run prefix-cache-tools datasets trace-panel \
  --trace artifacts/traces/wildchat.jsonl \
  --family wildchat \
  --output-dir artifacts/traces/wildchat-panel

uv run prefix-cache-evolve \
  --config artifacts/traces/wildchat-panel/evolution.yaml \
  --workload-manifest \
  --workload-manifest-output artifacts/traces/wildchat-panel/workload_manifest.json

uv run prefix-cache-evolve \
  --config artifacts/traces/wildchat-panel/evolution.yaml --baseline-report
```

The output directory must be new. It contains `train.jsonl`, `validation.jsonl`,
`hidden.jsonl`, `manifest.json`, `evolution.yaml`, and a copy of the converter's
manifest when available. The default split groups by tenant, keeping all that
tenant's sessions together; `--group-by session` permits different sessions from
the same tenant across splits. At least three groups are required. Groups are
sorted by a seeded SHA-256 rank, then assigned approximately 60%/20%/20% to
train/validation/hidden. Each split retains the input's timestamp ordering and
complete sessions. The manifest records actual group and request counts.
`--split-seed`, `--validation-fraction`, and `--hidden-fraction` control partitioning.

Train and validation are both visible to the evolution loop. The existing
scorer selects on validation, with train diagnostics available as feedback;
**hidden is the final holdout**. Search opens only its configured splits. The
default generated configuration replaces synthetic train/validation/hidden
families with trace streams and retains synthetic probes. Pass
`--keep-synthetic` to add traces alongside the base synthetic families instead.
Synthetic probes remain excluded from selection in both modes.

Each trace is evaluated once at each configured capacity, independently of
synthetic generator seeds. The loader checks its full-file SHA-256, request
count, block size, and session separation across evaluated splits. Workload and
score identities include the trace provenance and request-stream fingerprints.
Relative trace paths resolve from the YAML file, so an intact panel can move
between workspaces. Existing synthetic benchmark identities are unchanged.
Trace fingerprints now include `prefix-kv-cache-trace-replay-v2`. Container
results must advertise the same replay contract; an old image is rejected
with an instruction to rebuild and pin it. Historical result identities and
metrics are retained, rather than relabeled as corrected evaluations.

The [2026-09-07 replay audit](results/replay_audit_20260907.md) records the
corrected bundles and pinned container. Its combined configuration is
`configs/prefix_kv_cache_replay_audited.yaml`. All its retained hidden samples
have already been inspected and are diagnostic only. Use a fresh untouched
panel for a subsequent generalization claim.

Use calibration to inspect prompt lengths and prefix depths before setting
`problem.settings.capacity_sweep_blocks` in the generated YAML. The inherited
synthetic capacities may be too small for real conversations. For a larger
experiment, also size `evaluator.timeout` and
`problem.settings.max_memory_bytes`. `--quick` reduces synthetic workloads and
their seeds; it does **not** truncate or resample traces. Prepare a smaller
conversation sample for a trace smoke test. To change block size, reconvert
the source and prepare a new panel; opaque blocks cannot be reinterpreted at a
different geometry.

Start model-powered search after inspecting that configuration:

```bash
uv sync --frozen --extra wildchat --extra evolution
uv run prefix-cache-evolve \
  --config artifacts/traces/wildchat-panel/evolution.yaml --show-config
uv run prefix-cache-evolve \
  --config artifacts/traces/wildchat-panel/evolution.yaml \
  --iterations 100 --artifact-output artifacts/wildchat-runs
```

Saved trace-based runs copy their train, validation, and probe trace files into
`traces/` and write `config_snapshot.yaml`. Those search inputs can be replayed
after moving the run directory. The snapshot retains the original hidden path
and checksum without reading or copying hidden data. Automatic run saving also
defers generated-mutation decomposition and specialist promotion adjudication,
which normally evaluate hidden workloads. The original YAML is retained as
`source_config.yaml`.

Verify a saved run after moving it with its archived manifest:

```bash
uv run prefix-cache-evolve \
  --config artifacts/wildchat-runs/<run-id>/config_snapshot.yaml \
  --workload-manifest \
  --workload-manifest-reference artifacts/wildchat-runs/<run-id>/workload_manifest.json \
  --workload-manifest-output artifacts/wildchat-runs/<run-id>/verified_workload_manifest.json
```

Trace manifest generation defaults to the search splits. With a reference, it
uses exactly the split set recorded there, so verifying an ordinary saved run
does not require the original hidden data. Synthetic-only manifests retain their
existing default of all four splits.

Keep the original panel and converter manifest for provenance and final holdout
evaluation. Evaluate the final candidate explicitly using that panel:

```bash
uv run prefix-cache-evolve \
  --config artifacts/traces/wildchat-panel/evolution.yaml \
  --hidden-report \
  --candidate-program artifacts/wildchat-runs/<run-id>/best_program.py
```

Random group holdouts test transfer across sampled users or sessions, not future
time periods. Filtering groups also changes the arrival mix. Small, first-N
samples and multi-turn-only samples establish pipeline operation, not general
policy superiority on WildChat or production traffic.

The [September 2026 smoke record](results/wildchat_smoke_20260905.md) documents
the first pinned, 20-conversation check through both the baseline report and the
isolated evolution evaluator.

## Publishing a Result

For new mixed-traffic exploration and simplification, follow the
[behavior and policy-cost workflow](policy_costs.md). It records behavioral score,
source size and measured cost separately, keeps smaller archive alternatives,
and requires exact visible-trial behavior before accepting a simplification.

Archive these files for each reported experiment:

1. Candidate source or saved run directory.
2. Exact YAML configuration.
3. `run_summary.json`, `metadata.json`, and `workload_manifest.json`.
4. The committed `uv.lock` and Git revision.
5. Any trace input or download script plus its SHA-256.
6. Provider model identifiers, date, search seed, evaluation budget, wall time,
   and reported API cost.

For WildChat replay, also archive the conversion manifest and retain the HMAC
key securely outside the publication artifact.

Report evaluation reproducibility separately from search reproducibility. A
candidate score can be exactly replayable even when the search trajectory that
found the candidate is not.
