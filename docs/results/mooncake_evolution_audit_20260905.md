# Production Agentic Evolution: Preflight Audit

> [Replay audit update](replay_audit_20260907.md): this report preserves historical replay semantics. Missing-session handling and parts of the conversational replay were corrected afterward; compare policies again under the audited contract before drawing new conclusions.

> Historical preflight record. The audit, bounded search, and frozen final
> evaluation are complete in
> [mooncake_evolution_20260906.md](mooncake_evolution_20260906.md).

Date: 2026-09-05. Branch: `feat/real-data`.

The offline audit passes. Model-powered evolution and external code review have
not started: automatic approval review rejected the new private-code payload to
OpenAI, and explicit approval is pending. No search cost has been incurred, no
policy has been promoted, and the final holdout windows have not been evaluated.

This is a prepared experiment and baseline result, not evidence of a newly
evolved policy. The complete structured record is
[mooncake_evolution_audit_20260905.json](mooncake_evolution_audit_20260905.json).

## Production data and frozen experiment

The source is the original [Mooncake FAST25 tool-and-agent trace](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces),
released from Kimi production traffic. It provides timestamps, prompt/output
lengths, and cumulative prefix identities at native 512-token blocks. It does
not provide session IDs, tenant IDs, tool DAGs, priorities, or measured GPU
latency. This is not a claim about coding-agent traffic specifically.

The complete source has 23,608 requests. The chronological panel retains every
request in each selected half-open window, with original ordering and prefix
identity. Bounds were selected before measuring their validation performance.

| Split | Source-clock interval, ms | Requests | Input tokens |
|---|---|---:|---:|
| train | [0, 90,000) | 483 | 4,680,571 |
| validation | [1,200,000, 1,290,000) | 600 | 5,191,348 |
| validation | [2,100,000, 2,190,000) | 659 | 5,638,767 |
| hidden | [3,000,000, 3,180,000) | 1,296 | 11,089,082 |
| hidden | [3,300,000, 3,480,000) | 1,296 | 10,961,476 |

Search sees 1,742 requests; 2,592 later requests are reserved for final evaluation.
The 483-request training interval was already inspected in the initial production
replay. Metadata and hashes of all windows were prepared, but hidden records are
not mounted or scored during search. Unknown identity placeholders are retained;
these are chronological holdouts from one capture, not independent sessions.

Every window/capacity trial starts with fresh policy and cache state. Physical
geometry is 512 tokens with capacities 128/512/2,048 blocks and 100 ms arrival
buckets. Score weights remain those of verifier 1.0.0. Synthetic regression uses
its original geometry separately; no historical synthetic score is a target for
this production panel.

The frozen search has 32 evaluations, an $8 accounted-dollar threshold, and a
one-hour deadline, with two evaluation workers. In-flight calls can exceed the
dollar threshold; external review, offline preflight, and final reports are extra.
The configured providers are OpenAI `gpt-5.4-mini` for mutation and `gpt-5.5` for
diversity/paradigm work. No provider call has been made for this experiment.

## Audit findings and repairs

- Levi's generic loader executed candidate modules before the repository's
  static checks. The source-aware dispatcher now performs those checks first,
  across the normal initialization and mutation paths. A regression test checks
  that rejected top-level side effects never execute.
- Generated candidates needed OS isolation. The new Docker backend exposes only
  selected trace inputs and source, uses a non-root user, disables networking,
  makes the root and input mount read-only, drops capabilities, and limits memory,
  CPU, processes, and runtime. It passes no provider credentials and never falls
  back to host source execution. Search-scoped cleanup handles cancelled workers.
- Mutable candidate module/class/default-argument state could otherwise persist
  across trials. Full policy source is reloaded for every capacity/window trial;
  a mutable-global regression verifies the reset.
- Mooncake cannot support session-disjoint partitioning. Explicit chronological
  windows now pin the common source, bounds, counts, and content hashes, reject
  overlaps/reversed split order, and revalidate loaded timestamps. No session
  identity is invented.
- The original workflow could inherit unsuitable geometry/score targets and
  expensive default reports. A dedicated native-block template preserves score
  weights, exposes the actual budgets, rejects an empty search panel before any
  model call, and permits exactly the three requested baseline comparisons.

## Runtime verification

Formatting, lint, mypy, and all 444 functional pytest tests pass; measured coverage
is 87.22%. Four immutable incumbent bundles validate and their sources are
unchanged. No style-only edits were made to benchmark policy programs.

An actual spawned Levi worker evaluated the unchanged parent inside the frozen
container in 145.3 seconds, with zero model calls. Live inspection verified user
`10001:10001`, no network, read-only root and inputs, 2 GiB memory, dropped
capabilities, and absent OpenAI/Anthropic/AWS credentials. All 73 installed Python
source files match the audited checkout by SHA-256.

Host and container agree exactly on charged scores, evaluation identities,
per-request hit lengths, and cache metrics. Seven occurrences of reporting-only
shadow-price bias/MAE values differ by at most 4.44e-16. These diagnostics do
not affect policy inputs or charged scores; the raw differences are retained.
The audit tolerance is restricted to those two diagnostics below 1e-12, and does
not relax comparisons of scores or cache behavior.

## Baseline result on visible validation windows

| Policy | Charged score | Before complexity | Token hits | Evictions / 1,000 | Underfill |
|---|---:|---:|---:|---:|---:|
| tinylfu_lru | 63.709 | 63.709 | 36.26% | 273.4 | 32.43% |
| incumbent | 40.465 | 48.067 | 35.02% | 1146.0 | 16.22% |
| vllm_apc | 39.957 | 39.957 | 35.35% | 7080.8 | 0.20% |
| lru | 39.478 | 39.478 | 35.19% | 7809.6 | 0.00% |

Rates are the evaluator's equal-weight aggregation across two validation windows
and three capacities, not a pooled token fraction. The parent pays its 572-node
source charge of 7.603; built-in baselines have zero source complexity charge.
The before-complexity column makes that difference visible. All policies share
the same verifier, context, and panel identities and have zero invalid trials.

The production parent does not establish dominance on this native geometry.
TinyLFU-LRU provides a stronger target than merely improving the parent. Keep
score weights fixed, and assess hit rate, churn, admission waste, and underfill
alongside the charged score. The churn penalty caps at 25 above approximately
1,666.67 evictions per thousand requests; large raw-churn differences can be
invisible to that part of the score.

## Approval and final evaluation

The prepared external review receives changed code, tests, config, and docs.
Evolution receives policy source and aggregate feedback. Neither receives trace
records or credentials. The requested approval covers this new OpenAI payload
and the bounded search; local audit work is complete independently of that gate.

After review passes, run the frozen search, select one eligible candidate from
validation, and freeze its exact source before opening the hidden windows.
Compare that candidate, the unchanged parent, LRU, TinyLFU-LRU, and vLLM-style APC
there. Run the original synthetic regression separately, and report any failure
to improve or transfer. Do not feed hidden results back into mutation or
automatically promote a native-geometry winner.

The experiment measures cold-start cache-policy replay. Omitted time gaps,
output-length pin durations, unavailable tenant/session metadata, and simulator
latency proxies limit conclusions about end-to-end serving performance.

## Reproduction evidence

The [production evolution recipe](../reproducibility.md#production-trace-evolution-with-chronological-holdouts)
recreates the temporal panel. This workspace also retains the exact command
configuration and all offline trials under `artifacts/mooncake-evolution/`:
`search.yaml`, `frozen-inputs.json`, `seed.py`, `preflight.py`, `preflight.json`,
`isolation.json`, `image-source-verification.json`, `parity-differences.json`,
`seed-trials.json`, `seed-container-trials.json`, `seed-worker-evaluation.json`,
the three baseline trial files, and the full test log. Their hashes are recorded
in the structured audit result. The raw traces and full trial artifacts remain
local and are not checked into the repository.

- Parent source: `0c19deaec83d83f776e837c58822ac4cada611caa0a5504d17816e50e08f0037`
- Search config: `967e33ffd81f7fe6ad6205fc8db1213c2ff28413d88820b803b04e43d24ceb52`
- Container: `sha256:d2b930fa37ab0245e9dd90483125c0ffcf6440910c307c0caf9becd72afb7c95`
- Evaluation context: `0644b00177018ecedf2131f12d19208447f7c7693d3da8819d5809efa06e0de2`
- Visible panel: `39b9f190369bcd16aef3fd9a5c518249d35341dc5222d7c45df46b9dc614ad80`
