# Replay audit before further evolution

Date: 2026-09-07

The audit found consequential replay defects, repaired them, and prepared new
trace bundles. No evolution, model API calls, or incumbent promotion occurred.
Both frozen policies remain byte-for-byte unchanged. Historical reports retain
their original metrics and now link to this audit.

## Findings and repairs

| Finding | Consequence | Repair |
| --- | --- | --- |
| Mooncake's unavailable session was a shared identifier | Session-aware policies counted unrelated requests as recurring sessions | Emit `session_hash: null`; normalize the exact legacy placeholder to policy-visible `None`. Require chronological windows when sessions are unavailable. |
| WildChat model field was omitted by the Parquet reader and prefix hasher | Matching text could produce false KV sharing across different models | Preserve the model and namespace prefix blocks by it. Unknown models share only within a session. |
| WildChat content was concatenated with literal role delimiters | Content containing delimiters could impersonate message boundaries and create false prefix matches | JSON-escape content; bump canonical prompt serialization to chat v2 and conversion schema to v3. |
| LMCache skipped individual invalid rows | Relative tool gaps and cumulative histories could be spliced across a missing request | Discard the entire affected session, including buffered and later rows. If its identity is unavailable, fail conversion even with `--skip-invalid`. |
| LMCache invented a tenant per session | Tenant-sensitive policies and fairness metrics received fabricated tenant distinctions | Use one explicitly unknown accounting pool while preserving genuine session identity. |
| LMCache permitted zero-length decode duration | A following request could precede the simulator's minimum active lifetime | Use at least one bucket; reject nonfinite gaps, scheduled timestamps, and nonfinite message JSON. |
| Configured LMCache training hit rate was absent from flattened feedback | The requested behavior feature was unavailable to evolution | Emit all train and validation workload diagnostics. Hidden and probe workload metrics remain excluded from this feedback. Selection still scores validation. |
| Old containers could return results from older replay semantics | Host fixes could silently be bypassed by a previously pinned image | Include `prefix-kv-cache-trace-replay-v2` in trace fingerprints and result metadata; reject incompatible container results. Rebuild and pin the corrected image. |

The metadata schema now accepts null sessions and null predicted output length,
matching the parser. Empty or whitespace-only opaque identifiers are rejected.
All numeric score weights and policy sources were preserved.

## Measured impact of the Mooncake session correction

This is a controlled diagnostic on the first **483-request training window**,
with **512-token blocks and 128-block capacity**. A deliberately shared session
identifier emulates the earlier false recurrence; the corrected replay has no
session identity. Request order, lengths, prefix blocks, timing, and capacity
are identical. No hidden data was evaluated.

| Frozen policy | Shared-session control: token hit | Corrected: token hit | Control: evictions / 1,000 requests | Corrected: evictions / 1,000 requests |
| --- | ---: | ---: | ---: | ---: |
| Previous synthetic + Mooncake joint policy | 25.76% | 25.76% | 0.0 | 0.0 |
| Newer four-domain policy | 25.70% | 28.74% | 6,302.3 | 420.3 |

Removing fabricated session recurrence improves the newer policy's hit rate by
**3.04 percentage points** and reduces churn by **93.3%** in this window. The
previous joint policy is unaffected. The shared identifier is a neutral audit
control, not the original identifier's hash; these frozen policies do not
dispatch on literal session IDs. This diagnoses a causal replay issue and is
not a new full-suite ranking.

Both policies also completed corrected conversational smoke checks. Every row
below uses capacity **128 blocks**; WildChat blocks contain 16 tokens and LMCache
blocks contain 512 tokens. These are not averages across the earlier capacity
sweeps.

| Visible workload | Requests | Previous joint token hit | Newer four-domain token hit |
| --- | ---: | ---: | ---: |
| WildChat train | 35 | 1.73% | 21.71% |
| WildChat validation | 11 | 3.61% | 57.43% |
| LMCache train | 133 | 10.98% | 68.68% |
| LMCache validation | 99 | 83.85% | 94.17% |

The older policy still under-admits on these conversational samples. The newer
one gets more hits but incurs more churn. A hybrid's motivation therefore
survives, but its design should use corrected full-suite comparisons. This
audit did not re-rank baselines or optimize either policy.

## Corrected bundles and provenance

The combined configuration is
[`configs/prefix_kv_cache_replay_audited.yaml`](../../configs/prefix_kv_cache_replay_audited.yaml).
It retains the synthetic suite, native trace geometries, and capacity sweeps;
removes stale historical score targets from the prompt; documents absent
session identities; and points to new trace files and a rebuilt pinned image.
The recorded experiment-chain spend remains $3.11414110. The configuration's
future-run cap is the remaining $16.88585890 of the approved $20; no part of it
was spent by this audit.

Local artifacts are under `artifacts/replay-audit-20260907/`:

- `mooncake/`: reconverted the pinned 23,608-request source; rebuilt the exact
  five existing windows. Verified that their records differ only in null
  session metadata. Training has 483 requests, validation has 600 and 659, and
  the two historical diagnostic holdouts each have 1,296.
- `wildchat/`: reconverted the same 51 source rows at revision
  `7d6490e462285cf85d91eabea0f9a954fbddcd1f`, with the existing private HMAC key.
  Retained the same 20 conversations, 65 requests, and tenant partitions:
  35 train, 11 validation, 19 historical diagnostic holdout requests. Prompt
  tokenization and hashes changed with the corrected serialization and model
  namespace.
- `lmcache/`: **metadata migration, not a full v2 reconversion**. All 329
  retained requests had positive output lengths and the original conversion
  skipped zero rows; the timing fixes do not alter this sample. Verified that
  only tenant metadata changed, preserving 16 sessions and the 133/99/97
  request partitions. The original model-separated block hashes are retained.
  Raw LMCache rows and its original HMAC key were unavailable for reconversion;
  the migration manifest preserves original provenance and states this limit.
- `visible_workloads.json`: validated all 52 visible workload streams,
  including seven trace streams, without loading hidden traces in that check.
- `frozen_policy_checks.json` and individual result files: six sandbox
  evaluations, covering 12 workload/capacity trials, with source hashes,
  operative configs, replay identities, and complete diagnostics.
- `check_frozen_policies.py`: deterministic reproduction of the bounded
  comparisons; no search or hidden evaluations.

The accompanying [machine-readable audit](replay_audit_20260907.json) records
bundle, source, configuration, and image identities. No raw conversation text
or private HMAC keys were added to the report or replay artifacts.

## Remaining limits before drawing broader conclusions

1. These retained hidden samples have already been inspected. Their split
   labels are historical; they are diagnostic only. A later generalization
   or promotion claim needs an untouched panel.
2. WildChat is conversation-derived replay. Its recorded assistant timestamps
   are response completions, not arrivals, and canonical prompts approximate
   provider templates. See the
   [pinned WildChat card](https://huggingface.co/datasets/allenai/WildChat-1M/blob/7d6490e462285cf85d91eabea0f9a954fbddcd1f/README.md).
3. LMCache contains collected agent task runs. Its `pre_gap` measures time
   after the previous response; cross-session starts and decode duration remain
   simulator proxies. See the
   [LMCache dataset](https://huggingface.co/datasets/sammshen/lmcache-agentic-traces).
   The mixed-model replay is an abstract capacity pool with isolated prefix
   namespaces, not a measured single-model GPU deployment.
4. The LMCache sample is only 16 sessions and WildChat only 20 conversations.
   Filtering into splits changes concurrency; LMCache's ten training sessions
   and three validation sessions have different pressure. A larger sample and
   a predeclared arrival/concurrency sweep are needed for traffic-wide claims.
5. Missing tenant identities do not measure tenant fairness. Correcting their
   representation does not recover production tenant behavior. Mooncake's
   native 512-token identifiers also cannot recover finer-grained sharing.

These limits are not resolved by evolution. Rebaseline the frozen policies
under the audited full visible suite before selecting a new evolution seed.

That [full reassessment was completed on 2026-09-08](replay_reassessment_20260908.md),
covering 15 frozen policies and baselines across 1,815 trials.

## Validation

- Full pytest suite: **500 passed**.
- `ruff format .` and `ruff check .`: passed.
- Mypy: **37 source files passed**.
- Corrected combined configuration: CLI validation and all 52 visible workload
  stream preparations passed.
- Both frozen source hashes verified; all six final container checks succeeded
  under the required v2 replay contract, covering 12 trials.
- Historical score files and immutable incumbent sources were preserved;
  no new hidden scores or model charges were produced.
