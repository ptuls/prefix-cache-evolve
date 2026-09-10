# Qwen search and AgentX held-out validation

Qwen-Bailian provides an independent production source for policy development.
AgentX is reserved for final transfer validation. In this repository, the split
named `validation` is visible to evolution; **AgentX belongs in `hidden`**.
Freeze the candidate using the search panels before reading AgentX policy scores.
Dataset parsing and importer tests do not consume this holdout.

Routine AgentX evaluation uses **four randomly selected complete sessions**,
with seed `20260908` and a 20-million-input-token budget. The fixed draw contains
165 requests and 19,681,280 input tokens. The rest of the corpus remains reserved
for a broader final check. No policy scores were used to choose this sample.
The [frozen selection manifest](results/agentx_hidden_sample_20260908.json)
records all four session IDs and the source/replay checksums. The active local
bundle is `artifacts/traces/qwen-agentx-panel`; the previous complete-corpus
bundle is retained as `artifacts/traces/qwen-agentx-full-panel`.

## Pinned sources

The initial integration uses Qwen's API automation capture and AgentX's 256k
release. The Qwen converter also accepts chat, thinking, and coder captures in
the same native schema. Convert each capture independently: integer block IDs
do not establish sharing across captures.

| Source | Revision | Native SHA-256 |
| --- | --- | --- |
| [Qwen-Bailian API](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon) | `5f7439c51ec248a0c585f7d90a41a6f57773b912` | `68e3f98e2d601d60d0abf4b89bc8a3654372abab7b1cde6373a13d0054379d59` |
| [AgentX 256k](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k) | `8fecd2fc56694469f758f0afbbb6335ad3043740` | `e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef` |

Download into the ignored cache. Qwen uses Git LFS: the `media` URL retrieves
actual data, whereas a GitHub `raw` URL can return a pointer file.

```bash
mkdir -p .cache/public-traces
curl --fail --location \
  https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/5f7439c51ec248a0c585f7d90a41a6f57773b912/qwen_traceB_blksz_16.jsonl \
  --output .cache/public-traces/qwen_traceB_blksz_16.jsonl
curl --fail --location \
  https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/resolve/8fecd2fc56694469f758f0afbbb6335ad3043740/traces.jsonl \
  --output .cache/public-traces/agentx-256k.jsonl

uv run prefix-cache-tools datasets qwen \
  --input .cache/public-traces/qwen_traceB_blksz_16.jsonl \
  --expected-sha256 68e3f98e2d601d60d0abf4b89bc8a3654372abab7b1cde6373a13d0054379d59 \
  --output artifacts/traces/qwen-bailian-api.jsonl
uv run prefix-cache-tools datasets agentx \
  --input .cache/public-traces/agentx-256k.jsonl \
  --expected-sha256 e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef \
  --sample-sessions 4 --sample-seed 20260908 --max-input-tokens 20000000 \
  --output artifacts/traces/agentx-256k-sample.jsonl
```

No tokenizer, Hugging Face library, or private hash key is required. Conversion
checksums the complete source, validates selected requests, and writes JSONL
plus a `.manifest.json` with checksums, native geometry, counts, and assumptions.
Existing outputs are never overwritten. Sorting uses a temporary on-disk
database; allow space for both it and the output.

Sampling ranks session IDs by SHA-256 of the algorithm label, seed, and ID,
then selects the first `--sample-sessions` IDs. Membership does not depend on
source row order, session length, model, or policy performance. Complete nested
subagent requests and original session-clock offsets are retained. The manifest
pins the algorithm, seed, population size, selected IDs, and token budget.
`--max-input-tokens` rejects an oversized draw instead of truncating sessions or
replacing long sessions with shorter ones. Omitting sampling and budget options
converts the full corpus to a separately named output.

## Search and holdout configuration

Split Qwen by conversation component, then attach AgentX as another hidden
family. Specify capacities in **tokens** to compare equivalent storage budgets
across native block sizes. These example capacities are untuned starting points.

```bash
uv run prefix-cache-tools datasets trace-panel \
  --trace artifacts/traces/qwen-bailian-api.jsonl \
  --family qwen_bailian_api --group-by session --block-size-tokens 16 \
  --capacity-tokens 65536 --capacity-tokens 262144 \
  --output-dir artifacts/traces/qwen-bailian-panel
uv run prefix-cache-tools datasets attach-holdout \
  --trace artifacts/traces/agentx-256k-sample.jsonl --family agentx \
  --config artifacts/traces/qwen-bailian-panel/evolution.yaml \
  --capacity-tokens 65536 --capacity-tokens 262144 \
  --output-dir artifacts/traces/qwen-agentx-panel
```

`trace-panel` assigns 60%/20%/20% of Qwen groups by default. Qwen validation is
search-visible; Qwen hidden and the sampled AgentX sessions remain held out.
`attach-holdout` verifies converter provenance and rejects session overlap with
search/probe traces. It copies the new hidden source into its bundle. Existing
workloads retain their resolved paths and checksums: keep those original panels.
The command can also attach AgentX to another search configuration, including
the mixed synthetic/Mooncake suite.

Normal evolution does not open or copy AgentX into the worker's search inputs.
After selecting a candidate, run explicit final evaluation:

```bash
uv run prefix-cache-evolve \
  --config artifacts/traces/qwen-agentx-panel/evolution.yaml \
  --hidden-report --candidate-program path/to/frozen_candidate.py
```

The full releases have 172,800 Qwen requests and 68,266 AgentX requests.
The sampled AgentX panel reduces input-token volume by about 350 times, but four
sessions still provide limited coverage and high sampling variance. The simulator
expands prompt tokens in memory and has substantial cost on long contexts. Qwen
remains complete in this recipe; size its windows and evaluator limits before
running the whole suite. `--quick` does not truncate traces. Never redraw or tune
the AgentX subset based on policy scores.

## Replay semantics and limits

**Qwen:** native blocks contain 16 tokens. The source hashes individual blocks;
the replay loader derives token identities from the full ordered prefix path.
An equal block after different parents therefore cannot create a false hit.
Seconds become milliseconds. Connected `parent_chat_id` components stay together
in group splits, including forks and missing-parent anchors. In the API capture
every request is a root; these groups are not evidence of multi-turn sessions.
Tenant identities and priorities are unavailable. Split after conversion:
separately converted files receive separate capture namespaces. See the
[source schema](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon).

**AgentX:** native blocks contain 64 tokens, with chained IDs local to each
source session. Main and nested subagent requests share a namespace within the
same session/model. Different models and sessions cannot share cache entries.
Nested requests already carry session-relative `t` values; wrapper timestamps
are not added again. Whole-session identity keeps all branches/models together
in any later group partition.

Absolute cross-session arrival times are unavailable. By default session starts
are aligned; `--session-spacing-ms` adds a fixed offset per session in source
order and records that choice. Within-session offsets and observed overlap
remain intact. This fixed-arrival simulation does not reconstruct
completion-dependent spawn/join scheduling or reproduce the official AgentX
serving benchmark. Input counts are upstream block-count proxies; provider
templates and hidden content are approximated upstream. See the
[methodology](https://inferencex.semianalysis.com/agentx/methodology) and the
[release's input-length caveat](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126).
