# Production Agentic Prefix-Cache Evolution

> [Replay audit update](replay_audit_20260907.md): this report preserves historical replay semantics. Missing-session handling and parts of the conversational replay were corrected afterward; compare policies again under the audited contract before drawing new conclusions.

Date: 2026-09-06. Branch: `feat/real-data`.

The audited, bounded search did **not** produce a policy improvement. Its final
archive contained only the unchanged production parent, so validation-only
selection froze that exact source before hidden evaluation. TinyLFU-LRU remained
the strongest policy under the production objective on both visible validation
(`63.709`) and the untouched later windows (`50.491`). No policy was promoted.

The structured record is
[mooncake_evolution_20260906.json](mooncake_evolution_20260906.json).

## Data and protocol

The source is the original [Mooncake FAST25 tool-and-agent trace](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces)
from Kimi production traffic. The complete source has 23,608 requests and
contains timestamps, prompt/output lengths, and cumulative prefix identities at
native 512-token blocks. It contains no prompt content and does not expose real
session, tenant, priority, model, or tool-DAG identity.

| Split | Source-clock interval, ms | Requests | Input tokens |
|---|---|---:|---:|
| train | [0, 90,000) | 483 | 4,680,571 |
| validation | [1,200,000, 1,290,000) | 600 | 5,191,348 |
| validation | [2,100,000, 2,190,000) | 659 | 5,638,767 |
| hidden | [3,000,000, 3,180,000) | 1,296 | 11,089,082 |
| hidden | [3,300,000, 3,480,000) | 1,296 | 10,961,476 |

Search used 1,742 requests in train and visible validation. The final production
comparison opened 2,592 later requests only after `selection.json` fixed the
source and SHA-256. Every window/capacity trial started with cold policy and
cache state. Capacities were 128, 512, and 2,048 blocks. Search and final
evaluation used verifier 1.0.0 with unchanged weights.

## Audit result

The local audit repaired eight concrete isolation, result-integrity, import,
source-validation, and panel-provenance defects found across external review
rounds. The final OpenAI autoreview reported no actionable finding and rated the
patch correct with confidence `0.94`. The reviewed source passed
470 pytest tests at 87.18% coverage, Ruff format/check,
configured mypy across 35 source files, and validation of all four immutable
incumbent bundles.

The frozen image is `sha256:15972abc69f4141ae90ddd06d1b8fdd811ca3c33be56864133b4307a86d7e7f1`. It contains the same 73 Python files
as the checkout by SHA-256. A zero-model-call preflight exercised the real Levi
worker-to-container path. Candidate containers ran as user `10001:10001` with no
network, a read-only root and input mount, no provider credentials, dropped
capabilities, and bounded resources. Host and container agreed exactly on
scores, identities, per-request hit lengths, and cache metrics. Seven differences
in reporting-only shadow-price diagnostics were at most `4.44e-16`.

## Search outcome

The configured limits were 32 evaluations, $8 accounted model cost, and 3,600 seconds.
The wall deadline was binding: the run completed 3 evaluator
attempts in 3600.1 seconds and spent `$1.1775`.
`gpt-5.5` generated diversity/paradigm proposals and
`gpt-5.4-mini` generated mutations. Several initialization responses could not
be extracted as code; one candidate was rejected by the documented source
contract, and the one scored mutation reached `30.674962`, below the parent at
`40.464543`.

The final archive size was one. The selected source is byte-identical to the unchanged
parent: `0c19deaec83d83f776e837c58822ac4cada611caa0a5504d17816e50e08f0037`. It has effective complexity 572, zero invalid
trials, and pays a 7.603 complexity cost. OpenAI received policy
source, mutation instructions, and aggregate visible feedback after explicit
approval. Raw trace records and credentials remained local.

## Untouched production holdout

| Policy | Charged | Before complexity | Token hit | Worst quarter | Churn / 1k | Waste | Underfill |
|---|---:|---:|---:|---:|---:|---:|---:|
| TinyLFU-LRU | 50.491 | 50.491 | 35.52% | 32.22% | 1031.9 | 89.59% | 16.97% |
| vLLM-style APC | 37.256 | 37.256 | 34.04% | 31.18% | 7815.3 | 98.80% | 0.08% |
| LRU | 36.380 | 36.380 | 33.74% | 30.90% | 8541.9 | 98.21% | 0.00% |
| selected (unchanged parent) | 34.900 | 42.502 | 35.67% | 32.69% | 2621.0 | 95.90% | 2.92% |
| parent repeat | 34.900 | 42.502 | 35.67% | 32.69% | 2621.0 | 95.90% | 2.92% |

The selected parent has the highest token-hit rate at 35.67%, but its churn is
2,621 evictions per 1,000 requests and its complexity charge reduces its raw
42.502 score to 34.900. TinyLFU-LRU keeps nearly the same hit rate at 35.52%,
cuts churn to 1,032, and wins the charged objective by 15.592 points. The selected
parent's token hit is consistent across the two hidden windows:
35.48% in the first and 35.86% in the second.

## Original synthetic regression

Visible original panel:

| Policy | Charged | Before complexity | Token hit | Worst quarter | Churn / 1k | Waste | Underfill |
|---|---:|---:|---:|---:|---:|---:|---:|
| selected (unchanged parent) | 65.649 | 73.252 | 59.64% | 46.29% | 163.9 | 65.99% | 3.07% |
| parent repeat | 65.649 | 73.252 | 59.64% | 46.29% | 163.9 | 65.99% | 3.07% |
| TinyLFU-LRU | 63.548 | 63.548 | 58.01% | 43.22% | 499.0 | 69.67% | 2.65% |
| LRU | 49.499 | 49.499 | 56.92% | 43.07% | 999.2 | 80.37% | 0.00% |
| vLLM-style APC | 46.427 | 46.427 | 52.94% | 43.09% | 313.1 | 48.41% | 17.31% |

Hidden original panel:

| Policy | Charged | Before complexity | Token hit | Worst quarter | Churn / 1k | Waste | Underfill |
|---|---:|---:|---:|---:|---:|---:|---:|
| selected (unchanged parent) | 3.064 | 10.667 | 45.52% | 31.59% | 194.4 | 55.08% | 6.94% |
| parent repeat | 3.064 | 10.667 | 45.52% | 31.59% | 194.4 | 55.08% | 6.94% |
| vLLM-style APC | -2.725 | -2.725 | 43.79% | 31.19% | 800.3 | 59.07% | 8.05% |
| TinyLFU-LRU | -2.789 | -2.789 | 43.60% | 29.63% | 811.5 | 61.07% | 4.20% |
| LRU | -17.524 | -17.524 | 42.54% | 29.64% | 1641.1 | 79.10% | 0.00% |

Because selection retained the original parent, it reproduces the historical
synthetic visible result (`65.649`) and leads these comparators on synthetic
hidden (`3.064`). This establishes no newly evolved transfer gain; it confirms
that the failed production search did not change or regress the incumbent.

## Decision and limits

No new incumbent bundle is created. The search was too shallow to establish
that the parent cannot be improved, while its result is sufficient to reject a
claim of successful policy evolution. TinyLFU-LRU is the practical winner in
this controlled production replay and should be the baseline for a future run.

The panel is one same-capture chronological sample. Cold-window replay, missing
identity/content fields, output-length pinning, and latency proxies limit claims
about end-to-end serving. The trace represents production tool-and-agent traffic
and does not support a coding-agent-specific conclusion.

Exact local evidence is under `artifacts/mooncake-evolution/`, including the
frozen config, approval, review outputs, preflight, snapshot, selection, full
per-trial results, and SHA-256 manifests. The original synthetic config remained
`a2c34ace5741b5cfa2707b40c5a3ae7d419b6caf651a48aeb37cc0fafb10ef53`.
