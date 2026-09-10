# Qwen-inclusive policy evolution

Date: 2026-09-09

The new **587-node policy scores 60.818 before AST cost**, a
**1.962-point gain** over the 617-node parent on the frozen five-domain visible panel.
It is retained as an experimental seed; no incumbent was promoted. Fresh Qwen
holdout behavior is essentially tied with the parent and trails TinyLFU-LRU.
AgentX remains inconclusive after a 15-minute replay timeout. The existing
agentic probe still flags token-hit rate and admission waste.

## Policy change

The [saved policy](../../src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/qwen_mixed_20260909.py) retains admission-pressure history more strongly
(`0.86` instead of `0.78`), raises the base admission score from `-0.65` to `-0.50`,
and removes the eviction bonus for long previous access gaps. It adds no new state
table. Complexity falls by 30 nodes (4.9%). The winning changes were found through
local mutations; all six broad initialization proposals and both later paradigm
proposals exceeded the source-size limit.

## Visible comparison

| Policy | Raw behavior | Charged score | Charged AST nodes |
| --- | ---: | ---: | ---: |
| New policy | 60.818 | 53.066 | 587 |
| Parent | 58.856 | 50.809 | 617 |
| LRU | 45.102 | 45.102 | Not charged |
| TinyLFU-LRU | 58.741 | 58.741 | Not charged |

Scores are objective points, not hit percentages. Raw behavior removes only the AST
charge, retaining latency, churn, fairness, underfill, and admission costs. Registered
baselines receive no AST charge under the existing convention, and TinyLFU-LRU still
leads the charged score.

The improvement is primarily lower churn: the global churn charge falls from
9.410 to 7.588. Qwen validation hit rates fall from 31.91%/35.02% to 28.69%/32.76%,
while evictions per 1,000 requests fall from 1,941/861 to 205/319. Qwen underfill
rises from 16.28%/14.82% to 37.28%/31.63%. This is a cache-economics tradeoff,
not an across-the-board hit-rate improvement. These window metrics average the
two configured capacities.

## Fresh Qwen holdout, evaluated after selection

| Policy | Raw behavior | Charged score | Charged AST nodes |
| --- | ---: | ---: | ---: |
| New policy | 63.267 | 55.515 | 587 |
| Parent | 63.293 | 55.246 | 617 |
| LRU | 47.712 | 47.712 | Not charged |
| TinyLFU-LRU | 67.090 | 67.090 | Not charged |

The new policy changes the Qwen holdout raw score by **-0.026 points**
relative to the parent: effectively tied, with a small behavioral regression.
Its charged score improves because its source is smaller. TinyLFU-LRU leads
both comparisons. The following token-hit rates show each capacity separately;
the aggregate score includes cache costs and is not token weighted.

| Holdout | Capacity (tokens) | New policy hit | Parent hit | LRU hit | TinyLFU hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen_bailian_api_1 | 65,536 | 36.45% | 37.31% | 35.11% | 34.49% |
| qwen_bailian_api_1 | 262,144 | 36.70% | 39.45% | 40.95% | 34.49% |

Qwen uses a later, disjoint 15-second window (469 requests), with native 16-token
blocks and 65,536/262,144-token cache capacities. No policy tuning followed hidden
evaluation. This short window limits generalization claims.

## AgentX runtime limitation

The original combined hidden replay included the previously frozen uniform
hash-ranked AgentX sample: **four complete sessions, 165 requests, 19,681,280
prompt tokens**, native 64-token blocks, and the same token capacities. Its
selection seed is 20260908. The candidate container exited 137 after **900.059
seconds**, matching the configured 900-second timeout; no Docker OOM event was
observed. It produced no complete performance result. Queued controls were
cancelled after that failure, so there is no AgentX policy comparison.

Qwen was then evaluated separately, preserving its exact rows, geometry, scoring,
policy seed, and resource limits. All four policies completed that panel. The
AgentX sample was neither redrawn nor trimmed, and its results were not used to
modify the policy. AgentX validation remains **inconclusive**, not passed. A
longer or more efficient replay is needed before a general agent-traffic claim.

## Execution and reproducibility

- Search covered synthetic, Mooncake, WildChat, LMCache, and Qwen visible traffic.
  Historical consumed holdouts were excluded; only fresh Qwen and AgentX were hidden.
- The approved run used Luna mutations and Sol paradigms, with limits of $8,
  32 reserved evaluations, and one hour. It stopped after **30 completed attempts**
  (nine failures) when outstanding reservations reached the evaluation limit.
  Recorded model cost was **$0.71161388**, with **48.66 minutes** of search time.
  This raises the previously recorded $3.1141411 continuation-chain cost to
  **$3.82575498**. Local replay and reporting time are additional.
- Selection was frozen before hidden evaluation. The raw winner was retained,
  then a separately hashed packaged copy was Ruff-formatted. Their ASTs are
  exactly equal. The packaged source passed all 115 visible trials and reproduced
  the exact raw score under the preflight evaluation identity. All eight final Qwen
  hidden trials passed under one shared Qwen evaluation identity. The separate
  AgentX timeout remains recorded and excluded from score comparisons.
- Candidate execution used the pinned, networkless Docker image. Raw traces and
  hidden scores were never sent to the models. The original combined holdout
  used one worker; the smaller Qwen-only comparison used two workers.
- `ruff format .` and `ruff check .` passed. No policy was changed after holdout
  results. The immutable incumbent registry remains unchanged.

Full results, per-trial metrics, source hashes, frozen selection, and probe gates
are in the [machine-readable report](qwen_evolution_20260909.json). Local replay
artifacts and reproduction scripts are under `artifacts/qwen-evolution-20260908/`.
