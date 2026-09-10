# Qwen-inclusive evolution from a minimal LRU seed

Date: 2026-09-09

Starting from a **48-node LRU seed**, the one-hour scratch run found a
**115-node policy scoring 49.176 before AST cost**, up **4.074 points** from
45.102. It remains well below the continuation winner at 60.818 and TinyLFU-LRU
at 58.741 on the identical visible panel. On fresh Qwen traffic, its raw score
is only slightly above LRU and substantially below both stronger controls.
The policy is retained as an experimental seed; no incumbent was promoted.

## What the search found

The [saved policy](../../src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/qwen_scratch_20260909.py)
evicts the inactive resident leaf with the largest score:

```python
age / ((1 + hit_count) * (1 + subtree_hit_rate))
```

This protects frequently reused blocks while retaining recency. It adds no
policy-owned state. Admission remains unconditional under the simulator's actual
one-based block depths. The generated `depth == 0` rejection cannot fire, and
the other admission branch returns the same value as its fallback. These
redundancies are retained in the exact evaluated source, with their AST cost.
No manual policy changes followed selection or hidden evaluation.

The first improvement scored 49.041 with 65 nodes. Later mutations raised raw
score to 49.139 and then 49.176. The predeclared selection rule maximized raw
visible score among valid archived policies at or below 650 nodes, breaking
exact ties by lower complexity. It therefore selected the 115-node winner.

## Visible comparison

| Policy | Raw behavior | Charged score | Charged AST nodes |
| --- | ---: | ---: | ---: |
| Scratch winner | 49.176 | 46.893 | 115 |
| Continuation winner | 60.818 | 53.066 | 587 |
| LRU | 45.102 | 45.102 | Not charged |
| TinyLFU-LRU | 58.741 | 58.741 | Not charged |

Scores are objective points, not hit percentages. Raw behavior removes only
the AST charge; cache-quality costs remain. Registered baselines receive no
AST charge under the existing scoring convention.

The scratch policy still reaches the maximum global churn charge of 25.000,
compared with 7.588 for the continuation policy and 5.720 for TinyLFU-LRU.
Its Qwen validation token-hit rates are 34.55% and 35.31%, averaging the two
capacities in each window, but churn remains about 15,636 and 15,551 evictions
per 1,000 requests. The continuation policy has lower hit rates on those windows
(28.69% and 32.76%) but dramatically lower churn (205 and 319). The scratch
run improved eviction ranking without discovering effective admission control.

The saved scratch source passed all **115 visible trials** and reproduced the
exact selected raw score. The visible controls reuse retained trial measurements:
their source and visible configuration were verified unchanged, and their scores
were reaggregated under the new configuration's identity. Every retained trial
metric and raw score is unchanged. This is not a fresh replay of those controls.
A fresh preflight of the minimal LRU seed independently matched all 115 previous
LRU trials, excluding the source-complexity field. The visible panel hash is
unchanged; context hashes differ from the earlier run because they also include
unused hidden-trace pins.

## Fresh Qwen holdout

| Policy | Raw behavior | Charged score | Charged AST nodes |
| --- | ---: | ---: | ---: |
| Scratch winner | 2.204 | -0.078 | 115 |
| Continuation winner | 14.504 | 6.752 | 587 |
| LRU | 1.928 | 1.928 | Not charged |
| TinyLFU-LRU | 13.753 | 13.753 | Not charged |

All four policies were freshly replayed after selection on the same previously
unscored Qwen window: **[6,600,000, 6,615,000) milliseconds**, or 110 minutes
into the capture. It contains **361 requests and 356,567 prompt tokens**, using
native 16-token blocks. All eight trials passed under one evaluation identity.
This window is distinct from the earlier continuation run's holdout, so its
absolute scores should not be compared directly with that earlier holdout.

| Capacity (tokens) | Scratch hit | Continuation hit | LRU hit | TinyLFU hit |
| --- | ---: | ---: | ---: | ---: |
| 65,536 | 16.98% | 14.82% | 16.77% | 14.31% |
| 262,144 | 22.94% | 14.82% | 22.94% | 14.31% |

Scratch improves raw holdout score over LRU by only **0.276 points** and trails
the continuation winner by **12.299 points**. Its higher hit rates come with
40,346 and 2,629 evictions per 1,000 requests at the two capacities. Both
continuation and TinyLFU-LRU have zero evictions in this window, while accepting
lower hit rates and greater policy underfill. Continuation leads the raw
objective; TinyLFU-LRU leads after the candidate AST charge.

This is a short cold-start window, with modeled arrival buckets and serving
costs. It does not establish broad production performance. The existing agentic
probe still flags wasted admission tokens; the cyclic-working-set probe passes.
Those probes are reporting checks, separate from the fresh Qwen assessment.
**AgentX was deferred** under the frozen plan: its previous sample timeout
remains unresolved, and no new AgentX replay or resampling was performed.

## Search limits and interpretation

- Search used the same five visible traffic domains, scoring, source-size limits,
  model choices, initialization counts, and worker settings as the continuation
  run. It started with an empty archive and minimal LRU source, a neutral task
  prompt without the continuation policy or its scores, and search seed 20260909.
- The run stopped at the **one-hour time limit** after **28 completed attempts**:
  16 successful evaluations including the initial seed, and 12 failures. Two
  evaluations were in flight at the last status update before shutdown. The
  configured 32 evaluation slots were a ceiling, not 32 completed evaluations.
- All six broad initialization proposals were rejected by the source-size check.
  The initialization variant and five later attempts failed callback validation.
  Initialization therefore left one archive cell. Both later paradigm attempts
  were skipped because they required two elites. The run effectively continued
  through local mutations of one elite.
- Recorded model cost was **$0.6208524**, within the approved $8 limit. Search
  runtime was 3,600.25 seconds. Including the previously recorded continuation
  chain, total model cost is **$4.44660738**. Preflight, replay, reporting, and
  engineering time are additional and excluded from those API costs.

This pilot did not rediscover the stronger continuation policy. It also does
not isolate the effect of the seed alone: the prompt and search RNG changed,
and failed initialization severely restricted exploration. Before spending more
on repetitions, the next issue to investigate is reliably obtaining valid,
diverse initialization within the size limit and preserving multiple archive
cells when initialization is sparse. Any new search needs a fresh final holdout.

## Reproducibility

Selection froze at **2026-09-09 05:51:01 UTC**, before fresh hidden evaluation.
Raw and packaged winner sources are byte-for-byte identical, with SHA-256
`138dc6b5f102e33cc09a7f09a82aa6b754bbed29d858c2b2ad08bf38e3457c7b`.
Ruff left the source unchanged, and AST equality was checked. Candidate code ran
in the pinned, networkless Docker image; the offline comparisons made no model
calls. Hidden scores and raw traces were excluded from model requests.

`ruff format .` and `ruff check .` passed. Benchmark policy hashes were verified
unchanged by formatting, and the immutable incumbent registry was not modified.
The [machine-readable report](qwen_scratch_20260909.json) retains exact source,
frozen configuration and prompt, selection provenance, score history, per-trial
metrics, panel identities, and probe gates. Local run artifacts and reproduction
scripts are under `artifacts/qwen-scratch-20260909/`.
