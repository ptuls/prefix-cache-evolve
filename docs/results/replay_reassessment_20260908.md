# Policy reassessment after the replay audit

Date: 2026-09-08

**Use the newer four-domain policy as the primary mixed-traffic evolution
seed.** The earlier apparent Mooncake regression largely disappears after
correcting missing sessions. The previous joint policy remains a useful
reference for synthetic traffic and tight-capacity cache efficiency.
TinyLFU-LRU still wins the configured complexity-adjusted objective.

This reassessment ran three unchanged frozen policy sources and all 12
registered deployable baselines against
[`prefix_kv_cache_replay_audited.yaml`](../../configs/prefix_kv_cache_replay_audited.yaml):
58 workload streams and 121 trials per policy, **1,815 successful trials** in
total. Validation determines the score; training is diagnostic feedback, and
the two synthetic probes are reporting-only. No hidden traces, new evolution,
model API calls, or promotions were involved.

## Full corrected ranking

Higher scores are better; these are **objective points, not hit percentages**.
The raw column removes only the AST complexity charge and retains every other
score term, including churn, latency, fairness, waste, and underfill. Registered
baselines receive zero AST charge under the existing scorer.

| Policy | Configured score | Before AST charge | Effective AST nodes |
| --- | ---: | ---: | ---: |
| TinyLFU-LRU | 61.060 | 61.060 | Not charged |
| Newer four-domain | 54.680 | 62.844 | 629 |
| LFU | 51.318 | 51.318 | Not charged |
| Prefix anchor | 47.546 | 47.546 | Not charged |
| Cost-aware LRU | 47.400 | 47.400 | Not charged |
| LRU | 46.950 | 46.950 | Not charged |
| SGLang radix emulation | 46.950 | 46.950 | Not charged |
| Tenant-fair LRU | 46.927 | 46.927 | Not charged |
| Prefix fanout | 36.308 | 36.308 | Not charged |
| vLLM APC emulation | 35.813 | 35.813 | Not charged |
| Prefer shallow | 35.233 | 35.233 | Not charged |
| Previous synthetic + Mooncake joint | 25.445 | 33.453 | 613 |
| Recompute greedy | 19.870 | 19.870 | Not charged |
| Promoted production incumbent | 12.979 | 20.581 | 572 |
| No cache | -55.831 | -55.831 | Not charged |

The newer policy ranks **second on the configured objective and first before
AST cost**. It has a 1.784-point raw advantage over TinyLFU, but its 8.164-point
complexity charge produces a 6.380-point deficit. This is a meaningful
behavioral improvement over the earlier interpretation, not evidence that
the deployment or promotion criteria have already been met.

## Traffic tradeoffs

These are validation-domain scores derived from the same saved trials and
scoring formula. Each domain is rescored independently, including its own
weakest-group contribution and score caps. They are not additive contributions
to the overall objective or separately measured panel identities.

| Policy | Synthetic score | Mooncake score | WildChat score | LMCache score |
| --- | ---: | ---: | ---: | ---: |
| Previous joint | 65.422 | 57.970 | -66.969 | 182.326 |
| Newer four-domain | 57.166 | 58.061 | 23.119 | 207.607 |
| TinyLFU-LRU | 63.548 | 63.709 | 35.064 | 209.097 |
| Production incumbent | 65.649 | 40.465 | -66.564 | 40.636 |

| Policy | Synthetic token hit | Mooncake token hit | WildChat token hit | LMCache token hit |
| --- | ---: | ---: | ---: | ---: |
| Previous joint | 59.81% | 34.76% | 3.61% | 86.69% |
| Newer four-domain | 59.10% | 36.95% | 48.12% | 94.68% |
| TinyLFU-LRU | 58.01% | 36.26% | 41.77% | 91.02% |

The newer policy gets more token hits than TinyLFU in every validation domain,
yet TinyLFU wins the configured score through different cache economics and
the absence of an AST charge. Higher hit rate alone is not the objective.

The newer policy's synthetic score is 8.257 points below the previous joint
policy, despite a token-hit difference of only 0.71 percentage points. Mean
synthetic churn rises from **174.5 to 448.3 evictions per 1,000 requests**.
The largest local regressions are repeated tenant phase shifts, tenant skew,
and priority-burst recovery. The newer policy loses raw score on 16 of the 20
synthetic validation workload/capacity groups.

On the more concurrent LMCache training split, the newer policy retains its
substantial advantage: **82.67% token hit versus 10.98%** for the previous joint
policy and 78.06% for TinyLFU-LRU. Both training and validation results are archived in
the machine-readable report; the training result does not contribute to the
validation selection score.

## Mooncake: aggregate recovery with capacity-specific differences

All Mooncake replays use native **512-token blocks**. Each row below combines
both validation windows at the stated capacity.

| Capacity in blocks | Previous joint score | Newer score | Previous token hit | Newer token hit | Previous churn / 1,000 | Newer churn / 1,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 60.341 | 55.834 | 34.65% | 35.91% | 19.7 | 455.8 |
| 512 | 57.827 | 57.400 | 34.82% | 36.58% | 0.8 | 438.4 |
| 2,048 | 55.741 | 62.881 | 34.82% | 38.35% | 0.0 | 233.1 |

The older policy still has useful tight-capacity behavior. Its aggregate
Mooncake underfill is 39.23%, versus 3.98% for the newer policy; much of its
low churn comes from admitting very little. At larger capacity, the newer
policy makes better use of available cache space.

For the unchanged newer source, correcting replay semantics raises the full
score from **39.646 to 54.680**. Its mean Mooncake validation hit rate rises
from 35.53% to 36.95%, and churn falls from about 7,750 to 376 per 1,000 requests.
The global churn charge falls from 25.000 to 7.912. All 17 synthetic workload
metric dictionaries, including probes, are exactly equal to the historical
evaluation. These are effects of replay correction, not further learning.

## Revised evolution direction

1. **Start from the newer policy.** It already covers conversational and
   collected agent continuation and now reaches aggregate Mooncake parity
   with the older joint policy. Keep the older policy and production incumbent
   as synthetic-efficiency references.
2. **Test focused admission and eviction ablations before adding a selector.**
   Compare the older block-reuse eviction rule with the newer admission rule,
   and separately tighten admission during sustained pressure while retaining
   observed session-continuation evidence. These are proposed experiments,
   not measured hybrid policies. Use deployable online signals throughout.
3. **Recover synthetic efficiency and simplify together.** An offline
   counterfactual replacing only the newer policy's synthetic validation
   trials with the older policy's trials scores **58.568** at an assumed
   complexity of 629. Even with zero routing overhead, that remains below
   TinyLFU's 61.060. Restoring the old synthetic behavior alone is therefore
   insufficient; improve remaining cache economics and reduce unnecessary
   code. This calculation uses workload labels offline and is neither a
   deployable router nor an evaluated algorithm.

There is no basis here for claiming that a general hybrid has already won.
The newer policy still flags the existing agentic probe gate for hit rate and
admission waste. The older joint policy flags a request-tail check. These
diagnostics were not used to mutate or tune either source during reassessment.

## Scope, validation, and artifacts

- Synthetic traffic uses 16-token blocks with capacities 24/48; Mooncake and
  LMCache use 512-token blocks with capacities 128/512/2,048; WildChat uses
  16-token blocks with capacities 64/128. Every configured visible trace is
  replayed at every native capacity, with all configured synthetic seeds.
- The validation mean has 20 synthetic workload/capacity groups, six Mooncake,
  two WildChat, and three LMCache. The suite is not equally weighted by domain.
- Every final result has the same panel and evaluation-context identities.
  All 1,815 trials were valid, all three frozen source hashes matched their
  originals, and saved scores were independently reaggregated from trials.
- One initial source container exited 137 while three ran concurrently. Its
  isolated retry completed all 121 trials with unchanged policy and settings.
  The local Docker VM exposes about 3.53 GiB total memory, so memory pressure
  is a plausible cause; the exact kill cause was not independently established.
  The reproduction script now defaults to two workers and records failures
  promptly. The initial attempt and recovery are archived.
- Formatter, Ruff, and whitespace checks passed. Evaluator implementation
  hashes matched the audited version; this reassessment changed no runtime
  evaluator or policy logic.
- WildChat remains a 20-conversation reconstructed sample and LMCache a
  16-session collected-agent sample with modeled timing. Existing hidden sets
  have been inspected previously and were not evaluated here. An untouched,
  broader holdout is still required for a new generalization or promotion claim.

Local results, frozen source copies, manifests, execution notes, scripts, and
the offline counterfactual are in `artifacts/replay-reassessment-20260908/`.
The [machine-readable report](replay_reassessment_20260908.json) records the
ranking, domain and capacity diagnostics, score identities, and provenance.
See the [preceding audit](replay_audit_20260907.md) for data-conversion limits.
