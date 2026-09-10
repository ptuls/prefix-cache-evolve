# Joint Synthetic and Mooncake Evolution

> [Replay audit update](replay_audit_20260907.md): this report preserves historical replay semantics. Missing-session handling and parts of the conversational replay were corrected afterward; compare policies again under the audited contract before drawing new conclusions.

Date: 2026-09-07

Status: **completed joint improvement; promotion blocked by the agentic tail gate**

This experiment added the frozen Mooncake production tool-and-agent trace to
the full synthetic evolution suite while preserving each traffic source's
native geometry. Synthetic workloads used 16-token blocks at 24 and 48 blocks;
Mooncake used 512-token blocks at 128, 512, and 2,048 blocks. The selection
panel contained 60 synthetic validation trials and six Mooncake validation
trials. Synthetic hidden families and two later Mooncake windows remained
unopened until policy selection was complete.

The first 32-evaluation search started from the prior Mooncake specialist and
did not improve it. A second 48-evaluation search started from the general
synthetic incumbent and used the Mooncake winner's recurrence-gated admission
mechanism as crossover guidance. It used `openai/gpt-5.6-luna` for ordinary
mutations and `openai/gpt-5.6-sol` for diversity and paradigm shifts. Raw trace
records and credentials were not sent to OpenAI.

The second search found a joint policy with a visible score of **61.942**. A
post-search source audit removed one unused assignment, reducing effective
complexity from 627 to 613 without changing any cache decision. The required
full rerun raised the code-as-data score to **62.078**.

## Visible selection results

| Policy | Combined score | Validation token hit | Block hit | Churn / 1k | Waste | Underfill |
|---|---:|---:|---:|---:|---:|---:|
| Simplified joint candidate | **62.078** | **57.53%** | **53.64%** | **159.3** | **67.62%** | 6.31% |
| Generated joint winner | 61.942 | 57.53% | 53.64% | 159.3 | 67.62% | 6.31% |
| TinyLFU-LRU | 61.615 | 56.03% | 51.80% | 478.5 | 71.45% | **5.36%** |
| Mooncake-seeded policy | 60.235 | 57.32% | 53.36% | 544.3 | 79.96% | 2.30% |
| General synthetic parent | 58.523 | 57.40% | 53.53% | 253.2 | 68.73% | 4.26% |
| LRU | 37.325 | 54.94% | 50.92% | 1,618.3 | 81.99% | 0.0% |
| vLLM-style APC | 25.307 | 51.34% | 43.96% | 928.3 | **52.98%** | 15.76% |

The simplified candidate leads TinyLFU-LRU by 0.463 points while reducing
churn by 66.7% and admission waste by 3.9 percentage points. Its policy
underfill is 0.9 percentage points higher. The exact aggregate hit rates for
reference policies are retained in the run artifacts; rounded values above are
provided for orientation.

## Traffic-domain balance

| Visible domain | Joint candidate | General parent | Mooncake specialist |
|---|---:|---:|---:|
| Synthetic suite | **65.422** | **65.649** | 61.368 |
| Mooncake validation windows | **57.970** | 40.465 | **63.158** |

Relative to the general parent, the joint policy gives up only 0.227 points on
synthetic traffic and gains 17.505 points on Mooncake. Relative to the Mooncake
specialist, it gives up 5.188 Mooncake points but gains 4.054 synthetic points.
The combined objective therefore selected a compromise policy rather than a
domain-specific winner.

## Mechanism

The evolved change has two load-bearing parts:

- It normalizes recomputation cost by `6 * block_size_tokens`. At 16-token
  blocks this exactly preserves the parent's original denominator of 96; at
  512-token blocks it removes an absolute-token-scale bias.
- It attenuates the depth admission penalty by as much as 90% after online
  decayed frequency exceeds one. Recurring deep agent routes can therefore be
  admitted without broadly opening the cache to one-off suffixes.

Eviction scoring and the parent's pressure state remain unchanged. The result
is a compact crossover of the synthetic incumbent's low-churn scorer with the
Mooncake specialist's recurrence-evidence exception; it does not dispatch on a
workload name or contain two complete policy implementations.

## Frozen hidden adjudication

The simplified source was fixed before the hidden panel was opened. It was then
replayed over ten shifted synthetic families and two later Mooncake windows
containing 1,296 requests each.

| Policy | Hidden score | Token hit | Block hit | Worst-quarter hit | Request p10 | Waste | Churn / 1k | Underfill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Simplified joint candidate | **6.129** | **44.78%** | **41.14%** | **31.98%** | **15.35%** | **58.27%** | **200.0** | 7.91% |
| TinyLFU-LRU | -0.230 | 42.9% | 38.6% | 29.9% | 15.3% | 63.7% | 831.5 | **5.4%** |
| vLLM-style APC | -12.789 | 42.9% | 36.6% | 31.2% | 15.2% | 62.7% | 1,438.0 | 7.3% |
| LRU | -14.557 | 41.7% | 38.0% | 29.8% | 13.6% | 80.8% | 2,268.5 | 0.0% |

The joint candidate leads TinyLFU-LRU by 6.359 hidden points. On the two
Mooncake holdouts its token-hit rates are 35.12% and 35.45%, compared with
35.36% and 35.68% for TinyLFU-LRU. Its hidden advantage therefore comes from
much lower churn and waste plus stronger shifted synthetic performance, rather
than higher Mooncake hit rate alone.

## Gate and decision

The candidate passes six of seven agentic surrogate-to-probe checks. The only
failure is request-tail token hit: the surrogate/probe absolute gap is 0.1645
against a 0.1500 limit. Aggregate agentic token-hit gap, worst-quarter hit,
waste, underfill, short-reuse eviction, and churn all pass. The candidate also
improves the reporting-only agent-branching probe token hit to 44.3%, versus
36.4% for TinyLFU-LRU.

Do not promote this policy to the immutable incumbent registry yet. It is the
strongest joint synthetic-plus-Mooncake candidate from this experiment and
passes hidden adjudication, but the fail-closed request-tail gate requires an
independent trace or online shadow test rather than further tuning on the now
opened holdout.

## Reproducibility

- Run bundle:
  `artifacts/mooncake-synthetic-evolution/full-run-from-incumbent/20260906T141808Z`
- Generated winner: `best_program.py`, SHA-256
  `a115477d847ea2f47aa7913c8c3a8068af85ff3df150e3bf5d463626414f1eac`
- Audited simplification: `best_program_simplified.py`, SHA-256
  `94c8b8ca5c3267d7f9a5b6269fbaec9cec6755da6fcbc87923e748fa96c34015`
- Tracked future seed:
  `src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/joint_mooncake_synthetic_20260907.py`
  with SHA-256
  `ea310f1ec3139452e93eb560e57220c5477fc0ebe7ae6609575c92af21edb22f`;
  its Ruff-formatted source reproduced all four reported scores exactly
- Panel SHA-256:
  `dc7d97c2870a835cd68ad0179950026e11f40a35e0155ba34974917baa442da9`
- Evaluation context SHA-256:
  `c22962b345b7f11ae4547edf0ec55873252634140125a8791d4eee1a812ccb19`
- Pinned sandbox image:
  `sha256:8e6ce5586d4b6dd68f5bad1cd70dd83dc05c9f444ac930eec173e297898621b3`

The aborted incompatible-temperature attempt cost $0.4975, the
Mooncake-seeded search cost $0.6392, and the successful incumbent-seeded search
cost $1.0586, for $2.1954 in recorded model charges. This remained well below
the approved $20 cap. Local evaluator reruns incurred no model cost.
