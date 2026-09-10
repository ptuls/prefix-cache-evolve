# Project Overview And Documentation

This document summarizes the current results, research workflow, scope, and
contribution opportunities, then maps the repository's detailed documentation.

## Current Results

Evolution found deployable policies with strong but geometry-dependent results.
The historical 8-token discovery policy scores `77.113` under hardened
complexity accounting, ahead of TinyLFU-LRU at `70.362`. The same policy scores
`62.757` when transferred to the operative 16-token verifier, below
TinyLFU-LRU at `63.548`, although it reduces churn from `499.0` to `168.1` per
thousand requests.

A later production-oriented search and simplification stage produced a separate
16-token policy scoring `65.649`, ahead of TinyLFU-LRU at `63.548`, with churn
of `163.9` rather than `499.0`. In the broader geometry sweep, that policy wins
at 16- and 24-token blocks, trails at 32-, 48-, and 64-token blocks, and retains
substantially lower churn throughout. No single incumbent currently dominates
across geometries.

The original discovery result was reported as `77.230`; hardened complexity
accounting changes it to `77.113` without changing policy behavior. Scores from
different verifier geometries are not directly comparable.

Producing the headline production incumbent required two directly necessary
final-stage searches: the production search cost `$4.133`, and simplification
cost another `$0.937`, for `$5.070` in recorded model API charges. The honest
documented pre-promotion research total is at least `$44.845` across `2,989`
evaluations. These figures exclude engineering time, local compute, and
experiments without retained cost metadata.

The stronger result is methodological:

- Pressure-aware admission and online structural context were more useful than
  broad recurrence machinery.
- Fine-grained verifier feedback and specialist archive dimensions made compact
  improvements discoverable.
- Allowing bounded over-cap exploration followed by a separate simplification
  stage produced the deployable production incumbent.
- Specialized and held-out workloads caught promising-looking policies that
  overfit, churned excessively, or failed to transfer.
- Exact incumbent replay is reproducible, but independent rediscovery from a
  weak seed is not yet established. Three fresh 298-evaluation searches
  produced zero behaviorally close policies.

The incumbent was not found by one clean run from scratch. It emerged through a
staged research process that co-evolved the verifier, generator feedback,
archive, policy lineage, and promotion gates.

The incumbent headline remains a deterministic synthetic result. A separate
production-trace experiment now tests transfer and bounded continued evolution;
it does not replace the synthetic headline or establish broad production
generality.

The initial [Mooncake tool-and-agent replay](results/mooncake_toolagent_20260905.md)
adds a bounded production-traffic transfer check at the source's native
512-token geometry. It is separate from those headline experiments and does
not establish a general ranking across production agent workloads.

The completed [production agentic evolution](results/mooncake_evolution_20260906.md)
used frozen chronological search and holdout windows after a clean isolation
audit. The one-hour search reached only three evaluator attempts and found no
improvement over the unchanged parent. On untouched later production windows,
TinyLFU-LRU scored `50.491` against the parent's `34.900` while retaining nearly
the same token-hit rate. The parent still led the tested policies on the original
synthetic hidden panel. No policy was promoted.

The [GPT-5.6 full run](results/mooncake_evolution_gpt56_20260906.md) used Luna
for mutations and Sol for paradigm shifts. Its selected policy beat TinyLFU-LRU
by `10.603` points on untouched Mooncake windows, mainly by reducing churn and
admission waste, but regressed on the original synthetic panels. The general
incumbent therefore remains unchanged.

The [joint synthetic and Mooncake run](results/mooncake_synthetic_evolution_20260907.md)
then evaluated both traffic domains at native geometry during selection. Its
audited 613-node candidate scores `62.078` on visible joint validation versus
`61.615` for TinyLFU-LRU and leads the same baseline by `6.359` points on the
post-selection hidden panel. It nearly preserves the general incumbent's
synthetic score while gaining `17.505` points over that parent on visible
Mooncake traffic. One request-tail agentic gate remains flagged, so no incumbent
was promoted.

The subsequent
[four-domain agentic run](results/mooncake_synthetic_wildchat_lmcache_evolution_20260907.md)
added WildChat and a pinned, metadata-only replay of LMCache collected agent
sessions. Evolution raised the seed from `25.854` to `39.050`; a verified
source simplification reduced effective complexity from 691 to 629 and raised
the charged score to `39.646` without changing behavior. The policy recovered
WildChat validation hit from `4.0%` to `50.0%` and reached `94.7%` on LMCache
validation. TinyLFU-LRU still scored `62.491` visibly and `11.705` on the frozen
hidden panel, versus `-7.974` for the evolved policy. High churn, admission
waste, and a flagged agentic probe gate block promotion.

The [replay audit](results/replay_audit_20260907.md) documents corrections to
missing session identities, fabricated LMCache tenants, invalid-row scheduling,
WildChat model separation and serialization, and visible training feedback.
Historical results retain the earlier replay semantics. New bundles and a
container are pinned separately; no new evolution was launched.

The [corrected full-suite reassessment](results/replay_reassessment_20260908.md)
compares both evolved policies, the production incumbent, and all 12 deployable
baselines across 1,815 trials. The newer policy reaches `54.680` versus the old
joint policy's `25.445`; TinyLFU-LRU leads at `61.060`. The newer policy leads
before AST cost (`62.844`) and recovers aggregate Mooncake parity, but retains
synthetic churn and probe-gate weaknesses. It is the recommended next seed;
no new evolution or promotion was performed.

The [complexity review](results/complexity_review_20260908.md) breaks down the
newer policy's 629 effective AST nodes and 8.164-point penalty, then rescores
the frozen comparison under alternative coefficients. It recommends separating
behavioral performance, source size, and measured serving cost; registered
baselines currently receive no AST charge. No scoring configuration was changed.

The subsequent [implementation guide](policy_costs.md) provides new raw-score
exploration and simplification configurations, retained archive alternatives,
and container-based callback/state measurements with explicit operating budgets.
Historical configurations and results retain their original semantics.

The [Qwen-inclusive evolution run](results/qwen_evolution_20260909.md) adds native
Qwen production traffic to visible selection and reserves fresh Qwen and a frozen
four-session AgentX sample for final evaluation. Its experimental seed improves
raw behavior from `58.856` to `60.818` while reducing effective source complexity
from 617 to 587 nodes. Recorded model cost is `$0.712`. Qwen churn falls at the
expense of hit rate and cache utilization; the agentic probe remains flagged.
The fresh Qwen holdout essentially ties the parent and trails TinyLFU-LRU;
AgentX validation is inconclusive after the 15-minute evaluator timeout.

The subsequent [minimal-seed scratch run](results/qwen_scratch_20260909.md)
starts from 48-node LRU and reaches `49.176` raw visible score with a 115-node
policy, below the continuation winner's `60.818`. It costs `$0.621` over
28 attempts in one hour, including 12 failures. Failed initialization leaves
one archive cell and prevents later paradigm proposals. On a new Qwen holdout,
scratch scores `2.204` raw versus `14.504` for continuation and `13.753` for
TinyLFU-LRU; its improvement over LRU is only `0.276` points. No incumbent was
promoted, and AgentX remains deferred. This pilot does not establish independent
rediscovery or isolate the seed's causal effect.

The `vllm_apc` baseline behaviorally emulates the core APC cache policy inside
the controlled simulator: exact-prefix reuse of full blocks, active-reference
protection, and LRU eviction of reusable unreferenced blocks. It does not
reproduce vLLM's internal data structures or additional serving optimizations,
including scheduling, allocation, continuous batching, offload, and kernels.
The SGLang baseline has the same policy-level scope. The reported scores compare
cache-policy logic under a common contract, not end-to-end serving throughput.

## Research Workflow

The repository separates productive optimization from scientific
adjudication:

- **Production evolution** starts from the retained incumbent and searches for
  further improvements.
- **Weak-seed rediscovery** tests whether the search process can independently
  recover a behaviorally close policy without incumbent source or coefficients.
- **Specialist searches** isolate narrower surfaces, such as eviction ranking.
- **Simplification searches** turn useful over-cap candidates into deployable
  policies.
- **Analysis tools** run causal experiments, regret audits, ablations, geometry
  sweeps, and rediscovery adjudication.

The current weak-seed rediscovery verdict is negative. Exact replay and
independent discovery are deliberately reported as separate claims.

## Included Capabilities

- A deterministic prefix-tree cache simulator with root-contiguous hits,
  leaf-only eviction, active decode pins, forced bypass, and partial blocks.
- Train, validation, quarantined probe, and hidden synthetic workload panels.
- Deployable baselines including TinyLFU-LRU, vLLM APC, SGLang
  RadixAttention, LRU, LFU, recompute-aware, prefix-fanout, and tenant-fair
  policies.
- Reporting-only future-knowledge controls and decision-level regret audits.
- Versioned score identities, workload manifests, immutable incumbent bundles,
  and saved evolution artifacts.
- An anonymized metadata-only trace calibration, replay, and evolution path,
  including WildChat conversion, Mooncake production-trace replay, and pinned,
  disjoint trace panels where session identities are available.
- An interactive policy-comparison lab.

## Scope And Limitations

This repository studies online prefix-cache heuristics under controlled
workloads. It does not model a complete serving stack, and the current headline
does not include a public production trace.

The production trace workflow configures a Docker backend for untrusted source
candidates, with no network, a read-only filesystem, and only selected inputs
mounted. Other workflows default to process isolation and resource limits,
which are not a security sandbox. Configure OS/container isolation before
executing untrusted generated code on a credential-bearing workstation.

## Contributing

The highest-value contributions are:

- workload models and anonymized trace replays that expose real serving failure
  modes;
- comparisons against production cache managers;
- policy ideas that improve agentic branching or geometry transfer;
- verifier, feedback, and mutation improvements that make weak-seed
  rediscovery repeatable.

## Documentation Map

- [`technical_report.tex`](technical_report.tex) is the paper-like current
  account: method, current evidence, principles, limitations, and
  reproducibility.
- [`research_log.tex`](research_log.tex) preserves detailed run chronology,
  engineering turning points, historical verifier results, and rejected
  solutions.
- [`reproducibility.md`](reproducibility.md) documents installation,
  model-provider configuration, replay expectations, and publication metadata.
- [`results/`](results/) contains generated measurements and focused analysis
  summaries.
- [Analysis tools](../src/prefix_cache_evolve/tools/README.md) documents
  analysis, ablation, tuning, and report-generation commands.

Keep chronological additions and detailed failed-run analysis in the research
log. Update the technical report only when the current method, supported claims,
or limitations change.

Git history is intentionally preserved. To inspect code-oriented history without
documentation-only changes:

```bash
git log -- . ':(exclude)docs/**' ':(exclude)README.md'
```
