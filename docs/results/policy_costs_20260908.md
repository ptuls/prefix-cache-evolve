# Complexity recommendations: implementation and first measurements

Date: 2026-09-08

The [corrected v3 rerun](#corrected-v3-rerun) below supersedes the initial cost
measurements. Historical v1 data remains recorded for reproducibility.

The new workflow separates raw behavior, source size, and measured callback/state
cost. `prefix_kv_cache_replay_exploration.yaml` uses raw-score selection and the
new 617-node seed; `prefix_kv_cache_replay_simplification.yaml` prepares a dedicated
size-reduction stage. Exploration/promotion ceilings remain 750/650 nodes and the
historical AST coefficient remains 0.065. Historical configs and policy bundles
are unchanged. No evolution model calls or promotions were made.

## A completed simplification

The new `seeds/joint_four_domain_simplified_20260908.py` removes two constructor
assignments from the frozen four-domain policy. `_session_reuse` and `_priority`
are assigned on every request start before any admission/eviction scoring, so
their constructor values are unused in the simulator lifecycle. No thresholds,
formulas, arithmetic order, or state-update behavior changed.

Both sources were replayed in the same pinned container on **all 109 configured
train/validation trials**. Every deterministic trial metric matched exactly
apart from source complexity. The simplified policy meets the predeclared
visible-behavior equivalence rule and the size limit. Effective AST count falls
from **629 to 617**, and uncredited implementation count from 640 to 628.
The exploration config now defaults to that separately retained source.
This is a seed, not a promoted incumbent; existing generalization/probe limitations
remain. No probe or hidden input was opened for this check.

## First cost measurements

Four policies each completed one instrumented repeat: **436 successful trials**.
Synthetic workloads use their configured seeds and capacities, and every visible
trace is replayed at its native capacities. These 109 trials exclude the 12
reporting-only probe trials present in the earlier 121-trial reassessment.

| Policy | Raw behavior | Historical charged score | Implementation AST | Worst callback p99 upper (us) | Worst scan p99 upper (us) | Sampled instance bytes max |
|---|---:|---:|---:|---:|---:|---:|
| Simplified four-domain | 62.844 | 54.797 | 628 | 816.415 | 292.777 | 18,641 |
| Frozen four-domain | 62.844 | 54.680 | 640 | 816.374 | 313.662 | 18,641 |
| TinyLFU-LRU | 61.060 | 61.060 | 227 | 5.334 | 32.797 | 547,571 |
| LRU | 46.950 | 46.950 | 82 | 5.292 | 34.333 | 352 |

The measurements expose a tradeoff that AST size cannot capture: the evolved
policy retains less instance state than TinyLFU on these streams, but has much
slower worst-trial callbacks. The 12-node simplification establishes a source-size
reduction, not a measured runtime improvement. All operating budgets remain
**unassessed** because none were supplied. One instrumented repeat on this local
Docker runtime is not a deployment qualification or a stable timing comparison.

The ordinary deterministic evaluator and profiler matched every trial metric
for all 12 deployable baselines and the frozen candidate in focused tests.
The full 109-trial results for the frozen candidate, TinyLFU and LRU also match
the preceding corrected reassessment exactly, excluding the changed panel identity.
Profiling skips tracemalloc, retains Python timer/call overhead, and records cold
calls. p99 values are approximate histogram upper bounds; the table takes the
worst observed trial. Scan times sum ranking callbacks and exclude simulator
bookkeeping. State samples count the reachable instance graph, including primitive
delegates, at powers of two requests and the final request; they exclude code,
class/module state, profiler storage and simulator KV. They are not RSS or peaks.

## Reproduction and artifacts

The [workflow guide](../policy_costs.md) explains budgets, source accounting and
the two search stages. The [machine-readable result](policy_costs_20260908.json)
pins sources, image, identities, scores and measurements. Complete profiles,
individual trials, source copies and config are in
`artifacts/policy-costs-20260908/full-visible/`.

The measured image is
`sha256:3dc2bb5197f501ff7668c384e4864267aee1c2b3ed2b3766cecda57b1ad4c68f`.
These are historical v1 profiler measurements. The structured review later
identified callback-replacement and callback-owned-state defects for other valid
policy shapes; the four measured sources use neither affected pattern. Current
tools require the corrected v3 profiler and reject this old image.
Reproduce the historical measurement using the v1 tools preserved in that image,
with `prefix-cache-tools analyze policy-costs`, the exploration config,
that image, `--reference` pointing to the original four-domain seed, `--candidate`
pointing to the new simplified seed, `--baseline tinylfu_lru --baseline lru`,
`--repeats 1`, and a new `--output-dir`. The normal command default is three repeats.

Validation: 525 pytest tests passed with 85.89% coverage; Ruff formatting/lint,
Mypy (39 source files), and all four incumbent-bundle validations passed.
All pre-existing incumbent and seed files remained byte-for-byte unchanged.

## Corrected v3 rerun

Structured review identified five profiling defects for valid policy shapes:
callbacks replaced at runtime, captured/default callback state, custom size hooks
that mutate policy behavior, built-in callback owners, and container-subclass
attributes/slots. The profiler now resolves callbacks per invocation and walks
native object references without invoking Python inspection hooks. Regression
tests confirm ordinary/profiled behavior equality and state-budget detection.

The corrected protocol is `prefix-kv-cache-policy-costs-v3`. It uses native base
size descriptors and excludes interpreter GC/preheaders and unmaterialized
instance-dictionary bookkeeping. Its byte estimates are not comparable to v1.
The other sampling and measurement limitations above still apply.

All four policies completed a fresh instrumented repeat: **436 successful
trials**. Every deterministic trial field matches the historical v1 results
apart from panel identity. The simplified and frozen sources still match exactly
on all 109 visible trials, with effective AST size **617 versus 629**.

| Policy | Raw behavior | Historical charged score | Implementation AST | Worst callback p99 upper (us) | Worst scan p99 upper (us) | Sampled native state bytes max |
|---|---:|---:|---:|---:|---:|---:|
| Simplified four-domain | 62.844 | 54.797 | 628 | 76.542 | 298.662 | 14,988 |
| Frozen four-domain | 62.844 | 54.680 | 640 | 84.709 | 434.955 | 14,988 |
| TinyLFU-LRU | 61.060 | 61.060 | 227 | 5.209 | 30.590 | 547,384 |
| LRU | 46.950 | 46.950 | 82 | 10.291 | 24.822 | 24 |

These remain single-repeat instrumented timings with **unassessed** operating
budgets. The different timing values do not establish a speedup across profiler
versions or between the two policy sources.

The [v3 machine-readable result](policy_costs_20260908_v3.json) records the runtime,
source hashes, image, scores and checks. Full profiles and per-trial results are
in `artifacts/policy-costs-20260908/full-visible-v3/`. Reproduce with current tools:

```sh
prefix-cache-tools analyze policy-costs \
  --config configs/prefix_kv_cache_replay_exploration.yaml \
  --sandbox-image sha256:55d1c0455c9fe355353bfdcd03a654da1666f69a5180db993ff1da3a15e9e383 \
  --reference src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/joint_mooncake_synthetic_wildchat_lmcache_20260907.py \
  --candidate src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/joint_four_domain_simplified_20260908.py \
  --baseline tinylfu_lru --baseline lru --repeats 1 \
  --output-dir artifacts/policy-costs-20260908/new-v3-repeat
```

Validation: **535 tests passed**, 85.91% coverage, Ruff formatting/lint and Mypy
(39 source files) passed. All four incumbent bundles validate; all 30 pre-existing
seed/incumbent files remain byte-for-byte unchanged.

Review also corrected saved synthetic configurations to reference their archived
seed, including overrides. Relocation tests remove the original config/seed and
verify the saved configuration still resolves the correct source. This artifact
saving fix does not change the v3 profiling runtime or measurements.
