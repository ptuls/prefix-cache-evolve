# Complexity penalty review

Date: 2026-09-08

The AST penalty still provides useful pressure against code growth. Its
current asymmetric use is a poor standalone criterion for comparing serving
policies: evolved sources pay for their implementation while registered
baselines pay zero. Treat cache behavior, source size, and measured execution
cost as separate dimensions when deciding what to deploy.

This review inspects exact source and arithmetically rescores the frozen
[15-policy reassessment](replay_reassessment_20260908.md). No new replay,
evolution, or model call was performed. The operative configuration and all
policy sources remain unchanged.

## Formula and active controls

The configured objective subtracts

\[
P(C) = 0.065\,C^{0.75},
\]

where `C` is effective Python AST node count. The charge is applied once to
the combined score, not multiplied by requests, capacities, or workloads.
It is smooth, unbounded, and concave: the marginal cost of another node
decreases as source size grows.

The active audited configuration also has two separate limits:

- Above **750 effective nodes**, a candidate is rejected before evaluation.
- Above **650**, an evaluated candidate is marked exploration-only and is
  directed toward a simplification pass before promotion eligibility.

The newer policy has 629 nodes, leaving 21 nodes below the promotion limit
and 121 below the evaluation limit. Passing the AST limit does not pass the
other promotion or generalization checks.

The joint configuration currently selects with `search_score_mode: combined`.
The repository also supports `raw_before_complexity`; the eviction-specialist
configuration already uses that mode while retaining complexity reporting and
separate size limits. Statements in the technical report about raw-score
search should not be read as the active setting for the joint configuration.

## What the newer policy is charged for

Its source has **640 raw nodes**, receives **11 primitive credits**, and is
charged for **629 effective nodes**:

| Source component | Raw nodes | Credits | Effective contribution |
| --- | ---: | ---: | ---: |
| Admission formula | 232 | 2 | 230 |
| Eviction formula | 154 | 0 | 154 |
| Request-start pressure and session update | 150 | 6 | 144 |
| Constructor and state initialization | 51 | 3 | 48 |
| Shared block-feature helper | 35 | 0 | 35 |
| No-op cache callback, aliases, and class shell | 18 | 0 | 18 |
| **Total** | **640** | **11** | **629** |

The three-node credits are for the `MultiTimescaleDecay` constructor and its
`values` and `observe` call sites. Two `threshold_excess` call sites receive
one node each. Credits are capped at 25% of raw nodes, or 160 here, so the cap
does not bind. Shared primitive implementation code is not included in the
candidate's source count.

The counter includes arithmetic operators, constants, calls, variable and
attribute references, `Load`/`Store` contexts, function arguments, and control
flow. The source contains three `if` statements and 34 call sites, not 629
branches. There are 183 `Load`/`Store` nodes alone. Imports, the module
docstring, literal `__all__`, the top-level `candidate_factory = build_candidate`
alias, and thin recognized factory wrappers are excluded. Class and function
bodies, including their annotations and docstrings, are otherwise counted;
the newer policy's two class-level callback aliases are charged. Comments and
formatting do not affect AST count.

These component counts add to `C`; their individually exponentiated penalties
would not add to the actual charge. The exponent applies to the total.

## How strong is the charge?

| Effective nodes | Penalty in objective points |
| --- | ---: |
| 100 | 2.055 |
| 300 | 4.685 |
| 500 | 6.873 |
| Production incumbent: 572 | 7.603 |
| Previous joint: 613 | 8.008 |
| Newer four-domain: 629 | 8.164 |
| Promotion limit: 650 | 8.368 |
| Evaluation limit: 750 | 9.316 |

There are two different comparisons:

1. **Within the evolved family**, the newer policy pays only **0.156 more**
   than the previous 613-node policy. Removing 100 nodes from the newer source
   while preserving behavior would recover approximately **0.994 points**.
   This is moderate pressure to justify extra machinery.
2. **Against an uncharged baseline**, it pays the full **8.164 points**. Its
   raw score of 62.844 becomes 54.680, while TinyLFU remains at 61.060.
   Holding behavior fixed, source reduction alone would need to reach at most
   **82 effective nodes** to beat that baseline at the current coefficient.
   That is a substantial redesign, not routine cleanup.

This charge is in abstract objective points. It is not dollars, milliseconds,
memory consumption, or a measured maintenance cost. The documented rationale
is compactness, discouraging inert code, and breaking behavioral near-ties;
the examined documentation does not calibrate `0.065` to a serving-resource
budget.

## Sensitivity of the current ranking

All rows reuse exactly the same measured behavior. TinyLFU remains uncharged
in this table, matching the existing convention.

| Coefficient | Newer policy penalty | Newer score | TinyLFU score | Winner |
| --- | ---: | ---: | ---: | --- |
| 0.065, current | 8.164 | 54.680 | 61.060 | TinyLFU |
| 0.0325, half | 4.082 | 58.762 | 61.060 | TinyLFU |
| 0.020 | 2.512 | 60.332 | 61.060 | TinyLFU |
| 0.015 | 1.884 | 60.960 | 61.060 | TinyLFU |
| 0.010 | 1.256 | 61.588 | 61.060 | Newer policy |
| 0 | 0 | 62.844 | 61.060 | Newer policy |

The crossover coefficient is approximately **0.014203**. This is a sensitivity
result, not a reason to pick a coefficient that makes a preferred policy win.
These rescoring results do not predict what evolution would discover under a
different coefficient.

For an illustrative consistent-source comparison, counting TinyLFU's exact
policy class plus `_BasePolicy` yields **227 nodes**, a **3.801-point** charge,
and a score of **57.258**. It would still beat the newer policy's 54.680 by
2.579 points. LRU's same class-plus-base convention gives 82 nodes and a
1.771-point charge. These are not revised official baseline scores: inherited
methods, annotations, and packaging affect the counts. A production comparison
would need a declared source-accounting convention shared by all policies.

## Does it serve the current goal?

**For controlling search bloat: yes, with limits.** The history documents
useful simplification passes and rejected large implementations. Retain
bounded exploration, source validation, and a simplicity preference. AST size
is one way to express that preference, not a direct estimate of generalization.

**For deployment cost or a headline baseline ranking: insufficient by itself.**
Several concrete mismatches matter:

- Equivalent helper extraction, annotations, or expression structure can change
  AST size without changing cache behavior. Reusing provided primitives omits
  their implementation from the source charge.
- A compact state update can have growing memory cost. The registered TinyLFU
  baseline's frequency dictionary has no eviction bound in its implementation;
  the newer policy's session primitive is bounded to 1,024 keys by default.
  A smaller AST does not establish a smaller retained state.
- The latency proxy charges uncached recomputation, block lookup, and eviction
  counts. It does not time policy callbacks. Existing timeout and memory limits
  are coarse evaluator safeguards, not per-callback serving budgets.
- The separate 650-node promotion limit already constrains source size. The
  additional smooth penalty should therefore express a deliberate marginal
  preference, rather than stand in for an unmeasured deployment requirement.

## Recommendation

1. **Separate serving comparisons from search regularization.** Publish raw
   behavioral score, source complexity, and eventual measured CPU/state costs
   together. Use a consistent cost convention for baselines and candidates.
2. **For the next exploratory run, use the existing raw-score mode within the
   current exploration limit**, retain smaller alternatives, and perform a
   dedicated simplification pass. Prefer the simpler policy when a predeclared
   validation/generalization criterion treats their behavior as equivalent.
   The current 650-node promotion limit can remain provisional during this
   investigation; changing search mode alone does not remove it.
3. **Measure the deployment constraint directly:** callback time distributions
   on representative requests and eviction scans, retained state versus unique
   prefixes/sessions, and scaling with capacity. Set operating budgets before
   deciding whether a larger source is acceptable. If a scalar AST term remains
   useful, calibrate it to an explicit simplicity tradeoff rather than this
   observed winner's crossover point.

No coefficient or mode was changed by this review. The newer policy's existing
probe weaknesses and need for a fresh, broader holdout remain independent of
complexity accounting.

The [machine-readable analysis](complexity_review_20260908.json) records exact
source hashes, per-component counts, all 15 policies' coefficient sensitivities,
and the illustrative baseline counts. The reproduction script and complete
analysis are in `artifacts/complexity-review-20260908/`.
