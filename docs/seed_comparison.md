# Controlled, repeated seed comparison

Status: protocol and preparation tooling, **not experimental results**. No paid
search is launched by the preparation command. The historical scratch and
continuation pilots do not estimate the causal effect of starting source: their
prompts, randomness and initialization failures differed.

## Question and unit of replication

Under a frozen search protocol and visible evaluator, does starting from the
frozen 587-node Qwen continuation policy produce better final policies than
starting from the 48-node minimal LRU policy? This estimates the effect of these
two source programs, including their embedded prior knowledge. It does not
separate architecture, source length and coefficients, or establish a general
law about seed quality. Prior work used to obtain the strong seed is reported
separately from the marginal search cost in this experiment.

One complete search is a replicate. Pair the two starting conditions by search
RNG seed, use a different RNG seed in every pair, and randomize execution order
within each pair. Workload seeds, capacities, windows and individual requests
contribute to each search's evaluation; they are not independent repetitions
of search. Each run starts with a new archive and no shared advice or candidates.

Start with five pairs (ten searches) as an estimation pilot. Show all five paired
differences and their mean/median. Five pairs provide limited precision: even
five strictly positive differences have a minimum two-sided exact sign-test
p-value of 0.0625. Do not promise a statistically significant result or choose
the repetition count after seeing whether it crosses a threshold. A larger
confirmatory campaign should be preregistered separately using the pilot's
variance and a practically meaningful effect size.

## Fixed conditions

The preparation tool copies the supplied evaluator's visible workload settings
and resolved model choices, and creates a common neutral task description. It
replaces historical seed-specific advice, objectives and search notes. Both arms
receive the same prompt templates, model configuration, API routing, source
limits and workload inputs. Parent source and observed feedback necessarily
differ; matching templates does not mean identical complete model requests.

The initial protocol uses:

- 64 total evaluator attempts per search, including the supplied seed,
  initialization and failed candidate evaluations. Invalid candidates consume
  budget; successful evaluations are not topped up to a quota.
- Two diverse initialization proposals, up to Levi's three attempts per proposal,
  and one variant per valid initial source. Failures and missing extracted code
  are recorded, including model calls that never reach the evaluator.
- Eight fixed CVT cells, initialized from the paired RNG seed, rather than cells
  fitted to seed-dependent initialization data. One generation worker and one
  evaluation worker reduce ordering variation; remote generation remains
  nondeterministic even with provider-supported request seeds.
- No periodic paradigm shifts or meta-advice. Initialization still generates
  alternative algorithms. This removes the existing two-elite paradigm gate
  and asynchronous paradigm accounting from this first experiment. Conclusions
  apply to this protocol; comparison of the richer evolver is a later experiment.
- `raw_before_complexity` search scoring, 750 effective AST nodes for evaluation,
  and 650 for final selection. No post-selection simplification or manual repair.
- Equal dollar and wall-clock guardrails per run. These are backstops, not the
  primary comparison axis. Their configured totals are not a cost forecast or
  a strict billing cap: an in-flight model call can exceed an accounting limit.

Levi's evaluation-budget shutdown now drains already-reserved evaluations before
cancelling workers. Cost/time limits and task cancellation remain effective. The normal
evolution entry point holds a macOS idle-sleep assertion. This does not protect
against machine shutdown or forced sleep.

## Before the measured campaign

1. Run the offline initialization and budget regression checks. Verify that
   sparse/duplicate initialization retains archive capacity and can accept later
   distinct behavior. This checks software mechanics, not the validity rate of
   actual model proposals.
2. Replay both exact starting sources on the full frozen visible panel under the
   same candidate sandbox. Require deployability and consistent score identities.
   Establish their raw scores and the LRU/TinyLFU controls before any search.
3. Run an **excluded paired initialization pilot**, with different search seeds
   and the same templates/models/settings. Require a valid generated alternative
   and a completed mutation evaluation in each arm; inspect archive occupancy,
   callback/size failures and budget accounting. A single occupied cell does not
   by itself mean broken initialization. If the gate fails, diagnose it and freeze
   a new protocol before beginning the measured campaign. Never repair just one arm.
4. Preregister a fresh final holdout and its exact input hashes, capacities and
   aggregation rules. Historical scored Qwen windows are consumed, not fresh.
   Do not redraw AgentX because a previous sample was slow or unfavorable.
5. Verify the execution environment and sandbox image, freeze source/dependency
   identities, and set the campaign's approved model-spend limit. Preparation
   requires no model credentials; launching paid searches is a separate action.

The source/configuration hashes and visible trace file hashes are frozen locally.
Search configs omit hidden and probe workloads completely. Final holdout
preregistration is deliberately separate from search configuration. The preparer
does not invent a fresh holdout or label an already-used panel as unseen.

## Selection and analysis

For every run, freeze the highest visible raw-score valid archived policy with
at most 650 effective AST nodes, including the seed as a fallback. Consider the
final and retained periodic snapshots. Break ties by smaller effective AST,
then lexicographic SHA-256. Preserve the exact bytes, archive and selection record
before any holdout evaluation. Do not select on hidden results or manually edit
the winner. Report smaller alternatives separately.

The primary search outcome is final selected visible raw score at the fixed
attempt budget. Report `strong - minimal` within every pair. Also report gain
from each arm's own starting score: final score alone measures the advantage of
the supplied prior knowledge as well as subsequent improvement. Include per-run
score-versus-attempt curves, valid/failed counts, initialization outcomes, archive
occupancy, source size, model cost and active/wall runtime. A plateau or absence
of rediscovery is bounded by this budget; it is not evidence of impossibility.

Report every planned run, including initialization failure and early termination.
Do not silently discard failed searches, replace them until they succeed, or treat
an early stop as a completed 64-attempt run. A censored pair is incomplete for the
fixed-budget comparison. Publish partial results and the reason; any follow-up
campaign must preserve the original record and use a separately frozen protocol.

Evaluate **all** frozen winners plus unchanged controls on one common fresh
holdout after the whole campaign's selection is frozen. That is a secondary
generalization assessment, with the same search-level pairing. Holdout workload
variation is nested within search repetition; do not inflate the sample size by
pooling individual trial scores as independent searches. Report behavior, source
size and separately instrumented CPU/state measurements with consistent accounting.

## Commands

Use an existing prepared panel config to carry the trace pins and geometry:

```bash
prefix-cache-tools experiments seed-comparison prepare \
  --config PATH_TO_PREPARED_PANEL/evolution.yaml \
  --output artifacts/seed-comparison-v1 \
  --evaluations 64 --dollars-per-run 2 --seconds-per-run 14400

prefix-cache-tools experiments seed-comparison validate artifacts/seed-comparison-v1
```

The default independent search seeds are 104729, 130363, 155921, 181081 and
206369. Repeat `--search-seed` to preregister another set. `--minimal` and
`--strong` accept explicit source files. Existing output directories are never
overwritten. Changing implementation, configuration, seed or trace bytes requires
a new campaign directory; validation fails on drift.

After the launch gates pass, run each entry in `campaign.json` order through
the existing evolution command, in separate processes with `PYTHONHASHSEED=0`,
no `LEVI_MODEL` override, and the generated `evolution.yaml`. Pass
`--iterations 64` explicitly: the existing CLI's iteration default is separate
from YAML. Use a unique artifact directory per run. The preparation command
does not launch or supervise these runs and does not produce a statistical report.

## Implementation validation (2026-09-11)

The full repository suite passes 595 tests, including controls for paired
configuration equality, frozen-input drift, independent search seeds, fixed
centroid geometry and evaluation-budget draining. Ruff formatting/checks and
mypy also pass. All 35 tracked benchmark policy/seed/incumbent bundle files
remain byte-identical to HEAD.

A separate offline smoke run exercises the real pinned Levi pipeline with a
fixture text-generation client: six valid initialization programs, eight archive
cells (two occupied), two subsequent mutation evaluations, and exactly eight
completed evaluations at the eight-attempt limit. It makes zero remote model
calls. This validates execution mechanics; it does not satisfy the real-model
initialization pilot or provide evidence about cache-policy seed dependence.
