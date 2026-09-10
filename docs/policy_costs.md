# Behavior, complexity, and policy execution costs

Use `configs/prefix_kv_cache_replay_exploration.yaml` for new mixed-traffic
searches. It selects raw behavior, starts from the validated 617-node
simplification of the newer four-domain seed, and
retains the 750-node exploration and provisional 650-node promotion ceilings.
The 0.065 AST coefficient remains for diagnostic charged scores. Historical
configs and results are unchanged; timing never enters the selection objective.

```bash
uv run prefix-cache-evolve \
  --config configs/prefix_kv_cache_replay_exploration.yaml --show-config
```

`search.seed_program` resolves relative to the YAML file; `--seed-program` takes
precedence. Saved trace runs copy the actual seed and update the snapshot path.
After search, `behavior_size_frontier/` preserves successful archive sources
not dominated in raw validation score and effective AST size, with identities
and source hashes. This covers the final Levi archive, not every past proposal;
missing legacy raw scores are not guessed.

For a dedicated simplification pass, freeze the exploration winner and use
`configs/prefix_kv_cache_replay_simplification.yaml` with
`--seed-program path/to/frozen_winner.py`. Its combined search score rewards
source reduction, while the post-search acceptance rule requires **identical
deterministic train/validation trial metrics except source complexity**, a
strictly smaller effective AST, and the configured promotion size limit.
Equal aggregate scores alone are insufficient. Generalization and existing
probe gates remain separate requirements.

Build the updated sandbox and obtain its immutable image ID:

```bash
docker build --tag prefix-cache-evolve-policy-costs \
  --file docker/sandbox/Dockerfile .
docker image inspect --format '{{.Id}}' prefix-cache-evolve-policy-costs
```

Substitute that `sha256:...` value below. The cost command runs each source and
baseline sequentially in the same networkless container profile. Three fresh
repeats are the default, with all visible streams and native capacities.

```bash
uv run prefix-cache-tools analyze policy-costs \
  --config configs/prefix_kv_cache_replay_exploration.yaml \
  --sandbox-image sha256:REPLACE_WITH_BUILT_IMAGE_ID \
  --reference path/to/frozen_winner.py \
  --candidate path/to/simplified_candidate.py \
  --baseline lru --baseline tinylfu_lru --baseline vllm_apc \
  --output-dir artifacts/policy-cost-comparison
```

The output directory must be new. JSON/Markdown reports separate raw behavior,
historical charged score, implementation AST, callback time, scan cost and
sampled state. Sources, configuration, repeats, panel/context identities, image
ID and Python/platform details are retained. Worker source hashes must match
the host, including baselines. Old images without the profiling protocol fail.
The current protocol is `prefix-kv-cache-policy-costs-v3`; rebuild older images.
`--reference` lists eligible simplifications under the exact-behavior rule.
No probe/hidden inputs are opened and no policy is promoted.

Implementation AST uses the existing uncredited counter for both groups:
candidate implementations and baseline classes, bases and transitively
referenced module helpers. Imports and thin factories are excluded; annotations
and implementation docstrings remain. Shared library code is excluded for both.
Credited effective size remains a separate search/promotion diagnostic; baseline
historical charges remain zero. No revised scalar baseline score is invented.

Callbacks report wall time and thread CPU count, total, mean, maximum and
approximate p50/p95/p99 upper bounds (about 1% histogram resolution). Scan cost
sums ranking callbacks and records scan width, excluding simulator scan
construction, sorting and bookkeeping. Python timer/call overhead and cold calls
remain included. Tracemalloc is disabled for profiling; container limits remain.

State samples occur at initialization, powers of two requests and the final
request, against unique prefixes, known sessions, resident blocks and capacity.
The reachable policy instance graph includes primitive delegates and counts
shared objects once. Code, classes, module state, profiler storage and simulator
KV are excluded. This is not RSS or a transient peak; native allocations and
class/module state require additional measurement for policies using them.
Instance-held callbacks include captured closure state, positional/keyword
defaults, and Python/built-in bound owners, without following a function's module
globals. Native GC references include container-subclass attributes and slots.
Native base size descriptors bypass candidate inspection hooks. This v3 estimate
excludes interpreter GC/preheaders and unmaterialized instance-dictionary
bookkeeping, so byte counts are not comparable to earlier protocol versions.
Callbacks are resolved on each invocation so runtime replacements behave as in
ordinary replay.

Declare `--callback-p99-us`, `--scan-p99-us` and `--state-budget-bytes` before
comparison. They check the worst observed trial/repeat and are recorded in the
report. Unspecified budgets remain **unassessed**; a state budget applies only
to sampled instance state. Passing observed budgets never establishes deployment
suitability. Measure on target hardware with representative capacity, request
length, concurrency and unique-session ranges.

Existing WildChat/LMCache samples remain small and existing hidden samples are
diagnostic. New generalization or promotion claims need a broader untouched
holdout. This workflow does not download datasets or start paid evolution itself.

The [first full visible measurement](results/policy_costs_20260908.md) records
436 trials and the unchanged-behavior simplification from 629 to 617 nodes.
