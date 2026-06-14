# Reproducibility and Model Providers

This document separates deterministic policy evaluation from LLM-guided search,
which depends on external model services and asynchronous scheduling.

## Installation

Use the committed `uv.lock` and Python 3.11:

```bash
# Evaluator, simulator, reports, and lab only.
make setup

# Evolution support through Levi.
make setup-evolution

# Tests, formatter, and type checker; no Git-hosted Levi dependency.
make setup-dev
make check
```

Equivalent commands are `uv sync --frozen --no-default-groups`, `uv sync
--frozen --no-default-groups --extra evolution`, and `uv sync --frozen --group
dev`. To run the complete suite including Levi adapter tests, combine the last
two as `uv sync --frozen --group dev --extra evolution`.

## Seeded Components

The main configuration has three distinct seed controls:

```yaml
search:
  seed: 20260609

problem:
  settings:
    verifier_version: "1.0.0"
    seeds: [11, 23, 37]
    policy_seed: 0
```

- `verifier_version` must match the verifier implemented by the checked-out
  source.
- `problem.settings.seeds` deterministically generates synthetic workloads.
- `policy_seed` is passed to candidate factories independently of workload
  generation.
- `search.seed` seeds Python and NumPy selection inside the Levi process and
  supplies monotonically derived request seeds to model providers that support
  seeded generation.

Inspect all resolved values without making a model request:

```bash
uv run prefix-cache-evolve --show-config
uv run prefix-cache-evolve --show-config --search-seed 17
```

Policy evaluation is deterministic for the same source, config, Python
environment, and deterministic policy. LLM search is not guaranteed to be
bit-for-bit reproducible: remote providers may ignore seeds or change serving
implementations, and concurrent workers can complete in a different order.
For the strongest practical repeatability, set `temperature: 0`,
`pipeline.n_llm_workers: 1`, and `evaluator.parallel_evaluations: 1` in a copied
configuration.

Saved evolution runs contain the configuration snapshot, workload manifest,
resolved model identifiers, search seed, package versions, Git commit and dirty
state, Levi snapshot, model cost, and candidate source. New score records and
snapshot history entries carry a three-part identity:

- `verifier_version`: the semantic verifier contract.
- `panel_sha256`: ordered request streams plus panel geometry.
- `evaluation_context_sha256`: verifier version, normalized evaluator config,
  and panel SHA-256.

Reports reject missing or mixed identities inside an exact comparison.
Intentional cross-context reports, such as geometry sweeps, record and validate
one identity per geometry. Historical unstamped runs remain legacy records and
cannot be tabulated with current versioned results.

## Incumbent Registry

Promoted policies are committed as immutable bundles under
`src/prefix_cache_evolve/problems/prefix_kv_cache/incumbents/`. Each bundle
contains the exact candidate `policy.py` and a `manifest.json` recording:

- source SHA-256, import target, and effective complexity;
- verifier, panel, evaluation-context, score, and headline metric pins;
- originating run, source artifact, recorded evaluations, and API cost;
- lineage and any distinction between original promotion accounting and the
  current replay.

`registry.json` preserves historical incumbents and assigns the current
production and retained discovery roles. Promoting a new policy means adding a
new bundle and changing the registry; do not overwrite an existing bundle.
Validate all stored identities with:

```bash
uv run prefix-cache-tools incumbents validate
uv run prefix-cache-tools incumbents list
```

CI runs the same validation and fails on unregistered bundles, source drift,
complexity drift, stale import targets, or mismatched source-artifact hashes.

## Weak-Seed Rediscovery

Deterministic replay of the incumbent is different from independently finding a
similar policy. The normal search remains incumbent-seeded because it is the
productive optimization lane. Rediscovery uses
`configs/prefix_kv_cache_rediscovery.yaml`, the candidate-valid
`seeds/weak_initial.py` seed, and a neutral prompt with no incumbent source,
score, coefficients, or mechanism-preservation instructions. Search ranks the
minimum of ordinary selection score and a non-quarantined agentic-workflow
guidance score. Final rediscovery adjudication always re-evaluates generated
source with the unchanged canonical `configs/prefix_kv_cache.yaml`; probe and
hidden panels remain unavailable during search.

Run at least three independent search seeds at the normal 300-evaluation budget:

```bash
uv run prefix-cache-evolve \
  --iterations 300 \
  --config configs/prefix_kv_cache_rediscovery.yaml \
  --seed-program src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/weak_initial.py \
  --search-seed 101 \
  --artifact-output artifacts/prefix_kv_cache_rediscovery_runs/weak_initial
```

Repeat with additional search seeds, then adjudicate every saved run together:

```bash
uv run prefix-cache-tools analyze rediscovery \
  --run artifacts/prefix_kv_cache_rediscovery_runs/weak_initial/<run-a> \
  --run artifacts/prefix_kv_cache_rediscovery_runs/weak_initial/<run-b> \
  --run artifacts/prefix_kv_cache_rediscovery_runs/weak_initial/<run-c>
```

The primary criterion is behavioral rather than source similarity. A generated
policy must be valid, remain at or below 650 effective AST nodes, pass the
agentic surrogate gate, and recover at least 80% of the charged
weak-seed-to-incumbent gap on selection, quarantined probe, and hidden panels.
The discoverability claim is supported only after at least two distinct search
seeds pass.

The current result is negative. Three staged 298-evaluation runs on June 14,
2026 used search seeds 211, 307, and 401, totaling 894 evaluations, `$21.5381`,
and 8,705 seconds. None passed the behavioral criterion. The ordinary-score run
overfit selection and failed probe plus the agentic gate. Adding the robust
guidance floor improved probe and passed the gate but selected a compact,
high-churn policy that failed selection and hidden transfer. After documenting
generic stateful primitives and enabling inspirations, the final corrected run
did not beat the weak seed; its strongest generated mutation was 708 nodes,
failed the agentic gate, and scored below the weak seed on every canonical
charged panel. Exact incumbent replay is reproducible, but independent
weak-seed rediscovery is not yet established.

## Bring Your Own Model

Models use [LiteLLM](https://docs.litellm.ai/) provider-qualified identifiers.
The simplest override uses one model for every search role:

```bash
uv run prefix-cache-evolve --show-config --model openai/<model-id>
uv run prefix-cache-evolve --show-config --model anthropic/<model-id>
uv run prefix-cache-evolve --show-config --model gemini/<model-id>
uv run prefix-cache-evolve --show-config --model ollama/<model-id>
```

Use separate mutation and paradigm-shift models when desired:

```bash
uv run prefix-cache-evolve \
  --primary-model openai/<mutation-model-id> \
  --secondary-model anthropic/<paradigm-model-id> \
  --iterations 100
```

Standard provider environment variables are:

| Provider | Model prefix | Credential |
|---|---|---|
| OpenAI | `openai/` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini/` | `GEMINI_API_KEY` |
| Ollama | `ollama/` | Usually none |

Never place API keys in YAML. For a self-hosted OpenAI-compatible endpoint:

```bash
export LOCAL_MODEL_API_KEY=local-no-key-required

uv run prefix-cache-evolve \
  --model openai/<served-model-name> \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key-env LOCAL_MODEL_API_KEY \
  --search-seed 17 \
  --iterations 100
```

The equivalent YAML is:

```yaml
llm:
  default_provider: openai
  primary_model: <served-model-name>
  secondary_model: <served-model-name>
  api_base: http://127.0.0.1:8000/v1
  api_key_env: LOCAL_MODEL_API_KEY
  temperature: 0.3
  max_tokens: 6000

search:
  seed: 17

punctuated_equilibrium:
  reasoning_effort: medium
  max_tokens: 12000
```

Set `punctuated_equilibrium.max_tokens` separately for reasoning-capable
paradigm models. The repository compatibility layer overrides Levi's historical
4,096-token paradigm-generation default only for the configured paradigm model.
Mutation calls keep their own pipeline budget, and Levi versions that natively
forward the configured paradigm budget bypass the compatibility override.

Run `--show-config` before a paid search. It validates the YAML and prints the
resolved model, endpoint, search seed, workload seeds, policy seed, capacities,
and worker settings without contacting the provider.

## Publishing a Result

Archive these files for each reported experiment:

1. Candidate source or saved run directory.
2. Exact YAML configuration.
3. `run_summary.json`, `metadata.json`, and `workload_manifest.json`.
4. The committed `uv.lock` and Git revision.
5. Any trace input or download script plus its SHA-256.
6. Provider model identifiers, date, search seed, evaluation budget, wall time,
   and reported API cost.

Report evaluation reproducibility separately from search reproducibility. A
candidate score can be exactly replayable even when the search trajectory that
found the candidate is not.
