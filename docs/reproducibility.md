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

# Tests, formatter, and evolution support.
make setup-dev
make check
```

Equivalent commands are `uv sync --frozen --no-default-groups`, `uv sync
--frozen --no-default-groups --extra evolution`, and `uv sync --frozen --group
dev`.

## Seeded Components

The main configuration has three distinct seed controls:

```yaml
search:
  seed: 20260609

problem:
  settings:
    seeds: [11, 23, 37]
    policy_seed: 0
```

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
state, Levi snapshot, model cost, and candidate source.

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
```

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
