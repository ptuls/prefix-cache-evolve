# WildChat evolution pipeline smoke check — September 5, 2026

Real conversation-derived requests now run through the normal baseline report
and the isolated evaluator used by evolution. Both paths returned the same score
and evaluation identity for the unchanged production incumbent. All 14 policies
in the baseline comparison evaluated successfully. This check did not run a
model-powered policy search or evaluate the hidden panel.

The [machine-readable record](wildchat_smoke_20260905.json) retains the source
revision, conversion and partition manifests, full evaluator settings, policy
source hash, score identity, and summaries for all 14 policies. The implementation
was an uncommitted continuation of branch `feat/real-data` at the recorded base
commit. Conversation text, trace records, and the private HMAC key are not in
that record.

## Input and evaluation

The source is Allen AI's
[WildChat-1M dataset](https://huggingface.co/datasets/allenai/WildChat-1M/blob/7d6490e462285cf85d91eabea0f9a954fbddcd1f/README.md),
pinned to `7d6490e462285cf85d91eabea0f9a954fbddcd1f`, training split, ODC-BY.
The first 20 eligible conversations with at least two assistant responses were
converted: 51 rows scanned, 31 single-turn rows excluded, 65 requests, 20 sessions,
and 18 tenants. All 65 requests had recorded assistant response timestamps.

Conversion used `cl100k_base`, 16-token blocks, canonical cumulative conversation
prompts, and HMAC identifiers. WildChat timestamps mark response completion;
their use here does not recover real request arrivals or serving latency.

The panel used tenant grouping, split seed 0, and requested 60%/20%/20% fractions.
Whole-group allocation produced:

| Split | Requests | Sessions | Tenants | Used for |
|---|---:|---:|---:|---|
| Train | 35 | 12 | 12 | Search feedback |
| Validation | 11 | 3 | 3 | Selection score |
| Hidden | 19 | 5 | 3 | Reserved; not scored |

Cache capacity was 64 and 128 blocks, equivalent to 1,024 and 2,048 tokens.
Arrival buckets were 100 ms. The base configuration was
`configs/prefix_kv_cache.yaml`, with `capacity_blocks: 64`,
`capacity_sweep_blocks: [64, 128]`, `request_count: 12`, and `seeds: [11]`.
Panel preparation replaced synthetic selection and hidden families; it retained
the two synthetic probes, each with 12 requests. Probe scores were excluded from
selection. Each real trace ran once per capacity, independently of synthetic
seeds.

## Observations

The table shows the charged selection score and validation metrics averaged
equally across the two capacities. `vllm_apc` is the repository's simulated APC
baseline, not a measurement of a running vLLM server.

| Policy | Selection score | Validation token hit rate | Churn per 1k requests |
|---|---:|---:|---:|
| Production incumbent | -65.530 | 3.97% | 0.0 |
| vLLM-style APC | 52.030 | 57.77% | 1,818.2 |
| LRU | 51.717 | 57.77% | 2,454.5 |
| TinyLFU-LRU | 37.975 | 42.85% | 545.5 |
| No cache | -64.500 | 0.00% | 0.0 |

The incumbent had a validation underfill rate of 87.75% and bypassed 94.28% of
prompt tokens. Its low churn accompanied low reuse in this sample. These
diagnostics make admission behavior a useful next target for a larger real-data
experiment. They do not establish a general WildChat ranking: validation has
only 11 requests from three tenants, and the sample deliberately excludes
single-turn conversations. No policy was promoted.

Future-aware reporting policies are identified separately in the JSON record.
Their scores are diagnostic bounds and are excluded from deployable-policy
claims. These scores also have a different evaluation context from the
repository's synthetic headline results.

## Reproduction and artifacts

Follow the [trace evolution guide](../reproducibility.md#trace-panels-for-evolution).
For this sample, convert with the pinned revision above,
`--conversation-limit 20`, and `--minimum-requests-per-conversation 2`; prepare
the panel using the base-setting overrides above and default partition options.
The complete evaluator settings are in the JSON record. Reusing the original
private HMAC key is necessary for identical hashes **and split memberships**.

The local conversion and panel are retained under
`artifacts/traces/wildchat-smoke*`. The baseline report, complete evaluator
output, and policy copy are under `artifacts/wildchat-smoke/`. The hidden trace
remains in its original panel for an explicit future final evaluation.

The selection panel SHA-256 is
`f20ef11d1bfff7e58139725cb20a001c56c8b48309c513d2a8059dfa465d7351`;
the evaluation context is
`bef3b968b8190f9db763e81b5dbaaef2cd43a710d248d97a778129f21448b9c1`,
under verifier `1.0.0`.
