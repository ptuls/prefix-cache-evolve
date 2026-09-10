# Synthetic, Mooncake, WildChat, and LMCache Evolution

> [Replay audit update](replay_audit_20260907.md): this report preserves historical replay semantics. Missing-session handling and parts of the conversational replay were corrected afterward; compare policies again under the audited contract before drawing new conclusions.

Date: 2026-09-07

Status: **completed; agentic transfer improved, aggregate promotion rejected**

This experiment added the
[LMCache agentic traces](https://huggingface.co/datasets/sammshen/lmcache-agentic-traces)
to the existing synthetic, Mooncake, and WildChat selection suite. The source
revision was pinned at `6e043b9e89865df3aec19fd5679286b683bfd70e` and is licensed
CC BY 4.0. It contains collected agent trajectories with cumulative message
histories, output lengths, and tool gaps. It is more directly usable for cache
replay than a training corpus, but it is not production serving telemetry.

The converter wrote only opaque HMAC session, tenant, and block identifiers,
prompt and output lengths, request ordering, and derived timing. It wrote no
prompt text or original identifiers. A bounded panel sampled 16 complete
sessions and 329 requests spanning SWE-bench, GAIA, and WildClaw. The split was
fixed by session before outcome evaluation: 10 train sessions (133 requests),
3 validation sessions (99 requests), and 3 hidden sessions (97 requests).

Synthetic and WildChat traffic used 16-token blocks. Mooncake production and
LMCache collected-agent traffic used 512-token blocks. The search used
`gpt-5.6-luna` for ordinary mutations and `gpt-5.6-sol` for paradigm shifts.
The configured run allowed 48 evaluations, $17.50, and 12 hours. It completed
46 evaluator results in 4,537 seconds and spent **$0.9188** in model charges.
Including the prior work in this experiment chain, recorded charges are
**$3.1141**, below the approved $20 cap.

## Visible selection

The generated policy improved the four-domain seed from **25.854** to
**39.050**. It used 691 effective AST nodes, 41 above the promotion ceiling. A
post-search source audit factored duplicated block features and removed guards
around simulator fields that are nonnegative by construction. The resulting
629-node source reproduced every comparable visible behavior metric exactly;
only its complexity penalty changed, raising its score to **39.646**.

| Policy | Combined score | Validation block hit | Worst quarter | Request p10 | Churn / 1k | Waste | Underfill |
|---|---:|---:|---:|---:|---:|---:|---:|
| TinyLFU-LRU | **62.491** | 53.2% | 42.2% | 24.4% | **469.1** | 66.8% | 5.4% |
| LRU | 47.321 | 52.9% | 42.3% | 24.5% | 1,599.2 | 77.9% | **0.0%** |
| Simplified evolved policy | 39.646 | **54.31%** | **44.0%** | **24.8%** | 1,098.4 | 72.10% | 2.72% |
| Generated evolved policy | 39.050 | 54.31% | 44.0% | 24.8% | 1,098.4 | 72.10% | 2.72% |
| vLLM APC | 36.183 | 46.4% | 42.3% | **24.8%** | 924.6 | **49.9%** | 14.7% |

Per-domain hit rates are more useful for interpreting the mixed geometries.
The evolved policy does not beat the best deployable baselines. Its raw score
before complexity is 47.810, but it reaches the 25-point churn cap and pays
5.110 points in fairness cost.

| Visible trace | Four-domain seed | Evolved policy | LRU | vLLM APC | TinyLFU-LRU |
|---|---:|---:|---:|---:|---:|
| WildChat validation token hit | 4.0% | **50.0%** | **57.8%** | **57.8%** | 42.8% |
| LMCache train token hit | 13.8% | **80.5%** | 83.4% | 83.8% | 75.5% |
| LMCache validation token hit | 86.7% | **94.7%** | **94.7%** | **94.7%** | 91.0% |
| Mooncake validation 1 token hit | n/a | 36.1% | 35.7% | 35.9% | **36.6%** |
| Mooncake validation 2 token hit | n/a | 35.0% | 34.6% | 34.8% | **35.9%** |

The policy solved the seed's categorical under-admission on conversational and
concurrent agent sessions. Its broad admission path also increased churn and
admission waste, which kept the combined score well below TinyLFU-LRU.

## Mechanism

The evolved `CompactReusePolicy` keeps a two-timescale request-session counter
and a smoothed recent-pressure value. Admission combines block hits, current
session recurrence, access-gap evidence, recomputation value, fullness, and
prefix descendants. Recent pressure suppresses admission unless recurrence is
already visible. Eviction protects blocks with hits, descendants, subtree hit
rate, recomputation value, and access-gap evidence.

The same formula runs at both native block sizes and has no workload, dataset,
model, request-ID, or hash dispatch. Its main weakness is that session
recurrence opens admission broadly enough to retain useful agent histories but
does not sufficiently reject low-value Mooncake suffixes.

## Frozen hidden audit

The 629-node source and SHA-256 were frozen before the LMCache hidden split was
opened. The final panel contained ten shifted synthetic families, two later
Mooncake production windows, and three held-out LMCache sessions. WildChat is
validation-only because its earlier bounded sample has no untouched split.

| Policy | Hidden score | Token hit | Block hit | Churn / 1k | Waste | Underfill |
|---|---:|---:|---:|---:|---:|---:|
| TinyLFU-LRU | **11.705** | 44.93% | 40.80% | **804.5** | 61.21% | 5.29% |
| vLLM APC | -0.729 | 45.15% | 39.10% | 1,385.3 | **60.10%** | 7.04% |
| LRU | -2.637 | 44.01% | 40.37% | 2,201.2 | 78.23% | **0.0%** |
| Evolved policy | -7.974 | **46.39%** | **42.49%** | 1,277.3 | 68.42% | 11.88% |

The evolved policy has the highest aggregate hidden hit rates, yet trails
TinyLFU-LRU by **19.679 score points**. Its churn reaches the 25-point cap and
its 629-node source pays an 8.164-point complexity penalty.

| Hidden trace | Evolved policy | LRU | vLLM APC | TinyLFU-LRU |
|---|---:|---:|---:|---:|
| LMCache token hit | 93.93% | 94.02% | **94.40%** | 90.24% |
| LMCache churn / 1k | 450.2 | 721.6 | 226.8 | **209.6** |
| LMCache admission waste | 20.05% | 20.91% | **3.17%** | 7.33% |
| Mooncake hidden 1 token hit | 33.99% | 33.65% | 33.91% | **35.36%** |
| Mooncake hidden 2 token hit | 34.27% | 33.83% | 34.18% | **35.68%** |

The LMCache holdout verifies strong prefix continuity in the collected agent
sessions, but the evolved policy is not the best way to capture it. vLLM APC
gets a slightly higher LMCache hit rate with about half the candidate's churn
and much less waste. TinyLFU-LRU wins both Mooncake hidden hit rates and the
overall objective.

## Gate and decision

The agentic surrogate-to-probe gate remains flagged. The token-hit gap is
0.1219 against a 0.1200 threshold, and the admission-waste gap is 0.2986
against 0.2000. Five other checks pass. Together with the hidden loss, this
blocks promotion even though the simplified source now meets the complexity
ceiling.

Retain the policy as a future seed. A next experiment should target selective
agent-session admission using reuse evidence local to a prefix, while keeping
the LMCache continuity gain and restoring TinyLFU-like control of Mooncake
churn. The opened hidden panel must remain reporting-only and cannot guide
coefficient tuning of this source.

## Reproducibility

- Run bundle:
  `artifacts/mooncake-synthetic-wildchat-lmcache-evolution/full-run/20260907T031332Z`
- Generated winner SHA-256:
  `ea2e46980ada6b51d2c8fa436cb74853fac2c5e3a15fa6276e7383abd9573fec`
- Simplified candidate and tracked seed SHA-256:
  `fb42fd8dae5097fefd8d352fd0244442e9971dd5926aac49ac4ccbc36aa5aa35`
- Tracked seed:
  `src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/joint_mooncake_synthetic_wildchat_lmcache_20260907.py`
- Panel SHA-256:
  `6d38c40899eb5500ae61339f7d91c83eac04dd2d5f74775b771b351ff730e16a`
- Evaluation context SHA-256:
  `1fcac0b85b8b001746195a5206d9733d8ae80a89e31d158eda22f50cc1be5d66`
- LMCache converted trace SHA-256:
  `a1d4847b5d6dc099f560ffa32ee17cbb3c1e69dcc0c1b4f7a4622b214eb7ec84`
- Pinned sandbox image:
  `sha256:8e6ce5586d4b6dd68f5bad1cd70dd83dc05c9f444ac930eec173e297898621b3`

The data panel is too small for a population claim. Tokenization and prompt
serialization approximate provider templates, and cross-session starts use a
deterministic closed-loop scheduler because the source lacks absolute serving
timestamps.
