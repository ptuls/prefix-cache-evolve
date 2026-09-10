# Full Mooncake Agentic Evolution With GPT-5.6

> [Replay audit update](replay_audit_20260907.md): this report preserves historical replay semantics. Missing-session handling and parts of the conversational replay were corrected afterward; compare policies again under the audited contract before drawing new conclusions.

Date: 2026-09-06

Status: **completed; production holdout improvement, no general incumbent promotion**

This run evolved the existing prefix-cache policy on the frozen chronological
Mooncake tool-and-agent panel. It used `openai/gpt-5.6-luna` for ordinary
mutations and `openai/gpt-5.6-sol` at medium reasoning effort for diversity and
punctuated-equilibrium paradigm shifts. Raw trace records and credentials were
not sent to OpenAI; model requests contained policy source, mutation
instructions, and aggregate visible-panel feedback.

The selected `SegmentedGhostPolicy` improved the untouched production holdout
score from **34.900** for the incumbent and **50.491** for TinyLFU-LRU to
**61.094**. Its token hit rate was nearly unchanged from the incumbent, while
churn fell by 91.5% and admission waste fell by 11.2 percentage points. The
policy regressed on the original 16-token synthetic suite, so it was not
promoted as the general incumbent.

## Search

| Item | Value |
|---|---:|
| Configured evaluation budget | 32 |
| Completed evaluator results | 30 |
| Configured dollar cap | $20.00 |
| Accounted model cost | $0.4005 |
| Configured wall cap | 12 hours |
| Search runtime | 1,717.2 seconds |
| Archive size | 3 |
| Candidate source complexity | 417 |
| Candidate complexity cost | 5.998 |

Levi stopped when 30 completed results plus two worker reservations reached the
32-evaluation budget. The search therefore exhausted its configured concurrent
evaluation allowance without reaching the dollar or wall limit. Five generated
candidates failed source or policy-contract validation; those failures remained
isolated and did not stop the run.

The visible validation score improved from **40.465** for the incumbent to
**63.158** for the selected candidate. TinyLFU-LRU remained slightly higher on
visible validation at **63.709**. Selection was fixed before opening the hidden
windows.

| Visible policy | Charged score | Token hit | Churn / 1k | Waste | Underfill |
|---|---:|---:|---:|---:|---:|
| TinyLFU-LRU | 63.709 | 36.3% | 273.4 | 89.3% | 32.4% |
| Selected candidate | 63.158 | 36.4% | 101.6 | 88.1% | 19.8% |
| Incumbent | 40.465 | 35.0% | 1,146.0 | 96.0% | 16.2% |
| vLLM-style APC | 39.957 | 35.4% | 7,080.8 | 98.7% | 0.2% |
| LRU | 39.478 | 35.2% | 7,809.6 | 98.2% | 0.0% |

## Untouched production holdout

All policies share verifier `1.0.0`, evaluation context
`ba9904881dab4e83e55eccde7efa293f2ffb06e20ee10b72274b6fa7392cc349`,
and panel `0b85e659fe170ef8783239214410b539b9995b28b74e70be9cf32111f9beb8a2`.

| Policy | Charged | Before complexity | Token hit | Churn / 1k | Waste | Underfill | Admission utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| Selected candidate | **61.094** | 67.092 | **35.69%** | **222.7** | **84.73%** | 9.61% | **11.802** |
| TinyLFU-LRU | 50.491 | 50.491 | 35.52% | 1,031.9 | 89.59% | 16.97% | 3.452 |
| vLLM-style APC | 37.256 | 37.256 | 34.04% | 7,815.3 | 98.80% | 0.08% | 0.735 |
| LRU | 36.380 | 36.380 | 33.74% | 8,541.9 | 98.21% | 0.00% | 0.680 |
| Incumbent | 34.900 | 42.502 | 35.67% | 2,621.0 | 95.90% | **2.92%** | 1.790 |

The candidate beats TinyLFU-LRU by 10.603 charged points while retaining a
slightly higher token hit rate. Its main tradeoff against the incumbent is more
underfill from selective admission, but it converts admitted blocks into much
more reuse and avoids most churn.

## Synthetic transfer

| Panel | Candidate | Incumbent | Delta | Candidate token hit | Incumbent token hit | Candidate churn / 1k | Incumbent churn / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Visible | 61.368 | 65.649 | -4.281 | 59.41% | 59.64% | 588.6 | 163.9 |
| Hidden | -6.078 | 3.064 | -9.142 | 45.99% | 45.52% | 898.7 | 194.4 |

The candidate retains or slightly improves raw hit behavior on synthetic hidden
traffic, but higher churn and admission waste reduce its score. Its
`on_request_start` method divides prompt length by the literal production block
size `512` instead of the supplied `block_size_tokens`; this explains a material
part of the 16-token transfer risk and prevents a general promotion claim.

### Synthetic block-size sweep

The full validation suite was replayed at 8, 16, 32, 64, and 128 tokens per
block. Every point uses all three seeds and preserves the same 384/768-token
cache-capacity tiers, so the number of cache blocks changes with block size.
The frozen pre-evolution incumbent and the evolved policy were evaluated from
their immutable run sources.

| Block size | Capacity blocks | Evolved score | Incumbent score | Delta | Evolved token hit | Incumbent token hit | Evolved churn / 1k | Incumbent churn / 1k |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 48 / 96 | **70.615** | 65.451 | **+5.164** | 67.9% | 65.1% | 515.4 | 72.7 |
| 16 | 24 / 48 | 61.368 | **65.649** | -4.281 | 59.4% | 59.6% | 588.6 | 163.9 |
| 32 | 12 / 24 | -22.670 | **-17.602** | -5.068 | 36.3% | 38.0% | 663.6 | 174.0 |
| 64 | 6 / 12 | -47.066 | **-37.107** | -9.959 | 19.1% | 22.4% | 660.4 | 150.2 |
| 128 | 3 / 6 | -64.655 | **-53.541** | -11.114 | 9.2% | 13.6% | 710.2 | 169.8 |

The evolved policy is strongest at 8-token granularity, where it also edges
TinyLFU-LRU by 0.253 charged points. At 16 tokens it trails the incumbent and
TinyLFU-LRU. At 32 tokens it beats the three generic baselines but still trails
the incumbent. Its relative performance deteriorates further at 64 and 128
tokens as hit rate falls and churn remains high. This is evidence of a narrow
small-block operating region rather than block-size invariance. The absolute
score collapse above 16 tokens affects every policy because fixed token capacity
leaves only 12/24, 6/12, and 3/6 cache blocks, respectively.

Detailed verifier identities and baseline results are preserved in
`artifacts/mooncake-evolution/full-run/final/synthetic-block-size-sweep-evolved.md`
and
`artifacts/mooncake-evolution/full-run/final/synthetic-block-size-sweep-parent.md`.

## Independent WildChat smoke transfer

After Mooncake selection was frozen, the candidate was evaluated without model
calls on the existing 65-request WildChat conversation panel. WildChat uses
16-token blocks and whole-tenant splits, so it provides an independent geometry
and traffic source. The sample is deliberately small: validation has 11 requests
and the previously untouched hidden split has 19.

| Policy | Validation score | Validation hit | Hidden score | Hidden hit | Hidden churn / 1k |
|---|---:|---:|---:|---:|---:|
| vLLM-style APC | 52.030 | 57.77% | 8.859 | 41.13% | 15,473.7 |
| LRU | 51.717 | 57.77% | 7.928 | 41.04% | 16,315.8 |
| TinyLFU-LRU | 37.975 | 42.85% | -10.951 | 29.55% | 8,631.6 |
| Selected candidate | 30.502 | 48.48% | -8.452 | 35.48% | 7,815.8 |
| Incumbent | -65.530 | 3.97% | -68.411 | 1.30% | 0.0 |

The evolved policy transfers far better than the incumbent and beats TinyLFU-LRU
on the tiny hidden split, while APC and LRU retain higher hit rate and charged
score. The absolute churn rates are unstable because each eviction is divided by
only 11 or 19 requests. These results establish compatibility and a useful signal,
not a representative WildChat ranking.

## Decision and scope

Keep the existing general incumbent unchanged. The evolved policy is a strong
candidate for a Mooncake-like 512-token production specialist after independent
production captures or an online shadow test. Hidden results must not be used
for another mutation round on this same panel.

This experiment used Mooncake's production tool-and-agent trace. WildChat was
not included in the evolutionary search; it was used only for the independent
post-selection conversation-derived smoke test above and is not production
serving traffic.

The frozen search config is
`artifacts/mooncake-evolution/full-run/search.yaml`; the selected source is
`artifacts/mooncake-evolution/full-run/runs/20260906T064231Z/best_program.py`
with SHA-256
`760b6b8d8bf5e9aa1277cfeb38230a41d3669081384a1a4e5daffd2f1e342e3b`.
The sandbox image remained
`sha256:15972abc69f4141ae90ddd06d1b8fdd811ca3c33be56864133b4307a86d7e7f1`.
