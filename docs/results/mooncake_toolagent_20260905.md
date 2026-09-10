# Mooncake tool-and-agent production replay — September 5, 2026

> [Replay audit update](replay_audit_20260907.md): this report preserves historical replay semantics. Missing-session handling and parts of the conversational replay were corrected afterward; compare policies again under the audited contract before drawing new conclusions.

The unchanged production incumbent and three deployable baselines were replayed
on the first 1,024 requests from Kimi's published tool-and-agent traffic. The
incumbent had the highest token hit rate at each tested capacity; TinyLFU-LRU had
the best default combined score and substantially lower churn. These are local
simulator results on a bounded production sample. No policy was tuned or promoted.

## Source and method

The [published trace](https://github.com/kvcache-ai/Mooncake/blob/3cca71daccf2a7afb8fe3f0295358f70e3a69fdb/FAST25-release/traces/toolagent_trace.jsonl)
is pinned to `3cca71daccf2a7afb8fe3f0295358f70e3a69fdb` (released February 2025).
The [authors' paper, section 7.2 and appendix A](https://madsys.cs.tsinghua.edu.cn/publication/mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving/ToS2025-Qin.pdf)
identifies it as sampled online Kimi tool-and-agent traffic. This is separate from
the synthetic trace distributed in the same repository.

The complete input contains 23,608 requests, 202,940,084 input tokens, and 183,300
unique cumulative prefix hashes. The source checksum is
`48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71`.
Every record passed ordering, 512-token block geometry, and cumulative hash
parent/length consistency checks. The converter reproduced the complete metadata
trace byte for byte after implementation changes.

The scored prefix contains 9,962,059 input tokens and spans timestamps
0–176,999 ms, about three minutes. Arrival order was retained, timestamps were
bucketed at 100 ms, and each trial started with an empty cache. Capacities were
128, 512, and 2,048 blocks of 512 tokens: 65,536, 262,144, and 1,048,576 tokens.
The candidate ran in an isolated worker with a 1,200-second timeout and 2 GiB
memory limit. All 12 policy/capacity trials succeeded.

The [machine-readable record](mooncake_toolagent_20260905.json) retains source and
conversion provenance, complete evaluator settings, policy and implementation
hashes, all result summaries, and the exact score identity.

## Results

Token hit rates and churn below are arithmetic means across the three capacities.
The combined score uses the repository's configured score aggregation and penalties.
`vllm_apc` is the simulated policy baseline, not a running vLLM server.

| Policy | Token hit rate | Churn per 1k requests | Default combined score |
|---|---:|---:|---:|
| Production incumbent | 30.83% | 2,703.1 | 24.324 |
| TinyLFU-LRU | 30.04% | 1,265.6 | 35.582 |
| vLLM-style APC | 29.02% | 9,563.2 | 26.172 |
| LRU | 28.82% | 10,289.1 | 25.757 |

Token hit rate by capacity, at the same native 512-token block size:

| Policy | 128 blocks | 512 blocks | 2,048 blocks |
|---|---:|---:|---:|
| Production incumbent | 29.71% | 29.94% | 32.83% |
| TinyLFU-LRU | 29.49% | 29.78% | 30.86% |
| vLLM-style APC | 27.02% | 29.61% | 30.45% |
| LRU | 26.55% | 29.55% | 30.37% |

The score charges the incumbent 7.603 points for source complexity;
baseline complexity charges are zero. Without that candidate charge, its score
would be 31.926, still below TinyLFU-LRU's 35.582. The default churn penalty also
saturates for the incumbent, LRU, and APC. Consequently, the scalar ranking should
be read alongside the raw hit-rate and churn tradeoff. These results motivate
reducing admission/eviction churn while retaining the incumbent's hit-rate gains;
they do not establish a general production winner.

## Scope

The source provides no session, tenant, priority, model, or tool-dependency fields.
The adapter marks session and tenant identity as one unknown placeholder and
priority as zero. Actual session/user counts, session holdouts, and tenant fairness
are unavailable. Generic ID-group counts refer to the placeholder. The simulator's
`validation` label does not make this a held-out session experiment.

The 512-token source hashes cannot recover finer-grained sharing. This result is
separate from the 16-token synthetic headline benchmark. Recorded arrivals are
real metadata, while latency and decode lifetime remain simulator models. The
sample is an early, cold-start slice of an older mixed tool-and-agent trace; it
cannot characterize all contemporary coding-agent traffic. No model-powered
search or hidden-panel evaluation occurred.

## Reproduction and implementation checks

Use the pinned download, resource configuration, and selected-baseline command in
the [production replay guide](../reproducibility.md#agentic-production-replay-mooncake).
The converted trace is under `artifacts/traces/mooncake-toolagent.jsonl`; the
operative configuration, policy copy, and complete replay are under
`artifacts/mooncake-toolagent/`.

Long contexts exposed quadratic repeated prefix formatting in the simulator.
Incremental Blake2b hashing now preserves the exact historical tuple-representation
bytes and hash values. Complete LRU and incumbent summaries on a 16-request
production control slice matched before and after, including scores and identities.
Thirty boundary/representation tests cover that compatibility. A small profile of
this helper with cProfile and tracemalloc took 0.892 s before and 0.095 s after;
that approximately 9.4x component speedup is not a serving throughput measurement.

The full suite passed 410 tests with 86.52% coverage; Ruff format/check and mypy
passed. Immutable incumbent source files are unchanged.

The replay panel SHA-256 is
`f63afc826263260b3397aabd4a56fa02b6381d52ebd325e1703472d7950a0326`;
the evaluation context is
`d3fc7d610bf15d34c864ab2c29776b56af37b2dca9099fcf6496398ce94c0753`,
under verifier `1.0.0`.
