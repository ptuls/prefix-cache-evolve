# Future-Information Audit

## Verdict

The retained deployable policies do not use explicit future information. The promoted
pressure-aware incumbent, compact seed, eviction-specialist seed, and deployable baselines do
not reference future-reuse fields or workload labels. Future-reuse-aware policies remain
reporting-only oracle comparisons.

The audit did identify indirect leakage opportunities in the evaluator contract. These have
been removed so a deployable candidate receives only online policy signals:

| Surface | Previous behavior | Current behavior |
|---|---|---|
| Future reuse count and next-use distance | Gated by `expose_future_reuse` | Still gated; deployable callbacks receive `None` and deployable baselines cannot require the fields |
| Workload-generation seed | Passed to the candidate factory | Candidate factory receives a separate fixed `policy_seed` |
| Request ID | Sequential benchmark position | Stable opaque hash |
| Request type | Descriptive synthetic workload label | Normalized to `request` |
| Prompt tokens | Usually empty, but custom request streams could expose them | Always empty in candidate-visible metadata |
| Internal regret tracker | Simulator computes future reuse for reporting | Never copied into deployable candidate-visible block metadata |

Historical generated candidates occasionally attempted to reference explicit future-reuse
fields. Those fields were unavailable in normal evaluation, and the current evaluator also
rejects such source before running it. Candidate source that references the sanitized
`request_type` or `prompt_tokens` fields is rejected as well.

## Verification

Regression tests cover all candidate callback paths: miss observation, hit observation,
admission scoring, and eviction scoring. They also verify that the policy seed is independent
of the workload seed and that a future-aware baseline cannot be registered as deployable.

The retained policy source audit found future-field references only in the two reporting-only
future-aware baseline implementations.

## Remaining Limitation

Candidate Python runs in the simulator worker process. The evaluator establishes an
experimental information contract, not an adversarial Python security boundary. A deliberately
malicious candidate could attempt reflection or inspect local source and artifacts. Closing
that class of attack requires capability-separated RPC evaluation or a restricted policy DSL.

The current ranked policies do not use those techniques. Their existing reported results are
not tainted by the leakage surfaces found here because they do not consume the affected seed or
request metadata.
