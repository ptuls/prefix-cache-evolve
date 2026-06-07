# Structured Prefix KV-Cache Three-Run Adjudication

**Decision: keep the compact incumbent.**

No structured seed or generated candidate improved charged validation selection
without regressing the held-out probe or hidden panel.

| Run | Evals | Cost | Best generated | Raw before cx | Cx | Subsidy | Probe | Agent hit | Cyclic hit | Hidden |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `20260606T123835Z` | 98 | $2.474 | 49.326 | 64.433 | 1429 | 15 | 22.871 | 0.1648 | 0.7605 | -16.542 |
| `20260606T125829Z` | 98 | $2.722 | 51.219 | 65.470 | 1322 | 15 | 20.854 | 0.2727 | 0.8054 | -16.501 |
| `20260606T132003Z` | 98 | $2.486 | 49.702 | 62.938 | 1198 | 0 | 35.312 | 0.2142 | 0.7736 | -15.486 |

## Reference Candidates

| Candidate | Selection | Raw before cx | Cx | Subsidy | Probe | Agent hit | Cyclic hit | Hidden |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Compact incumbent | 57.788 | 64.381 | 473 | 0 | 40.303 | 0.2268 | 0.7817 | -18.616 |
| Structured seed | 57.426 | 63.819 | 454 | 15 | 34.980 | 0.2533 | 0.7445 | -10.815 |

Run 2 is the strongest functional-form result: it improves raw selection and both
probe-family hit rates, but loses after complexity and regresses aggregate probe and
hidden scores. All three runs retained the structured seed as the charged winner.
