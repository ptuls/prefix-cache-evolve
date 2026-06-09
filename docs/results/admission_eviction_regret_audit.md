# Admission vs Eviction Regret Audit

Policy: `pressure_aware_incumbent`

## Falsifiable Claim

Claim: admission-side regret dominates eviction-side regret across
workload-capacity-seed groups.

**Verdict: Falsified under the strict universal rule.**

- Regretful groups: `123`.
- Admission-dominant groups: `81` (`65.9%`).
- Aggregate admission-side regret: `181327.0` future tokens.
- Aggregate eviction-side regret: `31565.0` future tokens.
- Aggregate admission share: `85.2%`.
- Admission decisions: `21173`; surrogate regret per decision: `8.56`.
- Eviction decisions: `5402`; surrogate regret per decision: `5.84`.
- Decision-normalized admission dominance: `61` of `123` regretful groups (`49.6%`).

The universal claim passes only if every valid group with nonzero regret has
strictly greater admission-side regret. Zero-regret groups are reported but do not
decide the claim.

## Audit Definition

- The future-value surrogate upper bound is block token count times remaining
  request occurrences after the current request.
- Avoidable admission regret is the value lost when an accepted incoming block is
  worth less than the cheapest legal displacement.
- Avoidable rejection regret is the value lost when a rejected incoming block is
  worth more than the cheapest legal displacement; free space has displacement zero.
- Value-weighted avoidable-eviction regret is the chosen victim's value minus the
  cheapest legal victim's value.
- Admission plus eviction surrogate regret exactly decomposes this local same-state
  comparison. It is not realized causal regret or a full counterfactual replay.

## By Split

| Split | Regretful groups | Admission dominant | Dominance rate | Admission regret | Eviction regret | Margin |
|---|---:|---:|---:|---:|---:|---:|
| `hidden` | 45 | 31 | 68.9% | 30182.0 | 18255.0 | +11927.0 |
| `probe` | 9 | 6 | 66.7% | 49490.0 | 822.0 | +48668.0 |
| `train` | 24 | 12 | 50.0% | 83380.0 | 5746.0 | +77634.0 |
| `validation` | 45 | 32 | 71.1% | 18275.0 | 6742.0 | +11533.0 |

## By Workload

| Workload | Regretful groups | Admission dominant | Dominance rate | Admission regret | Eviction regret | Margin |
|---|---:|---:|---:|---:|---:|---:|
| `hidden/adversarial_unique_prompts` | 0 | 0 | 0.0% | 0.0 | 0.0 | +0.0 |
| `hidden/cross_family_mixture` | 3 | 3 | 100.0% | 1440.0 | 0.0 | +1440.0 |
| `hidden/cyclic_working_set_pressure_shifted` | 3 | 0 | 0.0% | 560.0 | 886.0 | -326.0 |
| `hidden/heavy_tailed_prefix_lengths_shifted` | 6 | 1 | 16.7% | 3840.0 | 7056.0 | -3216.0 |
| `hidden/priority_burst_recovery_shifted` | 6 | 5 | 83.3% | 514.0 | 343.0 | +171.0 |
| `hidden/priority_one_off_noise_shifted` | 6 | 3 | 50.0% | 870.0 | 552.0 | +318.0 |
| `hidden/rolling_template_versions_shifted` | 6 | 6 | 100.0% | 3984.0 | 0.0 | +3984.0 |
| `hidden/stochastic_serving_mix_shifted` | 6 | 6 | 100.0% | 4576.0 | 448.0 | +4128.0 |
| `hidden/tenant_phase_shift_cycles_shifted` | 6 | 4 | 66.7% | 13390.0 | 8970.0 | +4420.0 |
| `hidden/tenant_session_reentry` | 3 | 3 | 100.0% | 1008.0 | 0.0 | +1008.0 |
| `probe/agent_trace_branching` | 6 | 6 | 100.0% | 49344.0 | 496.0 | +48848.0 |
| `probe/cyclic_working_set_pressure` | 3 | 0 | 0.0% | 146.0 | 326.0 | -180.0 |
| `train/agentic_tool_workflows` | 6 | 6 | 100.0% | 44192.0 | 2688.0 | +41504.0 |
| `train/long_context_mixed` | 0 | 0 | 0.0% | 0.0 | 0.0 | +0.0 |
| `train/rag_template_reuse` | 6 | 0 | 0.0% | 0.0 | 372.0 | -372.0 |
| `train/session_continuation_growth` | 6 | 6 | 100.0% | 39040.0 | 416.0 | +38624.0 |
| `train/shared_system_prompt` | 6 | 0 | 0.0% | 148.0 | 2270.0 | -2122.0 |
| `validation/concurrent_long_generation` | 3 | 0 | 0.0% | 757.0 | 955.0 | -198.0 |
| `validation/heavy_tailed_prefix_lengths` | 6 | 5 | 83.3% | 1696.0 | 1472.0 | +224.0 |
| `validation/hotset_cold_scan` | 0 | 0 | 0.0% | 0.0 | 0.0 | +0.0 |
| `validation/multi_tenant_skew` | 6 | 0 | 0.0% | 33.0 | 384.0 | -351.0 |
| `validation/phase_shift_prompts` | 3 | 3 | 100.0% | 2256.0 | 0.0 | +2256.0 |
| `validation/priority_burst_recovery` | 6 | 6 | 100.0% | 306.0 | 0.0 | +306.0 |
| `validation/priority_one_off_noise` | 3 | 0 | 0.0% | 0.0 | 540.0 | -540.0 |
| `validation/rolling_template_versions` | 6 | 6 | 100.0% | 1488.0 | 0.0 | +1488.0 |
| `validation/stochastic_serving_mix` | 6 | 6 | 100.0% | 2336.0 | 448.0 | +1888.0 |
| `validation/tenant_phase_shift_cycles` | 6 | 6 | 100.0% | 9403.0 | 2943.0 | +6460.0 |

## Strongest Counterexamples

| Group | Admission regret | Eviction regret | Margin |
|---|---:|---:|---:|
| `hidden/heavy_tailed_prefix_lengths_shifted/capacity_24/seed_6023` | 880.0 | 2096.0 | -1216.0 |
| `hidden/tenant_phase_shift_cycles_shifted/capacity_24/seed_10023` | 2158.0 | 2955.0 | -797.0 |
| `hidden/heavy_tailed_prefix_lengths_shifted/capacity_24/seed_6011` | 576.0 | 1200.0 | -624.0 |
| `hidden/heavy_tailed_prefix_lengths_shifted/capacity_48/seed_6011` | 416.0 | 976.0 | -560.0 |
| `hidden/heavy_tailed_prefix_lengths_shifted/capacity_48/seed_6023` | 448.0 | 992.0 | -544.0 |
| `hidden/heavy_tailed_prefix_lengths_shifted/capacity_24/seed_6037` | 976.0 | 1488.0 | -512.0 |
| `train/shared_system_prompt/capacity_48/seed_1011` | 0.0 | 493.0 | -493.0 |
| `train/shared_system_prompt/capacity_48/seed_1023` | 0.0 | 493.0 | -493.0 |
| `train/shared_system_prompt/capacity_48/seed_1037` | 0.0 | 493.0 | -493.0 |
| `hidden/tenant_phase_shift_cycles_shifted/capacity_24/seed_10011` | 2316.0 | 2726.0 | -410.0 |
| `train/shared_system_prompt/capacity_24/seed_1011` | 46.0 | 267.0 | -221.0 |
| `train/shared_system_prompt/capacity_24/seed_1037` | 46.0 | 267.0 | -221.0 |
| `train/shared_system_prompt/capacity_24/seed_1023` | 56.0 | 257.0 | -201.0 |
| `validation/priority_one_off_noise/capacity_24/seed_9011` | 0.0 | 180.0 | -180.0 |
| `validation/priority_one_off_noise/capacity_24/seed_9023` | 0.0 | 180.0 | -180.0 |
| `validation/priority_one_off_noise/capacity_24/seed_9037` | 0.0 | 180.0 | -180.0 |
| `hidden/cyclic_working_set_pressure_shifted/capacity_24/seed_8023` | 184.0 | 297.0 | -113.0 |
| `validation/heavy_tailed_prefix_lengths/capacity_24/seed_7037` | 432.0 | 544.0 | -112.0 |
| `hidden/cyclic_working_set_pressure_shifted/capacity_24/seed_8011` | 185.0 | 297.0 | -112.0 |
| `hidden/cyclic_working_set_pressure_shifted/capacity_24/seed_8037` | 191.0 | 292.0 | -101.0 |

## Interpretation

The strict universal claim is `falsified`: admission dominates in
`81` of `123` regretful groups, leaving
`42` eviction-dominant counterexamples.
The weaker aggregate-total claim is supported under this surrogate: admission accounts for `85.2%` of total surrogate regret,
and the median regretful group has a positive admission-minus-eviction margin.
After normalizing each side by its own decision count, admission dominates in
`61` of `123` regretful groups (`49.6%`). The aggregate per-decision rate still favors admission, but groupwise
per-decision dominance does not hold in a majority. Total contribution and
per-decision severity answer different questions.

Workload-level counterexamples with negative aggregate margins include `hidden/heavy_tailed_prefix_lengths_shifted`, `train/shared_system_prompt`, `validation/priority_one_off_noise`, `train/rag_template_reuse`, `validation/multi_tenant_skew`. These groups are concrete targets for eviction-specific
follow-up rather than evidence for a uniform admission-first rule.

## Limitations

The audit uses future occurrence count as a stable token-value surrogate upper
bound. It does not model whether every future occurrence would remain
root-contiguous after other decisions. It audits only explicit admission scores;
descendants bypassed after a parent rejection are consequences of that scored
rejection, not additional decisions.
