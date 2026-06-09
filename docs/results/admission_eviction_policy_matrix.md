# Admission-Eviction Policy Matrix

Every distinct admission implementation is crossed with four deployable
eviction rules and one reporting-only constrained next-use control.

| Admission | Eviction | Scope | Mean token hit | Saved tokens | Eviction regret | Admission share |
|---|---|---|---:|---:|---:|---:|
| `reject_all` | `lru` | deployable | 0.0000 | 0 | 0 | 100.0% |
| `reject_all` | `lfu` | deployable | 0.0000 | 0 | 0 | 100.0% |
| `reject_all` | `cost_aware_lru` | deployable | 0.0000 | 0 | 0 | 100.0% |
| `reject_all` | `incumbent_value_aware` | deployable | 0.0000 | 0 | 0 | 100.0% |
| `reject_all` | `oracle_next_use` | reporting-only | 0.0000 | 0 | 0 | 100.0% |
| `admit_all` | `lru` | deployable | 0.5205 | 505058 | 317312 | 25.2% |
| `admit_all` | `lfu` | deployable | 0.5427 | 528384 | 115139 | 53.8% |
| `admit_all` | `cost_aware_lru` | deployable | 0.5218 | 503812 | 368962 | 21.5% |
| `admit_all` | `incumbent_value_aware` | deployable | 0.5431 | 529228 | 107471 | 55.8% |
| `admit_all` | `oracle_next_use` | reporting-only | 0.5642 | 546710 | 38260 | 79.6% |
| `full_blocks_only` | `lru` | deployable | 0.5123 | 508016 | 143328 | 58.0% |
| `full_blocks_only` | `lfu` | deployable | 0.5219 | 522448 | 50416 | 80.8% |
| `full_blocks_only` | `cost_aware_lru` | deployable | 0.5116 | 506144 | 160688 | 54.7% |
| `full_blocks_only` | `incumbent_value_aware` | deployable | 0.5212 | 521616 | 57744 | 78.2% |
| `full_blocks_only` | `oracle_next_use` | reporting-only | 0.5283 | 532576 | 18192 | 92.1% |
| `tinylfu` | `lru` | deployable | 0.5286 | 518889 | 129436 | 63.6% |
| `tinylfu` | `lfu` | deployable | 0.5426 | 531884 | 53180 | 81.2% |
| `tinylfu` | `cost_aware_lru` | deployable | 0.5335 | 522315 | 124570 | 64.4% |
| `tinylfu` | `incumbent_value_aware` | deployable | 0.5431 | 533647 | 51172 | 81.8% |
| `tinylfu` | `oracle_next_use` | reporting-only | 0.5619 | 549453 | 15593 | 93.5% |
| `compact_seed` | `lru` | deployable | 0.5247 | 516722 | 205323 | 45.0% |
| `compact_seed` | `lfu` | deployable | 0.5460 | 537137 | 79998 | 68.9% |
| `compact_seed` | `cost_aware_lru` | deployable | 0.5274 | 517721 | 212074 | 44.2% |
| `compact_seed` | `incumbent_value_aware` | deployable | 0.5463 | 538490 | 72800 | 70.9% |
| `compact_seed` | `oracle_next_use` | reporting-only | 0.5666 | 554207 | 27843 | 86.5% |
| `structured_seed` | `lru` | deployable | 0.5422 | 531085 | 93756 | 71.5% |
| `structured_seed` | `lfu` | deployable | 0.5558 | 546952 | 31308 | 88.0% |
| `structured_seed` | `cost_aware_lru` | deployable | 0.5425 | 531427 | 95901 | 71.1% |
| `structured_seed` | `incumbent_value_aware` | deployable | 0.5549 | 544668 | 36666 | 86.4% |
| `structured_seed` | `oracle_next_use` | reporting-only | 0.5673 | 557807 | 9567 | 95.9% |
| `structured_recurrence` | `lru` | deployable | 0.5380 | 529892 | 123960 | 63.0% |
| `structured_recurrence` | `lfu` | deployable | 0.5549 | 548851 | 46267 | 81.4% |
| `structured_recurrence` | `cost_aware_lru` | deployable | 0.5398 | 531081 | 124482 | 62.9% |
| `structured_recurrence` | `incumbent_value_aware` | deployable | 0.5543 | 547450 | 47394 | 81.4% |
| `structured_recurrence` | `oracle_next_use` | reporting-only | 0.5694 | 562433 | 14306 | 93.3% |
| `pressure_aware_incumbent` | `lru` | deployable | 0.5448 | 546022 | 68189 | 72.7% |
| `pressure_aware_incumbent` | `lfu` | deployable | 0.5523 | 551190 | 43839 | 80.6% |
| `pressure_aware_incumbent` | `cost_aware_lru` | deployable | 0.5464 | 546457 | 100130 | 64.5% |
| `pressure_aware_incumbent` | `incumbent_value_aware` | deployable | 0.5535 | 554861 | 31565 | 85.2% |
| `pressure_aware_incumbent` | `oracle_next_use` | reporting-only | 0.5684 | 566784 | 8173 | 95.6% |
| `oracle_future_reuse` | `lru` | reporting-only | 0.5677 | 557317 | 82028 | 40.6% |
| `oracle_future_reuse` | `lfu` | reporting-only | 0.5703 | 558246 | 60995 | 48.3% |
| `oracle_future_reuse` | `cost_aware_lru` | reporting-only | 0.5687 | 557904 | 91714 | 37.4% |
| `oracle_future_reuse` | `incumbent_value_aware` | reporting-only | 0.5714 | 561712 | 51503 | 52.7% |
| `oracle_future_reuse` | `oracle_next_use` | reporting-only | 0.5795 | 573206 | 17558 | 75.9% |

## Best Eviction Per Admission

| Admission | LRU token hit | Best deployable eviction | Best token hit | Gain over LRU | Oracle token hit | Oracle headroom |
|---|---:|---|---:|---:|---:|---:|
| `reject_all` | 0.0000 | `lru` | 0.0000 | +0.0000 | 0.0000 | +0.0000 |
| `admit_all` | 0.5205 | `incumbent_value_aware` | 0.5431 | +0.0226 | 0.5642 | +0.0211 |
| `full_blocks_only` | 0.5123 | `lfu` | 0.5219 | +0.0096 | 0.5283 | +0.0064 |
| `tinylfu` | 0.5286 | `incumbent_value_aware` | 0.5431 | +0.0145 | 0.5619 | +0.0188 |
| `compact_seed` | 0.5247 | `incumbent_value_aware` | 0.5463 | +0.0217 | 0.5666 | +0.0203 |
| `structured_seed` | 0.5422 | `lfu` | 0.5558 | +0.0136 | 0.5673 | +0.0114 |
| `structured_recurrence` | 0.5380 | `lfu` | 0.5549 | +0.0169 | 0.5694 | +0.0145 |
| `pressure_aware_incumbent` | 0.5448 | `incumbent_value_aware` | 0.5535 | +0.0087 | 0.5684 | +0.0150 |
| `oracle_future_reuse` | 0.5677 | `incumbent_value_aware` | 0.5714 | +0.0037 | 0.5795 | +0.0082 |

## Paired Group Comparisons Against LRU

Win/tie/loss counts compare token hit rate on the same
workload-capacity-seed group.

| Admission | Eviction | Wins | Ties | Losses | Mean token-hit delta |
|---|---|---:|---:|---:|---:|
| `admit_all` | `lfu` | 97 | 65 | 0 | +0.0222 |
| `admit_all` | `incumbent_value_aware` | 96 | 63 | 3 | +0.0226 |
| `admit_all` | `oracle_next_use` | 117 | 45 | 0 | +0.0437 |
| `full_blocks_only` | `lfu` | 68 | 93 | 1 | +0.0096 |
| `full_blocks_only` | `incumbent_value_aware` | 67 | 93 | 2 | +0.0090 |
| `full_blocks_only` | `oracle_next_use` | 74 | 88 | 0 | +0.0161 |
| `tinylfu` | `lfu` | 74 | 85 | 3 | +0.0140 |
| `tinylfu` | `incumbent_value_aware` | 70 | 87 | 5 | +0.0145 |
| `tinylfu` | `oracle_next_use` | 96 | 66 | 0 | +0.0333 |
| `compact_seed` | `lfu` | 87 | 74 | 1 | +0.0213 |
| `compact_seed` | `incumbent_value_aware` | 85 | 73 | 4 | +0.0217 |
| `compact_seed` | `oracle_next_use` | 108 | 54 | 0 | +0.0420 |
| `structured_seed` | `lfu` | 76 | 81 | 5 | +0.0136 |
| `structured_seed` | `incumbent_value_aware` | 77 | 80 | 5 | +0.0127 |
| `structured_seed` | `oracle_next_use` | 97 | 63 | 2 | +0.0251 |
| `structured_recurrence` | `lfu` | 77 | 80 | 5 | +0.0169 |
| `structured_recurrence` | `incumbent_value_aware` | 79 | 79 | 4 | +0.0163 |
| `structured_recurrence` | `oracle_next_use` | 100 | 59 | 3 | +0.0314 |
| `pressure_aware_incumbent` | `lfu` | 66 | 85 | 11 | +0.0075 |
| `pressure_aware_incumbent` | `incumbent_value_aware` | 76 | 80 | 6 | +0.0087 |
| `pressure_aware_incumbent` | `oracle_next_use` | 102 | 59 | 1 | +0.0237 |
| `oracle_future_reuse` | `lfu` | 31 | 118 | 13 | +0.0026 |
| `oracle_future_reuse` | `incumbent_value_aware` | 33 | 120 | 9 | +0.0037 |
| `oracle_future_reuse` | `oracle_next_use` | 46 | 113 | 3 | +0.0119 |

## Interpretation

The best deployable pairing is `structured_seed+lfu` at mean token hit `0.5558`.
LFU and the compact incumbent value-aware eviction rule deliver similar
realized gains across the selective admission policies; neither uniformly
wins every group. This supports a simple frequency-aware eviction rule,
not the stronger claim that eviction choice is unimportant.

Mean token-hit differences are realized outcomes on identical generated
request panels. Eviction regret remains a local future-count surrogate.
The oracle is constrained to legal resident leaves and is reporting-only;
it is not an unconstrained globally optimal cache replay.
