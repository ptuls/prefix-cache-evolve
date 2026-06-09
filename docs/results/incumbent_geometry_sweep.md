# Incumbent Geometry Sweep

This report compares the promoted 16-token incumbent with the immediately
preceding pressure-aware incumbent from commit
`220b26e8e0e50f53d9ce048a434f040c635cd515`.

Each geometry was evaluated independently on the full train and validation
panels with the committed seeds and canonical 8-token workload generation.
Scores include source-complexity cost. Fixing block count means the physical
token capacity changes with block size.

| Block tokens | Blocks | Cache tokens | Old score | New score | Score delta | Old validation hit | New validation hit | Hit delta | Old churn/1k | New churn/1k |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24 | 192 | 65.069 | 57.365 | -7.704 | 0.6339 | 0.6210 | -1.290 pp | 236.5 | 109.7 |
| 8 | 48 | 384 | 76.594 | 65.643 | -10.951 | 0.6658 | 0.6497 | -1.613 pp | 109.4 | 77.8 |
| 8 | 96 | 768 | 78.174 | 65.339 | -12.835 | 0.6723 | 0.6521 | -2.025 pp | 76.0 | 67.7 |
| 8 | 128 | 1024 | 78.539 | 65.290 | -13.249 | 0.6730 | 0.6522 | -2.080 pp | 43.1 | 41.7 |
| 16 | 24 | 384 | 59.033 | 62.042 | +3.009 | 0.5813 | 0.5794 | -0.192 pp | 212.3 | 198.7 |
| 16 | 48 | 768 | 67.184 | 69.346 | +2.161 | 0.6151 | 0.6134 | -0.164 pp | 124.0 | 129.2 |
| 16 | 96 | 1536 | 66.458 | 67.694 | +1.236 | 0.6245 | 0.6251 | +0.060 pp | 77.1 | 68.8 |
| 16 | 128 | 2048 | 67.455 | 68.670 | +1.215 | 0.6255 | 0.6257 | +0.020 pp | 16.0 | 24.3 |

## Aggregate by block size

Jointly scoring all four block-count tiers gives:

| Block tokens | Old charged | New charged | Delta | Old raw | New raw | Raw delta |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 74.068 | 63.154 | -10.915 | 82.300 | 70.756 | -11.544 |
| 16 | 64.702 | 66.710 | +2.008 | 72.934 | 74.312 | +1.378 |

The old policy has effective complexity 636 and a 8.232-point complexity cost.
The new policy has effective complexity 572 and a 7.603-point cost.

## Interpretation

The promoted policy is a real improvement for its 16-token search target, but
it is not block-size invariant. At 16 tokens it wins charged and raw score at
every tested block count. Mean validation hit is slightly lower at 24 and 48
blocks, then reaches parity at 96 and 128 blocks; lower complexity and better
cache economics provide the remaining score gain.

At 8 tokens, the new policy loses raw and charged score at every capacity.
Lower churn does not compensate for excessive admission selectivity:
avoidable-rejection regret rises by roughly 3.9--6.1 percentage points, and
validation hit falls by 1.3--2.1 percentage points. The deficit grows with
capacity because the policy continues rejecting useful fine-grained blocks
after eviction pressure has largely disappeared.

The production conclusion is geometry-specific:

- promote the new policy for 16-token blocks;
- retain the old policy as the stronger 8-token reference;
- include both block sizes in the evolutionary selection panel before claiming
  a single geometry-robust successor.

A follow-up fixed-token-capacity sweep from 16 through 64 tokens per block
showed that 16 tokens is also the peak of the tested coarse-granularity range;
see [block_granularity_sweep.md](block_granularity_sweep.md).
