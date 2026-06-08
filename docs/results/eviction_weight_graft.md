# Eviction-Weight Graft Adjudication

**Decision: do not promote.**

Run `20260608T040632Z` selected a simpler full-policy mutation with a charged score of
`63.543`, versus `62.757` for its seed. Its eviction equation increased the age,
recurrence-protection, and priority-protection coefficients while leaving the subtree
term unchanged:

```python
- 0.85 * age - 1.8 * recurrence - 0.2 * subtree - 0.55 * priority
+ 1.15 * age - 1.9 * recurrence - 0.2 * subtree - 0.90 * priority
```

The eviction-only equation was grafted into the unchanged pressure-aware incumbent and
evaluated with the current fail-closed promotion adjudicator. The apparent search gain
did not survive: the historical mutation's charged advantage came from simplifying
other policy code, while the isolated eviction change slightly reduced raw behavior.

| Check | Graft | Incumbent | Passed |
|---|---:|---:|---:|
| Charged selection | 62.703974 | 62.757243 | No |
| Raw selection before complexity | 70.955397 | 70.989259 | No |
| Validation avoidable-eviction rate | 0.154245 | 0.152951 | No |
| Validation short-reuse-after-eviction rate | 0.038023 | 0.037817 | No |
| Aggregate probe | 55.383478 | 55.336371 | Yes |
| Agent-trace token hit | 0.422828 | 0.422736 | Yes |
| Cyclic token hit | 0.874520 | 0.874376 | Yes |
| Hidden score | 2.012421 | 2.224480 | No |
| Agentic surrogate-to-probe tripwire | 0.102945 | 0.120000 limit | Yes |

The graft improves both recurrence-heavy probe-family hit rates by a very small amount,
but loses selection behavior, eviction regret, and hidden robustness. This is another
case where a promising specialist diagnostic is insufficient for deployable promotion.
