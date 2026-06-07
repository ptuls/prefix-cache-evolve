# Structured Prefix KV-Cache Ablation

Behavior-only ablations are evaluated with complexity zero; source deletion and charged evaluation follow after selecting useful terms.

| Variant | Selection raw | Mean | Min contrib. | Churn cost | Probe raw | Agent hit | Cyclic hit | Probe churn/1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all_terms` | 62.178 | 65.543 | 13.554 | 9.256 | 35.654 | 0.2626 | 0.7408 | 634.5 |
| `without_recurrence` | 63.762 | 65.436 | 12.826 | 7.198 | 41.373 | 0.2533 | 0.7445 | 306.4 |
| `without_subtree` | 59.029 | 63.127 | 11.228 | 6.227 | 37.906 | 0.2558 | 0.7362 | 430.6 |
| `without_regime` | 61.264 | 67.734 | 14.290 | 13.096 | 27.756 | 0.2662 | 0.7267 | 986.1 |
| `without_miss_state` | 62.287 | 65.668 | 13.546 | 9.265 | 32.501 | 0.2626 | 0.7182 | 678.8 |
| `without_priority_state` | 62.298 | 65.674 | 13.554 | 9.266 | 35.654 | 0.2626 | 0.7408 | 634.5 |
| `without_recurrence_and_priority_state` | 63.819 | 65.380 | 12.815 | 7.073 | 41.373 | 0.2533 | 0.7445 | 306.4 |
| `without_recurrence_priority_and_miss_state` | 63.718 | 65.280 | 12.815 | 7.075 | 41.038 | 0.2534 | 0.7411 | 313.4 |
