# Project Overview And Current Results

This document contains the project-level results, workflow, scope, and
contribution guidance intentionally omitted from the root README.

## Current Results

Under verifier `1.0.0`, the current production-oriented 16-token policy scores
`65.649`, ahead of TinyLFU-LRU at `63.548`, while passing held-out probe,
hidden-panel, and tripwire checks. On the historical 8-token discovery verifier,
the promoted discovery policy scores `77.113` under hardened complexity
accounting, ahead of TinyLFU-LRU at `70.362`, the vLLM APC-style approximation
at `60.178`, and LRU at `51.186`.

The stronger result is methodological:

- Pressure-aware admission and online structural context were more useful than
  broad recurrence machinery.
- Fine-grained verifier feedback and specialist archive dimensions made compact
  improvements discoverable.
- Allowing bounded over-cap exploration followed by a separate simplification
  stage produced the deployable production incumbent.
- Specialized and held-out workloads caught promising-looking policies that
  overfit, churned excessively, or failed to transfer.
- Exact incumbent replay is reproducible, but independent rediscovery from a
  weak seed is not yet established. Three fresh 298-evaluation searches
  produced zero behaviorally close policies.

The incumbent was not found by one clean run from scratch. It emerged through a
staged research process that co-evolved the verifier, generator feedback,
archive, policy lineage, and promotion gates. The documented pre-promotion
lineage totals at least `2,989` evaluations and `$44.845` in recorded API cost.

All headline results use synthetic traffic. No public production trace is yet
part of the headline comparison.

## Research Workflow

The repository separates productive optimization from scientific
adjudication:

- **Production evolution** starts from the retained incumbent and searches for
  further improvements.
- **Weak-seed rediscovery** tests whether the search process can independently
  recover a behaviorally close policy without incumbent source or coefficients.
- **Specialist searches** isolate narrower surfaces, such as eviction ranking.
- **Simplification searches** turn useful over-cap candidates into deployable
  policies.
- **Analysis tools** run causal experiments, regret audits, ablations, geometry
  sweeps, and rediscovery adjudication.

The current weak-seed rediscovery verdict is negative. Exact replay and
independent discovery are deliberately reported as separate claims.

## Included Capabilities

- A deterministic prefix-tree cache simulator with root-contiguous hits,
  leaf-only eviction, active decode pins, forced bypass, and partial blocks.
- Train, validation, quarantined probe, and hidden synthetic workload panels.
- Deployable baselines including TinyLFU-LRU, vLLM APC, SGLang
  RadixAttention, LRU, LFU, recompute-aware, prefix-fanout, and tenant-fair
  policies.
- Reporting-only future-knowledge controls and decision-level regret audits.
- Versioned score identities, workload manifests, immutable incumbent bundles,
  and saved evolution artifacts.
- An anonymized metadata-only trace calibration and replay path.
- An interactive policy-comparison lab.

## Scope And Limitations

This repository studies online prefix-cache heuristics under controlled
workloads. It does not model a complete serving stack, and the current headline
does not include a public production trace.

Candidate execution uses process isolation and resource limits as defense in
depth, not as a security sandbox. Run untrusted generated code inside an
OS/container sandbox with the repository and verifier mounted read-only.

## Contributing

The highest-value contributions are:

- workload models and anonymized trace replays that expose real serving failure
  modes;
- comparisons against production cache managers;
- policy ideas that improve agentic branching or geometry transfer;
- verifier, feedback, and mutation improvements that make weak-seed
  rediscovery repeatable.

For detailed evidence and chronology, see the
[technical report](technical_report.tex) and [research log](research_log.tex).
For commands and artifact contracts, see the
[analysis tools guide](../src/prefix_cache_evolve/tools/README.md).
