# Documentation Structure

The documentation is split by purpose:

- `technical_report.tex` is the paper-like current account: method, current
  evidence, principles, limitations, and reproducibility.
- `research_log.tex` preserves detailed run chronology, engineering turning
  points, historical verifier results, and rejected solutions.
- `project_overview.md` summarizes current results, research workflow, scope,
  and contribution opportunities.
- `results/` contains generated measurements and focused analysis summaries.
- `reproducibility.md` documents installation, model-provider configuration,
  replay expectations, and publication metadata.
- [`../src/prefix_cache_evolve/tools/README.md`](../src/prefix_cache_evolve/tools/README.md)
  documents analysis, ablation, tuning, and report-generation commands.

Keep chronological additions and detailed failed-run analysis in the research
log. Update the technical report only when the current method, supported claims,
or limitations change.

Git history is intentionally preserved. To inspect code-oriented history without
documentation-only changes:

```bash
git log -- . ':(exclude)docs/**' ':(exclude)README.md'
```
