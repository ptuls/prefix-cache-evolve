# Repository Engineering Conventions

- Follow the Google Python Style Guide for Python code.
- Keep third-party and standard-library imports at module scope unless a lazy import is
  required for correctness or optional dependency handling.
- Write Python tests as functional pytest tests.
- Build command-line interfaces with Click. Do not introduce `argparse` CLIs.
- Run `ruff format .` and `ruff check .` before considering a change complete.

Benchmark policy programs under `prefix_kv_cache` are code-as-data: their exact source affects
complexity measurements and reproducibility. Do not make style-only edits to those programs
without rerunning and documenting the affected experiments.
