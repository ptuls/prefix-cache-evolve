# Contributing

Use Python 3.11, 3.12, or 3.13 on Linux or macOS.

```bash
make setup-dev
make check
```

Python follows the Google style guide. Keep imports at module scope, use Click
for Python command-line interfaces, and write functional pytest tests.

Benchmark policy programs are code-as-data. Do not make style-only changes to
seed or incumbent policy source. New promoted policies must be added as
immutable bundles and registered; existing bundles must never be overwritten.

Pull requests should state:

1. the behavior being changed;
2. the verification commands run;
3. whether score identities, workload manifests, or benchmark results changed;
4. whether any policy source changed and which experiments were rerun.

Generated-code evaluation is not a security sandbox. Follow [SECURITY.md](SECURITY.md)
for untrusted candidates.
