# Incumbent Bundles

Each subdirectory is an immutable promoted-policy bundle:

- `policy.py` is the exact candidate source used for promotion.
- `manifest.json` pins its source hash, complexity, benchmark identity, and
  originating run artifact.
- `registry.json` assigns current roles without rewriting prior bundles.

To promote a new incumbent:

1. Complete fail-closed selection, probe, hidden, complexity, and tripwire
   adjudication.
2. Create a new role/geometry/date bundle; never edit an existing bundle.
3. Preserve the selected candidate source byte-for-byte as `policy.py`.
4. Record the current verifier replay and source-artifact provenance in
   `manifest.json`.
5. Add the bundle to `registry.json` and update the intended current role.
6. Run `uv run prefix-cache-tools incumbents validate` and the full test suite.

Ruff formatting excludes incumbent `policy.py` files because their exact source
is part of the measured artifact.
