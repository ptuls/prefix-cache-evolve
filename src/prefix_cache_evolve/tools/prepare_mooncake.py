"""Convert public Mooncake block-hash traces for prefix-cache replay."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import click

from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import file_sha256
from prefix_cache_evolve.problems.prefix_kv_cache.trace_replay import iter_trace_records

_BLOCK_SIZE_TOKENS = 512
_SOURCE_FIELDS = {"timestamp", "input_length", "output_length", "hash_ids"}
_UNKNOWN_IDENTITY = "mooncake:unknown"


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Native Mooncake JSONL with timestamp, input_length, output_length, and hash_ids.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("artifacts/traces/mooncake.jsonl"),
    show_default=True,
    help="New replay trace; provenance is written to OUTPUT.manifest.json.",
)
@click.option("--expected-sha256", help="Verify the complete source file before publishing output.")
def main(input_path: Path, output_path: Path, expected_sha256: str | None) -> None:
    """Preserve Mooncake's native 512-token prefix identities and arrival times.

    Session, tenant, and priority metadata are unavailable in this format. The
    converter uses a null session, one tenant accounting pool, and priority zero; it cannot create
    session-disjoint evolution panels or recover finer-grained prefix sharing.
    """
    try:
        manifest = convert_mooncake_trace(
            input_path,
            output_path=output_path,
            expected_sha256=expected_sha256,
        )
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"trace={output_path}")
    click.echo(f"manifest={output_path.with_suffix(output_path.suffix + '.manifest.json')}")
    click.echo(f"requests={manifest['request_count']} block_size_tokens={_BLOCK_SIZE_TOKENS}")


def convert_mooncake_trace(
    input_path: Path,
    *,
    output_path: Path,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and convert an opaque Mooncake trace without inventing sessions."""
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    for destination in (output_path, manifest_path):
        if destination.resolve() == input_path.resolve() or destination.exists():
            raise ValueError(f"{destination} already exists; choose a new output path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256()
    identities: dict[int, tuple[int | None, int]] = {}
    with tempfile.TemporaryDirectory(prefix=".mooncake-", dir=output_path.parent) as temporary:
        converted = Path(temporary) / "trace.jsonl"
        with input_path.open("rb") as source, converted.open("w", encoding="utf-8") as output:
            for line_number, line in enumerate(source, start=1):
                source_digest.update(line)
                if not line.strip():
                    continue
                try:
                    payload = _convert_record(json.loads(line), identities)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
                output.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        source_sha = source_digest.hexdigest()
        if expected_sha256 is not None and source_sha != expected_sha256:
            raise ValueError("source SHA-256 does not match --expected-sha256")

        count = 0
        input_tokens = 0
        output_tokens = 0
        first_timestamp = None
        last_timestamp = None
        for record in iter_trace_records(converted, block_size_tokens=_BLOCK_SIZE_TOKENS):
            count += 1
            input_tokens += record.prompt_length
            output_tokens += record.output_length
            if first_timestamp is None:
                first_timestamp = record.timestamp_ms
            last_timestamp = record.timestamp_ms
        manifest = {
            "schema": "prefix-kv-cache-mooncake-conversion-v2",
            "source": {"path": str(input_path), "sha256": source_sha},
            "output_sha256": file_sha256(converted),
            "block_size_tokens": _BLOCK_SIZE_TOKENS,
            "request_count": count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "unique_prefix_hashes": len(identities),
            "timestamp_ms": {"first": first_timestamp, "last": last_timestamp},
            "session_count": None,
            "tenant_count": None,
            "unavailable_metadata": {
                "session_hash": None,
                "tenant_hash": _UNKNOWN_IDENTITY,
                "priority": 0,
                "predicted_output_length": None,
            },
            "limitations": [
                "Source hashes identify cumulative prefixes at 512-token granularity only.",
                "Session IDs are null; the shared tenant placeholder is not a measured tenant.",
                "Session holdouts, tenant fairness, and priority effects cannot be measured.",
                "Arrival times are recorded, but replay uses simulator timing and decode proxies.",
                "The source checksum identifies the input; provenance must identify its publisher "
                "and whether it is production or synthetic traffic.",
            ],
        }
        temporary_manifest = Path(temporary) / "manifest.json"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        converted.rename(output_path)
        temporary_manifest.rename(manifest_path)
    return manifest


def _convert_record(
    payload: Any,
    identities: dict[int, tuple[int | None, int]],
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _SOURCE_FIELDS:
        raise ValueError("expected only timestamp, input_length, output_length, and hash_ids")
    prompt_length = payload["input_length"]
    if type(prompt_length) is not int or prompt_length <= 0:
        raise ValueError("input_length must be a positive integer")
    hashes = payload["hash_ids"]
    if not isinstance(hashes, list) or len(hashes) != (prompt_length + 511) // 512:
        raise ValueError("hash_ids depth must equal ceil(input_length / 512)")
    parent = None
    for depth, block in enumerate(hashes):
        if type(block) is not int or block < 0:
            raise ValueError("hash_ids must contain nonnegative integers")
        identity = (parent, min(_BLOCK_SIZE_TOKENS, prompt_length - depth * _BLOCK_SIZE_TOKENS))
        if identities.setdefault(block, identity) != identity:
            raise ValueError("a cumulative hash ID has conflicting parents or block lengths")
        parent = block
    return {
        "timestamp_ms": payload["timestamp"],
        "tenant_hash": _UNKNOWN_IDENTITY,
        "session_hash": None,
        "priority": 0,
        "request_type": "mooncake_trace",
        "prompt_length": prompt_length,
        "output_length": payload["output_length"],
        "prefix_path": hashes,
    }
