"""Shared validation and atomic publication for public block-hash traces."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import file_sha256
from prefix_cache_evolve.problems.prefix_kv_cache.trace_replay import iter_trace_records


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read native objects with file and line context on malformed input."""
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                yield row
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def integer(value: Any, name: str, *, minimum: int = 0) -> int:
    """Require an integer without accepting booleans or rounding values."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def seconds_to_ms(value: Any) -> float:
    """Validate source seconds before converting to replay milliseconds."""
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("timestamp must be finite nonnegative seconds")
    timestamp = value * 1000.0
    if not math.isfinite(timestamp):
        raise ValueError("timestamp overflows milliseconds")
    return timestamp


def scope_key(*parts: str) -> str:
    """Produce a compact deterministic namespace without ambiguous separators."""
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()[:32]


def prefix_path(hashes: Any, length: int, block_size: int, namespace: str) -> list[str]:
    """Namespace block identities; replay chains each complete path prefix."""
    if not isinstance(hashes, list) or len(hashes) != (length + block_size - 1) // block_size:
        raise ValueError(f"hash_ids depth must equal ceil(input length / {block_size})")
    return [f"{namespace}:{integer(block, 'hash_id')}" for block in hashes]


def publish_trace(
    input_path: Path,
    output_path: Path,
    *,
    block_size: int,
    expected_sha256: str | None,
    convert: Callable[[str], tuple[Iterator[dict[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    """Sort on disk, validate, and publish a trace with checksum-pinned provenance.

    Source sessions may be large and have overlapping clocks. SQLite keeps the
    global timestamp sort bounded in memory and preserves input order for ties.
    """
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    for destination in (output_path, manifest_path):
        if destination.exists() or destination.resolve() == input_path.resolve():
            raise ValueError(f"{destination} already exists; choose a new output path")
    source_sha = file_sha256(input_path)
    if expected_sha256 is not None and source_sha != expected_sha256:
        raise ValueError("source SHA-256 does not match --expected-sha256")
    records, metadata = convert(source_sha)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hashed-trace-", dir=output_path.parent) as temporary:
        directory = Path(temporary)
        connection = sqlite3.connect(directory / "sort.sqlite")
        try:
            connection.execute("CREATE TABLE requests (timestamp REAL, payload TEXT)")
            with connection:
                connection.executemany(
                    "INSERT INTO requests VALUES (?, ?)",
                    (
                        (
                            row["timestamp_ms"],
                            json.dumps(row, separators=(",", ":"), sort_keys=True, allow_nan=False),
                        )
                        for row in records
                    ),
                )
            converted = directory / "trace.jsonl"
            with converted.open("w", encoding="utf-8") as output:
                for (payload,) in connection.execute(
                    "SELECT payload FROM requests ORDER BY timestamp, rowid"
                ):
                    output.write(payload + "\n")
        finally:
            connection.close()
        if file_sha256(input_path) != source_sha:
            raise ValueError("source changed during conversion")
        sessions = set()
        count = input_tokens = output_tokens = 0
        first_timestamp = last_timestamp = None
        for record in iter_trace_records(converted, block_size_tokens=block_size):
            count += 1
            input_tokens += record.prompt_length
            output_tokens += record.output_length
            sessions.add(record.session_key)
            if first_timestamp is None:
                first_timestamp = record.timestamp_ms
            last_timestamp = record.timestamp_ms
        manifest = {
            **metadata,
            "source": {"path": str(input_path), "sha256": source_sha},
            "output_sha256": file_sha256(converted),
            "block_size_tokens": block_size,
            "request_count": count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "session_count": len(sessions - {None}),
            "tenant_count": None,
            "timestamp_ms": {"first": first_timestamp, "last": last_timestamp},
        }
        temporary_manifest = directory / "manifest.json"
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        converted.rename(output_path)
        temporary_manifest.rename(manifest_path)
    return manifest
