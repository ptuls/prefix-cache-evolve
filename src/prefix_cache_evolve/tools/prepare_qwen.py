"""Convert Qwen-Bailian production block hashes and conversation links."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click

from prefix_cache_evolve.tools.hashed_trace import (
    integer,
    prefix_path,
    publish_trace,
    read_jsonl,
    scope_key,
    seconds_to_ms,
)

_FIELDS = {
    "chat_id",
    "parent_chat_id",
    "timestamp",
    "input_length",
    "output_length",
    "type",
    "turn",
    "hash_ids",
}


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="One native Qwen-Bailian JSONL capture (not a Git LFS pointer).",
)
@click.option(
    "--output", "output_path", type=click.Path(path_type=Path, dir_okay=False), required=True
)
@click.option("--expected-sha256", help="Pin the complete native source file.")
def main(input_path: Path, output_path: Path, expected_sha256: str | None) -> None:
    """Preserve native 16-token blocks and group linked conversation turns."""
    try:
        manifest = convert_qwen_trace(
            input_path, output_path=output_path, expected_sha256=expected_sha256
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"trace={output_path} requests={manifest['request_count']} block_size_tokens=16")


def convert_qwen_trace(
    input_path: Path, *, output_path: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    """Convert one capture, keeping independently hashed blocks in its namespace."""

    def convert(source_sha: str) -> tuple[Iterator[dict[str, Any]], dict[str, Any]]:
        roots = _conversation_roots(input_path)
        namespace = scope_key("qwen-bailian", source_sha)
        return _records(input_path, roots, namespace), {
            "schema": "prefix-kv-cache-qwen-conversion-v1",
            "publisher": "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon",
            "session_identity": "connected parent_chat_id component within one capture",
            "limitations": [
                "Hashes identify individual 16-token blocks; replay chains the full prefix path.",
                "Hash and conversation namespaces are local to this source capture.",
                "Parent links group observed conversations, not measured users or tenants.",
                "Missing parents anchor components; unobserved ancestors cannot be recovered.",
                "Arrival seconds become milliseconds; decode and latency remain simulator proxies.",
            ],
        }

    return publish_trace(
        input_path, output_path, block_size=16, expected_sha256=expected_sha256, convert=convert
    )


def _conversation_roots(path: Path) -> dict[int, int]:
    parents = {}
    for row in read_jsonl(path):
        if set(row) != _FIELDS:
            raise ValueError("expected native Qwen-Bailian fields only")
        chat = integer(row["chat_id"], "chat_id")
        parent = integer(row["parent_chat_id"], "parent_chat_id", minimum=-1)
        if chat in parents:
            raise ValueError("duplicate chat_id in source capture")
        parents[chat] = parent
    roots: dict[int, int] = {}
    for chat in parents:
        chain = set()
        current = chat
        while current in parents and current not in roots and parents[current] != -1:
            if current in chain:
                raise ValueError("cycle in parent_chat_id links")
            chain.add(current)
            current = parents[current]
        root = roots.get(current, current)
        roots[current] = root
        for child in chain:
            roots[child] = root
    return roots


def _records(path: Path, roots: dict[int, int], namespace: str) -> Iterator[dict[str, Any]]:
    for row in read_jsonl(path):
        length = integer(row["input_length"], "input_length", minimum=1)
        integer(row["turn"], "turn", minimum=1)
        if not isinstance(row["type"], str) or not row["type"]:
            raise ValueError("type must be a nonempty string")
        yield {
            "timestamp_ms": seconds_to_ms(row["timestamp"]),
            "tenant_hash": "qwen:unknown",
            "session_hash": f"qwen:{namespace}:{roots[row['chat_id']]}",
            "priority": 0,
            "request_type": "qwen_trace",
            "prompt_length": length,
            "output_length": integer(row["output_length"], "output_length"),
            "prefix_path": prefix_path(row["hash_ids"], length, 16, namespace),
        }
