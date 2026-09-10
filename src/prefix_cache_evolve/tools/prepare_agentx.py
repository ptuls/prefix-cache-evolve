"""Convert AgentX WEKA sessions into a held-out cache replay source."""

from __future__ import annotations

import hashlib
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


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="AgentX traces.jsonl: one WEKA session object per line.",
)
@click.option(
    "--output", "output_path", type=click.Path(path_type=Path, dir_okay=False), required=True
)
@click.option("--expected-sha256", help="Pin the complete native source file.")
@click.option(
    "--sample-sessions",
    type=click.IntRange(min=1),
    help="Select this many whole sessions by seeded random hash ranking.",
)
@click.option("--sample-seed", type=int, default=0, show_default=True)
@click.option(
    "--max-input-tokens",
    type=click.IntRange(min=1),
    help="Fail if the selected requests exceed this budget; never trim or redraw sessions.",
)
@click.option(
    "--session-spacing-ms",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Modeled spacing between session starts; zero aligns all starts.",
)
def main(
    input_path: Path,
    output_path: Path,
    expected_sha256: str | None,
    session_spacing_ms: int,
    sample_sessions: int | None,
    sample_seed: int,
    max_input_tokens: int | None,
) -> None:
    """Preserve 64-token local hashes and flatten recorded subagent arrivals."""
    try:
        manifest = convert_agentx_trace(
            input_path,
            output_path=output_path,
            expected_sha256=expected_sha256,
            session_spacing_ms=session_spacing_ms,
            sample_sessions=sample_sessions,
            sample_seed=sample_seed,
            max_input_tokens=max_input_tokens,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"trace={output_path} requests={manifest['request_count']} block_size_tokens=64")


def convert_agentx_trace(
    input_path: Path,
    *,
    output_path: Path,
    expected_sha256: str | None = None,
    session_spacing_ms: int = 0,
    sample_sessions: int | None = None,
    sample_seed: int = 0,
    max_input_tokens: int | None = None,
) -> dict[str, Any]:
    """Convert native sessions, retaining whole-session identity across models/branches."""
    integer(session_spacing_ms, "session_spacing_ms")
    if sample_sessions is not None:
        integer(sample_sessions, "sample_sessions", minimum=1)
    if type(sample_seed) is not int:
        raise ValueError("sample_seed must be an integer")
    if max_input_tokens is not None:
        integer(max_input_tokens, "max_input_tokens", minimum=1)
    metadata: dict[str, Any] = {
        "schema": "prefix-kv-cache-agentx-conversion-v1",
        "publisher": "https://inferencex.semianalysis.com/agentx/methodology",
        "recommended_split": "hidden",
        "session_spacing_ms": session_spacing_ms,
        "subagent_request_count": 0,
        "subagent_group_count": 0,
        "limitations": [
            "Native input lengths are block-count proxies, not exact provider tokenizer counts.",
            "64-token hashes are session/model scoped; cross-session sharing is unknown.",
            "Subagent t values already use the session clock; wrapper times are not added again.",
            "Cross-session arrivals are unknown; starts use modeled spacing in source order.",
            "Recorded offsets are preserved; completion-dependent spawn/join is not modeled.",
            "Provider templates are approximated upstream; decode timing is modeled.",
            "This cache-policy replay is not the official AgentX serving benchmark.",
        ],
    }

    def convert(_: str) -> tuple[Iterator[dict[str, Any]], dict[str, Any]]:
        selected = None
        if sample_sessions is not None:
            ids = [session["id"] for session in _sessions(input_path)]
            if sample_sessions > len(ids):
                raise ValueError("sample_sessions exceeds the available source sessions")
            ranked = sorted(
                ids,
                key=lambda value: (
                    hashlib.sha256(
                        f"agentx-session-sample-v1\0{sample_seed}\0{value}".encode()
                    ).digest(),
                    value,
                ),
            )
            selected = set(ranked[:sample_sessions])
            metadata["sampling"] = {
                "algorithm": "agentx-session-sample-v1",
                "seed": sample_seed,
                "source_session_count": len(ids),
                "selected_session_ids": sorted(selected),
                "selected_session_count": sample_sessions,
                "max_input_tokens": max_input_tokens,
            }
            metadata["limitations"].append(
                "Sampling uses no policy scores; small session samples have high variance."
            )
        return (
            _records(input_path, session_spacing_ms, metadata, selected, max_input_tokens),
            metadata,
        )

    return publish_trace(
        input_path,
        output_path,
        block_size=64,
        expected_sha256=expected_sha256,
        convert=convert,
    )


def _requests(
    items: Any, metadata: dict[str, Any], *, nested: bool = False
) -> Iterator[dict[str, Any]]:
    if not isinstance(items, list) or not items:
        raise ValueError("requests must be a nonempty list")
    for row in items:
        if not isinstance(row, dict):
            raise ValueError("request must be an object")
        if row.get("type") == "subagent":
            metadata["subagent_group_count"] += 1
            seconds_to_ms(row["t"])
            yield from _requests(row["requests"], metadata, nested=True)
        else:
            if row.get("type") not in {"s", "n"}:
                raise ValueError("unsupported AgentX request type")
            if nested:
                metadata["subagent_request_count"] += 1
            yield row


def _sessions(path: Path) -> Iterator[dict[str, Any]]:
    seen = set()
    for session in read_jsonl(path):
        session_id = session.get("id")
        if not isinstance(session_id, str) or not session_id.strip() or session_id in seen:
            raise ValueError("AgentX session id must be a unique nonempty string")
        seen.add(session_id)
        if type(session.get("block_size")) is not int or session["block_size"] != 64:
            raise ValueError("AgentX requires native block_size 64")
        if session.get("hash_id_scope") != "local":
            raise ValueError("AgentX requires hash_id_scope local")
        yield session


def _records(
    path: Path,
    spacing: int,
    metadata: dict[str, Any],
    selected: set[str] | None,
    max_input_tokens: int | None,
) -> Iterator[dict[str, Any]]:
    total_input_tokens = 0
    for session_index, session in enumerate(_sessions(path)):
        session_id = session["id"]
        if selected is not None and session_id not in selected:
            continue
        identities: dict[tuple[str, int], tuple[int | None, int]] = {}
        for row in _requests(session.get("requests"), metadata):
            model = row.get("model")
            if not isinstance(model, str) or not model.strip():
                raise ValueError("each AgentX request must identify its model")
            length = integer(row["in"], "in", minimum=1)
            total_input_tokens += length
            if max_input_tokens is not None and total_input_tokens > max_input_tokens:
                raise ValueError(
                    "selected sessions exceed max_input_tokens; no partial sample was published"
                )
            namespace = scope_key("agentx", session_id, model)
            hashes = row["hash_ids"]
            path_ids = prefix_path(hashes, length, 64, namespace)
            parent = None
            for depth, block in enumerate(hashes):
                identity = (parent, min(64, length - depth * 64))
                if identities.setdefault((model, block), identity) != identity:
                    raise ValueError("a cumulative AgentX hash has conflicting parents or lengths")
                parent = block
            yield {
                "timestamp_ms": session_index * spacing + seconds_to_ms(row["t"]),
                "tenant_hash": "agentx:unknown",
                "session_hash": f"agentx:{scope_key(session_id)}",
                "priority": 0,
                "request_type": "agentx_trace",
                "prompt_length": length,
                "output_length": integer(row["out"], "out"),
                "prefix_path": path_ids,
            }
