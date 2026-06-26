"""Tests for replay-safe WildChat conversion."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from prefix_cache_evolve.problems.prefix_kv_cache.trace_replay import (
    load_anonymized_trace,
)
from prefix_cache_evolve.problems.prefix_kv_cache.wildchat import (
    WildChatConversionConfig,
    convert_wildchat_rows,
)
from prefix_cache_evolve.tools import prepare_wildchat
from prefix_cache_evolve.tools.cli import main as tools_main

_HASH_KEY = b"test-key-material-with-at-least-32-bytes"


def test_wildchat_conversion_writes_only_sorted_opaque_metadata(tmp_path) -> None:
    output_path = tmp_path / "wildchat.jsonl"
    manifest_path = tmp_path / "wildchat.manifest.json"
    rows = [
        _row(
            conversation_hash="later",
            timestamp="2023-04-09T00:01:00Z",
            hashed_ip="tenant-a",
            conversation=[
                _message("user", "first private question"),
                _message("assistant", "first private answer"),
                _message("user", "second private question"),
                _message("assistant", "second private answer"),
            ],
        ),
        _row(
            conversation_hash="earlier",
            timestamp="2023-04-09T00:00:00Z",
            hashed_ip="tenant-b",
            conversation=[
                _message("user", "another private question"),
                _message("assistant", "another private answer"),
            ],
        ),
    ]

    manifest = convert_wildchat_rows(
        rows,
        output_path,
        manifest_path,
        encode=_byte_encoder,
        hash_key=_HASH_KEY,
        source={"kind": "test"},
        config=WildChatConversionConfig(
            block_size_tokens=8,
            turn_spacing_ms=500,
            tokenizer_name="test-bytes",
        ),
    )

    records = _read_jsonl(output_path)
    assert [record["timestamp_ms"] for record in records] == sorted(
        record["timestamp_ms"] for record in records
    )
    assert manifest["conversations_converted"] == 2
    assert manifest["requests_written"] == 3
    assert manifest["privacy"]["raw_content_written"] is False
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest

    serialized = output_path.read_text(encoding="utf-8")
    assert "private" not in serialized
    assert "tenant-a" not in serialized
    assert "later" not in serialized
    assert all(set(record) <= _REPLAY_FIELDS for record in records)

    same_session = [
        record for record in records if record["session_hash"] == records[-1]["session_hash"]
    ]
    assert len(same_session) == 2
    first, second = same_session
    reusable_full_blocks = first["prompt_length"] // 8
    assert (
        second["prefix_path"][:reusable_full_blocks] == first["prefix_path"][:reusable_full_blocks]
    )

    requests = load_anonymized_trace(output_path, block_size_tokens=8)
    assert len(requests) == 3
    assert all(request.info.prompt_tokens == () for request in requests)


def test_wildchat_conversion_is_strict_by_default(tmp_path) -> None:
    rows = [
        _row(
            conversation_hash="invalid",
            timestamp="2023-04-09T00:00:00Z",
            hashed_ip="tenant-a",
            conversation=[_message("user", "no assistant response")],
        )
    ]

    with pytest.raises(ValueError, match="produced no replay requests"):
        convert_wildchat_rows(
            rows,
            tmp_path / "trace.jsonl",
            tmp_path / "manifest.json",
            encode=_byte_encoder,
            hash_key=_HASH_KEY,
            source={"kind": "test"},
        )


def test_wildchat_conversion_can_skip_invalid_rows(tmp_path) -> None:
    rows = [
        {"conversation_hash": "broken"},
        _row(
            conversation_hash="valid",
            timestamp="2023-04-09T00:00:00Z",
            hashed_ip="tenant-a",
            conversation=[
                _message("user", "question"),
                _message("assistant", "answer"),
            ],
        ),
    ]

    manifest = convert_wildchat_rows(
        rows,
        tmp_path / "trace.jsonl",
        tmp_path / "manifest.json",
        encode=_byte_encoder,
        hash_key=_HASH_KEY,
        source={"kind": "test"},
        config=WildChatConversionConfig(skip_invalid=True),
    )

    assert manifest["skipped_conversations"] == 1
    assert manifest["skipped_reasons"] == {"timestamp must be an ISO-8601 string or datetime": 1}


def test_wildchat_cli_converts_local_jsonl_without_optional_dataset_loader(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "wildchat-source.jsonl"
    output_path = tmp_path / "wildchat-trace.jsonl"
    input_path.write_text(
        json.dumps(
            _row(
                conversation_hash="conversation-a",
                timestamp="2023-04-09T00:00:00Z",
                hashed_ip="tenant-a",
                conversation=[
                    _message("user", "question"),
                    _message("assistant", "answer"),
                ],
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PREFIX_CACHE_TRACE_HASH_KEY", _HASH_KEY.decode())
    monkeypatch.setattr(prepare_wildchat, "_build_encoder", lambda _: _byte_encoder)

    result = CliRunner().invoke(
        tools_main,
        [
            "datasets",
            "wildchat",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--encoding",
            "test-bytes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"wildchat_trace={output_path}" in result.output
    assert output_path.is_file()
    assert output_path.with_suffix(".jsonl.manifest.json").is_file()
    assert len(_read_jsonl(output_path)) == 1


def test_wildchat_cli_requires_private_hash_key(monkeypatch) -> None:
    monkeypatch.delenv("PREFIX_CACHE_TRACE_HASH_KEY", raising=False)

    result = CliRunner().invoke(tools_main, ["datasets", "wildchat"])

    assert result.exit_code != 0
    assert "PREFIX_CACHE_TRACE_HASH_KEY is not set" in result.output


def _byte_encoder(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _row(
    *,
    conversation_hash: str,
    timestamp: str,
    hashed_ip: str,
    conversation: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "conversation_hash": conversation_hash,
        "timestamp": timestamp,
        "hashed_ip": hashed_ip,
        "conversation": conversation,
    }


def _message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


_REPLAY_FIELDS = {
    "timestamp_ms",
    "tenant_hash",
    "session_hash",
    "request_type",
    "priority",
    "prompt_length",
    "output_length",
    "prefix_path",
}
