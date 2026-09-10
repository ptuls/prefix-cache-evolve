"""Tests for replay-safe LMCache agentic conversion."""

from __future__ import annotations

import json

import pytest

from prefix_cache_evolve.problems.prefix_kv_cache.lmcache_agentic import (
    LmcacheAgenticConversionConfig,
    convert_lmcache_agentic_rows,
)
from prefix_cache_evolve.problems.prefix_kv_cache.trace_replay import load_anonymized_trace

_HASH_KEY = b"test-key-material-with-at-least-32-bytes"


def test_lmcache_conversion_preserves_prefix_growth_and_gaps(tmp_path) -> None:
    first = [_message("system", "private system"), _message("user", "private task")]
    second = first + [
        _message("assistant", "private command"),
        _message("tool", "private result"),
    ]
    rows = [
        _row("session-a", first, output_length=65, pre_gap=0),
        _row("session-a", second, output_length=10, pre_gap=1.25),
        _row("session-b", first, output_length=1, pre_gap=0, model="other-model"),
    ]
    output = tmp_path / "trace.jsonl"
    manifest_path = tmp_path / "manifest.json"

    manifest = convert_lmcache_agentic_rows(
        rows,
        output,
        manifest_path,
        encode=_byte_encoder,
        hash_key=_HASH_KEY,
        source={"kind": "test"},
        config=LmcacheAgenticConversionConfig(
            block_size_tokens=8,
            arrival_bucket_ms=100,
            active_tokens_per_step=64,
            concurrency=2,
            tokenizer_name="test-bytes",
        ),
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert manifest["requests_written"] == 3
    assert manifest["sessions_written"] == 2
    assert manifest["privacy"]["raw_content_written"] is False
    assert "private" not in output.read_text()
    assert {record["request_type"] for record in records} == {"lmcache_other"}
    session_a = [record for record in records if record["output_length"] in {10, 65}]
    assert session_a[1]["timestamp_ms"] - session_a[0]["timestamp_ms"] == 1_450
    reusable_blocks = session_a[0]["prompt_length"] // 8
    assert (
        session_a[1]["prefix_path"][:reusable_blocks]
        == session_a[0]["prefix_path"][:reusable_blocks]
    )
    other_model = next(record for record in records if record["output_length"] == 1)
    assert other_model["prefix_path"] != session_a[0]["prefix_path"]
    assert len(load_anonymized_trace(output, block_size_tokens=8)) == 3


def test_lmcache_conversion_rejects_non_cumulative_session(tmp_path) -> None:
    rows = [
        _row("session-a", [_message("user", "one")]),
        _row("session-a", [_message("user", "different")]),
    ]

    with pytest.raises(ValueError, match="strict cumulative message prefix"):
        convert_lmcache_agentic_rows(
            rows,
            tmp_path / "trace.jsonl",
            tmp_path / "manifest.json",
            encode=_byte_encoder,
            hash_key=_HASH_KEY,
            source={"kind": "test"},
        )


def test_lmcache_conversion_requires_zero_first_gap(tmp_path) -> None:
    with pytest.raises(ValueError, match="first request"):
        convert_lmcache_agentic_rows(
            [_row("session-a", [_message("user", "one")], pre_gap=1)],
            tmp_path / "trace.jsonl",
            tmp_path / "manifest.json",
            encode=_byte_encoder,
            hash_key=_HASH_KEY,
            source={"kind": "test"},
        )


def test_zero_output_sessions_still_wait_for_simulator_release(tmp_path):
    first = [_message("user", "first")]
    rows = [
        _row("a", first, output_length=0),
        _row(
            "a", first + [_message("assistant", "reply"), _message("user", "next")], output_length=0
        ),
        _row("b", first, output_length=0),
    ]
    manifest, requests = _convert(tmp_path, rows, concurrency=1)
    assert [request.arrival_step for request in requests] == [0, 1, 2]
    assert len({request.info.tenant_id for request in requests}) == 1
    assert manifest["models"] == {"test-model": 2}


def test_skipping_invalid_row_discards_whole_session_without_splicing_gaps(tmp_path):
    first = [_message("user", "first")]
    second = first + [_message("assistant", "reply"), _message("user", "next")]
    rows = [
        _row("bad", first),
        _row("bad", second, output_length=-1, pre_gap=3),
        _row("bad", second + [_message("assistant", "reply")], pre_gap=5),
        _row("good", first, model="good-model"),
    ]
    manifest, requests = _convert(tmp_path, rows, skip_invalid=True)
    assert len(requests) == 1
    assert manifest["skipped_rows"] == 3
    assert manifest["invalid_rows"] == 1
    assert manifest["discarded_sessions"] == 1
    assert manifest["models"] == {"good-model": 1}


@pytest.mark.parametrize(
    "updates",
    [
        {"pre_gap": 1},
        {"pre_gap": 1e308},
        {"input": [{"role": "user", "content": float("nan")}]},
    ],
)
def test_skipped_invalid_first_rows_leave_no_session_state(tmp_path, updates):
    bad_row = {**_row("bad", [_message("user", "first")]), **updates}
    manifest, requests = _convert(
        tmp_path, [bad_row, _row("good", [_message("user", "good")])], skip_invalid=True
    )
    assert len(requests) == 1
    assert manifest["sessions_written"] == 1
    assert manifest["models"] == {"test-model": 1}


@pytest.mark.parametrize("session_id", [None, "", 123])
def test_missing_session_cannot_be_safely_skipped(tmp_path, session_id):
    first = [_message("user", "first")]
    anonymous = {**_row("session-a", first), "session_id": session_id}
    rows = [_row("session-a", first), anonymous, _row("good", first)]

    with pytest.raises(ValueError, match="session_id"):
        _convert(tmp_path, rows, skip_invalid=True)

    assert not (tmp_path / "trace.jsonl").exists()
    assert not (tmp_path / "manifest.json").exists()


def _convert(tmp_path, rows, **settings):
    output = tmp_path / "trace.jsonl"
    config = LmcacheAgenticConversionConfig(
        block_size_tokens=8, tokenizer_name="test-bytes", **settings
    )
    manifest = convert_lmcache_agentic_rows(
        rows,
        output,
        tmp_path / "manifest.json",
        encode=_byte_encoder,
        hash_key=_HASH_KEY,
        source={"kind": "test"},
        config=config,
    )
    return manifest, load_anonymized_trace(output, block_size_tokens=8)


def _row(
    session_id: str,
    messages: list[dict],
    *,
    output_length: int = 8,
    pre_gap: float = 0,
    model: str = "test-model",
) -> dict:
    return {
        "session_id": session_id,
        "model": model,
        "input": messages,
        "output_length": output_length,
        "pre_gap": pre_gap,
    }


def _message(role: str, content: str) -> dict:
    return {"role": role, "content": content, "tool_calls": None}


def _byte_encoder(text: str) -> list[int]:
    return list(text.encode())
