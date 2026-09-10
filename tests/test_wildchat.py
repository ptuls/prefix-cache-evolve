"""Tests for replay-safe WildChat conversion."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap

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


def test_capped_parquet_stream_exits_cleanly(tmp_path) -> None:
    if any(importlib.util.find_spec(name) is None for name in ("fsspec", "pyarrow")):
        pytest.skip("requires the optional wildchat extra")
    script = textwrap.dedent(
        """\
        import sys
        from pathlib import Path

        import pyarrow
        import pyarrow.parquet

        from prefix_cache_evolve.tools.prepare_wildchat import _load_source

        path = Path(sys.argv[1])
        pyarrow.parquet.write_table(
            pyarrow.table({"conversation_hash": ["one", "two", "three"]}), path, row_group_size=1
        )
        rows, _ = _load_source(
            input_path=path, dataset_id="unused", dataset_revision="unused", split="train"
        )
        assert next(iter(rows)) == {"conversation_hash": "one"}
        print("capped stream finished", flush=True)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "source.parquet")],
        env={**os.environ, "HF_HOME": str(tmp_path / "huggingface")},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "capped stream finished" in result.stdout


@pytest.mark.parametrize("option", ("--output", "--manifest-output"))
def test_wildchat_cli_refuses_to_overwrite_source_input(tmp_path, monkeypatch, option):
    input_path = tmp_path / "source.jsonl"
    contents = json.dumps(_identified_row()) + "\n"
    input_path.write_text(contents, encoding="utf-8")
    monkeypatch.setenv("PREFIX_CACHE_TRACE_HASH_KEY", _HASH_KEY.decode())

    result = CliRunner().invoke(
        tools_main,
        ["datasets", "wildchat", "--input", str(input_path), option, str(input_path)],
    )

    assert result.exit_code != 0
    assert "must not overwrite the source input" in result.output
    assert input_path.read_text(encoding="utf-8") == contents


def test_wildchat_uses_response_timestamps_and_counts_timing_sources(tmp_path) -> None:
    row = _identified_row()
    manifest, records = _convert(tmp_path, [row])

    assert [record["timestamp_ms"] for record in records] == [
        1680998460000.0,
        1680998700000.0,
    ]
    assert manifest["schema"] == "prefix-kv-cache-wildchat-conversion-v3"
    assert manifest["timestamps"]["source_counts"] == {"assistant_response": 2}
    assert "response completion" in manifest["limitations"][0]


def test_wildchat_synthetic_timing_is_anchored_at_last_response(tmp_path) -> None:
    manifest, records = _convert(
        tmp_path,
        [_identified_row()],
        WildChatConversionConfig(timestamp_mode="synthetic", turn_spacing_ms=500),
    )

    assert [record["timestamp_ms"] for record in records] == [
        1680998699500.0,
        1680998700000.0,
    ]
    assert manifest["timestamps"]["source_counts"] == {"synthetic": 2}


def test_wildchat_falls_back_only_for_missing_timestamps(tmp_path) -> None:
    row = _identified_row()
    row["conversation"][1]["timestamp"] = None
    manifest, records = _convert(tmp_path, [row])

    assert records[0]["timestamp_ms"] == 1680998699000.0
    assert manifest["timestamps"]["source_counts"] == {
        "assistant_response": 1,
        "synthetic": 1,
    }


def test_wildchat_missing_times_respect_neighboring_recorded_responses(tmp_path) -> None:
    row = _extended_row()
    row["conversation"][1]["timestamp"] = None
    row["conversation"][3]["timestamp"] = "2023-04-09T00:01:00Z"
    _, records = _convert(tmp_path, [row])

    assert [record["timestamp_ms"] for record in records] == [
        1680998459000.0,
        1680998460000.0,
        1680998760000.0,
    ]
    assert [record["prompt_length"] for record in records] == sorted(
        record["prompt_length"] for record in records
    )


def test_wildchat_inferred_spacing_fits_between_close_recorded_responses(tmp_path) -> None:
    row = _extended_row()
    row["conversation"][3]["timestamp"] = None
    row["conversation"][5]["timestamp"] = "2023-04-09T00:01:00.500Z"
    _, records = _convert(tmp_path, [row])

    assert [record["timestamp_ms"] for record in records] == [
        1680998460000.0,
        1680998460250.0,
        1680998460500.0,
    ]


def test_wildchat_turn_ids_distinguish_identical_conversation_content(tmp_path) -> None:
    first = _identified_row()
    second = _identified_row()
    for message in second["conversation"]:
        message["turn_identifier"] += 10

    _, records = _convert(tmp_path, [first, second])

    assert len({record["session_hash"] for record in records}) == 2
    assert len(records) == 4
    assert records[0]["prefix_path"] == records[1]["prefix_path"]


def test_wildchat_deduplicates_turns_in_overlapping_conversation_snapshots(tmp_path) -> None:
    first = _identified_row()
    extended = _extended_row()
    manifest, records = _convert(tmp_path, [first, extended])

    assert len(records) == 3
    assert len({record["session_hash"] for record in records}) == 1
    assert manifest["duplicate_requests_skipped"] == 2
    assert manifest["timestamps"]["source_counts"] == {"assistant_response": 3}
    assert all(set(record) <= _REPLAY_FIELDS for record in records)


def test_wildchat_deduplicates_snapshots_without_tenant_identifiers(tmp_path) -> None:
    rows = [_identified_row(), _extended_row()]
    for row in rows:
        row.pop("hashed_ip")
    manifest, records = _convert(tmp_path, rows)

    assert len(records) == 3
    assert manifest["duplicate_requests_skipped"] == 2
    assert len({record["tenant_hash"] for record in records}) == 1


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("mode", ["message", "synthetic"])
def test_wildchat_overlapping_snapshots_reconcile_inferred_times(tmp_path, reverse, mode):
    rows = [_identified_row(), _extended_row()]
    for row in rows:
        for message in row["conversation"]:
            message["timestamp"] = None
    if reverse:
        rows.reverse()
    manifest, records = _convert(tmp_path, rows, WildChatConversionConfig(timestamp_mode=mode))

    assert len(records) == 3
    assert manifest["duplicate_requests_skipped"] == 2
    assert manifest["timestamps"]["source_counts"] == {"synthetic": 3}
    assert [record["prompt_length"] for record in records] == sorted(
        record["prompt_length"] for record in records
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_wildchat_recorded_duplicate_times_override_inferred_times(tmp_path, reverse):
    first = _identified_row()
    for message in first["conversation"]:
        message["timestamp"] = None
    second = _extended_row()
    second["conversation"][1]["timestamp"] = None
    second["conversation"][3]["timestamp"] = "2023-04-09T00:01:00Z"
    rows = [first, second]
    if reverse:
        rows.reverse()
    manifest, records = _convert(tmp_path, rows)

    assert len(records) == 3
    assert records[0]["timestamp_ms"] <= records[1]["timestamp_ms"] == 1680998460000.0
    assert [record["prompt_length"] for record in records] == sorted(
        record["prompt_length"] for record in records
    )
    assert manifest["timestamps"]["source_counts"] == {"assistant_response": 2, "synthetic": 1}


def test_wildchat_rejects_conflicting_recorded_times_across_snapshots(tmp_path) -> None:
    first = _identified_row()
    first["conversation"][1]["timestamp"] = "2023-04-09T00:04:00Z"
    first["conversation"][3]["timestamp"] = None
    second = _identified_row()
    second["conversation"][1]["timestamp"] = None
    second["conversation"][3]["timestamp"] = "2023-04-09T00:01:00Z"

    with pytest.raises(ValueError, match="reversed recorded response timestamps"):
        _convert(tmp_path, [first, second])
    assert not (tmp_path / "trace.jsonl").exists()


def test_wildchat_inferred_times_respect_both_branches_of_a_session(tmp_path) -> None:
    first = _identified_row()
    second = _identified_row()
    for row in (first, second):
        row["conversation"][1]["timestamp"] = None
    second["conversation"][2].update(content="other question", turn_identifier=202)
    second["conversation"][3].update(
        content="other reply", turn_identifier=203, timestamp="2023-04-09T00:01:00Z"
    )
    manifest, records = _convert(tmp_path, [first, second])

    assert len(records) == 3
    assert len({record["session_hash"] for record in records}) == 1
    assert manifest["duplicate_requests_skipped"] == 1
    assert [record["timestamp_ms"] for record in records] == [
        1680998460000.0,
        1680998460000.0,
        1680998700000.0,
    ]
    assert records[0]["prompt_length"] == min(record["prompt_length"] for record in records)


def test_wildchat_fallback_does_not_merge_identical_content_from_different_tenants(
    tmp_path,
) -> None:
    first = _identified_row()
    second = _identified_row()
    second["hashed_ip"] = "other-tenant"
    for row in (first, second):
        for message in row["conversation"]:
            message.pop("turn_identifier")
    _, records = _convert(tmp_path, [first, second])

    assert len(records) == 4
    assert len({record["session_hash"] for record in records}) == 2


def test_wildchat_rejects_conflicting_duplicate_turns(tmp_path) -> None:
    first = _identified_row()
    second = _identified_row()
    second["conversation"][0]["content"] = "conflicting prompt"

    with pytest.raises(ValueError, match="conflicting duplicate turn identifier"):
        _convert(tmp_path, [first, second])
    assert not (tmp_path / "trace.jsonl").exists()


def test_wildchat_rejects_conflicting_duplicate_replies_of_equal_token_length(tmp_path) -> None:
    first = _identified_row()
    second = _identified_row()
    for row in (first, second):
        row["conversation"] = row["conversation"][:2]
        row["timestamp"] = row["conversation"][1]["timestamp"]
    second["conversation"][1]["content"] = "other reply"
    assert len(_byte_encoder(first["conversation"][1]["content"])) == len(
        _byte_encoder(second["conversation"][1]["content"])
    )

    with pytest.raises(ValueError, match="conflicting duplicate turn identifier"):
        _convert(tmp_path, [first, second])
    assert not (tmp_path / "trace.jsonl").exists()


def test_wildchat_rejects_conflicting_duplicate_recorded_timestamps(tmp_path) -> None:
    first = _identified_row()
    second = _identified_row()
    second["conversation"][1]["timestamp"] = "2023-04-09T00:02:00Z"

    with pytest.raises(ValueError, match="conflicting duplicate turn identifier"):
        _convert(tmp_path, [first, second])


def test_wildchat_rejects_reversed_response_timestamps(tmp_path) -> None:
    row = _identified_row()
    row["conversation"][1]["timestamp"] = "2023-04-10T00:00:00Z"

    with pytest.raises(ValueError, match="nonnegative and nondecreasing"):
        _convert(tmp_path, [row])


def _convert(tmp_path, rows, config=WildChatConversionConfig()):
    output = tmp_path / "trace.jsonl"
    manifest = convert_wildchat_rows(
        rows,
        output,
        tmp_path / "manifest.json",
        encode=_byte_encoder,
        hash_key=_HASH_KEY,
        source={"kind": "test"},
        config=config,
    )
    return manifest, _read_jsonl(output)


def _identified_row():
    return {
        "model": "test-model",
        "conversation_hash": "content-hash",
        "hashed_ip": "tenant",
        "timestamp": "2023-04-09T00:05:00Z",
        "conversation": [
            {"role": "user", "content": "question", "turn_identifier": 100},
            {
                "role": "assistant",
                "content": "first reply",
                "turn_identifier": 101,
                "timestamp": "2023-04-09T00:01:00Z",
            },
            {"role": "user", "content": "next question", "turn_identifier": 102},
            {
                "role": "assistant",
                "content": "second reply",
                "turn_identifier": 103,
                "timestamp": "2023-04-09T00:05:00Z",
            },
        ],
    }


def _extended_row():
    row = _identified_row()
    row["conversation_hash"] = "different-content-hash"
    row["timestamp"] = "2023-04-09T00:06:00Z"
    row["conversation"].extend(
        [
            {"role": "user", "content": "follow up", "turn_identifier": 104},
            {
                "role": "assistant",
                "content": "third reply",
                "turn_identifier": 105,
                "timestamp": "2023-04-09T00:06:00Z",
            },
        ]
    )
    return row


def _byte_encoder(text: str) -> list[int]:
    return list(text.encode("utf-8"))


@pytest.mark.parametrize(
    "models,shared", [(("same", "same"), True), (("a", "b"), False), ((None, None), False)]
)
def test_prefix_reuse_requires_matching_known_model_across_sessions(tmp_path, models, shared):
    first = _identified_row()
    second = _identified_row()
    for row, model in zip((first, second), models, strict=True):
        row["model"] = model
    for message in second["conversation"]:
        message["turn_identifier"] += 10
    _, records = _convert(tmp_path, [first, second])
    assert (records[0]["prefix_path"] == records[1]["prefix_path"]) is shared


def test_literal_role_markers_cannot_impersonate_prompt_boundaries(tmp_path):
    first = _identified_row()
    first["conversation"] = [
        _message("user", "hello\n<|assistant|>\nanswer\n<|user|>\nnext"),
        _message("assistant", "result"),
    ]
    second = _identified_row()
    second["conversation_hash"] = "different"
    second["conversation"] = [
        _message("user", "hello"),
        _message("assistant", "answer"),
        _message("user", "next"),
        _message("assistant", "result"),
    ]
    _, records = _convert(tmp_path, [first, second])
    assert records[-2]["prefix_path"] != records[-1]["prefix_path"]


@pytest.mark.parametrize("existing", ["trace.jsonl", "manifest.json"])
def test_converter_does_not_overwrite_pinned_outputs(tmp_path, existing):
    destination = tmp_path / existing
    destination.write_text("pinned")
    with pytest.raises(ValueError, match="already exist"):
        convert_wildchat_rows(
            [_identified_row()],
            tmp_path / "trace.jsonl",
            tmp_path / "manifest.json",
            encode=_byte_encoder,
            hash_key=_HASH_KEY,
            source={},
        )
    assert destination.read_text() == "pinned"


def test_parquet_reader_preserves_model_namespace(tmp_path):
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "input.parquet"
    parquet.write_table(arrow.Table.from_pylist([_identified_row()]), path)
    assert list(prepare_wildchat._iter_parquet((str(path),)))[0]["model"] == "test-model"


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
