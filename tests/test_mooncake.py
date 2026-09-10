"""Tests for conversion of native Mooncake production trace metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from prefix_cache_evolve.problems.prefix_kv_cache.runner import main as replay_main
from prefix_cache_evolve.problems.prefix_kv_cache.trace_replay import load_anonymized_trace
from prefix_cache_evolve.tools.cli import main as tools_main
from prefix_cache_evolve.tools.prepare_mooncake import convert_mooncake_trace


def _write_source(path: Path, records: list) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _record(**updates) -> dict:
    return {
        "timestamp": 0,
        "input_length": 513,
        "output_length": 10,
        "hash_ids": [1, 2],
        **updates,
    }


def test_mooncake_cli_preserves_prefixes_timing_and_missing_metadata(tmp_path):
    source = _write_source(
        tmp_path / "native.jsonl",
        [
            _record(timestamp=5),
            _record(timestamp=105),
            _record(timestamp=505, input_length=1_026, hash_ids=[1, 3, 4]),
        ],
    )
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "replay.jsonl"
    result = CliRunner().invoke(
        tools_main,
        [
            "datasets",
            "mooncake",
            "--input",
            str(source),
            "--output",
            str(output),
            "--expected-sha256",
            source_sha,
        ],
    )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["timestamp_ms"] for row in records] == [5, 105, 505]
    assert [row["prefix_path"] for row in records] == [[1, 2], [1, 2], [1, 3, 4]]
    assert {row["tenant_hash"] for row in records} == {"mooncake:unknown"}
    assert {row["session_hash"] for row in records} == {None}
    assert all(row["priority"] == 0 and "predicted_output_length" not in row for row in records)

    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["source"]["sha256"] == source_sha
    assert manifest["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest["request_count"] == 3
    assert manifest["input_tokens"] == 2_052
    assert manifest["output_tokens"] == 30
    assert manifest["unique_prefix_hashes"] == 4
    assert manifest["block_size_tokens"] == 512
    assert manifest["session_count"] is None
    assert manifest["tenant_count"] is None

    requests = load_anonymized_trace(output, block_size_tokens=512)
    assert all(request.info.session_id is None for request in requests)
    assert [request.arrival_step for request in requests] == [0, 1, 5]
    assert all(request.info.prompt_tokens == () for request in requests)
    assert requests[0].prompt_tokens == requests[1].prompt_tokens
    assert requests[0].prompt_tokens[:512] == requests[2].prompt_tokens[:512]
    assert requests[0].prompt_tokens[512:] != requests[2].prompt_tokens[512:513]
    with pytest.raises(ValueError, match="prefix_path depth"):
        load_anonymized_trace(output, block_size_tokens=16)


@pytest.mark.parametrize(
    "records",
    [
        [],
        [[]],
        [_record(messages=[{"content": "not a metadata trace"}])],
        [_record(input_length=True)],
        [_record(input_length=0)],
        [_record(hash_ids=[1])],
        [_record(hash_ids=[True, 2])],
        [_record(hash_ids=[-1, 2])],
        [_record(timestamp="0")],
        [_record(timestamp=True)],
        [_record(timestamp=-1)],
        [_record(timestamp=float("nan"))],
        [_record(timestamp=float("inf"))],
        [_record(output_length=-1)],
        [_record(output_length=True)],
        [_record(timestamp=10), _record(timestamp=0)],
        [_record(), _record(hash_ids=[3, 2])],
        [_record(), _record(input_length=514)],
    ],
)
def test_invalid_mooncake_trace_does_not_publish_partial_output(tmp_path, records):
    source = _write_source(tmp_path / "native.jsonl", records)
    output = tmp_path / "replay.jsonl"
    with pytest.raises(ValueError):
        convert_mooncake_trace(source, output_path=output)
    assert not output.exists()
    assert not output.with_suffix(".jsonl.manifest.json").exists()


def test_mooncake_checksum_covers_complete_source(tmp_path):
    source = _write_source(tmp_path / "native.jsonl", [_record()])
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_bytes(source.read_bytes() + b"\n")
    output = tmp_path / "replay.jsonl"
    with pytest.raises(ValueError, match="source SHA-256"):
        convert_mooncake_trace(source, output_path=output, expected_sha256=expected_sha)
    assert not output.exists()


@pytest.mark.parametrize("existing", ["input", "output", "manifest"])
def test_mooncake_conversion_does_not_overwrite_existing_files(tmp_path, existing):
    source = _write_source(tmp_path / "native.jsonl", [_record()])
    output = source if existing == "input" else tmp_path / "replay.jsonl"
    destination = output.with_suffix(".jsonl.manifest.json") if existing == "manifest" else output
    if existing != "input":
        destination.write_text("existing content", encoding="utf-8")
    original = destination.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        convert_mooncake_trace(source, output_path=output)
    assert destination.read_bytes() == original


@pytest.mark.parametrize("group_by", ["session", "tenant"])
def test_mooncake_unknown_identities_cannot_be_used_as_disjoint_groups(tmp_path, group_by):
    source = _write_source(tmp_path / "native.jsonl", [_record(timestamp=t) for t in range(6)])
    output = tmp_path / "replay.jsonl"
    convert_mooncake_trace(source, output_path=output)
    panel = tmp_path / "panel"
    result = CliRunner().invoke(
        tools_main,
        [
            "datasets",
            "trace-panel",
            "--trace",
            str(output),
            "--output-dir",
            str(panel),
            "--block-size-tokens",
            "512",
            "--group-by",
            group_by,
        ],
    )
    assert result.exit_code != 0
    assert "require known identities" in result.output
    assert not panel.exists()


def test_trace_cli_evaluates_only_selected_deployable_baselines(tmp_path):
    source = _write_source(tmp_path / "native.jsonl", [_record(timestamp=t) for t in range(3)])
    trace = tmp_path / "replay.jsonl"
    convert_mooncake_trace(source, output_path=trace)
    output = tmp_path / "results.json"
    result = CliRunner().invoke(
        replay_main,
        [
            "--replay-trace",
            str(trace),
            "--block-size-tokens",
            "512",
            "--capacity-blocks",
            "4",
            "--capacity-sweep-blocks",
            "4",
            "--trace-baseline",
            "lru",
            "--trace-baseline",
            "lru",
            "--trace-baseline",
            "vllm_apc",
            "--trace-output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert set(payload["results"]) == {"lru", "vllm_apc"}
    assert result.output.count("replaying_policy=lru ") == 1
    assert all(policy["success"] for policy in payload["results"].values())


@pytest.mark.parametrize(
    "args, message",
    [
        (["--trace-baseline", "lru"], "requires --replay-trace"),
        (["--trace-baseline", "not-a-baseline"], "Invalid value for '--trace-baseline'"),
    ],
)
def test_trace_baseline_selection_requires_valid_replay_options(args, message):
    result = CliRunner().invoke(replay_main, args)
    assert result.exit_code == 2
    assert message in result.output
