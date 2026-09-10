"""Public serving trace semantics and independent holdout integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from prefix_cache_evolve.evaluators.panels import prepare_workloads
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    PrefixKVCacheEvaluator,
    baseline_lru_blocks,
)
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import file_sha256
from prefix_cache_evolve.problems.prefix_kv_cache.trace_replay import load_anonymized_trace
from prefix_cache_evolve.tools.attach_holdout import attach_holdout
from prefix_cache_evolve.tools.cli import main
from prefix_cache_evolve.tools.hashed_trace import scope_key
from prefix_cache_evolve.tools.prepare_agentx import convert_agentx_trace
from prefix_cache_evolve.tools.prepare_qwen import convert_qwen_trace


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _qwen(chat=0, **updates):
    return {
        "chat_id": chat,
        "parent_chat_id": -1,
        "timestamp": chat / 10,
        "input_length": 32,
        "output_length": 2,
        "hash_ids": [7, 8],
        "type": "api",
        "turn": 1,
        **updates,
    }


def _agentx(session="a", **updates):
    return {
        "id": session,
        "models": ["opus"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [_request()],
        **updates,
    }


def _request(t=0, **updates):
    return {
        "t": t,
        "type": "s",
        "model": "opus",
        "in": 128,
        "out": 2,
        "hash_ids": [0, 1],
        **updates,
    }


def test_qwen_reconstructs_prefixes_and_groups_parent_components(tmp_path):
    source = _write(
        tmp_path / "source.jsonl",
        [
            _qwen(2, parent_chat_id=1, timestamp=0.5),
            _qwen(0),
            _qwen(1, parent_chat_id=0),
            _qwen(3, hash_ids=[9, 8]),
            _qwen(4, parent_chat_id=99, input_length=17),
            _qwen(5, parent_chat_id=99),
        ],
    )
    output = tmp_path / "trace.jsonl"
    manifest = convert_qwen_trace(source, output_path=output, expected_sha256=file_sha256(source))
    requests = load_anonymized_trace(output, block_size_tokens=16)
    assert [r.arrival_step for r in requests] == [0, 1, 3, 4, 5, 5]
    assert requests[0].prompt_tokens == requests[1].prompt_tokens == requests[-2].prompt_tokens
    assert requests[2].prompt_tokens[16:] != requests[0].prompt_tokens[16:]
    assert requests[3].prompt_tokens == requests[-1].prompt_tokens[:17]
    assert (
        requests[0].info.session_id == requests[1].info.session_id == requests[-2].info.session_id
    )
    assert requests[3].info.session_id == requests[-1].info.session_id
    assert manifest["session_count"] == 3
    assert manifest["output_sha256"] == file_sha256(output)
    assert all(r.info.predicted_output_length is None for r in requests)


def test_agentx_preserves_nested_clocks_and_scopes_sharing_by_session_and_model(tmp_path):
    source = _write(
        tmp_path / "source.jsonl",
        [
            _agentx(
                requests=[
                    _request(2),
                    {
                        "type": "subagent",
                        "t": 5,
                        "requests": [
                            _request(6),
                            _request(7, model="haiku"),
                        ],
                    },
                ]
            ),
            _agentx("b", requests=[_request(0)]),
        ],
    )
    output = tmp_path / "trace.jsonl"
    result = CliRunner().invoke(
        main,
        [
            "datasets",
            "agentx",
            "--input",
            str(source),
            "--output",
            str(output),
            "--session-spacing-ms",
            "3000",
        ],
    )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["timestamp_ms"] for r in records] == [2000, 3000, 6000, 7000]
    requests = load_anonymized_trace(output, block_size_tokens=64)
    assert requests[0].prompt_tokens == requests[2].prompt_tokens
    assert requests[0].prompt_tokens != requests[1].prompt_tokens
    assert requests[0].prompt_tokens != requests[3].prompt_tokens
    assert requests[0].info.session_id == requests[2].info.session_id == requests[3].info.session_id
    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["subagent_request_count"] == 2
    assert manifest["subagent_group_count"] == 1
    assert manifest["recommended_split"] == "hidden"


@pytest.mark.parametrize(
    "convert,rows,message",
    [
        (convert_qwen_trace, [_qwen(0, parent_chat_id=1), _qwen(1, parent_chat_id=0)], "cycle"),
        (convert_qwen_trace, [_qwen(hash_ids=[0])], "depth"),
        (convert_agentx_trace, [_agentx(hash_id_scope="global")], "local"),
        (
            convert_agentx_trace,
            [_agentx(requests=[_request(), _request(1, hash_ids=[2, 1])])],
            "conflicting parents",
        ),
        (convert_agentx_trace, [_agentx(requests=[_request(t=float("nan"))])], "timestamp"),
    ],
)
def test_invalid_native_data_is_not_published(tmp_path, convert, rows, message):
    source = _write(tmp_path / "source.jsonl", rows)
    output = tmp_path / "trace.jsonl"
    with pytest.raises(ValueError, match=message):
        convert(source, output_path=output)
    assert not output.exists()
    assert not output.with_suffix(".jsonl.manifest.json").exists()


def test_converter_checks_full_source_and_refuses_overwrite(tmp_path):
    source = _write(tmp_path / "source.jsonl", [_qwen()])
    sha = file_sha256(source)
    source.write_text(source.read_text() + "\n")
    output = tmp_path / "trace.jsonl"
    with pytest.raises(ValueError, match="SHA-256"):
        convert_qwen_trace(source, output_path=output, expected_sha256=sha)
    assert not output.exists()
    output.write_text("keep this")
    with pytest.raises(ValueError, match="already exists"):
        convert_qwen_trace(source, output_path=output)
    assert output.read_text() == "keep this"


def test_agentx_sampling_is_reproducible_and_keeps_complete_sessions(tmp_path):
    source = _write(
        tmp_path / "source.jsonl",
        [
            _agentx(
                session,
                requests=[
                    _request(),
                    {
                        "type": "subagent",
                        "t": 5,
                        "requests": [
                            _request(6),
                            _request(7, model="haiku"),
                        ],
                    },
                ],
            )
            for session in ("a", "b", "c", "d")
        ],
    )
    full = tmp_path / "full.jsonl"
    convert_agentx_trace(source, output_path=full, session_spacing_ms=1000)
    sample = tmp_path / "sample.jsonl"
    args = [
        "datasets",
        "agentx",
        "--input",
        str(source),
        "--output",
        str(sample),
        "--sample-sessions",
        "2",
        "--sample-seed",
        "7",
        "--max-input-tokens",
        "768",
        "--session-spacing-ms",
        "1000",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    manifest = json.loads(sample.with_suffix(".jsonl.manifest.json").read_text())
    selected = manifest["sampling"]["selected_session_ids"]
    assert selected == ["b", "c"]
    assert manifest["sampling"]["source_session_count"] == 4
    assert manifest["request_count"] == 6
    assert manifest["input_tokens"] == 768
    assert manifest["subagent_request_count"] == 4
    sessions = {f"agentx:{scope_key(session)}" for session in selected}
    expected = [
        line
        for line in full.read_text().splitlines()
        if json.loads(line)["session_hash"] in sessions
    ]
    assert sample.read_text().splitlines() == expected

    repeated = tmp_path / "repeated.jsonl"
    repeated_manifest = convert_agentx_trace(
        source,
        output_path=repeated,
        session_spacing_ms=1000,
        sample_sessions=2,
        sample_seed=7,
        max_input_tokens=768,
    )
    assert repeated.read_bytes() == sample.read_bytes()
    assert repeated_manifest["sampling"] == manifest["sampling"]
    reversed_source = _write(
        tmp_path / "reversed.jsonl",
        [json.loads(line) for line in reversed(source.read_text().splitlines())],
    )
    reordered = convert_agentx_trace(
        reversed_source,
        output_path=tmp_path / "reordered.jsonl",
        sample_sessions=2,
        sample_seed=7,
    )
    assert reordered["sampling"]["selected_session_ids"] == selected

    # A size guard fails without preferentially replacing a large selected session.
    for options, message in [
        ({"sample_sessions": 2, "max_input_tokens": 767}, "exceed max_input_tokens"),
        ({"sample_sessions": 5}, "exceeds the available"),
    ]:
        failed = tmp_path / "failed.jsonl"
        with pytest.raises(ValueError, match=message):
            convert_agentx_trace(source, output_path=failed, sample_seed=7, **options)
        assert not failed.exists()
        assert not failed.with_suffix(".jsonl.manifest.json").exists()


def test_qwen_search_and_agentx_holdout_remain_separate(tmp_path):
    qwen = tmp_path / "qwen.jsonl"
    source = _write(tmp_path / "qwen-source.jsonl", [_qwen(n) for n in range(10)])
    result = CliRunner().invoke(
        main, ["datasets", "qwen", "--input", str(source), "--output", str(qwen)]
    )
    assert result.exit_code == 0, result.output
    agentx = tmp_path / "agentx.jsonl"
    convert_agentx_trace(_write(tmp_path / "agentx-source.jsonl", [_agentx()]), output_path=agentx)
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "problem": {
                    "settings": {
                        "verifier_version": "1.0.0",
                        "probe_families": [],
                        "seeds": [11],
                    }
                }
            }
        )
    )
    qwen_panel = tmp_path / "qwen-panel"
    result = CliRunner().invoke(
        main,
        [
            "datasets",
            "trace-panel",
            "--trace",
            str(qwen),
            "--config",
            str(base),
            "--output-dir",
            str(qwen_panel),
            "--family",
            "qwen",
            "--group-by",
            "session",
            "--block-size-tokens",
            "16",
            "--capacity-tokens",
            "128",
        ],
    )
    assert result.exit_code == 0, result.output
    combined = tmp_path / "combined"
    result = CliRunner().invoke(
        main,
        [
            "datasets",
            "attach-holdout",
            "--trace",
            str(agentx),
            "--family",
            "agentx",
            "--config",
            str(qwen_panel / "evolution.yaml"),
            "--output-dir",
            str(combined),
            "--capacity-tokens",
            "128",
        ],
    )
    assert result.exit_code == 0, result.output
    config = load_evaluator_config(combined / "evolution.yaml")
    assert [
        (t.family, t.split, t.block_size_tokens, t.capacity_sweep_blocks)
        for t in config.trace_workloads
    ] == [
        ("qwen", "train", 16, (8,)),
        ("qwen", "validation", 16, (8,)),
        ("qwen", "hidden", 16, (8,)),
        ("agentx", "hidden", 64, (2,)),
    ]
    hidden = combined / "hidden.jsonl"
    hidden.rename(combined / "quarantined.jsonl")
    assert {s.family for s in prepare_workloads(config, splits=("train", "validation"))} == {"qwen"}
    assert PrefixKVCacheEvaluator(config)(baseline_lru_blocks).success
    hidden.with_name("quarantined.jsonl").rename(hidden)
    assert PrefixKVCacheEvaluator(config, splits=("hidden",))(baseline_lru_blocks).success

    # A second copy of a search trace cannot masquerade as an independent holdout.
    trace = next(t for t in config.trace_workloads if t.split == "train")
    search = Path(trace.path)
    search.with_suffix(".jsonl.manifest.json").write_text(
        json.dumps(
            {
                "output_sha256": trace.sha256,
                "request_count": trace.request_count,
                "block_size_tokens": 16,
            }
        )
    )
    with pytest.raises(ValueError, match="overlap"):
        attach_holdout(
            search,
            combined / "evolution.yaml",
            tmp_path / "leaked",
            family="leaked",
            capacity_tokens=(128,),
        )
