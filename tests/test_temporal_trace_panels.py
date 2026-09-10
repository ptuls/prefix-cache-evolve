"""Chronological trace splits preserve traffic and keep final windows quarantined."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from pydantic import TypeAdapter

from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.evaluators.fingerprints import evaluation_context_sha256
from prefix_cache_evolve.evaluators.panels import prepare_workloads
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluationResult,
    PrefixKVCacheEvaluator,
    baseline_lru_blocks,
    scoring_fn_complexity,
)
from prefix_cache_evolve.evaluators.verifier import VERIFIER_VERSION
from prefix_cache_evolve.problems.prefix_kv_cache import evaluator, runner, sandbox
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import (
    load_evaluator_config,
    prefix_kv_config_environment,
)
from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import (
    build_workload_manifest,
    file_sha256,
    stable_workload_manifest_payload,
)
from prefix_cache_evolve.problems.prefix_kv_cache.trace_replay import iter_trace_records
from prefix_cache_evolve.tools.cli import main as tools_main
from prefix_cache_evolve.workflow.config import ConfigLoader

_WINDOWS = (("train", 0, 200), ("validation", 300, 500), ("hidden", 600, 800))
_SEED = (
    Path(__file__).resolve().parents[1]
    / "src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/lru.py"
).read_text(encoding="utf-8")


@pytest.fixture
def temporal_source(tmp_path):
    trace = tmp_path / "source.jsonl"
    timestamps = (0, 0, 100, 150, 200, 250, 300, 300, 450, 500, 550, 600, 700, 750)
    records = [
        {
            "timestamp_ms": timestamp,
            "tenant_hash": "unknown",
            "session_hash": "unknown",
            "prompt_length": 8,
            "output_length": 4,
            "prefix_path": ["common", f"route-{index % 3}"],
        }
        for index, timestamp in enumerate(timestamps)
    ]
    trace.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    config = tmp_path / "base.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "problem": {
                    "settings": {
                        "verifier_version": VERIFIER_VERSION,
                        "block_size_tokens": 4,
                        "capacity_sweep_blocks": [2, 4],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return trace, config


def _prepare(temporal_source, output, windows=_WINDOWS):
    trace, config = temporal_source
    return CliRunner().invoke(
        tools_main,
        [
            "datasets",
            "temporal-trace-panel",
            "--trace",
            str(trace),
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--family",
            "mooncake",
            *[value for window in windows for value in ("--window", *map(str, window))],
        ],
    )


@pytest.fixture
def temporal_panel(tmp_path, temporal_source):
    output = tmp_path / "panel"
    result = _prepare(temporal_source, output)
    assert result.exit_code == 0, result.output
    return output


def test_temporal_panel_retains_all_window_requests_and_original_identities(
    temporal_source, temporal_panel
):
    source = [record.as_dict() for record in iter_trace_records(temporal_source[0])]
    manifest = json.loads((temporal_panel / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["sha256"] == file_sha256(temporal_source[0])
    assert manifest["source"]["request_count"] == 14
    assert manifest["excluded_request_count"] == 4
    assert manifest["partition"]["cold_start_per_window"]
    for trace, (split, start, end) in zip(manifest["traces"], _WINDOWS, strict=True):
        records = list(iter_trace_records(temporal_panel / trace["path"]))
        expected = [record for record in source if start <= record["timestamp_ms"] < end]
        assert [record.as_dict() for record in records] == expected
        assert trace["request_count"] == len(expected)
        assert trace["source_record_stop"] - trace["source_record_start"] == len(expected)
        assert trace["sha256"] == file_sha256(temporal_panel / trace["path"])
        assert trace["split"] == split
        assert {record.session_key for record in records} == {"unknown"}
        assert {record.tenant_key for record in records} == {"unknown"}
    config = load_evaluator_config(temporal_panel / "evolution.yaml")
    assert not config.workload_configs(("train", "validation", "probe", "hidden"))
    assert len(prepare_workloads(config, splits=("train", "validation", "hidden"))) == 3


def test_temporal_search_worker_and_manifest_do_not_open_hidden(temporal_panel):
    path = temporal_panel / "evolution.yaml"
    config = load_evaluator_config(path)
    (temporal_panel / "hidden-1.jsonl").unlink()
    manifest = build_workload_manifest(config, splits=("train", "validation", "probe"))
    with prefix_kv_config_environment(path):
        result = evaluator.evaluate_source(_SEED)
    assert result.metrics["success"], result.metrics
    assert result.metrics["panel_sha256"] == manifest["panel_sha256"]
    assert result.metrics["evaluation_context_sha256"] == manifest["evaluation_context_sha256"]
    assert set(result.artifacts["workload_metrics"]) == {
        "train/mooncake_1",
        "validation/mooncake_1",
    }
    assert not any(key.startswith("hidden_") for key in result.metrics)
    assert all("time_window" in stream["source"] for stream in manifest["streams"])


def test_temporal_manifest_can_move_and_cannot_be_mislabeled_as_session_disjoint(
    temporal_panel, tmp_path
):
    config = load_evaluator_config(temporal_panel / "evolution.yaml")
    original = build_workload_manifest(config, splits=("train", "validation"))
    relocated = tmp_path / "relocated"
    shutil.copytree(temporal_panel, relocated)
    moved = load_evaluator_config(relocated / "evolution.yaml")
    assert stable_workload_manifest_payload(
        build_workload_manifest(moved, splits=("train", "validation"))
    ) == stable_workload_manifest_payload(original)
    grouped = config.with_updates(
        trace_workloads=[
            trace.model_copy(update={"time_window": None}) for trace in config.trace_workloads
        ]
    )
    with pytest.raises(ValueError, match="sessions must not overlap"):
        prepare_workloads(grouped, splits=("train", "validation"))


@pytest.mark.parametrize("extra_first", [False, True])
@pytest.mark.parametrize("extra_origin", [None, "b" * 64])
def test_cross_split_session_checks_every_training_origin(
    temporal_panel, extra_first, extra_origin
):
    config = load_evaluator_config(temporal_panel / "evolution.yaml")
    train, validation, hidden = config.trace_workloads
    extra_window = (
        None
        if extra_origin is None
        else train.time_window.model_copy(update={"source_sha256": extra_origin})
    )
    extra = train.model_copy(update={"family": "extra_train", "time_window": extra_window})
    training = [extra, train] if extra_first else [train, extra]
    mixed = config.with_updates(trace_workloads=[*training, validation, hidden])
    with pytest.raises(ValueError, match="sessions must not overlap"):
        prepare_workloads(mixed, splits=("train", "validation"))


@pytest.mark.parametrize(
    "windows, message",
    [
        ((("train", 0, 400), ("validation", 300, 500), ("hidden", 600, 800)), "overlap"),
        ((("validation", 0, 200), ("train", 300, 500), ("hidden", 600, 800)), "order"),
        ((("train", 0, 200), ("validation", 300, 500)), "each of"),
        ((("train", 0, 200), ("validation", 310, 320), ("hidden", 600, 800)), "contain"),
        ((("train", 0, 0), ("validation", 300, 500), ("hidden", 600, 800)), "greater than"),
        ((("train", 0, float("inf")), ("validation", 300, 500), ("hidden", 600, 800)), "finite"),
    ],
)
def test_temporal_builder_rejects_invalid_windows_without_publishing(
    temporal_source, tmp_path, windows, message
):
    output = tmp_path / "bad-panel"
    result = _prepare(temporal_source, output, windows)
    assert result.exit_code != 0
    assert message in result.output
    assert not output.exists()


def test_temporal_builder_accepts_multiple_ordered_windows_per_split(temporal_source, tmp_path):
    output = tmp_path / "multiple"
    result = _prepare(
        temporal_source,
        output,
        (
            ("train", 0, 200),
            ("validation", 300, 400),
            ("validation", 400, 500),
            ("hidden", 600, 800),
        ),
    )
    assert result.exit_code == 0, result.output
    config = load_evaluator_config(output / "evolution.yaml")
    assert [trace.family for trace in config.trace_workloads if trace.split == "validation"] == [
        "mooncake_1",
        "mooncake_2",
    ]


@pytest.mark.parametrize("change", ["overlap", "reorder", "outside", "different_source"])
def test_temporal_runtime_rejects_inconsistent_window_descriptors(temporal_panel, change):
    config = load_evaluator_config(temporal_panel / "evolution.yaml")
    document = config.model_dump()
    validation = document["trace_workloads"][1]
    if change == "overlap":
        validation["time_window"]["start_ms"] = 100
    elif change == "reorder":
        validation["split"] = "hidden"
        document["trace_workloads"][2]["split"] = "validation"
    elif change == "outside":
        validation["time_window"]["start_ms"] = 301
    else:
        validation["time_window"]["source_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        changed = EvaluatorConfig.model_validate(document)
        prepare_workloads(changed, splits=("train", "validation"))


def test_temporal_builder_checks_unselected_records_and_source_provenance(
    temporal_source, tmp_path
):
    trace, _ = temporal_source
    with trace.open("a", encoding="utf-8") as handle:
        handle.write('{"timestamp_ms": 900, "content": "forbidden"}\n')
    output = tmp_path / "invalid-source"
    result = _prepare(temporal_source, output)
    assert result.exit_code != 0
    assert "raw-content" in result.output
    assert not output.exists()


def test_temporal_builder_refuses_changed_manifest_and_existing_output(
    temporal_source, temporal_panel, tmp_path
):
    assert _prepare(temporal_source, temporal_panel).exit_code != 0
    trace, _ = temporal_source
    manifest = trace.with_suffix(".jsonl.manifest.json")
    manifest.write_text(json.dumps({"output_sha256": "0" * 64}), encoding="utf-8")
    result = _prepare(temporal_source, tmp_path / "invalid-manifest")
    assert result.exit_code != 0
    assert "output_sha256" in result.output


def test_absent_time_windows_preserve_existing_trace_context_identity(temporal_panel):
    config = load_evaluator_config(temporal_panel / "evolution.yaml")
    old = config.model_dump(mode="json")
    for trace in old["trace_workloads"]:
        trace.pop("time_window")
    new = EvaluatorConfig.model_validate(old).model_dump(mode="json")
    arguments = {"verifier_version": config.verifier_version, "panel_sha": "a" * 64}
    assert evaluation_context_sha256(
        evaluator_config=old, **arguments
    ) == evaluation_context_sha256(evaluator_config=new, **arguments)


def test_unprepared_production_config_cannot_start_model_search(monkeypatch):
    def unexpected_workflow(*args, **kwargs):
        pytest.fail("an empty panel must fail before constructing a model workflow")

    monkeypatch.setattr(runner, "_build_workflow", unexpected_workflow)
    config_path = Path(__file__).resolve().parents[1] / "configs/prefix_kv_cache_mooncake.yaml"
    with pytest.raises(ValueError, match="prepare traces first"):
        runner.demo_run_evolution(config_file=str(config_path))


def test_selected_panel_baselines_run_once_and_reject_unknown_names(temporal_panel):
    config = load_evaluator_config(temporal_panel / "evolution.yaml")
    results = runner._evaluate_baselines(config, baseline_names=("lru", "lru", "tinylfu_lru"))
    assert list(results) == ["lru", "tinylfu_lru"]
    assert all(result.success for result in results.values())
    with pytest.raises(ValueError, match="unknown report baselines"):
        runner._evaluate_baselines(config, baseline_names=("typo",))


@pytest.mark.parametrize(
    "mode", [[], ["--baseline-report"], ["--hidden-report"], ["--probe-report"]]
)
def test_report_baseline_selection_reaches_the_requested_runner(monkeypatch, mode):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)

    for name in ("demo_run_evolution", "compare_baselines", "hidden_report", "probe_report"):
        monkeypatch.setattr(runner, name, capture)
    result = CliRunner().invoke(runner.main, [*mode, "--report-baseline", "lru"])
    assert result.exit_code == 0, result.output
    assert captured["baseline_names"] == ("lru",)


def test_container_snapshot_excludes_hidden_and_credentials(temporal_panel, monkeypatch):
    config = load_evaluator_config(temporal_panel / "evolution.yaml").with_updates(
        sandbox_image="sha256:" + "a" * 64
    )
    (temporal_panel / "hidden-1.jsonl").unlink()
    expected = PrefixKVCacheEvaluator(config)(baseline_lru_blocks)
    monkeypatch.setenv("OPENAI_API_KEY", "secret-preflight-sentinel")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/docker")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            mount = command[command.index("--mount") + 1]
            inputs = Path(mount.split("src=", 1)[1].split(",", 1)[0])
            payload = json.loads((inputs / "request.json").read_text(encoding="utf-8"))
            assert payload["source"] == _SEED
            assert payload["splits"] == ["train", "validation"]
            assert {path.name for path in inputs.iterdir()} == {
                "request.json",
                "trace-0.jsonl",
                "trace-1.jsonl",
            }
            assert payload["config"]["trace_workloads"][2]["path"].startswith("/unavailable/")
            assert all(
                "capacity_sweep_blocks" not in trace
                for trace in payload["config"]["trace_workloads"]
            )
            assert "secret-preflight-sentinel" not in json.dumps(payload) + " ".join(command)
            assert "--network" in command and command[command.index("--network") + 1] == "none"
            assert "--read-only" in command
            assert command[command.index("--entrypoint") + 1] == "timeout"
            return subprocess.CompletedProcess(command, 0, json.dumps(asdict(expected)), "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox, "_run_container_process", fake_run)
    result = sandbox.evaluate_in_docker(_SEED, config, splits=("train", "validation"))
    assert result == expected
    assert commands[-1][1:3] == ["rm", "--force"]
    assert commands[-1][-1] == commands[0][commands[0].index("--name") + 1]


def test_stale_container_replay_semantics_fail_closed(temporal_panel, monkeypatch):
    config = load_evaluator_config(temporal_panel / "evolution.yaml").with_updates(
        sandbox_image="stale-image"
    )
    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)
    result.candidate_metadata.pop("trace_replay_contract")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        sandbox,
        "_run_container_process",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(asdict(result)), ""
        ),
    )
    monkeypatch.setattr(sandbox.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="contract is stale"):
        sandbox.evaluate_in_docker(_SEED, config, splits=("validation",))


def test_unknown_sessions_require_temporal_provenance_and_never_form_a_group(temporal_panel):
    config = load_evaluator_config(temporal_panel / "evolution.yaml")
    traces = []
    for trace in config.trace_workloads:
        path = Path(trace.path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["session_hash"] = None
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        traces.append(trace.model_copy(update={"sha256": file_sha256(path)}))
    config = config.with_updates(trace_workloads=traces)
    workloads = prepare_workloads(config, splits=("train", "validation", "hidden"))
    assert all(r.info.session_id is None for workload in workloads for r in workload.requests)
    assert all(w.fingerprint()["source"]["replay_contract"].endswith("v2") for w in workloads)
    without_windows = config.with_updates(
        trace_workloads=[t.model_copy(update={"time_window": None}) for t in traces]
    )
    with pytest.raises(ValueError, match="require chronological time windows"):
        prepare_workloads(without_windows, splits=("train", "validation"))


@pytest.mark.parametrize("failure", ["timeout", "container_error", "invalid_json", "output_limit"])
def test_container_failures_remove_the_named_container(temporal_panel, monkeypatch, failure):
    config = load_evaluator_config(temporal_panel / "evolution.yaml").with_updates(
        sandbox_image="test-image"
    )
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/docker")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            if failure == "container_error":
                return subprocess.CompletedProcess(command, 1, "", "worker failure")
            if failure == "output_limit":
                raise RuntimeError("sandbox stdout exceeded output bytes")
            return subprocess.CompletedProcess(command, 0, "not json", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox, "_run_container_process", fake_run)
    with pytest.raises((TimeoutError, RuntimeError, ValueError)):
        sandbox.evaluate_in_docker(_SEED, config, splits=("validation",))
    assert commands[-1][1:3] == ["rm", "--force"]
    assert commands[-1][-1] == commands[0][commands[0].index("--name") + 1]


def test_container_process_drains_both_streams():
    command = [
        sys.executable,
        "-c",
        "import os; os.write(1, b'x' * 131072); "
        "os.write(2, b'worker diagnostic'); raise SystemExit(3)",
    ]
    result = sandbox._run_container_process(command, timeout=5)
    assert result.returncode == 3
    assert result.stdout == "x" * 131072
    assert result.stderr == "worker diagnostic"


@pytest.mark.parametrize("stream, descriptor", [("stdout", 1), ("stderr", 2)])
def test_container_process_bounds_each_output_stream(monkeypatch, stream, descriptor):
    monkeypatch.setattr(sandbox, f"_MAX_{stream.upper()}_BYTES", 128)
    command = [sys.executable, "-c", f"import os; os.write({descriptor}, b'x' * 65536)"]
    with pytest.raises(RuntimeError, match=f"sandbox {stream} exceeded 128 output bytes"):
        sandbox._run_container_process(command, timeout=5)


@pytest.mark.parametrize("close_streams", [False, True])
def test_container_process_enforces_deadline_even_after_pipes_close(close_streams):
    code = "import os, time; " + ("os.close(1); os.close(2); " if close_streams else "")
    with pytest.raises(subprocess.TimeoutExpired):
        sandbox._run_container_process([sys.executable, "-c", code + "time.sleep(5)"], timeout=0.1)


@pytest.mark.parametrize("termination", ["SystemExit(0)", "KeyboardInterrupt()", "BaseException()"])
def test_candidate_cannot_print_a_forged_result_and_exit(temporal_panel, tmp_path, termination):
    config = load_evaluator_config(temporal_panel / "evolution.yaml").with_updates(
        reject_unsupported_source_patterns=True
    )
    forged = asdict(PrefixKVCacheEvaluator(config)(baseline_lru_blocks))
    forged.update(combined_score=1e12, success=True, trials=[])
    # The old host decoder accepted this payload, making the stream boundary material.
    assert TypeAdapter(EvaluationResult).validate_json(json.dumps(forged), strict=True).success
    source = (
        "def build_candidate(capacity_blocks, block_size_tokens, seed=None):\n"
        f"    print({json.dumps(forged)!r})\n"
        f"    raise {termination}\n"
    )
    assert not evaluator._candidate_source_violations(source, scoring_fn_complexity(source), config)
    request = tmp_path / "forged.json"
    request.write_text(
        json.dumps(
            {
                "source": source,
                "config": config.model_dump(mode="json"),
                "splits": ["train", "validation"],
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(sandbox.main, [str(request)])
    assert result.exit_code != 0
    assert "1000000000000" not in result.output
    assert "Candidate evaluation failed with" in result.output


def test_candidate_diagnostics_do_not_corrupt_result_json(temporal_panel, tmp_path):
    config = load_evaluator_config(temporal_panel / "evolution.yaml")
    source = """
class Policy:
    def on_request_start(self, request, now):
        print("request diagnostic")

    def on_cache_hit(self, block, request, now):
        pass

    def on_cache_miss(self, block, request, now):
        pass

    def score_admission(self, block, now):
        print("admission diagnostic")
        return 1

    def score_eviction(self, block, now):
        return now - block.last_accessed_at

def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    print("factory diagnostic")
    return Policy()
"""
    request = tmp_path / "printing.json"
    request.write_text(
        json.dumps(
            {
                "source": source,
                "config": config.model_dump(mode="json"),
                "splits": ["train", "validation"],
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(sandbox.main, [str(request)])
    assert result.exit_code == 0, result.output
    parsed = TypeAdapter(EvaluationResult).validate_json(result.output, strict=True)
    assert parsed.success and len(parsed.trials) == 4
    assert "diagnostic" not in result.output


def test_candidate_import_facades_prevent_shared_module_mutation(temporal_panel, tmp_path):
    config = load_evaluator_config(temporal_panel / "evolution.yaml")
    original_log1p = sandbox.math.log1p
    source = """
import math

def poison(self):
    try:
        self.log1p = lambda value: 1000000000000.0
    except AttributeError:
        pass

class Policy:
    def on_request_start(self, request, now):
        pass

    def on_cache_hit(self, block, request, now):
        pass

    def on_cache_miss(self, block, request, now):
        pass

    def score_admission(self, block, now):
        return math.log1p(block.hit_count) + 1

    def score_eviction(self, block, now):
        return now - block.last_accessed_at

def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    poison(math)
    return Policy()
"""
    assert "attribute writes must target candidate-owned self state" in (
        evaluator._candidate_source_violations(
            source,
            scoring_fn_complexity(source),
            config.with_updates(reject_unsupported_source_patterns=True),
        )
    )
    # The import facade remains an independent runtime boundary.
    assert sandbox._fresh_policy(source, ("build_candidate",), 4, 4, 0) is not None
    assert sandbox.math.log1p is original_log1p


def test_permitted_future_annotations_import_works_in_sandbox(temporal_panel, tmp_path):
    config = load_evaluator_config(temporal_panel / "evolution.yaml").with_updates(
        reject_unsupported_source_patterns=True
    )
    source = """
from __future__ import annotations

class Policy:
    def on_request_start(self, request, now):
        pass

    def on_cache_hit(self, block, request, now):
        pass

    def on_cache_miss(self, block, request, now):
        pass

    def score_admission(self, block, now) -> float:
        return 1.0

    def score_eviction(self, block, now) -> float:
        return now - block.last_accessed_at

def build_candidate(capacity_blocks, block_size_tokens, seed=None) -> Policy:
    return Policy()
"""
    assert not evaluator._candidate_source_violations(source, scoring_fn_complexity(source), config)
    request = tmp_path / "future-annotations.json"
    request.write_text(
        json.dumps(
            {
                "source": source,
                "config": config.model_dump(mode="json"),
                "splits": ["train", "validation"],
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(sandbox.main, [str(request)])
    assert result.exit_code == 0, result.output
    assert TypeAdapter(EvaluationResult).validate_json(result.output, strict=True).success


def test_sandbox_always_enforces_source_grammar(temporal_panel, tmp_path):
    config = load_evaluator_config(temporal_panel / "evolution.yaml").with_updates(
        reject_unsupported_source_patterns=False
    )
    marker = tmp_path / "must-not-exist"
    source = f"open({str(marker)!r}, 'w').write('leaked')\n"
    request = tmp_path / "fail-closed.json"
    request.write_text(
        json.dumps(
            {
                "source": source,
                "config": config.model_dump(mode="json"),
                "splits": ["train", "validation"],
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(sandbox.main, [str(request)])
    assert result.exit_code != 0
    assert "Static policy violations" in result.output
    assert not marker.exists()


def test_container_mode_does_not_fall_back_to_loading_source_on_host(temporal_panel, monkeypatch):
    path = temporal_panel / "evolution.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["problem"]["settings"]["sandbox_image"] = "test-image"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)

    def forbidden(*args, **kwargs):
        pytest.fail("configured container mode must never load a candidate on the host")

    monkeypatch.setattr(evaluator, "load_candidate_factory_from_source", forbidden)
    with prefix_kv_config_environment(path):
        result = evaluator.evaluate_source(_SEED)
    assert not result.metrics["success"]
    assert "Docker is required" in result.artifacts["error_message"]


def test_search_scope_cleanup_only_targets_its_own_containers(monkeypatch):
    monkeypatch.setenv("PREFIX_CACHE_EVOLVE_SANDBOX_SCOPE", "outer-scope")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/docker")
    commands = []
    scope = None

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[1] == "ps":
            assert command[-1] == f"label=prefix-cache-evolve.scope={scope}"
            return subprocess.CompletedProcess(command, 0, "our-container\n", "")
        assert command == ["/usr/bin/docker", "rm", "--force", "our-container"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="interrupted search"):
        with sandbox.docker_evaluation_scope("test-image"):
            scope = os.environ["PREFIX_CACHE_EVOLVE_SANDBOX_SCOPE"]
            assert scope != "outer-scope"
            raise RuntimeError("interrupted search")
    assert len(commands) == 2
    assert os.environ["PREFIX_CACHE_EVOLVE_SANDBOX_SCOPE"] == "outer-scope"


def test_mooncake_search_allows_time_for_container_cleanup():
    path = Path(__file__).resolve().parents[1] / "configs/prefix_kv_cache_mooncake.yaml"
    workflow = ConfigLoader().load(path)
    config = load_evaluator_config(path)
    assert config.sandbox_image
    assert workflow.pipeline["eval_timeout"] >= config.timeout_s + 30


def test_fixed_request_replay_cannot_bypass_a_configured_sandbox(tmp_path):
    config = EvaluatorConfig(sandbox_image="test-image")
    with pytest.raises(ValueError, match="requires a configured trace panel"):
        runner._evaluate_replay_candidate_program(config, tmp_path / "not-loaded.py", ())


def test_container_trials_reset_candidate_module_state(temporal_panel, tmp_path):
    config = load_evaluator_config(temporal_panel / "evolution.yaml")
    source = """
CALLS = []

class Policy:
    def __init__(self, capacity_blocks, block_size_tokens, seed=None):
        CALLS.append(1)
        self.admit = len(CALLS) > 1

    def on_request_start(self, request, now):
        pass

    def on_cache_hit(self, block, request, now):
        pass

    def on_cache_miss(self, block, request, now):
        pass

    def score_admission(self, block, now):
        return 1 if self.admit else 0

    def score_eviction(self, block, now):
        return now - block.last_accessed_at

def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return Policy(capacity_blocks, block_size_tokens, seed)
"""
    request = tmp_path / "sandbox-request.json"
    request.write_text(
        json.dumps(
            {
                "source": source,
                "config": config.model_dump(mode="json"),
                "splits": ["train", "validation"],
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(sandbox.main, [str(request)])
    assert result.exit_code == 0, result.output
    trials = json.loads(result.output)["trials"]
    assert len(trials) == 4
    assert all(trial["admission_count"] == 0 and trial["token_hit_rate"] == 0 for trial in trials)
