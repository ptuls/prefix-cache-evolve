"""Trace-panel preparation, quarantine, and evolution integration tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from prefix_cache_evolve.evaluators.panels import prepare_workloads
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    PrefixKVCacheEvaluator,
    baseline_lru_blocks,
)
from prefix_cache_evolve.evaluators.verifier import VERIFIER_VERSION
from prefix_cache_evolve.problems.prefix_kv_cache import evaluator as levi_evaluator
from prefix_cache_evolve.problems.prefix_kv_cache import runner
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import (
    load_evaluator_config,
    prefix_kv_config_environment,
)
from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import (
    build_workload_manifest,
    file_sha256,
)
from prefix_cache_evolve.tools.cli import main as tools_main

_SEED_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/lru.py"
).read_text(encoding="utf-8")


@pytest.fixture
def trace_input(tmp_path):
    path = tmp_path / "trace.jsonl"
    records = [
        {
            "timestamp_ms": turn * 1_000 + session * 10,
            "tenant_hash": f"tenant-{session // 2}",
            "session_hash": f"session-{session}",
            "prompt_length": 8 + turn * 4,
            "output_length": 4,
            "prefix_path": ["common", f"prefix-{session}"] + (["continuation"] if turn else []),
        }
        for turn in range(2)
        for session in range(12)
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


@pytest.fixture
def base_config(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "problem": {
                    "settings": {
                        "verifier_version": VERIFIER_VERSION,
                        "block_size_tokens": 4,
                        "capacity_sweep_blocks": [4, 8],
                        "request_count": 8,
                        "seeds": [11, 23],
                        "train_families": ["shared_system_prompt"],
                        "validation_families": ["rag_template_reuse"],
                        "probe_families": [],
                        "hidden_families": ["adversarial_unique_prompts"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def panel_dir(tmp_path, trace_input, base_config):
    output = tmp_path / "panel"
    result = _prepare(trace_input, base_config, output)
    assert result.exit_code == 0, result.output
    return output


def _prepare(trace, config, output, *extra):
    return CliRunner().invoke(
        tools_main,
        [
            "datasets",
            "trace-panel",
            "--trace",
            str(trace),
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--family",
            "wildchat",
            *extra,
        ],
    )


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_panel_preserves_whole_sessions_order_and_disjoint_tenants(panel_dir, trace_input):
    manifest = json.loads((panel_dir / "manifest.json").read_text(encoding="utf-8"))
    session_splits = {}
    tenant_splits = {}
    records = []
    for trace in manifest["traces"]:
        split_records = _records(panel_dir / trace["path"])
        timestamps = [record["timestamp_ms"] for record in split_records]
        assert timestamps == sorted(timestamps)
        assert trace["request_count"] == len(split_records)
        assert trace["sha256"] == file_sha256(panel_dir / trace["path"])
        for record in split_records:
            assert (
                session_splits.setdefault(record["session_hash"], trace["split"]) == trace["split"]
            )
            assert tenant_splits.setdefault(record["tenant_hash"], trace["split"]) == trace["split"]
        records.extend(split_records)
    assert len(records) == len(_records(trace_input)) == 24
    assert set(session_splits.values()) == {"train", "validation", "hidden"}
    assert manifest["source"]["sha256"] == file_sha256(trace_input)


def test_trace_streams_are_not_repeated_for_synthetic_seeds(panel_dir):
    config = load_evaluator_config(panel_dir / "evolution.yaml")
    prepared = prepare_workloads(config, splits=("train", "validation"))
    result = PrefixKVCacheEvaluator(config)(baseline_lru_blocks)
    manifest = build_workload_manifest(config, splits=("train", "validation"))

    assert config.seeds == (11, 23)
    assert len(prepared) == 2
    assert len(result.trials) == 4  # Two fixed streams at two cache capacities.
    assert {trial.seed for trial in result.trials} == {0}
    assert result.success
    assert result.panel_sha256 == manifest["panel_sha256"]
    assert result.evaluation_context_sha256 == manifest["evaluation_context_sha256"]
    assert all(stream["source"]["kind"] == "anonymized_trace" for stream in manifest["streams"])


def test_evolution_worker_uses_traces_without_opening_hidden_data(panel_dir):
    config_path = panel_dir / "evolution.yaml"
    config = load_evaluator_config(config_path)
    hidden = PrefixKVCacheEvaluator(config, splits=("hidden",))(baseline_lru_blocks)
    assert set(hidden.workload_metrics) == {"hidden/wildchat"}
    (panel_dir / "hidden.jsonl").unlink()

    with prefix_kv_config_environment(config_path):
        result = levi_evaluator.evaluate_source(_SEED_SOURCE)

    assert result.metrics["success"], result.metrics
    assert set(result.artifacts["workload_metrics"]) == {"train/wildchat", "validation/wildchat"}
    assert "hidden_token_hit_rate" not in result.metrics
    manifest = build_workload_manifest(config, splits=("train", "validation", "probe"))
    assert result.metrics["panel_sha256"] == manifest["panel_sha256"]


def test_panel_paths_resolve_relative_to_config_and_can_move(panel_dir, tmp_path, monkeypatch):
    original = load_evaluator_config(panel_dir / "evolution.yaml")
    relocated = tmp_path / "relocated"
    shutil.copytree(panel_dir, relocated)
    monkeypatch.chdir(tmp_path.parent)
    moved = load_evaluator_config(relocated / "evolution.yaml")

    first = PrefixKVCacheEvaluator(original)(baseline_lru_blocks)
    second = PrefixKVCacheEvaluator(moved)(baseline_lru_blocks)

    assert first == second
    assert all(Path(trace.path).parent == relocated for trace in moved.trace_workloads)


@pytest.mark.parametrize("fixed_admission_policy", [None, "pressure_aware_incumbent"])
@pytest.mark.parametrize("use_manifest_reference", [False, True])
def test_saved_run_replays_without_hidden_or_original_panel(
    panel_dir, tmp_path, monkeypatch, fixed_admission_policy, use_manifest_reference
):
    config_path = panel_dir / "evolution.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["problem"]["settings"]["fixed_admission_policy"] = fixed_admission_policy
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    config = load_evaluator_config(config_path)
    (panel_dir / "hidden.jsonl").unlink()
    with prefix_kv_config_environment(config_path):
        evaluation = levi_evaluator.evaluate_source(_SEED_SOURCE)
    manifest = build_workload_manifest(config, splits=("train", "validation", "probe"))
    generated_source = _SEED_SOURCE + "\n_GENERATED_MARKER = 1\n"
    levi_snapshot = tmp_path / "levi_snapshot.json"
    levi_snapshot.write_text(
        json.dumps(
            {
                "elites": [
                    {
                        "program_id": "generated",
                        "content": generated_source,
                        "primary_score": evaluation.metrics["combined_score"],
                        **{
                            key: evaluation.metrics[key]
                            for key in (
                                "verifier_version",
                                "evaluation_context_sha256",
                                "panel_sha256",
                            )
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def unexpected_adjudication(*args, **kwargs):
        pytest.fail("automatic run saving must not evaluate the hidden panel")

    monkeypatch.setattr(runner, "_candidate_panel_decomposition", unexpected_adjudication)
    monkeypatch.setattr(
        runner,
        "_evaluate_baselines",
        lambda config, **kwargs: {
            "lru": PrefixKVCacheEvaluator(config, splits=("train", "validation", "probe"))(
                baseline_lru_blocks
            )
        },
    )
    result = SimpleNamespace(
        best_program=_SEED_SOURCE,
        metrics=evaluation.metrics,
        artifacts=evaluation.artifacts,
        metadata={"levi_snapshot_path": str(levi_snapshot)},
    )

    saved = runner.save_run_artifacts(
        result,
        tmp_path / "runs",
        iterations=0,
        config_label="trace-smoke",
        seed_source=_SEED_SOURCE,
        report_config=config,
        report_config_file=str(config_path),
        config_snapshot=config_path,
    )
    shutil.rmtree(panel_dir)
    snapshot = load_evaluator_config(saved / "config_snapshot.yaml")
    replay = build_workload_manifest(snapshot, splits=("train", "validation", "probe"))

    assert replay["evaluation_context_sha256"] == manifest["evaluation_context_sha256"]
    assert replay["panel_sha256"] == manifest["panel_sha256"]
    assert {path.name for path in (saved / "traces").iterdir()} == {
        "train-wildchat.jsonl",
        "validation-wildchat.jsonl",
    }
    assert (saved / "best_generated_mutation.py").read_text(encoding="utf-8") == generated_source
    assert (saved / "best_generated_mutation_snapshot.json").is_file()
    assert not (saved / "best_generated_mutation_decomposition_error.json").exists()
    assert (saved / "source_config.yaml").is_file()
    assert (saved / "baseline_comparison.md").is_file()
    assert not (saved / "baseline_comparison_error.json").exists()
    with prefix_kv_config_environment(saved / "config_snapshot.yaml"):
        repeated = levi_evaluator.evaluate_source(_SEED_SOURCE)
    assert repeated.metrics["combined_score"] == evaluation.metrics["combined_score"]
    assert (
        repeated.metrics["evaluation_context_sha256"]
        == evaluation.metrics["evaluation_context_sha256"]
    )
    cli_manifest = tmp_path / "regenerated_manifest.json"
    reference_args = (
        ["--workload-manifest-reference", str(saved / "workload_manifest.json")]
        if use_manifest_reference
        else []
    )
    verified = CliRunner().invoke(
        runner.main,
        [
            "--config",
            str(saved / "config_snapshot.yaml"),
            "--workload-manifest",
            "--workload-manifest-output",
            str(cli_manifest),
            *reference_args,
        ],
    )
    assert verified.exit_code == 0, verified.output or str(verified.exception)
    regenerated = json.loads(cli_manifest.read_text(encoding="utf-8"))
    assert regenerated["panel_sha256"] == manifest["panel_sha256"]
    assert regenerated["evaluation_context_sha256"] == manifest["evaluation_context_sha256"]
    if use_manifest_reference:
        assert "workload_manifest_reference=verified:" in verified.output


def test_manifest_verification_uses_only_the_reference_splits(panel_dir, tmp_path):
    config_path = panel_dir / "evolution.yaml"
    reference = build_workload_manifest(load_evaluator_config(config_path), splits=("hidden",))
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    (panel_dir / "train.jsonl").unlink()
    (panel_dir / "validation.jsonl").unlink()
    output = tmp_path / "verified_manifest.json"

    result = CliRunner().invoke(
        runner.main,
        [
            "--config",
            str(config_path),
            "--workload-manifest",
            "--workload-manifest-reference",
            str(reference_path),
            "--workload-manifest-output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output or str(result.exception)
    assert (
        json.loads(output.read_text(encoding="utf-8"))["panel_sha256"] == reference["panel_sha256"]
    )
    assert "workload_manifest_reference=verified:" in result.output


def test_trace_corruption_fails_before_starting_evolution(panel_dir, monkeypatch):
    train_path = panel_dir / "train.jsonl"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    def unexpected_workflow(*args, **kwargs):
        pytest.fail("a model workflow must not start with changed trace inputs")

    monkeypatch.setattr(runner, "_build_workflow", unexpected_workflow)

    with pytest.raises(ValueError, match="SHA-256"):
        runner.demo_run_evolution(config_file=str(panel_dir / "evolution.yaml"))


def test_trace_geometry_cannot_be_reinterpreted(panel_dir):
    config = load_evaluator_config(panel_dir / "evolution.yaml")

    with pytest.raises(ValueError, match="reconvert the source"):
        config.with_updates(block_size_tokens=8)


def test_trace_session_leakage_is_rejected_even_with_updated_hashes(panel_dir):
    config = load_evaluator_config(panel_dir / "evolution.yaml")
    train_session = _records(panel_dir / "train.jsonl")[0]["session_hash"]
    validation_path = panel_dir / "validation.jsonl"
    records = _records(validation_path)
    records[0]["session_hash"] = train_session
    validation_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    traces = [
        trace.model_copy(update={"sha256": file_sha256(validation_path)})
        if trace.split == "validation"
        else trace
        for trace in config.trace_workloads
    ]

    with pytest.raises(ValueError, match="sessions must not overlap"):
        PrefixKVCacheEvaluator(config.with_updates(trace_workloads=traces))(baseline_lru_blocks)


def test_panel_is_repeatable_and_refuses_to_overwrite(
    panel_dir, trace_input, base_config, tmp_path
):
    repeated = tmp_path / "repeated"
    result = _prepare(trace_input, base_config, repeated)
    assert result.exit_code == 0, result.output
    for filename in (
        "train.jsonl",
        "validation.jsonl",
        "hidden.jsonl",
        "evolution.yaml",
        "manifest.json",
    ):
        assert (panel_dir / filename).read_bytes() == (repeated / filename).read_bytes()

    result = _prepare(trace_input, base_config, repeated)
    assert result.exit_code != 0
    assert "already exists" in result.output


def test_panel_can_mix_traces_with_synthetic_workloads(trace_input, base_config, tmp_path):
    output = tmp_path / "mixed"
    result = _prepare(trace_input, base_config, output, "--keep-synthetic")
    assert result.exit_code == 0, result.output
    config = load_evaluator_config(output / "evolution.yaml")
    manifest = build_workload_manifest(config, splits=("train", "validation"))

    assert len(manifest["streams"]) == 6  # Four synthetic streams plus two trace streams.
    assert {stream["family"] for stream in manifest["streams"]} == {
        "shared_system_prompt",
        "rag_template_reuse",
        "wildchat",
    }
    evaluation = PrefixKVCacheEvaluator(config)(baseline_lru_blocks)
    assert evaluation.panel_sha256 == manifest["panel_sha256"]


def test_mixed_panel_preserves_per_trace_block_geometry(
    trace_input,
    base_config,
    tmp_path,
) -> None:
    output = tmp_path / "mixed-geometry"
    result = _prepare(trace_input, base_config, output, "--keep-synthetic")
    assert result.exit_code == 0, result.output
    config_path = output / "evolution.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = document["problem"]["settings"]
    settings["block_size_tokens"] = 8
    settings["capacity_blocks"] = 2
    settings["capacity_sweep_blocks"] = [2, 4]
    for trace in settings["trace_workloads"]:
        trace["capacity_sweep_blocks"] = [4, 8]
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    config = load_evaluator_config(config_path)
    evaluation = PrefixKVCacheEvaluator(config)(baseline_lru_blocks)
    manifest = build_workload_manifest(config, splits=("train", "validation"))

    synthetic_trials = [trial for trial in evaluation.trials if trial.workload != "wildchat"]
    trace_trials = [trial for trial in evaluation.trials if trial.workload == "wildchat"]
    assert {(trial.block_size_tokens, trial.capacity_blocks) for trial in synthetic_trials} == {
        (8, 2),
        (8, 4),
    }
    assert {(trial.block_size_tokens, trial.capacity_blocks) for trial in trace_trials} == {
        (4, 4),
        (4, 8),
    }
    assert "block_8_capacity_4" in evaluation.capacity_metrics
    assert "block_4_capacity_4" in evaluation.capacity_metrics
    assert evaluation.panel_sha256 == manifest["panel_sha256"]
    assert evaluation.evaluation_context_sha256 == manifest["evaluation_context_sha256"]
    assert manifest["evaluation"]["trace_geometry_overrides"] == [
        {
            "split": trace.split,
            "family": trace.family,
            "block_size_tokens": 4,
            "capacity_blocks": [4, 8],
            "capacity_tokens": [16, 32],
        }
        for trace in config.trace_workloads
        if trace.split in {"train", "validation"}
    ]


def test_mixed_block_size_trace_requires_capacity_override(
    panel_dir,
) -> None:
    config_path = panel_dir / "evolution.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["problem"]["settings"]["block_size_tokens"] = 8
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match="define its own capacity_sweep_blocks"):
        load_evaluator_config(config_path)


def test_panel_retains_and_checks_converter_provenance(trace_input, base_config, tmp_path):
    source_manifest = trace_input.with_suffix(".jsonl.manifest.json")
    source_manifest.write_text(
        json.dumps({"output_sha256": file_sha256(trace_input), "block_size_tokens": 4}),
        encoding="utf-8",
    )
    output = tmp_path / "provenance"
    result = _prepare(trace_input, base_config, output)
    assert result.exit_code == 0, result.output
    assert (output / "source.manifest.json").read_bytes() == source_manifest.read_bytes()

    source_manifest.write_text(
        json.dumps({"output_sha256": "a" * 64, "block_size_tokens": 4}), encoding="utf-8"
    )
    result = _prepare(trace_input, base_config, tmp_path / "mismatch")
    assert result.exit_code != 0
    assert "output_sha256 does not match" in result.output
    assert not (tmp_path / "mismatch").exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--validation-fraction", "0.8", "--hidden-fraction", "0.3"), "nonempty train"),
        (("--family", "../invalid"), "String should match pattern"),
        (("--block-size-tokens", "8"), "prefix_path depth"),
    ],
)
def test_panel_rejects_invalid_settings_atomically(
    trace_input, base_config, tmp_path, extra, message
):
    output = tmp_path / "invalid"
    result = _prepare(trace_input, base_config, output, *extra)

    assert result.exit_code != 0
    assert message in result.output
    assert not output.exists()


def test_panel_refuses_insufficient_groups(trace_input, base_config, tmp_path):
    records = _records(trace_input)
    for record in records:
        record["tenant_hash"] = "one-tenant"
    trace_input.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    result = _prepare(trace_input, base_config, tmp_path / "insufficient")

    assert result.exit_code != 0
    assert "at least three distinct groups" in result.output
