"""Run archival, promotion adjudication, and diagnostic report tests."""

import json
import textwrap
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.problems.prefix_kv_cache import run_artifacts
from prefix_cache_evolve.problems.prefix_kv_cache.candidate_panels import CandidatePanelBuilder
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import DEFAULT_CONFIG_PATH
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents.registry import current_incumbent
from prefix_cache_evolve.problems.prefix_kv_cache.run_artifacts import RunArtifactWriter
from prefix_cache_evolve.problems.prefix_kv_cache.utilities import (
    evaluation_result_summary,
    surrogate_probe_tripwire_suite,
)
from tests.support import report_result as _report_result
from tests.support import score_identity


def _decomposition_writer(decompose):
    return RunArtifactWriter(
        panels=SimpleNamespace(build_decomposition=decompose),
        evaluate_baselines=lambda *_args, **_kwargs: {},
        incumbent_path=current_incumbent("production").source_path,
    )


def _agentic_gate_metrics(
    token_hit_rate: float,
    **overrides: float,
) -> dict[str, float]:
    metrics = {
        "token_hit_rate": token_hit_rate,
        "request_token_hit_rate_p10": 0.25,
        "worst_quarter_token_hit_rate": 0.35,
        "wasted_admission_token_rate": 0.10,
        "policy_underfill_rate": 0.05,
        "short_reuse_after_eviction_missed_token_rate": 0.02,
        "cache_churn_per_1k": 100.0,
    }
    metrics.update(overrides)
    return metrics


def test_surrogate_probe_tripwire_suite_passes_all_configured_channels() -> None:
    suite = surrogate_probe_tripwire_suite(
        {
            "train/agentic_tool_workflows": _agentic_gate_metrics(0.48),
            "probe/agent_trace_branching": _agentic_gate_metrics(0.38),
            "validation/hotset_cold_scan": {"token_hit_rate": 0.64},
            "probe/cyclic_working_set_pressure": {"token_hit_rate": 0.86},
        }
    )

    assert suite["status"] == "pass"
    assert suite["flagged"] is False
    assert suite["flagged_channels"] == []
    assert suite["passed_channels"] == ["agentic_branching", "cyclic_working_set"]
    assert suite["channels"]["agentic_branching"]["checks"]["token_hit_rate"][
        "absolute_gap"
    ] == pytest.approx(0.10)
    assert suite["channels"]["cyclic_working_set"]["absolute_gap"] == pytest.approx(0.22)
    assert suite["max_threshold_ratio"] == pytest.approx(0.88)


def test_surrogate_probe_tripwire_suite_flags_only_divergent_channel() -> None:
    suite = surrogate_probe_tripwire_suite(
        {
            "train/agentic_tool_workflows": _agentic_gate_metrics(0.48),
            "probe/agent_trace_branching": _agentic_gate_metrics(0.38),
            "validation/hotset_cold_scan": {"token_hit_rate": 0.70},
            "probe/cyclic_working_set_pressure": {"token_hit_rate": 0.40},
        }
    )

    assert suite["status"] == "flagged"
    assert suite["flagged_channels"] == ["cyclic_working_set"]
    assert suite["channels"]["agentic_branching"]["status"] == "pass"
    assert suite["channels"]["cyclic_working_set"]["flag_reason"] == (
        "divergence_exceeds_threshold"
    )


def test_surrogate_probe_tripwire_suite_fails_closed_for_missing_channel() -> None:
    suite = surrogate_probe_tripwire_suite(
        {
            "train/agentic_tool_workflows": _agentic_gate_metrics(0.48),
            "probe/agent_trace_branching": _agentic_gate_metrics(0.38),
        }
    )

    assert suite["status"] == "flagged"
    assert suite["flagged_channels"] == ["cyclic_working_set"]
    assert suite["channels"]["cyclic_working_set"]["flag_reason"] == ("missing_or_invalid_metric")


def test_surrogate_probe_tripwire_suite_accepts_per_family_thresholds() -> None:
    suite = surrogate_probe_tripwire_suite(
        {
            "train/agentic_tool_workflows": _agentic_gate_metrics(0.48),
            "probe/agent_trace_branching": _agentic_gate_metrics(0.38),
            "validation/hotset_cold_scan": {"token_hit_rate": 0.64},
            "probe/cyclic_working_set_pressure": {"token_hit_rate": 0.86},
        },
        thresholds={
            "agentic_branching": 0.09,
            "cyclic_working_set": 0.23,
        },
    )

    assert suite["status"] == "flagged"
    assert suite["flagged_channels"] == ["agentic_branching"]
    assert suite["channels"]["agentic_branching"]["checks"]["token_hit_rate"][
        "threshold"
    ] == pytest.approx(0.09)
    assert suite["channels"]["cyclic_working_set"]["threshold"] == pytest.approx(0.23)


def test_surrogate_probe_tripwire_suite_flags_non_hit_agentic_divergence() -> None:
    suite = surrogate_probe_tripwire_suite(
        {
            "train/agentic_tool_workflows": _agentic_gate_metrics(
                0.48,
                wasted_admission_token_rate=0.50,
            ),
            "probe/agent_trace_branching": _agentic_gate_metrics(
                0.38,
                wasted_admission_token_rate=0.10,
            ),
            "validation/hotset_cold_scan": {"token_hit_rate": 0.64},
            "probe/cyclic_working_set_pressure": {"token_hit_rate": 0.86},
        }
    )

    assert suite["flagged_channels"] == ["agentic_branching"]
    assert suite["channels"]["agentic_branching"]["checks"]["token_hit_rate"]["status"] == "pass"
    assert suite["channels"]["agentic_branching"]["failed_metrics"] == [
        "wasted_admission_token_rate"
    ]


def test_save_run_artifacts_persists_best_program_and_metadata(tmp_path, monkeypatch) -> None:
    config = EvaluatorConfig(request_count=4, seeds=(3,), capacity_sweep_blocks=(8,))
    writer = RunArtifactWriter(
        panels=CandidatePanelBuilder(
            evaluate_program=lambda *_args, **_kwargs: _report_result(12.0, capacities=(8,)),
            summarize_result=evaluation_result_summary,
        ),
        evaluate_baselines=lambda *_args, **_kwargs: {
            "lru": _report_result(10.0, capacities=(8,)),
            "oracle_future_reuse": _report_result(16.0, capacities=(8,)),
        },
        incumbent_path=current_incumbent("production").source_path,
    )
    monkeypatch.setattr(
        run_artifacts,
        "_repository_state",
        lambda: {"commit": "abc123", "dirty": True},
    )
    best_program = textwrap.dedent(
        """
        class NoCachePolicy:
            def on_request_start(self, request, now):
                pass

            def score_admission(self, block, now):
                return -1.0

            def score_eviction(self, block, now):
                return 0.0

            def on_cache_hit(self, block, request, now):
                pass

            def on_cache_miss(self, block, request, now):
                pass


        def build_candidate(capacity_blocks, block_size_tokens, seed=None):
            return NoCachePolicy()
        """
    )
    result = SimpleNamespace(
        best_program=best_program,
        best_score=12.5,
        total_evaluations=7,
        total_cost=0.25,
        archive_size=3,
        runtime_seconds=4.0,
        metrics={
            **score_identity(),
            "combined_score": 12.5,
        },
        artifacts={
            **score_identity(),
            "split_metrics": {"validation": {"token_hit_rate": 0.5}},
            "workload_metrics": {
                "train/agentic_tool_workflows": _agentic_gate_metrics(0.48),
                "probe/agent_trace_branching": _agentic_gate_metrics(0.38),
                "validation/hotset_cold_scan": {"token_hit_rate": 0.64},
                "probe/cyclic_working_set_pressure": {"token_hit_rate": 0.86},
            },
        },
        metadata={"levi_runtime_seconds": 4.0},
    )
    config_snapshot = tmp_path / "input-config.yaml"
    config_snapshot.write_text("max_iterations: 3\n", encoding="utf-8")
    paradigm_candidates_dir = tmp_path / "levi-paradigm-candidates" / "eval_0010"
    paradigm_candidates_dir.mkdir(parents=True)
    (paradigm_candidates_dir / "00_paradigm_shift.py").write_text(
        "def build_candidate():\n    return 'paradigm'\n",
        encoding="utf-8",
    )
    result.metadata["levi_paradigm_candidates_dir"] = str(paradigm_candidates_dir.parent)

    run_dir = writer.save(
        result,
        tmp_path,
        iterations=3,
        report_config=config,
        config_label="unit-config",
        seed_label="artifacts/source-run",
        config_snapshot=config_snapshot,
        timestamp=datetime(2026, 6, 2, 1, 2, 3, tzinfo=UTC),
    )

    assert run_dir == tmp_path / "20260602T010203Z"
    assert "def build_candidate" in (run_dir / "best_program.py").read_text(encoding="utf-8")
    assert '"combined_score": 12.5' in (run_dir / "metrics.json").read_text(encoding="utf-8")
    assert '"config": "unit-config"' in (run_dir / "run_summary.json").read_text(encoding="utf-8")
    assert '"seed_program": "artifacts/source-run"' in (run_dir / "run_summary.json").read_text(
        encoding="utf-8"
    )
    assert '"config_snapshot": "config_snapshot.yaml"' in (run_dir / "run_summary.json").read_text(
        encoding="utf-8"
    )
    workload_manifest = json.loads((run_dir / "workload_manifest.json").read_text(encoding="utf-8"))
    assert len(workload_manifest["panel_sha256"]) == 64
    gate = json.loads((run_dir / "agentic_surrogate_probe_gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    assert gate["checks"]["token_hit_rate"]["absolute_gap"] == pytest.approx(0.10)
    assert "Status: PASS" in (run_dir / "agentic_surrogate_probe_gate.md").read_text(
        encoding="utf-8"
    )
    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["agentic_surrogate_probe_gate"]["status"] == "pass"
    assert run_summary["surrogate_probe_tripwires"]["status"] == "pass"
    assert run_summary["surrogate_probe_tripwires"]["passed_channels"] == [
        "agentic_branching",
        "cyclic_working_set",
    ]
    assert (run_dir / "surrogate_probe_tripwires.json").is_file()
    assert "cyclic_working_set" in (run_dir / "surrogate_probe_tripwires.md").read_text(
        encoding="utf-8"
    )
    assert run_summary["repository"] == {"commit": "abc123", "dirty": True}
    assert (run_dir / "config_snapshot.yaml").read_text(
        encoding="utf-8"
    ) == config_snapshot.read_text(encoding="utf-8")
    assert (run_dir / "paradigm_candidates" / "eval_0010" / "00_paradigm_shift.py").is_file()
    assert (tmp_path / "latest_run.txt").read_text(encoding="utf-8") == str(run_dir)
    report = (run_dir / "baseline_comparison.md").read_text(encoding="utf-8")
    assert "Prefix KV-Cache Best Program Baseline Comparison" in report
    assert "`candidate`" in report
    assert "`oracle_future_reuse`" in report
    assert "reporting-only/future-knowledge" in report
    assert "Held-Out Structure-Generalization Probe" in report
    assert "--baseline-report --capacity-sweep-blocks 8" in report
    assert f"--config {DEFAULT_CONFIG_PATH}" in report
    assert "--baseline-report --quick" not in report


def test_persist_best_generated_mutation_decomposes_strongest_non_seed(
    tmp_path,
) -> None:
    seed_source = "def build_candidate():\n    return 'seed'\n"
    strongest_source = "def build_candidate():\n    return 'strongest'\n"
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "elites": [
                    {
                        "program_id": "seed",
                        "content": seed_source,
                        "primary_score": 8.0,
                        **score_identity(),
                    },
                    {
                        "program_id": "weaker",
                        "content": "def build_candidate():\n    return 'weaker'\n",
                        "primary_score": 3.0,
                        **score_identity(),
                    },
                    {
                        "program_id": "strongest",
                        "content": strongest_source,
                        "primary_score": 7.0,
                        **score_identity(),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_decomposition(_config, candidate_path, *, include_hidden=False):
        assert include_hidden is False
        is_seed = candidate_path.name == "seed_program.py"
        return {
            "candidate": str(candidate_path),
            "raw_complexity": 10,
            "effective_complexity": 8,
            "primitive_subsidy_nodes": 2,
            "primitive_subsidy_exercised": True,
            "selection": {
                **score_identity(),
                "combined_score": 8.0 if is_seed else 7.0,
                "score_breakdown": {
                    "mean_workload_score": 10.0,
                    "min_workload_contribution": 2.0,
                    "churn_cost": 1.0,
                    "complexity_cost": 2.0,
                },
            },
            "probe": {
                **score_identity(),
                "combined_score": 4.0,
                "workload_metrics": {
                    "probe/agent_trace_branching": {"token_hit_rate": 0.2},
                    "probe/cyclic_working_set_pressure": {"token_hit_rate": 0.7},
                },
            },
        }

    writer = _decomposition_writer(fake_decomposition)

    writer.persist_generated_mutation(
        tmp_path,
        metadata={"levi_snapshot_path": str(snapshot_path)},
        seed_source=seed_source,
        config=EvaluatorConfig(),
    )

    assert (tmp_path / "best_generated_mutation.py").read_text(encoding="utf-8") == strongest_source
    decomposition = json.loads(
        (tmp_path / "best_generated_mutation_decomposition.json").read_text(encoding="utf-8")
    )
    assert decomposition["generated_program_id"] == "strongest"
    assert decomposition["best_generated_mutation"]["primitive_subsidy_exercised"]
    assert "hidden" not in decomposition["seed"]
    assert "hidden" not in decomposition["best_generated_mutation"]
    report = (tmp_path / "best_generated_mutation_decomposition.md").read_text(encoding="utf-8")
    assert "Best generated mutation" in report
    assert "Agent hit" in report
    assert "hidden panel remains quarantined" in report


def test_specialist_promotion_adjudication_fails_over_complexity_limit(
    tmp_path,
) -> None:
    (tmp_path / "best_program.py").write_text("candidate\n", encoding="utf-8")

    def fake_decomposition(config, candidate_path, *, include_hidden=False):
        assert include_hidden is True
        assert config.fixed_admission_policy is None
        assert config.max_candidate_complexity is None
        assert config.promotion_max_candidate_complexity is None
        is_candidate = candidate_path.name == "best_program.py"
        score = 12.0 if is_candidate else 10.0
        return {
            "candidate": str(candidate_path),
            "raw_complexity": 700 if is_candidate else 600,
            "effective_complexity": 651 if is_candidate else 600,
            "primitive_subsidy_nodes": 49 if is_candidate else 0,
            "primitive_subsidy_exercised": is_candidate,
            "selection": {
                **score_identity(),
                "combined_score": score,
                "score_breakdown": {"complexity_cost": 0.0},
                "split_metrics": {
                    "validation": {
                        "avoidable_eviction_rate": 0.1,
                        "short_reuse_after_eviction_missed_token_rate": 0.1,
                    }
                },
                "workload_metrics": {
                    "train/agentic_tool_workflows": _agentic_gate_metrics(0.48),
                    "validation/hotset_cold_scan": {"token_hit_rate": 0.64},
                },
            },
            "probe": {
                **score_identity(),
                "combined_score": score,
                "workload_metrics": {
                    "probe/agent_trace_branching": _agentic_gate_metrics(0.40),
                    "probe/cyclic_working_set_pressure": {"token_hit_rate": 0.80},
                },
            },
            "hidden": {
                **score_identity(),
                "combined_score": score,
            },
        }

    writer = _decomposition_writer(fake_decomposition)

    payload = writer.adjudicate_specialist(
        tmp_path,
        config=EvaluatorConfig(
            max_candidate_complexity=900,
            promotion_max_candidate_complexity=650,
            fixed_admission_policy="pressure_aware_incumbent",
        ),
    )

    assert payload is not None
    assert payload["status"] == "fail"
    assert payload["eligible"] is False
    assert payload["checks"]["complexity_within_promotion_limit"]["passed"] is False
    assert payload["checks"]["selection_non_regression"]["passed"] is True
    assert (
        json.loads((tmp_path / "promotion_adjudication.json").read_text(encoding="utf-8"))["status"]
        == "fail"
    )
    assert "exploration-only" in (tmp_path / "promotion_adjudication.md").read_text(
        encoding="utf-8"
    )


def test_eviction_only_promotion_adjudication_composes_complete_candidate(
    tmp_path,
) -> None:
    (tmp_path / "best_program.py").write_text(
        "def score_eviction(block, now, frequency, priority):\n"
        "    return now - block.last_accessed_at - frequency - priority\n",
        encoding="utf-8",
    )
    captured_paths = []

    def fake_decomposition(config, candidate_path, *, include_hidden=False):
        assert include_hidden is True
        assert config.fixed_admission_policy is None
        assert config.candidate_policy_surface == "full"
        captured_paths.append(candidate_path)
        is_candidate = candidate_path.name == "promotion_candidate.py"
        score = 11.0 if is_candidate else 10.0
        return {
            "candidate": str(candidate_path),
            "raw_complexity": 640 if is_candidate else 636,
            "effective_complexity": 640 if is_candidate else 636,
            "primitive_subsidy_nodes": 0,
            "primitive_subsidy_exercised": False,
            "selection": {
                **score_identity(),
                "combined_score": score,
                "score_breakdown": {"complexity_cost": 1.0},
                "split_metrics": {
                    "validation": {
                        "avoidable_eviction_rate": 0.1,
                        "short_reuse_after_eviction_missed_token_rate": 0.1,
                    }
                },
                "workload_metrics": {
                    "train/agentic_tool_workflows": _agentic_gate_metrics(0.50),
                    "validation/hotset_cold_scan": {"token_hit_rate": 0.64},
                },
            },
            "probe": {
                **score_identity(),
                "combined_score": score,
                "workload_metrics": {
                    "probe/agent_trace_branching": _agentic_gate_metrics(0.40),
                    "probe/cyclic_working_set_pressure": {"token_hit_rate": 0.80},
                },
            },
            "hidden": {
                **score_identity(),
                "combined_score": score,
            },
        }

    writer = _decomposition_writer(fake_decomposition)

    payload = writer.adjudicate_specialist(
        tmp_path,
        config=EvaluatorConfig(
            max_candidate_complexity=1000,
            promotion_max_candidate_complexity=650,
            fixed_admission_policy="pressure_aware_incumbent",
            candidate_policy_surface="eviction_only",
            search_score_mode="raw_before_complexity",
        ),
    )

    assert payload is not None
    assert payload["status"] == "pass"
    assert (tmp_path / "promotion_candidate.py").is_file()
    assert captured_paths[0].name == "promotion_candidate.py"
    assert "def score_admission(self, block, now):" in (
        tmp_path / "promotion_candidate.py"
    ).read_text(encoding="utf-8")
