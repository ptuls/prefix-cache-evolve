"""Tests for weak-seed incumbent rediscovery adjudication."""

from pathlib import Path

import pytest

from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.tools import analyze_rediscovery
from prefix_cache_evolve.tools.analyze_rediscovery import (
    _experiment_summary,
    _gap_recovery,
    source_design_signals,
)
from prefix_cache_evolve.workflow.configuration import ConfigLoader

_POLICY_ROOT = Path("src/prefix_cache_evolve/problems/prefix_kv_cache")


def _agentic_metrics(*, wasted_admission_token_rate: float = 0.1) -> dict[str, float]:
    return {
        "token_hit_rate": 0.4,
        "request_token_hit_rate_p10": 0.2,
        "worst_quarter_token_hit_rate": 0.3,
        "wasted_admission_token_rate": wasted_admission_token_rate,
        "policy_underfill_rate": 0.05,
        "short_reuse_after_eviction_missed_token_rate": 0.02,
        "cache_churn_per_1k": 100.0,
    }


def test_rediscovery_config_matches_operative_evaluator_without_incumbent_prompt_leakage() -> None:
    operative = load_evaluator_config(Path("configs/prefix_kv_cache.yaml"))
    rediscovery = load_evaluator_config(Path("configs/prefix_kv_cache_rediscovery.yaml"))
    prompt = (
        ConfigLoader().load(Path("configs/prefix_kv_cache_rediscovery.yaml")).problem_description
    )

    assert rediscovery == operative
    assert "65.649" not in prompt
    assert "TinyLFU" not in prompt
    assert "pressure-aware" not in prompt
    assert "preserve the supplied parent" not in prompt
    assert "MultiTimescaleDecay" not in prompt
    assert "threshold_excess" not in prompt
    assert "deliberately weak" in prompt


def test_source_design_signals_distinguish_weak_seed_from_incumbent_family() -> None:
    initial = source_design_signals((_POLICY_ROOT / "initial_program.py").read_text())
    compact = source_design_signals((_POLICY_ROOT / "compact_seed.py").read_text())
    incumbent = source_design_signals((_POLICY_ROOT / "pressure_aware_incumbent.py").read_text())

    assert not initial["observed_reuse_state"]
    assert not initial["pressure_conditioned_admission"]
    assert compact["observed_reuse_state"]
    assert compact["bounded_decay_state"]
    assert not compact["pressure_conditioned_admission"]
    assert incumbent["pressure_conditioned_admission"]
    assert incumbent["incumbent_design_family"]


@pytest.mark.parametrize(
    ("candidate", "seed", "incumbent", "expected"),
    (
        (6.0, 2.0, 10.0, 0.5),
        (12.0, 2.0, 10.0, 1.25),
        (8.0, 10.0, 9.0, 0.0),
        (10.0, 10.0, 9.0, 1.0),
    ),
)
def test_gap_recovery_is_normalized_to_seed_and_incumbent(
    candidate: float,
    seed: float,
    incumbent: float,
    expected: float,
) -> None:
    assert _gap_recovery(candidate, seed, incumbent) == pytest.approx(expected)


def test_experiment_requires_repeated_weak_seed_mechanism_rediscovery() -> None:
    runs = [
        {
            "seed_tier": "weak_initial",
            "search_seed": seed,
            "checks": {
                "mechanism_rediscovered": True,
                "behaviorally_rediscovered": True,
            },
        }
        for seed in (11, 23)
    ]

    summary = _experiment_summary(runs)

    assert summary["verdict"] == "supported"
    assert summary["weak_initial_mechanism_rediscovery_count"] == 2
    assert summary["distinct_mechanism_rediscovery_search_seed_count"] == 2


def test_run_analysis_prefers_generated_mutation_and_reports_preliminary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/prefix_kv_cache_rediscovery.yaml")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    initial_source = (_POLICY_ROOT / "initial_program.py").read_text()
    incumbent_source = (_POLICY_ROOT / "pressure_aware_incumbent.py").read_text()
    (run_dir / "seed_program.py").write_text(initial_source)
    (run_dir / "best_program.py").write_text(initial_source)
    (run_dir / "best_generated_mutation.py").write_text(incumbent_source)
    (run_dir / "run_summary.json").write_text(
        '{"iterations": 100, "total_evaluations": 98, "total_cost": 2.5}\n'
    )
    (run_dir / "metadata.json").write_text('{"search_seed": 17}\n')
    (run_dir / "config_snapshot.yaml").write_text(config_path.read_text())

    def fake_decomposition(_config, candidate_path):
        source = candidate_path.read_text()
        score = 10.0 if source == incumbent_source else 0.0
        panel = {
            "combined_score": score,
            "success": True,
            "score_breakdown": {"complexity_cost": 1.0},
        }
        return {
            "effective_complexity": 100,
            "selection": {
                **panel,
                "workload_metrics": {
                    "train/agentic_tool_workflows": _agentic_metrics(),
                },
            },
            "probe": {
                **panel,
                "workload_metrics": {
                    "probe/agent_trace_branching": _agentic_metrics(),
                },
            },
            "hidden": {**panel, "workload_metrics": {}},
        }

    monkeypatch.setattr(
        analyze_rediscovery,
        "_candidate_panel_decomposition",
        fake_decomposition,
    )

    payload = analyze_rediscovery.run_analysis(config_path, run_dirs=(run_dir,))

    assert payload["summary"]["verdict"] == "preliminary"
    assert payload["runs"][0]["candidate_kind"] == "best_generated_mutation"
    assert payload["runs"][0]["seed_tier"] == "weak_initial"
    assert payload["runs"][0]["search_seed"] == 17
    assert payload["runs"][0]["total_evaluations"] == 98
    assert payload["runs"][0]["total_cost"] == 2.5
    assert payload["runs"][0]["checks"]["agentic_surrogate_probe_gate"]
    assert payload["runs"][0]["checks"]["mechanism_rediscovered"]
    assert payload["targets"]["weak_initial"]["selection"]["charged_score"] == pytest.approx(8.0)


def test_run_analysis_rejects_behavioral_rediscovery_when_surrogate_gate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/prefix_kv_cache_rediscovery.yaml")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    initial_source = (_POLICY_ROOT / "initial_program.py").read_text()
    incumbent_source = (_POLICY_ROOT / "pressure_aware_incumbent.py").read_text()
    (run_dir / "seed_program.py").write_text(initial_source)
    (run_dir / "best_generated_mutation.py").write_text(incumbent_source)
    (run_dir / "config_snapshot.yaml").write_text(config_path.read_text())

    def fake_decomposition(_config, candidate_path):
        source = candidate_path.read_text()
        score = 10.0 if source == incumbent_source else 0.0
        panel = {
            "combined_score": score,
            "success": True,
            "score_breakdown": {"complexity_cost": 0.0},
        }
        return {
            "effective_complexity": 100,
            "selection": {
                **panel,
                "workload_metrics": {
                    "train/agentic_tool_workflows": _agentic_metrics(
                        wasted_admission_token_rate=0.5
                    ),
                },
            },
            "probe": {
                **panel,
                "workload_metrics": {
                    "probe/agent_trace_branching": _agentic_metrics(
                        wasted_admission_token_rate=0.1
                    ),
                },
            },
            "hidden": {**panel, "workload_metrics": {}},
        }

    monkeypatch.setattr(
        analyze_rediscovery,
        "_candidate_panel_decomposition",
        fake_decomposition,
    )

    payload = analyze_rediscovery.run_analysis(config_path, run_dirs=(run_dir,))
    run = payload["runs"][0]

    assert run["checks"]["charged_gap_recovery_at_least_threshold_all_panels"]
    assert not run["checks"]["agentic_surrogate_probe_gate"]
    assert not run["checks"]["behaviorally_rediscovered"]
    assert run["agentic_surrogate_probe_gate"]["failed_metrics"] == ["wasted_admission_token_rate"]
