"""Functional tests for evolution orchestration and terminal reporting."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from prefix_cache_evolve.workflow import reporting
from prefix_cache_evolve.workflow.execution import LeviRunResult
from prefix_cache_evolve.workflow.program import ProgramSource, TemporaryProgramFile
from prefix_cache_evolve.workflow.reporting import EvolutionReporter
from prefix_cache_evolve.workflow.workflow import EvolutionWorkflow


def test_temporary_program_file_normalizes_source_and_removes_file() -> None:
    source = ProgramSource("    def build_candidate():\n        return 3\n")

    with TemporaryProgramFile(source) as path:
        assert path.read_text(encoding="utf-8") == "def build_candidate():\n    return 3\n"

    assert not path.exists()


def test_temporary_program_file_is_removed_after_failure() -> None:
    path: Path | None = None

    with pytest.raises(ValueError, match="workflow failed"):
        with TemporaryProgramFile(ProgramSource("candidate\n")) as path:
            raise ValueError("workflow failed")

    assert path is not None
    assert not path.exists()


def test_evolution_workflow_runs_and_reports_normalized_program() -> None:
    calls = []
    config = SimpleNamespace(label="unit")
    result = SimpleNamespace(score=3.0)

    def run(program_path, received_config):
        calls.append(("run", program_path.read_text(encoding="utf-8"), received_config))
        return result

    workflow = EvolutionWorkflow(
        program_source=ProgramSource("    candidate\n"),
        config_provider=SimpleNamespace(
            load=lambda iterations: calls.append(("load", iterations)) or config,
            describe=lambda: "unit-config",
        ),
        runner=SimpleNamespace(run=run),
        reporter=SimpleNamespace(
            report=lambda received_result, iterations, label: calls.append(
                ("report", received_result, iterations, label)
            )
        ),
    )

    assert workflow.execute(7) is result
    assert calls == [
        ("load", 7),
        ("run", "candidate\n", config),
        ("report", result, 7, "unit-config"),
    ]


@pytest.mark.parametrize("estimated_cost", (None, 0.125))
def test_evolution_reporter_logs_identity_and_cost(
    monkeypatch: pytest.MonkeyPatch,
    estimated_cost: float | None,
) -> None:
    messages = []
    monkeypatch.setattr(
        reporting.logger,
        "info",
        lambda template, *args: messages.append(template.format(*args)),
    )
    result = LeviRunResult(
        best_program="def build_candidate(): pass",
        best_score=3.5,
        metrics={},
        artifacts={},
        total_evaluations=4,
        total_cost=0.25,
        metadata={
            "verifier_version": "1.0.0",
            "evaluation_context_sha256": "context",
            "panel_sha256": "panel",
            "run_cost": {
                "requests": 2,
                "prompt_tokens": 10,
                "cached_prompt_tokens": 3,
                "completion_tokens": 5,
                "total_tokens": 15,
                "estimated_cost_usd": estimated_cost,
            },
            "run_cost_summary_path": "artifacts/cost.json",
        },
    )

    EvolutionReporter().report(result, 7, "unit-config")

    rendered = "\n".join(messages)
    assert "Verifier version: 1.0.0" in rendered
    assert "Evaluation context: context" in rendered
    assert "Panel SHA-256: panel" in rendered
    assert "LLM usage: 2 requests" in rendered
    assert "Run cost summary: artifacts/cost.json" in rendered
    if estimated_cost is None:
        assert "Estimated LLM cost (USD): n/a" in rendered
    else:
        assert "Estimated LLM cost (USD): $0.1250" in rendered
