"""Functional tests for candidate validation and holdout boundaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.problems.prefix_kv_cache import candidate_panels, runner
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents.registry import current_incumbent
from tests.support import score_record

_VALID_CANDIDATE = """\
class Policy:
    def on_request_start(self, request, now):
        pass

    def score_admission(self, block, now):
        return 1.0

    def score_eviction(self, block, now):
        return now - block.last_accessed_at

    def on_cache_hit(self, block, request, now):
        pass

    def on_cache_miss(self, block, request, now):
        pass


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return Policy()
"""


@pytest.fixture
def candidate_path(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.py"
    path.write_text(_VALID_CANDIDATE, encoding="utf-8")
    return path


@pytest.mark.parametrize("evaluation", ("panel", "replay", "analysis"))
def test_candidate_evaluations_reject_unsupported_imports_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation: str,
) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text(f"import os\n{_VALID_CANDIDATE}", encoding="utf-8")
    monkeypatch.setattr(
        candidate_panels,
        "run_with_timeout",
        lambda *_args, **_kwargs: pytest.fail("invalid source reached the worker"),
    )
    config = EvaluatorConfig(reject_unsupported_source_patterns=True)

    with pytest.raises(ValueError, match="import from unsupported module os"):
        if evaluation == "panel":
            candidate_panels.evaluate_candidate_program(config, candidate)
        elif evaluation == "replay":
            candidate_panels.evaluate_replay_candidate_program(config, candidate, ())
        else:
            candidate_panels.load_validated_candidate_factory(config, candidate)


def test_candidate_evaluation_enforces_configured_complexity_cap(candidate_path: Path) -> None:
    with pytest.raises(ValueError, match="effective complexity .* exceeds limit 1"):
        candidate_panels.evaluate_candidate_program(
            EvaluatorConfig(max_candidate_complexity=1),
            candidate_path,
        )


def test_candidate_comparison_evaluates_candidate_before_baselines(
    candidate_path: Path,
) -> None:
    calls = []

    def evaluate_candidate(config, path, *, splits):
        del config
        calls.append(("candidate", path, splits))
        return SimpleNamespace(combined_score=12.0)

    def build_baselines():
        calls.append(("baselines",))
        return {"lru": SimpleNamespace(combined_score=10.0)}

    builder = candidate_panels.CandidatePanelBuilder(
        evaluate_program=evaluate_candidate,
        summarize_result=lambda result: {"combined_score": result.combined_score},
    )

    results = builder.build_comparison(
        EvaluatorConfig(),
        candidate_path,
        build_baselines,
        panel=candidate_panels.PROBE_PANEL,
    )

    assert list(results) == ["candidate", "lru"]
    assert calls == [
        ("candidate", candidate_path, candidate_panels.PROBE_PANEL.splits),
        ("baselines",),
    ]


@pytest.mark.parametrize("include_hidden", (False, True), ids=("public", "adjudication"))
def test_candidate_decomposition_preserves_schema_and_hidden_quarantine(
    candidate_path: Path,
    include_hidden: bool,
) -> None:
    evaluated_splits = []
    complexity_calls = []
    scores = {
        candidate_panels.SELECTION_PANEL.splits: 12.0,
        candidate_panels.PROBE_PANEL.splits: 8.0,
        candidate_panels.HIDDEN_PANEL.splits: 6.0,
    }

    def evaluate_candidate(config, path, *, splits):
        del config
        assert path == candidate_path
        evaluated_splits.append(splits)
        return SimpleNamespace(combined_score=scores[splits])

    def evaluate_complexity(source, *, form_aware=False):
        complexity_calls.append((source, form_aware))
        return 7 if form_aware else 10

    builder = candidate_panels.CandidatePanelBuilder(
        evaluate_program=evaluate_candidate,
        summarize_result=lambda result: {"combined_score": result.combined_score},
        evaluate_complexity=evaluate_complexity,
    )

    result = builder.build_decomposition(
        EvaluatorConfig(form_aware_complexity=True),
        candidate_path,
        include_hidden=include_hidden,
    )

    expected = {
        "candidate": str(candidate_path),
        "raw_complexity": 10,
        "effective_complexity": 7,
        "primitive_subsidy_nodes": 3,
        "primitive_subsidy_exercised": True,
        "selection": {"combined_score": 12.0},
        "probe": {"combined_score": 8.0},
    }
    expected_splits = [
        candidate_panels.SELECTION_PANEL.splits,
        candidate_panels.PROBE_PANEL.splits,
    ]
    if include_hidden:
        expected["hidden"] = {"combined_score": 6.0}
        expected_splits.append(candidate_panels.HIDDEN_PANEL.splits)

    assert result == expected
    assert evaluated_splits == expected_splits
    assert complexity_calls == [(_VALID_CANDIDATE, False), (_VALID_CANDIDATE, True)]


def test_runner_candidate_decomposition_preserves_artifact_schema(
    candidate_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = {
        candidate_panels.SELECTION_PANEL.splits: 12.0,
        candidate_panels.PROBE_PANEL.splits: 8.0,
        candidate_panels.HIDDEN_PANEL.splits: 6.0,
    }

    def evaluate_candidate(config, path, *, splits):
        del config
        assert path == candidate_path
        return score_record(
            scores[splits],
            success=True,
            invalid_fraction=0.0,
            split_metrics={},
            workload_metrics={},
            capacity_metrics={},
            candidate_metadata={},
            score_breakdown={},
        )

    monkeypatch.setattr(runner, "_evaluate_candidate_program", evaluate_candidate)

    payload = runner._candidate_panel_decomposition(
        EvaluatorConfig(),
        candidate_path,
        include_hidden=True,
    )

    assert list(payload) == [
        "candidate",
        "raw_complexity",
        "effective_complexity",
        "primitive_subsidy_nodes",
        "primitive_subsidy_exercised",
        "selection",
        "probe",
        "hidden",
    ]
    assert list(payload["selection"]) == [
        "verifier_version",
        "evaluation_context_sha256",
        "panel_sha256",
        "combined_score",
        "success",
        "invalid_fraction",
        "split_metrics",
        "workload_metrics",
        "capacity_metrics",
        "candidate_metadata",
        "score_breakdown",
    ]
    assert payload["selection"]["combined_score"] == 12.0
    assert payload["probe"]["combined_score"] == 8.0
    assert payload["hidden"]["combined_score"] == 6.0


def test_registered_incumbent_reports_do_not_modify_immutable_bundles() -> None:
    candidate = current_incumbent("production").source_path

    report = runner._baseline_comparison_output_path(candidate)

    assert report == (
        Path("artifacts")
        / "prefix_kv_cache_reports"
        / "production_16tok_20260609"
        / "baseline_comparison.md"
    )
    assert candidate.parent not in report.resolve().parents


def test_external_candidate_reports_remain_beside_candidate(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.py"

    assert runner._baseline_comparison_output_path(candidate) == tmp_path / "baseline_comparison.md"
