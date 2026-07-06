"""Tests for the score-gap robustness verification tool."""

from __future__ import annotations

from prefix_cache_evolve.problems.prefix_kv_cache.configuration import DEFAULT_CONFIG_PATH
from prefix_cache_evolve.tools.verify_significance import (
    PairedUnit,
    bootstrap_mean_ci,
    cluster_mean_differences,
    format_report,
    run_significance_analysis,
    seed_degeneracy_report,
)


def _unit(workload: str, capacity: int, seed: int, difference: float) -> PairedUnit:
    """Return a paired unit whose difference equals ``difference``."""
    return PairedUnit(
        group=f"validation/{workload}/capacity_{capacity}/seed_{seed}",
        split="validation",
        workload=workload,
        capacity_blocks=capacity,
        base_seed=seed,
        seed=seed,
        candidate_score=difference,
        baseline_score=0.0,
    )


def test_bootstrap_is_deterministic_for_fixed_seed() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    first = bootstrap_mean_ci(values, resamples=2000, seed=7)
    second = bootstrap_mean_ci(values, resamples=2000, seed=7)

    assert first == second
    assert first["lower"] < first["upper"]


def test_bootstrap_on_constant_input_collapses_to_the_constant() -> None:
    result = bootstrap_mean_ci([2.5, 2.5, 2.5, 2.5], resamples=500, seed=1)

    assert result["lower"] == 2.5
    assert result["upper"] == 2.5


def test_bootstrap_on_positive_differences_excludes_zero() -> None:
    values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    result = bootstrap_mean_ci(values, confidence=0.95, resamples=5000, seed=0)

    assert result["lower"] > 0.0
    assert result["confidence"] == 0.95


def test_cluster_mean_differences_collapses_replicates() -> None:
    units = [
        _unit("a", 24, 1, 2.0),
        _unit("a", 24, 2, 4.0),
        _unit("b", 24, 1, -1.0),
    ]
    clusters = cluster_mean_differences(units, lambda unit: (unit.workload,))

    assert clusters[("a",)] == 3.0
    assert clusters[("b",)] == -1.0


def test_seed_degeneracy_flags_identical_replicates() -> None:
    units = [
        _unit("degenerate", 24, 1, 5.0),
        _unit("degenerate", 24, 2, 5.0),
        _unit("varying", 24, 1, 1.0),
        _unit("varying", 24, 2, 2.0),
    ]
    report = seed_degeneracy_report(units)

    assert report["cell_count"] == 2
    assert report["seed_invariant_cell_count"] == 1
    assert report["seed_invariant_cells"] == ["degenerate/capacity_24"]


def test_run_significance_analysis_reports_descriptive_robustness() -> None:
    kwargs = dict(
        request_count=4,
        seeds=(3, 7, 11),
        splits=("validation",),
        workloads=("shared_system_prompt", "rag_template_reuse"),
        bootstrap_resamples=2000,
    )
    payload = run_significance_analysis(DEFAULT_CONFIG_PATH, **kwargs)
    repeat = run_significance_analysis(DEFAULT_CONFIG_PATH, **kwargs)

    assert payload["schema"] == "prefix-kv-cache-score-gap-robustness-v2"
    assert payload["candidate"] == "production_incumbent"
    assert payload["baseline"] == "tinylfu_lru"
    # Two capacities, three seeds, and two workload families.
    assert payload["paired_unit_count"] == 12
    assert len(payload["units"]) == 12
    assert payload["descriptive_family_summary"]["family_count"] == 2
    seed_summary = payload["whole_panel_outer_seed_summary"]
    assert seed_summary["outer_seed_count"] == 3
    assert not seed_summary["inference_ready"]
    assert seed_summary["paired_outer_seed_bootstrap_confidence_interval"] is None
    assert len(payload["leave_one_family_out_sensitivity"]["omissions"]) == 2
    assert "seed_invariant_cell_count" in payload["seed_degeneracy"]
    assert payload["mean_difference"] == repeat["mean_difference"]
    assert payload["descriptive_family_summary"] == repeat["descriptive_family_summary"]
    assert payload["whole_panel_outer_seed_summary"] == repeat["whole_panel_outer_seed_summary"]
    assert "difference" in payload["charged_score"]

    report = format_report(payload)
    assert "Score-gap robustness analysis" in report
    assert "confidence interval: omitted" in report


def test_outer_seed_interval_requires_twenty_whole_panel_replicates() -> None:
    payload = run_significance_analysis(
        DEFAULT_CONFIG_PATH,
        request_count=2,
        seeds=tuple(range(20)),
        splits=("validation",),
        workloads=("shared_system_prompt",),
        bootstrap_resamples=1000,
    )

    seed_summary = payload["whole_panel_outer_seed_summary"]
    assert seed_summary["inference_ready"]
    assert seed_summary["paired_outer_seed_bootstrap_confidence_interval"] is not None


def test_run_significance_analysis_rejects_unknown_baseline() -> None:
    try:
        run_significance_analysis(
            DEFAULT_CONFIG_PATH,
            baseline_name="not_a_real_baseline",
            request_count=4,
            seeds=(3,),
            splits=("validation",),
            workloads=("shared_system_prompt",),
        )
    except ValueError as error:
        assert "unknown baseline" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for unknown baseline")
