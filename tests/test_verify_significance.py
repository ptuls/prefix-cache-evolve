"""Tests for the score-gap significance verification tool."""

from __future__ import annotations

from prefix_cache_evolve.problems.prefix_kv_cache.configuration import DEFAULT_CONFIG_PATH
from prefix_cache_evolve.tools.verify_significance import (
    PairedUnit,
    bootstrap_mean_ci,
    cluster_mean_differences,
    format_report,
    permutation_p_value,
    run_significance_analysis,
    seed_degeneracy_report,
    sign_test_p_value,
)


def _unit(workload: str, capacity: int, seed: int, difference: float) -> PairedUnit:
    """Return a paired unit whose difference equals ``difference``."""
    return PairedUnit(
        group=f"validation/{workload}/capacity_{capacity}/seed_{seed}",
        split="validation",
        workload=workload,
        capacity_blocks=capacity,
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


def test_sign_test_matches_exact_binomial_tail() -> None:
    result = sign_test_p_value([1.0, 1.0, 1.0, -1.0])

    assert result["positive"] == 3
    assert result["negative"] == 1
    assert result["ties"] == 0
    # Two-sided exact tail: 2 * (C(4,0) + C(4,1)) / 2**4 = 2 * 5 / 16.
    assert abs(float(result["p_value"]) - 0.625) < 1e-12


def test_sign_test_reports_all_ties_as_inconclusive() -> None:
    result = sign_test_p_value([0.0, 0.0, 0.0])

    assert result["ties"] == 3
    assert result["p_value"] == 1.0


def test_permutation_is_deterministic_and_small_for_strong_signal() -> None:
    values = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    first = permutation_p_value(values, resamples=3000, seed=3)
    second = permutation_p_value(values, resamples=3000, seed=3)

    assert first == second
    assert float(first["p_value"]) < 0.05


def test_permutation_is_large_for_symmetric_noise() -> None:
    values = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    result = permutation_p_value(values, resamples=3000, seed=0)

    assert float(result["p_value"]) > 0.2


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


def test_run_significance_analysis_clusters_and_is_reproducible() -> None:
    kwargs = dict(
        request_count=4,
        seeds=(3,),
        splits=("validation",),
        workloads=("shared_system_prompt",),
        bootstrap_resamples=2000,
        permutation_resamples=2000,
    )
    payload = run_significance_analysis(DEFAULT_CONFIG_PATH, **kwargs)
    repeat = run_significance_analysis(DEFAULT_CONFIG_PATH, **kwargs)

    assert payload["schema"] == "prefix-kv-cache-score-gap-significance-v1"
    assert payload["candidate"] == "production_incumbent"
    assert payload["baseline"] == "tinylfu_lru"
    # Two capacity tiers, one seed, one workload family.
    assert payload["paired_unit_count"] == 2
    assert len(payload["units"]) == 2
    assert payload["primary_clustering"] == "workload_family"
    # The primary verdict is driven by the family-level clustering.
    assert payload["verdict"] == payload["clustered"]["by_workload_family"]["verdict"]
    assert payload["clustered"]["by_workload_family"]["cluster_count"] == 1
    assert "seed_invariant_cell_count" in payload["seed_degeneracy"]
    assert payload["mean_difference"] == repeat["mean_difference"]
    assert payload["clustered"] == repeat["clustered"]
    assert payload["per_group_naive"] == repeat["per_group_naive"]
    assert "candidate_combined_score" in payload["combined_score_context"]

    report = format_report(payload)
    assert "Score-gap significance analysis" in report
    assert "Cluster-robust significance" in report


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
