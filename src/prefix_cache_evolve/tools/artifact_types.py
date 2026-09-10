"""Typed records shared by research-analysis artifact producers."""

from __future__ import annotations

from typing import Any, TypeAlias, TypedDict

# Analysis artifacts intentionally mix nested records, numbers, and strings. Keep
# that serialization boundary explicit while checking the surrounding algorithms.
ArtifactRecord: TypeAlias = dict[str, Any]


class ScoreIdentityRecord(TypedDict):
    """Identity fields required to compare score-bearing research records."""

    verifier_version: str
    evaluation_context_sha256: str
    panel_sha256: str


class BootstrapConfidenceInterval(TypedDict):
    """Deterministic percentile-bootstrap interval and its configuration."""

    lower: float
    upper: float
    confidence: float
    resamples: int
    seed: int


class PermutationTestResult(TypedDict):
    """Paired permutation-test result and its deterministic configuration."""

    p_value: float
    at_least_as_extreme: int
    resamples: int
    seed: int


class SignTestResult(TypedDict):
    """Exact paired sign-test result."""

    p_value: float
    positive: int
    negative: int
    ties: int


class ClusteredSignificanceResult(TypedDict):
    """Cluster-robust statistical comparison for one clustering strategy."""

    clustering: str
    cluster_count: int
    cluster_wins: int
    cluster_losses: int
    cluster_ties: int
    mean_difference: float
    bootstrap_confidence_interval: BootstrapConfidenceInterval
    confidence_interval_excludes_zero: bool
    permutation_test: PermutationTestResult
    sign_test: SignTestResult
    verdict: str


class SeedDegeneracyResult(TypedDict):
    """Diagnostic identifying pseudo-replicated workload/capacity cells."""

    cell_count: int
    seed_invariant_cell_count: int
    seed_invariant_cells: list[str]
    note: str


class CombinedScoreContext(TypedDict):
    """Headline combined scores retained alongside paired behavioral statistics."""

    candidate_combined_score: float
    baseline_combined_score: float
    combined_score_difference: float
    note: str


class ClusteredComparisons(TypedDict):
    """Supported independent-unit clustering strategies."""

    by_workload_family: ClusteredSignificanceResult
    by_workload_family_capacity: ClusteredSignificanceResult


class NaiveSignificanceResult(TypedDict):
    """Descriptive per-group statistics that do not correct pseudo-replication."""

    note: str
    bootstrap_confidence_interval: BootstrapConfidenceInterval
    confidence_interval_excludes_zero: bool
    permutation_test: PermutationTestResult
    sign_test: SignTestResult


class PairedUnitRecord(TypedDict):
    """Serializable candidate-versus-baseline comparison for one benchmark cell."""

    group: str
    split: str
    workload: str
    capacity_blocks: int
    seed: int
    candidate_score: float
    baseline_score: float
    difference: float


class SignificanceReport(ScoreIdentityRecord):
    """Complete schema for the headline score-gap significance artifact."""

    schema: str
    config: str
    candidate: str
    baseline: str
    splits: list[str]
    workloads: list[str] | None
    request_count: int
    seeds: list[int]
    capacity_blocks: list[int]
    pairing_unit: str
    unit_score_definition: str
    combined_score_context: CombinedScoreContext
    paired_unit_count: int
    candidate_wins: int
    candidate_losses: int
    ties: int
    mean_candidate_score: float
    mean_baseline_score: float
    mean_difference: float
    primary_clustering: str
    verdict: str
    seed_degeneracy: SeedDegeneracyResult
    clustered: ClusteredComparisons
    per_group_naive: NaiveSignificanceResult
    units: list[PairedUnitRecord]
