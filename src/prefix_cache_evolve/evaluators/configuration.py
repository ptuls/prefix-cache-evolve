"""Validated configuration for prefix KV-cache evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from prefix_cache_evolve.evaluators.verifier import (
    VERIFIER_VERSION,
    VERIFIER_VERSION_PATTERN,
)


def _flatten_scoring_settings(value: object) -> object:
    """Flatten a nested scoring mapping into evaluator settings."""
    if not isinstance(value, Mapping):
        return value
    values = dict(value)
    scoring = values.pop("scoring", None)
    if scoring is None:
        return values
    if not isinstance(scoring, Mapping):
        raise ValueError("scoring must be a mapping")
    duplicates = sorted(set(values).intersection(scoring))
    if duplicates:
        raise ValueError(
            "scoring fields must not also appear at the settings root: " + ", ".join(duplicates)
        )
    values.update(scoring)
    return values


@dataclass
class WorkloadConfig:
    """Configures one workload family inside one split."""

    family: str
    split: str
    request_count: int = 96
    seed_offset: int = 0


class TraceTimeWindow(BaseModel):
    """A half-open chronological window in one hash-pinned source capture.

    This establishes time separation, not session or tenant independence.
    Timestamps retain the source clock; each simulator trial starts cold.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    start_ms: float = Field(ge=0, allow_inf_nan=False)
    end_ms: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_bounds(self) -> TraceTimeWindow:
        if self.end_ms <= self.start_ms:
            raise ValueError("trace time window must have start_ms < end_ms")
        return self


class TraceWorkloadConfig(BaseModel):
    """Pin one complete metadata-only trace to an evaluation split."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    family: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    split: Literal["train", "validation", "probe", "hidden"]
    block_size_tokens: PositiveInt
    capacity_sweep_blocks: tuple[PositiveInt, ...] = ()
    request_count: PositiveInt
    arrival_bucket_ms: PositiveInt = 100
    time_window: TraceTimeWindow | None = None

    def effective_capacity_blocks(
        self,
        default_capacity_blocks: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Return trace-specific capacities or the evaluator defaults."""
        values = self.capacity_sweep_blocks or default_capacity_blocks
        return tuple(dict.fromkeys(int(value) for value in values))


class EvaluatorConfig(BaseModel):
    """Configuration for prefix KV-cache evaluation and scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    verifier_version: str = Field(default=VERIFIER_VERSION, pattern=VERIFIER_VERSION_PATTERN)
    capacity_blocks: PositiveInt = 24
    capacity_sweep_blocks: tuple[PositiveInt, ...] = ()
    block_size_tokens: PositiveInt = 16
    workload_token_granularity: PositiveInt = 8
    seeds: tuple[int, ...] = (11, 23, 37)
    policy_seed: int = 0
    train_families: tuple[str, ...] = (
        "shared_system_prompt",
        "rag_template_reuse",
        "long_context_mixed",
        "session_continuation_growth",
        "agentic_tool_workflows",
    )
    validation_families: tuple[str, ...] = (
        "phase_shift_prompts",
        "multi_tenant_skew",
        "hotset_cold_scan",
        "concurrent_long_generation",
        "stochastic_serving_mix",
        "rolling_template_versions",
        "heavy_tailed_prefix_lengths",
        "priority_burst_recovery",
        "priority_one_off_noise",
        "tenant_phase_shift_cycles",
    )
    probe_families: tuple[str, ...] = (
        "agent_trace_branching",
        "cyclic_working_set_pressure",
    )
    hidden_families: tuple[str, ...] = (
        "adversarial_unique_prompts",
        "cross_family_mixture",
        "tenant_session_reentry",
        "stochastic_serving_mix_shifted",
        "rolling_template_versions_shifted",
        "heavy_tailed_prefix_lengths_shifted",
        "priority_burst_recovery_shifted",
        "cyclic_working_set_pressure_shifted",
        "priority_one_off_noise_shifted",
        "tenant_phase_shift_cycles_shifted",
    )
    trace_workloads: tuple[TraceWorkloadConfig, ...] = ()
    request_count: PositiveInt = 96
    family_request_multipliers: dict[str, PositiveInt] = Field(
        default_factory=lambda: {
            "tenant_phase_shift_cycles": 3,
            "tenant_phase_shift_cycles_shifted": 4,
        }
    )
    prefill_cost_per_token: NonNegativeFloat = 1.0
    lookup_cost_per_block: NonNegativeFloat = 0.035
    eviction_cost_per_block: NonNegativeFloat = 0.2
    active_tokens_per_step: PositiveInt = 64
    kv_capacity_mode: Literal["prefix_only", "shared"] = "prefix_only"
    w_avg_tok: NonNegativeFloat = 80.0
    w_avg_blk: NonNegativeFloat = 60.0
    min_workload_weight: NonNegativeFloat = 0.5
    min_seed_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    request_tail_weight: NonNegativeFloat = 12.0
    worst_window_weight: NonNegativeFloat = 12.0
    priority_hit_weight: NonNegativeFloat = 8.0
    wasted_admission_weight: NonNegativeFloat = 6.0
    admission_utility_weight: NonNegativeFloat = 1.0
    avoidable_eviction_weight: NonNegativeFloat = 8.0
    latency_norm: NonNegativeFloat = 0.0
    latency_weight: NonNegativeFloat = 35.0
    latency_cap: NonNegativeFloat = 40.0
    churn_weight: NonNegativeFloat = 0.015
    churn_cap: NonNegativeFloat = 25.0
    underfill_weight: NonNegativeFloat = 12.0
    underfill_cap: NonNegativeFloat = 15.0
    fairness_weight: NonNegativeFloat = 80.0
    fairness_cap: NonNegativeFloat = 30.0
    k_complex: NonNegativeFloat = 0.065
    complexity_exponent: PositiveFloat = 0.75
    v_min: float = -1_000.0
    invalid_surcharge: NonNegativeFloat = 1_000.0
    timeout_s: PositiveFloat = 30.0
    max_memory_bytes: PositiveInt = 64 * 1024 * 1024
    sandbox_image: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:/@-]*$")
    form_aware_complexity: bool = False
    max_candidate_complexity: PositiveInt | None = None
    promotion_max_candidate_complexity: PositiveInt | None = None
    surrogate_probe_tripwire_thresholds: dict[str, NonNegativeFloat] = Field(
        default_factory=lambda: {
            "agentic_branching": 0.12,
            "cyclic_working_set": 0.25,
        }
    )
    fixed_admission_policy: str | None = None
    candidate_policy_surface: Literal["full", "eviction_only"] = "full"
    search_score_mode: Literal["combined", "raw_before_complexity", "robust_min"] = "combined"
    search_guidance_families: tuple[str, ...] = ()
    reject_unsupported_source_patterns: bool = False

    @field_validator("verifier_version")
    @classmethod
    def _require_implemented_verifier_version(cls, value: str) -> str:
        if value != VERIFIER_VERSION:
            raise ValueError(f"this checkout implements verifier {VERIFIER_VERSION}, not {value}")
        return value

    @model_validator(mode="before")
    @classmethod
    def _flatten_scoring_settings(cls, value: object) -> object:
        """Accept the YAML scoring subsection while retaining a flat runtime API."""
        return _flatten_scoring_settings(value)

    @model_validator(mode="after")
    def _validate_tripwire_channels(self) -> EvaluatorConfig:
        """Require an explicit threshold for every supported tripwire channel."""
        expected = {"agentic_branching", "cyclic_working_set"}
        configured = set(self.surrogate_probe_tripwire_thresholds)
        if configured != expected:
            missing = sorted(expected - configured)
            unknown = sorted(configured - expected)
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            raise ValueError(
                "surrogate_probe_tripwire_thresholds must configure exactly "
                f"{sorted(expected)} ({'; '.join(details)})"
            )
        return self

    @model_validator(mode="after")
    def _validate_search_guidance(self) -> EvaluatorConfig:
        """Require robust search guidance to use non-quarantined train families."""
        guidance = set(self.search_guidance_families)
        if self.search_score_mode != "robust_min":
            if guidance:
                raise ValueError("search_guidance_families requires search_score_mode='robust_min'")
            return self
        if not guidance:
            raise ValueError("robust_min search requires at least one search guidance family")
        trace_train = {trace.family for trace in self.trace_workloads if trace.split == "train"}
        unknown = guidance - (set(self.train_families) | trace_train)
        if unknown:
            raise ValueError(
                "search guidance families must be configured train families: "
                + ", ".join(sorted(unknown))
            )
        quarantined = guidance & (set(self.probe_families) | set(self.hidden_families))
        if quarantined:
            raise ValueError(
                "search guidance families must not be probe or hidden families: "
                + ", ".join(sorted(quarantined))
            )
        return self

    @model_validator(mode="after")
    def _validate_trace_workloads(self) -> EvaluatorConfig:
        """Reject ambiguous streams and attempts to reinterpret opaque blocks."""
        names = {
            (workload.split, workload.family)
            for workload in self.workload_configs(("train", "validation", "probe", "hidden"))
        }
        content_splits: dict[str, str] = {}
        for trace in self.trace_workloads:
            if (
                trace.block_size_tokens != self.block_size_tokens
                and not trace.capacity_sweep_blocks
            ):
                raise ValueError(
                    "trace block_size_tokens must match the evaluator; reconvert the "
                    "source or define its own capacity_sweep_blocks for mixed geometry"
                )
            name = (trace.split, trace.family)
            if name in names:
                raise ValueError(f"duplicate workload {trace.split}/{trace.family}")
            names.add(name)
            previous_split = content_splits.setdefault(trace.sha256, trace.split)
            if previous_split != trace.split:
                raise ValueError("the same trace content must not appear in different splits")
        by_source: dict[str, list[TraceWorkloadConfig]] = {}
        for trace in self.trace_workloads:
            if trace.time_window is not None:
                by_source.setdefault(trace.time_window.source_sha256, []).append(trace)
        split_order = {"train": 0, "validation": 1, "probe": 2, "hidden": 3}
        for traces in by_source.values():
            ordered = sorted(traces, key=lambda trace: trace.time_window.start_ms)  # type: ignore[union-attr]
            for left, right in zip(ordered, ordered[1:]):
                assert left.time_window is not None and right.time_window is not None
                if left.time_window.end_ms > right.time_window.start_ms:
                    raise ValueError("trace time windows from the same source must not overlap")
                if split_order[left.split] > split_order[right.split]:
                    raise ValueError(
                        "trace time splits must follow train/validation/probe/hidden order"
                    )
        return self

    def with_updates(self, **updates: object) -> EvaluatorConfig:
        """Return a validated copy with the supplied settings overlaid."""
        normalized = _flatten_scoring_settings(updates)
        if not isinstance(normalized, Mapping):
            raise TypeError("evaluator config updates must be a mapping")
        return type(self).model_validate({**self.model_dump(), **dict(normalized)})

    def effective_capacity_blocks(self) -> tuple[int, ...]:
        """Returns the capacities evaluated for each workload and seed."""
        values = self.capacity_sweep_blocks or (self.capacity_blocks,)
        capacities: list[int] = []
        for value in values:
            capacity = int(value)
            if capacity <= 0:
                raise ValueError("capacity blocks must be positive")
            if capacity not in capacities:
                capacities.append(capacity)
        return tuple(capacities)

    def effective_capacity_tokens(self) -> tuple[int, ...]:
        """Returns evaluated cache capacities expressed in tokens."""
        if self.block_size_tokens <= 0:
            raise ValueError("block size tokens must be positive")
        return tuple(
            capacity * self.block_size_tokens for capacity in self.effective_capacity_blocks()
        )

    def effective_workload_token_granularity(self) -> int:
        """Returns the canonical token granularity used to build synthetic traffic."""
        granularity = int(self.workload_token_granularity)
        if granularity <= 0:
            raise ValueError("workload token granularity must be positive")
        return granularity

    def workload_configs(self, splits: Iterable[str]) -> tuple[WorkloadConfig, ...]:
        """Return expanded workload configurations for the requested splits."""
        configs: list[WorkloadConfig] = []
        for split in splits:
            families = {
                "train": self.train_families,
                "validation": self.validation_families,
                "probe": self.probe_families,
                "hidden": self.hidden_families,
            }[split]
            for index, family in enumerate(families):
                multiplier = int(self.family_request_multipliers.get(family, 1))
                if multiplier <= 0:
                    raise ValueError("family request multipliers must be positive")
                configs.append(
                    WorkloadConfig(
                        family=family,
                        split=split,
                        request_count=self.request_count * multiplier,
                        seed_offset=1000 * (index + 1),
                    )
                )
        return tuple(configs)
