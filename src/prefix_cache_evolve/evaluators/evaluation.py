"""Prepare evaluation panels, execute trials, and aggregate policy scores."""

from __future__ import annotations

import tracemalloc
from statistics import mean
from typing import Callable, Iterable, Protocol

from prefix_cache_evolve.evaluators.baselines import baseline_lru_blocks
from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.evaluators.contracts import PrefixKVPolicy
from prefix_cache_evolve.evaluators.fingerprints import (
    evaluation_context_sha256,
    panel_sha256,
    request_stream_fingerprint_record,
)
from prefix_cache_evolve.evaluators.panels import (
    PreparedWorkload,
    prepare_workloads,
    trace_geometry_overrides,
)
from prefix_cache_evolve.evaluators.policy_factory import FixedAdmissionPolicy, build_policy
from prefix_cache_evolve.evaluators.results import EvaluationResult, TrialMetrics
from prefix_cache_evolve.evaluators.scoring import aggregate_by as _aggregate_by
from prefix_cache_evolve.evaluators.scoring import workload_base_score as _workload_base_score
from prefix_cache_evolve.evaluators.simulator import PrefixKVCacheSimulator
from prefix_cache_evolve.evaluators.trace_replay import TRACE_REPLAY_CONTRACT
from prefix_cache_evolve.evaluators.workloads import WorkloadRequest, build_workload


class Simulation(Protocol):
    """The trial execution capability required by the evaluator."""

    def run(
        self,
        policy: PrefixKVPolicy,
        requests: tuple[WorkloadRequest, ...],
        *,
        split: str,
        workload: str,
        seed: int,
        scoring_fn_complexity: int = 0,
    ) -> TrialMetrics:
        """Run one policy on a fixed request stream."""
        ...


class PrefixKVCacheEvaluator:
    """Callable evaluator compatible with Levi-style candidate factories."""

    def __init__(
        self,
        config: EvaluatorConfig | None = None,
        *,
        splits: tuple[str, ...] = ("train", "validation"),
        expose_future_reuse: bool = False,
        simulator_factory: Callable[..., Simulation] = PrefixKVCacheSimulator,
        workload_builder: Callable[..., tuple[WorkloadRequest, ...]] | None = None,
        fixed_admission_factory: Callable[..., PrefixKVPolicy] | None = None,
        record_admission_diagnostics: bool = False,
        oracle_admission: bool = False,
        oracle_eviction: bool = False,
    ) -> None:
        self.config = config or EvaluatorConfig()
        self.splits = splits
        self.expose_future_reuse = expose_future_reuse
        self._simulator_factory = simulator_factory
        self._workload_builder = workload_builder or build_workload
        self._fixed_admission_factory = fixed_admission_factory
        self._record_admission_diagnostics = record_admission_diagnostics
        self._oracle_admission = oracle_admission
        self._oracle_eviction = oracle_eviction

    def __call__(
        self,
        factory: Callable[..., PrefixKVPolicy] | None = None,
        *,
        scoring_fn_complexity: int = 0,
    ) -> EvaluationResult:
        """Evaluate a policy factory across configured workloads."""
        factory = factory or baseline_lru_blocks
        trials: list[TrialMetrics] = []
        prepared = prepare_workloads(
            self.config,
            splits=self.splits,
            workload_builder=self._workload_builder,
        )
        workload_streams: dict[tuple[str, str], list[PreparedWorkload]] = {}
        for stream in prepared:
            workload_streams.setdefault((stream.split, stream.family), []).append(stream)
        panel_sha = self._workload_panel_sha(
            self.config.effective_capacity_blocks(),
            [stream.fingerprint() for stream in prepared],
        )
        for streams in workload_streams.values():
            geometries = {(stream.block_size_tokens, stream.capacity_blocks) for stream in streams}
            if len(geometries) != 1:
                raise ValueError("one workload family must use one evaluation geometry")
            block_size_tokens, capacity_blocks_values = next(iter(geometries))
            for capacity_blocks in capacity_blocks_values:
                for stream in streams:
                    trials.append(
                        self._run_trial(
                            factory,
                            stream.requests,
                            split=stream.split,
                            workload=stream.family,
                            seed=stream.actual_seed,
                            capacity_blocks=capacity_blocks,
                            block_size_tokens=block_size_tokens,
                            scoring_fn_complexity=scoring_fn_complexity,
                        )
                    )

        return self._result_from_trials(
            trials,
            scoring_fn_complexity,
            panel_sha=panel_sha,
        )

    def evaluate_requests(
        self,
        factory: Callable[..., PrefixKVPolicy] | None,
        requests: Iterable[WorkloadRequest],
        *,
        workload: str = "trace_replay",
        split: str = "validation",
        seed: int = 0,
        scoring_fn_complexity: int = 0,
    ) -> EvaluationResult:
        """Evaluate a fixed metadata-derived request sequence."""
        factory = factory or baseline_lru_blocks
        request_tuple = tuple(requests)
        stream = request_stream_fingerprint_record(
            request_tuple,
            split=split,
            family=workload,
            base_seed=seed,
            seed_offset=0,
            actual_seed=seed,
        )
        panel_sha = panel_sha256(
            evaluation=self._panel_evaluation_metadata(
                self.config.effective_capacity_blocks(),
                splits=(split,),
                request_count=len(request_tuple),
                base_seeds=(seed,),
                family_request_multipliers={},
                stream_count=1,
            ),
            streams=(stream,),
        )
        trials = [
            self._run_trial(
                factory,
                request_tuple,
                split=split,
                workload=workload,
                seed=seed,
                capacity_blocks=capacity_blocks,
                block_size_tokens=self.config.block_size_tokens,
                scoring_fn_complexity=scoring_fn_complexity,
            )
            for capacity_blocks in self.config.effective_capacity_blocks()
        ]
        return self._result_from_trials(
            trials,
            scoring_fn_complexity,
            panel_sha=panel_sha,
        )

    def rescore_trials(
        self,
        trials: Iterable[TrialMetrics],
        *,
        scoring_fn_complexity: int = 0,
    ) -> EvaluationResult:
        """Reaggregate existing trials under this evaluator's score weights."""
        trial_list = list(trials)
        panel_shas = {trial.panel_sha256 for trial in trial_list}
        if "" in panel_shas or len(panel_shas) != 1:
            raise ValueError("rescoring requires trials from exactly one fingerprinted panel")
        return self._result_from_trials(
            trial_list,
            scoring_fn_complexity,
            panel_sha=next(iter(panel_shas)),
        )

    def _workload_panel_sha(
        self,
        capacity_blocks: tuple[int, ...],
        streams: list[dict[str, object]],
    ) -> str:
        """Return the manifest-compatible hash for the configured workload panel."""
        return panel_sha256(
            evaluation=self._panel_evaluation_metadata(
                capacity_blocks,
                splits=self.splits,
                request_count=self.config.request_count,
                base_seeds=self.config.seeds,
                family_request_multipliers=dict(
                    sorted(self.config.family_request_multipliers.items())
                ),
                stream_count=len(streams),
            ),
            streams=streams,
        )

    def _panel_evaluation_metadata(
        self,
        capacity_blocks: tuple[int, ...],
        *,
        splits: tuple[str, ...],
        request_count: int,
        base_seeds: tuple[int, ...],
        family_request_multipliers: dict[str, int],
        stream_count: int,
    ) -> dict[str, object]:
        """Return the canonical panel configuration used by workload manifests."""
        metadata = {
            "splits": list(splits),
            "capacity_blocks": list(capacity_blocks),
            "capacity_tokens": [
                capacity * self.config.block_size_tokens for capacity in capacity_blocks
            ],
            "physical_block_size_tokens": self.config.block_size_tokens,
            "workload_token_granularity": (self.config.effective_workload_token_granularity()),
            "request_count": request_count,
            "base_seeds": list(base_seeds),
            "family_request_multipliers": family_request_multipliers,
            "stream_count": stream_count,
        }
        geometry_overrides = trace_geometry_overrides(self.config, splits=splits)
        if geometry_overrides:
            metadata["trace_geometry_overrides"] = geometry_overrides
        return metadata

    def _run_trial(
        self,
        factory: Callable[..., PrefixKVPolicy],
        requests: tuple[WorkloadRequest, ...],
        *,
        split: str,
        workload: str,
        seed: int,
        capacity_blocks: int,
        block_size_tokens: int,
        scoring_fn_complexity: int,
    ) -> TrialMetrics:
        simulator = self._simulator_factory(
            capacity_blocks=capacity_blocks,
            block_size_tokens=block_size_tokens,
            prefill_cost_per_token=self.config.prefill_cost_per_token,
            lookup_cost_per_block=self.config.lookup_cost_per_block,
            eviction_cost_per_block=self.config.eviction_cost_per_block,
            active_tokens_per_step=self.config.active_tokens_per_step,
            kv_capacity_mode=self.config.kv_capacity_mode,
            expose_future_reuse=self.expose_future_reuse,
            max_memory_bytes=self.config.max_memory_bytes,
            record_admission_diagnostics=self._record_admission_diagnostics,
            oracle_admission=self._oracle_admission,
            oracle_eviction=self._oracle_eviction,
        )
        tracing_already_started = tracemalloc.is_tracing()
        if tracing_already_started:
            tracemalloc.reset_peak()
        else:
            tracemalloc.start()
        try:
            try:
                policy = build_policy(
                    factory,
                    capacity_blocks,
                    block_size_tokens,
                    self.config.policy_seed,
                )
                if self._fixed_admission_factory is not None:
                    admission_policy = build_policy(
                        self._fixed_admission_factory,
                        capacity_blocks,
                        block_size_tokens,
                        self.config.policy_seed,
                    )
                    policy = FixedAdmissionPolicy(
                        admission_policy=admission_policy,
                        eviction_policy=policy,
                    )
            except Exception as exc:
                return TrialMetrics(
                    split=split,
                    workload=workload,
                    seed=seed,
                    capacity_blocks=capacity_blocks,
                    block_size_tokens=block_size_tokens,
                    scoring_fn_complexity=scoring_fn_complexity,
                    invalid=True,
                    invalid_reason=f"factory raised {type(exc).__name__}",
                )
            trial = simulator.run(
                policy,
                requests,
                split=split,
                workload=workload,
                seed=seed,
                scoring_fn_complexity=scoring_fn_complexity,
            )
            _, peak_memory_bytes = tracemalloc.get_traced_memory()
            if self.config.max_memory_bytes and peak_memory_bytes > self.config.max_memory_bytes:
                return TrialMetrics(
                    split=split,
                    workload=workload,
                    seed=seed,
                    capacity_blocks=capacity_blocks,
                    block_size_tokens=block_size_tokens,
                    scoring_fn_complexity=scoring_fn_complexity,
                    invalid=True,
                    invalid_reason=(
                        f"candidate used {peak_memory_bytes} bytes "
                        f"(> {self.config.max_memory_bytes})"
                    ),
                )
            return trial
        finally:
            if not tracing_already_started:
                tracemalloc.stop()

    def _result_from_trials(
        self,
        trials: list[TrialMetrics],
        scoring_fn_complexity: int,
        *,
        panel_sha: str,
    ) -> EvaluationResult:
        for trial in trials:
            trial.panel_sha256 = panel_sha
        context_sha = evaluation_context_sha256(
            verifier_version=self.config.verifier_version,
            evaluator_config=self.config.model_dump(mode="json"),
            panel_sha=panel_sha,
        )
        invalid_fraction = (
            sum(1 for trial in trials if trial.invalid) / len(trials) if trials else 1.0
        )
        selection_trials = [trial for trial in trials if trial.split != "probe"]
        if not selection_trials:
            selection_trials = trials
        selection_invalid_fraction = (
            sum(1 for trial in selection_trials if trial.invalid) / len(selection_trials)
            if selection_trials
            else 1.0
        )
        split_metrics = _aggregate_by((trial.split for trial in trials), trials)
        workload_metrics = _aggregate_by(
            (f"{trial.split}/{trial.workload}" for trial in trials), trials
        )
        block_sizes = {trial.block_size_tokens for trial in trials}
        capacity_metrics = _aggregate_by(
            (
                (
                    f"block_{trial.block_size_tokens}_capacity_{trial.capacity_blocks}"
                    if len(block_sizes) > 1
                    else f"capacity_{trial.capacity_blocks}"
                )
                for trial in trials
            ),
            trials,
        )
        score_breakdown = self._score_breakdown(
            trials,
            selection_invalid_fraction,
            scoring_fn_complexity,
        )
        return EvaluationResult(
            verifier_version=self.config.verifier_version,
            evaluation_context_sha256=context_sha,
            panel_sha256=panel_sha,
            combined_score=score_breakdown["combined_score"],
            success=selection_invalid_fraction == 0.0,
            invalid_fraction=selection_invalid_fraction,
            split_metrics=split_metrics,
            workload_metrics=workload_metrics,
            capacity_metrics=capacity_metrics,
            candidate_metadata={
                **(
                    {"trace_replay_contract": TRACE_REPLAY_CONTRACT}
                    if self.config.trace_workloads
                    else {}
                ),
                "verifier_version": self.config.verifier_version,
                "evaluation_context_sha256": context_sha,
                "panel_sha256": panel_sha,
                "capacity_blocks": self.config.capacity_blocks,
                "capacity_sweep_blocks": ",".join(
                    str(value) for value in self.config.effective_capacity_blocks()
                ),
                "capacity_sweep_tokens": ",".join(
                    str(value) for value in self.config.effective_capacity_tokens()
                ),
                "block_size_tokens": self.config.block_size_tokens,
                "kv_capacity_mode": self.config.kv_capacity_mode,
                "workload_token_granularity": self.config.effective_workload_token_granularity(),
                "policy_seed": self.config.policy_seed,
                "scoring_fn_complexity": scoring_fn_complexity,
                "churn_weight": self.config.churn_weight,
                "underfill_weight": self.config.underfill_weight,
                "fairness_weight": self.config.fairness_weight,
                "complexity_weight": self.config.k_complex,
                "complexity_exponent": self.config.complexity_exponent,
                "complexity_mode": (
                    "form_aware" if self.config.form_aware_complexity else "legacy_ast_nodes"
                ),
                "min_workload_weight": self.config.min_workload_weight,
                "min_seed_weight": self.config.min_seed_weight,
                "request_tail_weight": self.config.request_tail_weight,
                "worst_window_weight": self.config.worst_window_weight,
                "priority_hit_weight": self.config.priority_hit_weight,
                "wasted_admission_weight": self.config.wasted_admission_weight,
                "wasted_admission_metric": "wasted_admission_token_rate",
                "admission_utility_weight": self.config.admission_utility_weight,
                "avoidable_eviction_weight": self.config.avoidable_eviction_weight,
                "latency_norm": self.config.latency_norm,
                "latency_norm_scope": (
                    "configured" if self.config.latency_norm > 0.0 else "workload_capacity"
                ),
                "expose_future_reuse": self.expose_future_reuse,
                "oracle_admission": self._oracle_admission,
                "oracle_eviction": self._oracle_eviction,
                "fixed_admission_policy": self.config.fixed_admission_policy or "",
                "reporting_invalid_fraction": invalid_fraction,
                "selection_invalid_fraction": selection_invalid_fraction,
            },
            score_breakdown=score_breakdown,
            trials=tuple(trials),
        )

    def _score_trials(
        self,
        trials: list[TrialMetrics],
        invalid_fraction: float,
        complexity: int,
    ) -> float:
        return self._score_breakdown(
            trials,
            invalid_fraction,
            complexity,
        )["combined_score"]

    def _score_breakdown(
        self,
        trials: list[TrialMetrics],
        invalid_fraction: float,
        complexity: int,
    ) -> dict[str, float]:
        """Returns the combined score and its top-level weighted components."""
        if invalid_fraction > 0.0:
            combined_score = (
                self.config.v_min - 1.0 - self.config.invalid_surcharge * invalid_fraction
            )
            return {
                "combined_score": combined_score,
                "invalid_fraction": invalid_fraction,
                "invalid_surcharge_cost": (self.config.invalid_surcharge * invalid_fraction),
            }
        validation = [trial for trial in trials if trial.split == "validation"]
        if not validation:
            validation = [trial for trial in trials if trial.split not in {"hidden", "probe"}]
        if not validation:
            validation = [trial for trial in trials if trial.split == "hidden"]
        if not validation:
            validation = [trial for trial in trials if trial.split == "probe"]
        by_workload_capacity: dict[tuple[str, int, int], list[TrialMetrics]] = {}
        for trial in validation:
            by_workload_capacity.setdefault(
                (trial.workload, trial.block_size_tokens, trial.capacity_blocks),
                [],
            ).append(trial)
        workload_scores = []
        min_seed_weight = min(1.0, max(0.0, self.config.min_seed_weight))
        for workload_trials in by_workload_capacity.values():
            average_score = _workload_base_score(
                workload_trials,
                token_weight=self.config.w_avg_tok,
                block_weight=self.config.w_avg_blk,
                request_tail_weight=self.config.request_tail_weight,
                worst_window_weight=self.config.worst_window_weight,
                priority_hit_weight=self.config.priority_hit_weight,
                wasted_admission_weight=self.config.wasted_admission_weight,
                admission_utility_weight=self.config.admission_utility_weight,
                avoidable_eviction_weight=self.config.avoidable_eviction_weight,
                latency_weight=self.config.latency_weight,
                latency_cap=self.config.latency_cap,
                latency_norm=self.config.latency_norm,
            )
            seed_floor = min(
                _workload_base_score(
                    [trial],
                    token_weight=self.config.w_avg_tok,
                    block_weight=self.config.w_avg_blk,
                    request_tail_weight=self.config.request_tail_weight,
                    worst_window_weight=self.config.worst_window_weight,
                    priority_hit_weight=self.config.priority_hit_weight,
                    wasted_admission_weight=self.config.wasted_admission_weight,
                    admission_utility_weight=self.config.admission_utility_weight,
                    avoidable_eviction_weight=self.config.avoidable_eviction_weight,
                    latency_weight=self.config.latency_weight,
                    latency_cap=self.config.latency_cap,
                    latency_norm=self.config.latency_norm,
                )
                for trial in workload_trials
            )
            workload_scores.append(
                (1.0 - min_seed_weight) * average_score + min_seed_weight * seed_floor
            )
        mean_score = mean(workload_scores) if workload_scores else 0.0
        min_workload_score = min(workload_scores) if workload_scores else 0.0
        workload_capacity_groups = tuple(by_workload_capacity.values())
        churn = (
            mean(
                mean(trial.cache_churn_per_1k for trial in workload_trials)
                for workload_trials in workload_capacity_groups
            )
            if workload_capacity_groups
            else 0.0
        )
        underfill = (
            mean(
                mean(trial.policy_underfill_rate for trial in workload_trials)
                for workload_trials in workload_capacity_groups
            )
            if workload_capacity_groups
            else 0.0
        )
        fairness = (
            mean(
                trial.tenant_fairness_penalty
                for trial in validation
                if trial.workload == "multi_tenant_skew"
            )
            if any(trial.workload == "multi_tenant_skew" for trial in validation)
            else 0.0
        )
        churn_cost = min(self.config.churn_cap, self.config.churn_weight * churn)
        underfill_cost = min(
            self.config.underfill_cap,
            self.config.underfill_weight * underfill,
        )
        fairness_cost = min(
            self.config.fairness_cap,
            self.config.fairness_weight * fairness,
        )
        complexity_cost = self.config.k_complex * complexity**self.config.complexity_exponent
        min_workload_contribution = self.config.min_workload_weight * min_workload_score
        combined_score = (
            mean_score
            + min_workload_contribution
            - churn_cost
            - underfill_cost
            - fairness_cost
            - complexity_cost
        )
        return {
            "combined_score": combined_score,
            "mean_workload_score": mean_score,
            "min_workload_score": min_workload_score,
            "min_workload_contribution": min_workload_contribution,
            "churn_cost": churn_cost,
            "policy_underfill_rate": underfill,
            "underfill_cost": underfill_cost,
            "fairness_cost": fairness_cost,
            "complexity_cost": complexity_cost,
            "validation_trial_count": float(len(validation)),
        }
