"""Tests for the prefix KV-cache evaluator."""

from __future__ import annotations

import json
import math
import os
import textwrap
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluationResult,
    EvaluatorConfig,
    PrefixBlockInfo,
    PrefixKVCacheEvaluator,
    PrefixKVCacheSimulator,
    RequestInfo,
    TrialMetrics,
    WorkloadRequest,
    _AdmissionAudit,
    _aggregate_trials,
    baseline_depth_prefer_shallow,
    baseline_future_reuse_heuristic,
    baseline_lfu_blocks,
    baseline_lru_blocks,
    baseline_no_cache,
    baseline_oracle_future_reuse,
    baseline_prefix_fanout,
    baseline_tenant_fair_lru,
    baseline_tinylfu_lru,
    baseline_vllm_apc,
    build_workload,
    scoring_fn_complexity,
)
from prefix_cache_evolve.evaluators.verifier import (
    VERIFIER_VERSION,
    require_single_score_identity,
)
from prefix_cache_evolve.problems.prefix_kv_cache import evaluator as levi_evaluator
from prefix_cache_evolve.problems.prefix_kv_cache import runner as prefix_runner
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import (
    DEFAULT_CONFIG_PATH,
    PREFIX_KV_CONFIG_ENV,
    PREFIX_KV_QUICK_ENV,
    active_evaluator_config,
    evaluator_config_from_settings,
    load_evaluator_config,
)
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents.registry import current_incumbent
from prefix_cache_evolve.problems.prefix_kv_cache.runner import (
    _baseline_report_headline,
    _config_from_args,
    _score_weight_sensitivity_rows,
    compare_baselines,
    save_run_artifacts,
    write_baseline_plots,
)
from prefix_cache_evolve.problems.prefix_kv_cache.specialist import (
    candidate_evaluator,
    compose_eviction_specialist_source,
    eviction_only_source_violations,
    fixed_admission_factory,
)
from prefix_cache_evolve.problems.prefix_kv_cache.utilities import (
    agentic_surrogate_probe_tripwire,
)
from tests.support import score_identity, score_record


class AdmitAllLRU:
    def on_request_start(self, request, now: int) -> None:
        return None

    def score_admission(self, block, now: int) -> float:
        return 1.0

    def score_eviction(self, block, now: int) -> float:
        return float(now - block.last_accessed_at)

    def on_cache_hit(self, block, request, now: int) -> None:
        return None

    def on_cache_miss(self, block, request, now: int) -> None:
        return None


def _block_info(**overrides) -> PrefixBlockInfo:
    values = {
        "block_id": 1,
        "prefix_hash": 1,
        "parent_hash": None,
        "depth": 2,
        "start_token": 0,
        "end_token": 8,
        "token_count": 8,
        "tenant_id": 0,
        "created_at": 0,
        "last_accessed_at": 3,
        "hit_count": 0,
        "descendant_count": 5,
        "active_ref_count": 0,
        "estimated_recompute_cost": 8.0,
    }
    values.update(overrides)
    return PrefixBlockInfo(**values)


def _report_result(
    score: float,
    *,
    capacities: tuple[int, ...] = (8, 16),
) -> EvaluationResult:
    split_metrics = {
        "block_hit_rate": 0.4,
        "token_hit_rate": 0.5,
        "worst_quarter_token_hit_rate": 0.3,
        "request_token_hit_rate_p10": 0.2,
        "wasted_admission_token_rate": 0.1,
        "admission_token_utility": 1.5,
        "avoidable_eviction_rate": 0.05,
        "policy_underfill_rate": 0.04,
        "cache_churn_per_1k": 10.0,
    }
    workload_metrics = {
        "token_hit_rate": 0.5,
        "block_hit_rate": 0.4,
        "cache_churn_per_1k": 10.0,
        "priority_weighted_token_hit_rate": 0.55,
    }
    return EvaluationResult(
        **score_identity(),
        combined_score=score,
        success=True,
        invalid_fraction=0.0,
        split_metrics={
            "validation": dict(split_metrics),
            "probe": dict(split_metrics),
        },
        workload_metrics={
            "validation/priority_burst_recovery": dict(workload_metrics),
            "validation/priority_one_off_noise": dict(workload_metrics),
            "probe/agent_trace_branching": dict(workload_metrics),
        },
        capacity_metrics={
            f"capacity_{capacity}": dict(workload_metrics) for capacity in capacities
        },
        candidate_metadata={"scoring_fn_complexity": 3},
        score_breakdown={
            "combined_score": score,
            "mean_workload_score": score,
            "min_workload_contribution": 0.0,
            "churn_cost": 0.0,
            "underfill_cost": 0.0,
            "fairness_cost": 0.0,
            "complexity_cost": 0.0,
        },
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


def test_evaluator_accepts_injected_workload_and_simulator_dependencies() -> None:
    calls = []

    def workload_builder(family, *, request_count, block_size_tokens, seed):
        calls.append(("workload", family, request_count, block_size_tokens, seed))
        return ()

    class FakeSimulator:
        def __init__(self, **kwargs):
            calls.append(
                (
                    "simulator",
                    kwargs["capacity_blocks"],
                    kwargs["block_size_tokens"],
                )
            )

        def run(
            self,
            policy,
            requests,
            *,
            split,
            workload,
            seed,
            scoring_fn_complexity,
        ):
            calls.append(("run", split, workload, seed, len(requests)))
            return TrialMetrics(
                split=split,
                workload=workload,
                seed=seed,
                capacity_blocks=4,
                scoring_fn_complexity=scoring_fn_complexity,
            )

    config = EvaluatorConfig(
        capacity_blocks=4,
        request_count=1,
        seeds=(3,),
        validation_families=("shared_system_prompt",),
    )
    evaluator = PrefixKVCacheEvaluator(
        config,
        splits=("validation",),
        simulator_factory=FakeSimulator,
        workload_builder=workload_builder,
    )

    result = evaluator(baseline_lru_blocks)

    assert result.success is True
    assert calls == [
        ("workload", "shared_system_prompt", 1, 8, 1003),
        ("simulator", 4, 16),
        ("run", "validation", "shared_system_prompt", 1003, 0),
    ]


def test_evaluator_can_fix_admission_while_candidate_controls_eviction() -> None:
    config = EvaluatorConfig(
        capacity_blocks=4,
        request_count=12,
        seeds=(3,),
        train_families=("shared_system_prompt",),
        validation_families=(),
        probe_families=(),
        fixed_admission_policy="unit_no_cache",
    )
    normal = PrefixKVCacheEvaluator(
        config.with_updates(fixed_admission_policy=None),
        splits=("train",),
    )(lambda *_: AdmitAllLRU())
    specialist = PrefixKVCacheEvaluator(
        config,
        splits=("train",),
        fixed_admission_factory=baseline_no_cache,
    )(lambda *_: AdmitAllLRU())

    assert normal.split_metrics["train"]["admission_count"] > 0
    assert specialist.split_metrics["train"]["admission_count"] == 0
    assert specialist.split_metrics["train"]["token_hit_rate"] == 0.0
    assert specialist.candidate_metadata["fixed_admission_policy"] == "unit_no_cache"


def test_eviction_only_specialist_freezes_admission_and_callbacks() -> None:
    calls = []

    class BasePolicy(AdmitAllLRU):
        def on_request_start(self, request, now: int) -> None:
            calls.append("base_request")

        def on_cache_hit(self, block, request, now: int) -> None:
            calls.append("base_hit")

        def on_cache_miss(self, block, request, now: int) -> None:
            calls.append("base_miss")

        def _values(self, key, now):
            return 2.0, 3.0

    config = EvaluatorConfig(
        capacity_blocks=4,
        request_count=12,
        seeds=(3,),
        train_families=("shared_system_prompt",),
        validation_families=(),
        probe_families=(),
        fixed_admission_policy="pressure_aware_incumbent",
        candidate_policy_surface="eviction_only",
    )
    evaluator = candidate_evaluator(config, splits=("train",))
    evaluator._base_factory = lambda *_args: BasePolicy()

    def score_eviction(block, now, frequency, priority):
        calls.append(("eviction", frequency, priority))
        return float(now - block.last_accessed_at)

    result = evaluator(score_eviction)

    assert result.success is True
    assert "base_request" in calls
    assert "base_hit" in calls
    assert "base_miss" in calls
    assert ("eviction", 2.0, 3.0) in calls


def test_fixed_admission_factory_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="unknown fixed_admission_policy"):
        fixed_admission_factory("unknown")


def test_no_cache_zero_hits() -> None:
    config = EvaluatorConfig(request_count=24, seeds=(3,))
    result = PrefixKVCacheEvaluator(config)(baseline_no_cache)

    assert result.split_metrics["train"]["token_hit_rate"] == 0.0
    assert result.split_metrics["validation"]["token_hit_rate"] == 0.0
    assert result.invalid_fraction == 0.0


def test_block_recurrence_timestamps_use_only_prior_accesses() -> None:
    class CaptureRecurrence(AdmitAllLRU):
        def __init__(self) -> None:
            self.observations = []

        def on_cache_hit(self, block, request, now: int) -> None:
            self.observations.append((now, block.prev_last_accessed_at, block.last_access_gap))

        def on_cache_miss(self, block, request, now: int) -> None:
            self.observations.append((now, block.prev_last_accessed_at, block.last_access_gap))

    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=4,
                priority=0,
                request_type="unit",
                prompt_tokens=(),
            ),
            true_output_length=1,
            prompt_tokens=(1, 2, 3, 4),
            arrival_step=arrival_step,
        )
        for request_id, arrival_step in enumerate((2, 7, 11))
    )
    policy = CaptureRecurrence()
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=4,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )

    simulator.run(policy, requests, split="train", workload="unit", seed=1)

    assert policy.observations == [
        (2, None, None),
        (7, 2, 5),
        (11, 7, 4),
    ]


def test_block_access_gap_summary_is_bounded_and_deterministic() -> None:
    class CaptureGapSummary(AdmitAllLRU):
        def __init__(self) -> None:
            self.observations = []

        def on_cache_hit(self, block, request, now: int) -> None:
            self.observations.append((now, block.access_gap_mean, block.access_gap_var))

        def on_cache_miss(self, block, request, now: int) -> None:
            self.observations.append((now, block.access_gap_mean, block.access_gap_var))

    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=4,
                priority=0,
                request_type="unit",
                prompt_tokens=(),
            ),
            true_output_length=1,
            prompt_tokens=(1, 2, 3, 4),
            arrival_step=arrival_step,
        )
        for request_id, arrival_step in enumerate((0, 4, 10, 12))
    )
    policy = CaptureGapSummary()
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=4,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )

    simulator.run(policy, requests, split="train", workload="unit", seed=1)

    assert policy.observations[:2] == [(0, None, None), (4, None, None)]
    assert policy.observations[2] == pytest.approx((10, 4.5, 0.75))
    assert policy.observations[3] == pytest.approx((12, 3.875, 1.734375))
    block = next(iter(simulator.blocks.values()))
    for now in range(13, 10_013):
        simulator._record_access(block, now)
    assert block.access_gap_sample_count == 2
    assert isinstance(block.access_gap_mean, float)
    assert isinstance(block.access_gap_mean_square, float)


def test_subtree_aggregates_include_known_descendants() -> None:
    class CaptureSubtree(AdmitAllLRU):
        def __init__(self) -> None:
            self.hit_observations = []

        def on_cache_hit(self, block, request, now: int) -> None:
            self.hit_observations.append(
                (
                    request.request_id,
                    block.depth,
                    block.subtree_hit_rate,
                    block.active_ref_count,
                    block.subtree_active_ref_count,
                )
            )

    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=8,
                priority=0,
                request_type="unit",
                prompt_tokens=(),
            ),
            true_output_length=128,
            prompt_tokens=tuple(range(8)),
            arrival_step=request_id,
        )
        for request_id in range(2)
    )
    policy = CaptureSubtree()
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=4,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )

    simulator.run(policy, requests, split="train", workload="unit", seed=1)

    assert policy.hit_observations[0][0] == policy.hit_observations[1][0]
    assert policy.hit_observations[0][0] != 1
    assert [observation[1:] for observation in policy.hit_observations] == [
        (1, 0.25, 2, 3),
        (2, 0.5, 2, 2),
    ]
    root = min(simulator.blocks.values(), key=lambda block: block.depth)
    assert simulator._subtree_hit_counts[root.prefix_hash] >= root.hit_count
    assert simulator._subtree_active_ref_counts[root.prefix_hash] >= root.active_ref_count


def test_request_regime_context_is_bounded_and_independent_of_request_type() -> None:
    class CaptureRegime(AdmitAllLRU):
        def __init__(self) -> None:
            self.observations = []

        def on_request_start(self, request, now: int) -> None:
            self.observations.append(
                (
                    request.request_type,
                    request.recent_admission_pressure,
                    request.recent_miss_rate,
                )
            )

    def run(request_types):
        prompts = ((1, 1, 1, 1), (2, 2, 2, 2)) + ((2, 2, 2, 2),) * 38
        requests = tuple(
            WorkloadRequest(
                info=RequestInfo(
                    request_id=request_id,
                    tenant_id=0,
                    session_id=0,
                    prompt_length=4,
                    priority=0,
                    request_type=request_type,
                    prompt_tokens=(),
                ),
                true_output_length=1,
                prompt_tokens=prompt,
                arrival_step=request_id,
            )
            for request_id, (request_type, prompt) in enumerate(
                zip(request_types, prompts, strict=True)
            )
        )
        policy = CaptureRegime()
        simulator = PrefixKVCacheSimulator(
            capacity_blocks=1,
            block_size_tokens=4,
            prefill_cost_per_token=1.0,
            lookup_cost_per_block=0.0,
            eviction_cost_per_block=0.0,
        )
        simulator.run(policy, requests, split="train", workload="unit", seed=1)
        return policy.observations, simulator

    first, simulator = run(tuple(f"type_{index}" for index in range(40)))
    second, _ = run(tuple("different" for _ in range(40)))

    assert {observation[0] for observation in first} == {"request"}
    assert {observation[0] for observation in second} == {"request"}
    assert first[0][1:] == (0.0, 0.0)
    assert first[1][1:] == (1.0, 1.0)
    assert first[2][1:] == (1.0, 1.0)
    assert first[3][1:] == pytest.approx((1.0, 2.0 / 3.0))
    assert [observation[1:] for observation in first] == [observation[1:] for observation in second]
    assert simulator._recent_admission_pressure.maxlen == 32
    assert simulator._recent_miss_rates.maxlen == 32
    assert len(simulator._recent_admission_pressure) == 32
    assert len(simulator._recent_miss_rates) == 32


def test_candidate_visible_request_metadata_is_scrubbed() -> None:
    class CaptureRequest(AdmitAllLRU):
        def __init__(self) -> None:
            self.observed = []

        def on_request_start(self, request, now: int) -> None:
            self.observed.append(request)

        def score_admission(self, block, now: int) -> float:
            return -1.0

    request = WorkloadRequest(
        info=RequestInfo(
            request_id=7,
            tenant_id=3,
            session_id=5,
            prompt_length=4,
            priority=2,
            request_type="priority_one_off_noise",
            prompt_tokens=(91, 92, 93, 94),
            predicted_output_length=32,
        ),
        true_output_length=128,
        prompt_tokens=(1, 2, 3, 4),
    )
    policy = CaptureRequest()
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=2,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )

    simulator.run(policy, (request,), split="train", workload="unit", seed=1003)

    assert len(policy.observed) == 1
    visible = policy.observed[0]
    assert visible.request_id != request.info.request_id
    assert visible.request_type == "request"
    assert visible.prompt_tokens == ()
    assert visible.tenant_id == request.info.tenant_id
    assert visible.session_id == request.info.session_id
    assert visible.priority == request.info.priority
    assert visible.predicted_output_length == request.info.predicted_output_length


def test_deployable_candidate_never_receives_future_reuse_metadata() -> None:
    class CaptureFutureReuse(AdmitAllLRU):
        def __init__(self) -> None:
            self.observed = []

        def _record(self, channel, block) -> None:
            self.observed.append(
                (
                    channel,
                    block.estimated_future_reuse,
                    block.estimated_next_reuse_distance,
                )
            )

        def on_cache_hit(self, block, request, now: int) -> None:
            self._record("hit", block)

        def on_cache_miss(self, block, request, now: int) -> None:
            self._record("miss", block)

        def score_admission(self, block, now: int) -> float:
            self._record("admission", block)
            return 1.0

        def score_eviction(self, block, now: int) -> float:
            self._record("eviction", block)
            return float(now - block.last_accessed_at)

    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=4,
                priority=0,
                request_type="unit",
                prompt_tokens=(),
            ),
            true_output_length=1,
            prompt_tokens=tuple([token] * 4),
        )
        for request_id, token in enumerate((1, 2, 1, 3))
    )
    policy = CaptureFutureReuse()
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=2,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
        expose_future_reuse=False,
    )

    simulator.run(policy, requests, split="train", workload="unit", seed=1003)

    assert policy.observed
    assert {channel for channel, _, _ in policy.observed} == {
        "admission",
        "eviction",
        "hit",
        "miss",
    }
    assert {(future, distance) for _, future, distance in policy.observed} == {(None, None)}


def test_discrete_baselines_break_equal_priority_ties_with_lru() -> None:
    older = _block_info(last_accessed_at=1)
    newer = _block_info(last_accessed_at=9)

    for factory in (
        baseline_lfu_blocks,
        baseline_depth_prefer_shallow,
        baseline_prefix_fanout,
    ):
        policy = factory(8, 4)
        assert policy.score_eviction(older, now=10) > policy.score_eviction(newer, now=10)


def test_vllm_apc_admits_full_blocks_and_uses_documented_eviction_order() -> None:
    policy = baseline_vllm_apc(8, 4)
    full = _block_info(token_count=4, depth=2, last_accessed_at=3)
    partial = _block_info(token_count=3, depth=2, last_accessed_at=3)
    older_shallow = _block_info(depth=1, last_accessed_at=2)
    newer_deep = _block_info(depth=8, last_accessed_at=3)
    tied_shallow = _block_info(depth=1, last_accessed_at=3)
    tied_deep = _block_info(depth=8, last_accessed_at=3)

    assert policy.score_admission(full, now=10) > 0.0
    assert policy.score_admission(partial, now=10) < 0.0
    assert policy.score_eviction(older_shallow, now=10) > policy.score_eviction(newer_deep, now=10)
    assert policy.score_eviction(tied_deep, now=10) > policy.score_eviction(tied_shallow, now=10)


def test_lfu_still_prefers_to_evict_a_less_frequent_block() -> None:
    unused = _block_info(last_accessed_at=9, hit_count=0)
    frequent = _block_info(last_accessed_at=1, hit_count=1)
    policy = baseline_lfu_blocks(8, 4)

    assert policy.score_eviction(unused, now=10) > policy.score_eviction(frequent, now=10)


def test_oracle_evicts_furthest_next_reuse_even_if_it_is_more_frequent() -> None:
    sooner_once = _block_info(
        estimated_future_reuse=1.0,
        estimated_next_reuse_distance=2.0,
    )
    later_often = _block_info(
        block_id=2,
        prefix_hash=2,
        estimated_future_reuse=10.0,
        estimated_next_reuse_distance=10.0,
    )
    heuristic = baseline_future_reuse_heuristic(8, 4)
    oracle = baseline_oracle_future_reuse(8, 4)

    assert heuristic.score_eviction(sooner_once, now=0) > heuristic.score_eviction(
        later_often, now=0
    )
    assert oracle.score_eviction(later_often, now=0) > oracle.score_eviction(sooner_once, now=0)


def test_tenant_fair_lru_prefers_eviction_from_better_served_tenant() -> None:
    served = _block_info(tenant_id=0)
    underserved = _block_info(block_id=2, prefix_hash=2, tenant_id=1)
    request = RequestInfo(
        request_id=0,
        tenant_id=0,
        session_id=0,
        prompt_length=8,
        priority=0,
        request_type="unit",
        prompt_tokens=(),
    )
    policy = baseline_tenant_fair_lru(8, 4)

    policy.on_cache_hit(served, request, now=0)
    policy.on_cache_miss(underserved, request, now=0)

    assert policy.score_eviction(served, now=10) > policy.score_eviction(underserved, now=10)


def test_tenant_fair_lru_reduces_multi_tenant_fairness_gap() -> None:
    config = EvaluatorConfig(
        request_count=96,
        seeds=(3,),
        capacity_blocks=12,
        block_size_tokens=8,
        validation_families=("multi_tenant_skew",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))

    lru = evaluator(baseline_lru_blocks)
    tenant_fair = evaluator(baseline_tenant_fair_lru)

    lru_gap = lru.workload_metrics["validation/multi_tenant_skew"]["tenant_fairness_penalty"]
    tenant_fair_gap = tenant_fair.workload_metrics["validation/multi_tenant_skew"][
        "tenant_fairness_penalty"
    ]
    assert tenant_fair_gap < lru_gap


def test_prefix_fanout_does_not_regress_lru_on_branching() -> None:
    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        validation_families=("agent_trace_branching",),
    )
    lru = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)
    fanout = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_prefix_fanout)

    lru_hit_rate = lru.workload_metrics["validation/agent_trace_branching"]["token_hit_rate"]
    fanout_hit_rate = fanout.workload_metrics["validation/agent_trace_branching"]["token_hit_rate"]
    assert fanout_hit_rate >= lru_hit_rate


def test_adversarial_over_admission_high_churn() -> None:
    config = EvaluatorConfig(
        request_count=36,
        seeds=(3,),
        capacity_blocks=8,
        block_size_tokens=8,
        hidden_families=("adversarial_unique_prompts",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("hidden",))(lambda *_: AdmitAllLRU())
    metrics = result.workload_metrics["hidden/adversarial_unique_prompts"]

    assert metrics["token_hit_rate"] == 0.0
    assert metrics["cache_churn_per_1k"] > 2500.0
    assert result.invalid_fraction == 0.0


def test_invalid_candidate_penalized() -> None:
    class BadPolicy(AdmitAllLRU):
        def score_admission(self, block, now: int) -> float:
            return float("nan")

    config = EvaluatorConfig(request_count=12, seeds=(3,))
    invalid = PrefixKVCacheEvaluator(config)(lambda *_: BadPolicy())
    no_cache = PrefixKVCacheEvaluator(config)(baseline_no_cache)

    assert invalid.invalid_fraction > 0.0
    assert invalid.combined_score < no_cache.combined_score
    assert invalid.success is False


def test_factory_internal_type_error_is_not_retried() -> None:
    calls = []

    def factory(capacity_blocks, block_size_tokens, seed=None):
        calls.append(seed)
        raise TypeError("internal construction failure")

    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        policy_seed=17,
        train_families=("shared_system_prompt",),
    )

    result = PrefixKVCacheEvaluator(config, splits=("train",))(factory)

    assert result.invalid_fraction == 1.0
    assert calls == [17]


def test_policy_seed_is_independent_of_workload_seed() -> None:
    seen_policy_seeds = []

    def factory(capacity_blocks, block_size_tokens, seed=None):
        seen_policy_seeds.append(seed)
        return AdmitAllLRU()

    config = EvaluatorConfig(
        request_count=1,
        seeds=(3, 7),
        policy_seed=19,
        train_families=("shared_system_prompt", "rag_template_reuse"),
    )

    result = PrefixKVCacheEvaluator(config, splits=("train",))(factory)

    assert result.success is True
    assert len({trial.seed for trial in result.trials}) == 4
    assert seen_policy_seeds == [19, 19, 19, 19]
    assert result.candidate_metadata["policy_seed"] == 19


def test_missing_policy_hooks_are_structured_invalid_results() -> None:
    class MissingHooks:
        def score_admission(self, block, now: int) -> float:
            return -1.0

        def score_eviction(self, block, now: int) -> float:
            return 0.0

    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        train_families=("shared_system_prompt",),
    )

    result = PrefixKVCacheEvaluator(config, splits=("train",))(lambda *_: MissingHooks())

    assert result.invalid_fraction == 1.0
    assert (
        result.workload_metrics["train/shared_system_prompt"]["invalid_reason"]
        == "policy must implement on_request_start()"
    )


def test_candidate_memory_limit_is_enforced() -> None:
    class MemoryHeavyPolicy(AdmitAllLRU):
        def __init__(self) -> None:
            self.payload = bytearray(16 * 1024)

    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        train_families=("shared_system_prompt",),
        max_memory_bytes=1024,
    )

    result = PrefixKVCacheEvaluator(config, splits=("train",))(lambda *_: MemoryHeavyPolicy())

    assert result.invalid_fraction == 1.0
    assert (
        "candidate used" in result.workload_metrics["train/shared_system_prompt"]["invalid_reason"]
    )


def test_evaluate_source_minimal_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(request_count=12, seeds=(3,)),
    )
    source = _minimal_policy_source("-1.0", "0.0")

    result = levi_evaluator.evaluate_source(source)

    assert result.metrics["success"] is True
    assert result.metrics["combined_score"] < 0.0
    assert result.artifacts["candidate_metadata"]["scoring_fn_complexity"] > 0
    assert result.artifacts["score_breakdown"]["combined_score"] == result.metrics["combined_score"]
    assert result.artifacts["score_breakdown"]["complexity_cost"] > 0.0
    assert (
        result.metrics["selection_raw_score_before_complexity"] > result.metrics["combined_score"]
    )
    assert result.metrics["selection_complexity_cost"] > 0.0
    assert result.metrics["per_example_scores"]
    assert len(result.metrics["per_example_scores"]) == len(result.metrics["feedback_per_example"])
    assert all(
        "Weak validation workload validation/" in feedback
        or "Targeted mutation-guidance workload" in feedback
        for feedback in result.metrics["feedback_per_example"]
    )
    assert any(
        "Targeted mutation-guidance workload train/agentic_tool_workflows" in feedback
        for feedback in result.metrics["feedback_per_example"]
    )
    assert any(
        "Targeted mutation-guidance workload validation/stochastic_serving_mix" in feedback
        for feedback in result.metrics["feedback_per_example"]
    )
    assert "train_workload_agentic_tool_workflows_token_hit_rate" in result.metrics
    assert "train_workload_agentic_tool_workflows_policy_underfill_rate" in result.metrics
    assert (
        "train_workload_agentic_tool_workflows_avoidable_admission_regret_token_rate"
        in result.metrics
    )
    assert (
        "train_workload_agentic_tool_workflows_"
        "value_weighted_avoidable_eviction_regret_token_rate" in result.metrics
    )
    assert "validation_workload_stochastic_serving_mix_token_hit_rate" in result.metrics
    assert all(
        "Admission audit:" in feedback
        and "Eviction audit:" in feedback
        and "Cache economics:" in feedback
        and "dominant_regret=" in feedback
        for feedback in result.metrics["feedback_per_example"]
    )
    assert all(
        "probe/" not in feedback and "hidden/" not in feedback
        for feedback in result.metrics["feedback_per_example"]
    )


def test_workload_feedback_reports_measured_regret_side() -> None:
    result = _report_result(0.5)
    metrics = result.workload_metrics["validation/priority_burst_recovery"]
    metrics.update(
        {
            "avoidable_admission_rate": 0.25,
            "avoidable_admission_regret_token_rate": 0.03,
            "avoidable_rejection_rate": 0.10,
            "avoidable_rejection_regret_token_rate": 0.02,
            "value_weighted_avoidable_eviction_rate": 0.20,
            "value_weighted_avoidable_eviction_regret_token_rate": 0.01,
        }
    )

    _, feedback = levi_evaluator._workload_failure_feedback(result)

    diagnostic = next(item for item in feedback if "validation/priority_burst_recovery" in item)
    assert "avoidable_accept_regret_token_rate=0.0300" in diagnostic
    assert "avoidable_reject_regret_token_rate=0.0200" in diagnostic
    assert "value_weighted_eviction_regret_token_rate=0.0100" in diagnostic
    assert "admission_regret_token_rate=0.0500" in diagnostic
    assert "dominant_regret=admission" in diagnostic


def test_evaluate_source_rejects_static_policy_violations(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(
            max_candidate_complexity=10_000,
            reject_unsupported_source_patterns=True,
        ),
    )
    source = textwrap.dedent(
        """
        import math
        import random

        class Policy:
            def on_request_start(self, request, now):
                pass

            def on_cache_hit(self, block, request, now):
                pass

            def on_cache_miss(self, block, request, now):
                pass

            def score_admission(self, block, now):
                try:
                    return block.estimated_future_reuse or 0.0
                except Exception:
                    return 0.0

            def score_eviction(self, block, now):
                return 0.0

        def build_candidate(capacity_blocks, block_size_tokens, seed=None):
            return Policy()
        """
    )

    result = levi_evaluator.evaluate_source(source)

    assert result.metrics["success"] is False
    identity = require_single_score_identity(
        (result.metrics, result.artifacts),
        context="static rejection",
    )
    assert len(identity.evaluation_context_sha256) == 64
    assert len(identity.panel_sha256) == 64
    assert result.metrics["error"].startswith("Repair before retry:")
    assert "future-knowledge field estimated_future_reuse" in result.metrics["error"]
    assert "broad exception handlers are not allowed" in result.metrics["error"]
    assert "unused import math" in result.metrics["error"]
    assert "unused import random" in result.metrics["error"]
    assert any(
        "Remove estimated_future_reuse" in repair for repair in result.artifacts["repair_feedback"]
    )
    assert any(
        "Delete math from the imports" in repair for repair in result.artifacts["repair_feedback"]
    )


def test_evaluate_source_rejects_excessive_complexity(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(max_candidate_complexity=1),
    )
    source = _minimal_policy_source("-1.0", "0.0")

    result = levi_evaluator.evaluate_source(source)

    assert result.metrics["success"] is False
    assert "effective complexity" in result.metrics["error"]
    assert "exceeds limit 1" in result.metrics["error"]
    assert "Delete or simplify at least" in result.artifacts["suggestion"]


def test_evaluate_source_scores_over_promotion_cap_and_requests_simplification(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(
            capacity_blocks=4,
            request_count=2,
            seeds=(3,),
            train_families=("shared_system_prompt",),
            validation_families=("shared_system_prompt",),
            probe_families=(),
            max_candidate_complexity=10_000,
            promotion_max_candidate_complexity=1,
        ),
    )

    result = levi_evaluator.evaluate_source(_minimal_policy_source("-1.0", "0.0"))

    assert result.metrics["success"] is True
    assert result.metrics["promotion_eligible"] is False
    assert result.metrics["promotion_complexity_excess"] > 0
    assert any(
        "Exploration-only candidate: behavioral evaluation completed" in feedback
        for feedback in result.metrics["feedback_per_example"]
    )
    assert any(
        "dedicated simplification mutation" in feedback
        for feedback in result.metrics["feedback_per_example"]
    )
    assert result.artifacts["workload_metrics"]


def test_evaluate_source_eviction_only_uses_raw_behavioral_search_score(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(
            capacity_blocks=4,
            request_count=2,
            seeds=(3,),
            train_families=("shared_system_prompt",),
            validation_families=("shared_system_prompt",),
            probe_families=(),
            fixed_admission_policy="pressure_aware_incumbent",
            candidate_policy_surface="eviction_only",
            search_score_mode="raw_before_complexity",
        ),
    )
    source = textwrap.dedent(
        """
        def score_eviction(block, now, frequency, priority):
            return now - block.last_accessed_at - frequency - priority
        """
    )

    result = levi_evaluator.evaluate_source(source)

    assert result.metrics["success"] is True
    assert result.metrics["combined_score"] == result.metrics["raw_score_before_complexity"]
    assert result.metrics["combined_score"] > result.metrics["charged_combined_score"]


def test_evaluate_source_robust_search_uses_non_quarantined_guidance_floor(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(
            capacity_blocks=4,
            request_count=8,
            seeds=(3,),
            train_families=("agentic_tool_workflows",),
            validation_families=("shared_system_prompt",),
            probe_families=(),
            hidden_families=(),
            search_score_mode="robust_min",
            search_guidance_families=("agentic_tool_workflows",),
        ),
    )

    result = levi_evaluator.evaluate_source(_minimal_policy_source("1.0", "0.0"))

    guidance = result.metrics["search_guidance_floor_score"]
    canonical = result.metrics["charged_combined_score"]
    assert result.metrics["success"] is True
    assert result.metrics["combined_score"] == min(canonical, guidance)


def test_evaluate_source_eviction_only_rejects_full_policy_surface(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(
            candidate_policy_surface="eviction_only",
            fixed_admission_policy="pressure_aware_incumbent",
        ),
    )

    result = levi_evaluator.evaluate_source(_minimal_policy_source("-1.0", "0.0"))

    assert result.metrics["success"] is False
    assert "eviction-only specialist must not define policy classes" in result.metrics["error"]
    assert "must define exactly one score_eviction function" in result.metrics["error"]


def test_evaluate_source_eviction_only_reports_decorator_repair(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(
            candidate_policy_surface="eviction_only",
            fixed_admission_policy="pressure_aware_incumbent",
        ),
    )
    source = textwrap.dedent(
        """
        @staticmethod
        def score_eviction(block, now, frequency, priority):
            return 0.0
        """
    )

    result = levi_evaluator.evaluate_source(source)

    assert result.metrics["success"] is False
    assert result.artifacts["repair_feedback"] == [
        "Remove decorators from score_eviction so exploration and promotion execute "
        "the same function body."
    ]


def test_evaluate_source_reports_syntax_repair_before_complexity(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(max_candidate_complexity=1),
    )

    result = levi_evaluator.evaluate_source("def build_candidate(:\n    pass\n")

    assert result.metrics["success"] is False
    assert "syntax error at line 1" in result.metrics["error"]
    assert "effective complexity" not in result.metrics["error"]
    assert result.artifacts["violations"] == ["syntax error at line 1: invalid syntax"]


def test_evaluate_source_reports_runtime_contract_repairs(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(
            request_count=1,
            seeds=(3,),
            train_families=("shared_system_prompt",),
            validation_families=(),
            probe_families=(),
        ),
    )
    source = textwrap.dedent(
        """
        class Policy:
            def score_admission(self, block, now):
                return 0.0

            def score_eviction(self, block, now):
                return 0.0

        def build_candidate(capacity_blocks, block_size_tokens, seed=None):
            return Policy()
        """
    )

    result = levi_evaluator.evaluate_source(source)

    assert result.metrics["success"] is False
    assert "policy must implement on_request_start()" in result.metrics["error"]
    assert "Implement on_request_start()" in result.artifacts["suggestion"]


def test_static_policy_checks_validate_multi_timescale_decay_constructor() -> None:
    config = EvaluatorConfig(reject_unsupported_source_patterns=True)
    invalid_source = textwrap.dedent(
        """
        from prefix_cache_evolve.problems.prefix_kv_cache.primitives import MultiTimescaleDecay

        class Policy:
            def __init__(self):
                self._state = MultiTimescaleDecay(4, 10)
        """
    )
    valid_source = textwrap.dedent(
        """
        from prefix_cache_evolve.problems.prefix_kv_cache.primitives import MultiTimescaleDecay

        class Policy:
            def __init__(self):
                self._state = MultiTimescaleDecay(half_lives=(4.0, 20.0), max_keys=64)
        """
    )

    invalid = levi_evaluator._candidate_source_violations(
        invalid_source,
        complexity=1,
        config=config,
    )
    valid = levi_evaluator._candidate_source_violations(
        valid_source,
        complexity=1,
        config=config,
    )

    assert "MultiTimescaleDecay accepts only one positional argument" in invalid
    assert "MultiTimescaleDecay half-lives must be a sequence" in invalid
    assert valid == ()


def test_static_policy_checks_validate_threshold_excess_signature() -> None:
    config = EvaluatorConfig(reject_unsupported_source_patterns=True)
    invalid_source = textwrap.dedent(
        """
        from prefix_cache_evolve.problems.prefix_kv_cache.primitives import threshold_excess

        class Policy:
            def score_admission(self, block, now):
                return threshold_excess(block.depth)
        """
    )
    valid_source = textwrap.dedent(
        """
        from prefix_cache_evolve.problems.prefix_kv_cache.primitives import threshold_excess

        class Policy:
            def score_admission(self, block, now):
                return threshold_excess(block.depth, 2.0)
        """
    )

    invalid = levi_evaluator._candidate_source_violations(
        invalid_source,
        complexity=1,
        config=config,
    )
    valid = levi_evaluator._candidate_source_violations(
        valid_source,
        complexity=1,
        config=config,
    )

    assert "threshold_excess requires value and threshold" in invalid
    assert valid == ()


def test_static_policy_checks_reject_scrubbed_request_fields() -> None:
    config = EvaluatorConfig(reject_unsupported_source_patterns=True)
    source = textwrap.dedent(
        """
        class Policy:
            def on_request_start(self, request, now):
                self.kind = request.request_type
                self.tokens = request.prompt_tokens
        """
    )

    violations = levi_evaluator._candidate_source_violations(
        source,
        complexity=1,
        config=config,
    )

    assert "sanitized request field request_type is not a policy signal" in violations
    assert "sanitized request field prompt_tokens is not a policy signal" in violations


@pytest.mark.parametrize("builtin_name", ["exec", "eval", "compile", "vars"])
def test_static_policy_checks_reject_dynamic_builtins(builtin_name: str) -> None:
    config = EvaluatorConfig(reject_unsupported_source_patterns=True)
    source = f"""
class Policy:
    def score_admission(self, block, now):
        return {builtin_name}("0")
"""

    violations = levi_evaluator._candidate_source_violations(
        source,
        complexity=scoring_fn_complexity(source),
        config=config,
    )

    assert f"{builtin_name}() is not allowed in candidate code" in violations


def test_static_policy_checks_reject_aliased_dynamic_builtin() -> None:
    config = EvaluatorConfig(reject_unsupported_source_patterns=True)
    source = """
class Policy:
    def score_admission(self, block, now):
        runner = exec
        runner("pass")
        return 0.0
"""

    violations = levi_evaluator._candidate_source_violations(
        source,
        complexity=scoring_fn_complexity(source),
        config=config,
    )

    assert "exec() is not allowed in candidate code" in violations


def test_evaluate_source_rejects_exec_laundering_end_to_end(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(
            max_candidate_complexity=650,
            reject_unsupported_source_patterns=True,
        ),
    )
    source = 'exec("class Policy:\\n    pass")'

    result = levi_evaluator.evaluate_source(source)

    assert result.metrics["success"] is False
    assert "exec() is not allowed in candidate code" in result.metrics["error"]
    assert "unsupported top-level statement Expr" in result.metrics["error"]


def test_static_policy_checks_reject_dunder_attribute_access() -> None:
    config = EvaluatorConfig(reject_unsupported_source_patterns=True)
    source = """
class Policy:
    def score_admission(self, block, now):
        return block.__class__
"""

    violations = levi_evaluator._candidate_source_violations(
        source,
        complexity=scoring_fn_complexity(source),
        config=config,
    )

    assert "dunder attribute __class__ is not allowed" in violations


def test_static_policy_checks_reject_decorators() -> None:
    config = EvaluatorConfig(reject_unsupported_source_patterns=True)
    source = """
def decorate(policy):
    return policy

@decorate
class Policy:
    pass
"""

    violations = levi_evaluator._candidate_source_violations(
        source,
        complexity=scoring_fn_complexity(source),
        config=config,
    )

    assert "decorators are not allowed in candidate code" in violations


def test_static_policy_checks_reject_unsupported_top_level_statements() -> None:
    config = EvaluatorConfig(reject_unsupported_source_patterns=True)
    source = """
if True:
    class Policy:
        pass
"""

    violations = levi_evaluator._candidate_source_violations(
        source,
        complexity=scoring_fn_complexity(source),
        config=config,
    )

    assert "unsupported top-level statement If" in violations


def test_static_policy_checks_reject_nonliteral_module_lambda() -> None:
    config = EvaluatorConfig(reject_unsupported_source_patterns=True)
    source = "score = lambda block, now: block.depth"

    violations = levi_evaluator._candidate_source_violations(
        source,
        complexity=scoring_fn_complexity(source),
        config=config,
    )

    assert "top-level assignments must define uppercase literal constants" in violations


def test_static_policy_checks_restrict_candidate_imports() -> None:
    config = EvaluatorConfig(reject_unsupported_source_patterns=True)
    source = """
from prefix_cache_evolve.evaluators.baselines import baseline_tinylfu_lru

def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return baseline_tinylfu_lru(capacity_blocks, block_size_tokens, seed)
"""

    violations = levi_evaluator._candidate_source_violations(
        source,
        complexity=scoring_fn_complexity(source),
        config=config,
    )

    assert "import from unsupported module prefix_cache_evolve.evaluators.baselines" in violations


@pytest.mark.parametrize(
    "path",
    [
        current_incumbent("discovery").source_path,
        current_incumbent("production").source_path,
    ],
)
def test_committed_incumbents_pass_static_source_contract(path: Path) -> None:
    source = Path(path).read_text(encoding="utf-8")
    complexity = scoring_fn_complexity(source, form_aware=True)
    config = EvaluatorConfig(
        max_candidate_complexity=650,
        reject_unsupported_source_patterns=True,
    )

    assert levi_evaluator._candidate_source_violations(source, complexity, config) == ()


def test_eviction_specialist_seed_passes_static_source_contract() -> None:
    source = Path(
        "src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/eviction_specialist.py"
    ).read_text(encoding="utf-8")
    complexity = scoring_fn_complexity(source, form_aware=True)
    config = EvaluatorConfig(
        max_candidate_complexity=1000,
        reject_unsupported_source_patterns=True,
        candidate_policy_surface="eviction_only",
    )

    assert levi_evaluator._candidate_source_violations(source, complexity, config) == ()


def test_evaluate_factory_uses_configured_timeout(monkeypatch) -> None:
    captured = {}
    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        timeout_s=0.25,
    )
    monkeypatch.setattr(levi_evaluator, "DEFAULT_CONFIG", config)

    def fake_run_with_timeout(
        func,
        *args,
        timeout_seconds,
        memory_limit_bytes=None,
        cpu_limit_seconds=None,
        **kwargs,
    ):
        captured["timeout_seconds"] = timeout_seconds
        captured["memory_limit_bytes"] = memory_limit_bytes
        captured["cpu_limit_seconds"] = cpu_limit_seconds
        return func(*args, **kwargs)

    monkeypatch.setattr(levi_evaluator, "run_with_timeout", fake_run_with_timeout)

    result = levi_evaluator.evaluate_factory(baseline_no_cache)

    assert result.metrics["success"] is True
    assert captured["timeout_seconds"] == 0.25
    assert captured["memory_limit_bytes"] == config.max_memory_bytes
    assert captured["cpu_limit_seconds"] == 0.25


def test_evaluate_source_times_out_during_module_loading(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(timeout_s=0.01),
    )
    source = """
import time
time.sleep(0.5)
"""

    started = time.perf_counter()
    result = levi_evaluator.evaluate_source(source)

    assert result.metrics["error"] == "evaluation timed out"
    assert time.perf_counter() - started < 0.3


def test_root_anchored_match() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=4,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )
    request = WorkloadRequest(
        info=RequestInfo(
            request_id=0,
            tenant_id=0,
            session_id=0,
            prompt_length=8,
            priority=0,
            request_type="unit",
            prompt_tokens=tuple(range(8)),
        ),
        true_output_length=64,
    )
    blocks = simulator._materialize_chain(request, now=0)
    blocks[1].resident = True

    assert simulator.match_resident_prefix(blocks) == 0


def test_forced_bypass_not_invalid() -> None:
    class AdmitEverything(AdmitAllLRU):
        pass

    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        capacity_blocks=1,
        train_families=("shared_system_prompt",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("train",))(lambda *_: AdmitEverything())
    metrics = result.workload_metrics["train/shared_system_prompt"]

    assert result.invalid_fraction == 0.0
    assert metrics["forced_bypass_count"] > 0


def test_pinned_blocks_are_released_after_generation_finishes() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=1,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
        active_tokens_per_step=64,
    )
    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id,
                prompt_length=4,
                priority=0,
                request_type="unit",
                prompt_tokens=tuple(range(request_id * 4, request_id * 4 + 4)),
            ),
            true_output_length=128 if request_id == 0 else 64,
        )
        for request_id in range(3)
    )

    metrics = simulator.run(
        AdmitAllLRU(),
        requests,
        split="train",
        workload="unit",
        seed=1,
    )

    assert metrics.forced_bypass_count == 1
    assert metrics.forced_bypass_tokens == 4
    assert metrics.admission_count == 2
    assert metrics.eviction_count == 1


def test_re_admitted_block_becomes_most_recently_used() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=2,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
        active_tokens_per_step=64,
    )
    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id,
                prompt_length=4,
                priority=0,
                request_type="unit",
                prompt_tokens=tuple([token] * 4),
            ),
            true_output_length=1,
        )
        for request_id, token in enumerate((1, 2, 3, 1, 4))
    )

    simulator.run(
        AdmitAllLRU(),
        requests,
        split="train",
        workload="unit",
        seed=1,
    )

    resident_token_sets = {
        request.info.prompt_tokens
        for request in requests
        if simulator.blocks[simulator._materialize_chain(request, now=5)[0].prefix_hash].resident
    }
    assert resident_token_sets == {(1, 1, 1, 1), (4, 4, 4, 4)}


def test_cache_miss_charges_failed_lookup_probe() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=1,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=2.0,
        eviction_cost_per_block=0.0,
    )
    request = WorkloadRequest(
        info=RequestInfo(
            request_id=0,
            tenant_id=0,
            session_id=0,
            prompt_length=4,
            priority=0,
            request_type="unit",
            prompt_tokens=(1, 2, 3, 4),
        ),
        true_output_length=1,
    )

    metrics = simulator.run(
        baseline_no_cache(1, 4),
        (request,),
        split="train",
        workload="unit",
        seed=1,
    )

    assert metrics.lookup_block_count == 1
    assert metrics.lookup_blocks_per_request == 1.0
    assert metrics.admission_score_count == 1
    assert metrics.admission_rejection_count == 1
    assert metrics.admission_rate == 0.0
    assert metrics.policy_bypass_tokens == 4
    assert metrics.policy_underfill_rate == 1.0
    assert metrics.forced_bypass_tokens == 0
    assert metrics.p95_latency_proxy == 6.0


def test_underfill_does_not_penalize_natural_unused_capacity() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=8,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )
    request = WorkloadRequest(
        info=RequestInfo(
            request_id=0,
            tenant_id=0,
            session_id=0,
            prompt_length=4,
            priority=0,
            request_type="unit",
            prompt_tokens=(1, 2, 3, 4),
        ),
        true_output_length=1,
    )

    metrics = simulator.run(
        baseline_lru_blocks(8, 4),
        (request,),
        split="train",
        workload="unit",
        seed=1,
    )

    assert metrics.memory_occupancy_mean == 1.0
    assert metrics.policy_bypass_token_rate == 0.0
    assert metrics.policy_underfill_rate == 0.0


def test_admission_lifecycle_and_short_reuse_regret_are_reported() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=1,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )
    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=4,
                priority=0,
                request_type="unit",
                prompt_tokens=tuple([token] * 4),
            ),
            true_output_length=1,
        )
        for request_id, token in enumerate((1, 1, 2, 1))
    )

    metrics = simulator.run(
        AdmitAllLRU(),
        requests,
        split="train",
        workload="unit",
        seed=1,
    )

    assert metrics.admission_count == 3
    assert metrics.useful_admission_count == 1
    assert metrics.wasted_admission_count == 2
    assert metrics.useful_admission_rate == 1 / 3
    assert metrics.wasted_admission_rate == 2 / 3
    assert metrics.admitted_token_count == 12
    assert metrics.useful_admission_token_count == 4
    assert metrics.wasted_admission_token_count == 8
    assert metrics.useful_admission_token_rate == 1 / 3
    assert metrics.wasted_admission_token_rate == 2 / 3
    assert metrics.admission_saved_tokens == 4
    assert metrics.admission_saved_tokens_per_admission == 4 / 3
    assert metrics.admission_token_utility == 1 / 3
    assert metrics.evicted_without_hit_count == 1
    assert metrics.short_reuse_after_eviction_missed_tokens == 4
    assert metrics.short_reuse_after_eviction_missed_token_rate > 0.0
    assert metrics.eviction_reuse_distance_p50 == 1.0


def test_admission_token_utility_distinguishes_full_and_partial_blocks() -> None:
    def run(tokens: tuple[int, ...]) -> TrialMetrics:
        simulator = PrefixKVCacheSimulator(
            capacity_blocks=1,
            block_size_tokens=4,
            prefill_cost_per_token=1.0,
            lookup_cost_per_block=0.0,
            eviction_cost_per_block=0.0,
        )
        requests = tuple(
            WorkloadRequest(
                info=RequestInfo(
                    request_id=request_id,
                    tenant_id=0,
                    session_id=0,
                    prompt_length=len(tokens),
                    priority=0,
                    request_type="unit",
                    prompt_tokens=tokens,
                ),
                true_output_length=1,
            )
            for request_id in range(2)
        )
        return simulator.run(
            AdmitAllLRU(),
            requests,
            split="train",
            workload="unit",
            seed=1,
        )

    full = run((1, 1, 1, 1))
    partial = run((2,))

    assert full.useful_admission_rate == partial.useful_admission_rate == 1.0
    assert full.admission_saved_tokens_per_admission == 4.0
    assert partial.admission_saved_tokens_per_admission == 1.0
    assert full.admission_token_utility == 1.0
    assert partial.admission_token_utility == 0.25


def test_token_weighted_admission_waste_reflects_block_size() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=2,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )
    request_tokens = ((1,), (1,), (2, 2, 2, 2))
    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=len(tokens),
                priority=0,
                request_type="unit",
                prompt_tokens=tokens,
            ),
            true_output_length=1,
        )
        for request_id, tokens in enumerate(request_tokens)
    )

    metrics = simulator.run(
        AdmitAllLRU(),
        requests,
        split="train",
        workload="unit",
        seed=1,
    )

    assert metrics.useful_admission_count == 1
    assert metrics.wasted_admission_count == 1
    assert metrics.wasted_admission_rate == 0.5
    assert metrics.admitted_token_count == 5
    assert metrics.useful_admission_token_count == 1
    assert metrics.wasted_admission_token_count == 4
    assert metrics.wasted_admission_token_rate == 0.8


def test_avoidable_eviction_audit_distinguishes_lru_from_oracle() -> None:
    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=4,
                priority=0,
                request_type="unit",
                prompt_tokens=tuple([token] * 4),
            ),
            true_output_length=1,
        )
        for request_id, token in enumerate((1, 2, 1, 3, 2))
    )

    def run(factory, *, expose_future_reuse: bool) -> TrialMetrics:
        simulator = PrefixKVCacheSimulator(
            capacity_blocks=2,
            block_size_tokens=4,
            prefill_cost_per_token=1.0,
            lookup_cost_per_block=0.0,
            eviction_cost_per_block=0.0,
            expose_future_reuse=expose_future_reuse,
        )
        return simulator.run(
            factory(2, 4),
            requests,
            split="train",
            workload="unit",
            seed=1,
        )

    lru = run(baseline_lru_blocks, expose_future_reuse=False)
    oracle = run(baseline_oracle_future_reuse, expose_future_reuse=True)

    assert lru.avoidable_eviction_count == 1
    assert lru.avoidable_short_reuse_eviction_count == 1
    assert lru.value_weighted_avoidable_eviction_count == 1
    assert lru.value_weighted_avoidable_eviction_regret_tokens == 4.0
    assert lru.avoidable_admission_regret_tokens == 0.0
    assert lru.avoidable_rejection_regret_tokens == 0.0
    assert oracle.avoidable_eviction_count == 0
    assert oracle.value_weighted_avoidable_eviction_count == 0
    assert oracle.token_hit_rate > lru.token_hit_rate


def test_admission_audit_distinguishes_avoidable_admission_and_rejection() -> None:
    def requests(tokens: tuple[int, ...]) -> tuple[WorkloadRequest, ...]:
        return tuple(
            WorkloadRequest(
                info=RequestInfo(
                    request_id=request_id,
                    tenant_id=0,
                    session_id=0,
                    prompt_length=4,
                    priority=0,
                    request_type="unit",
                    prompt_tokens=tuple([token] * 4),
                ),
                true_output_length=1,
            )
            for request_id, token in enumerate(tokens)
        )

    def run(
        policy,
        token_stream: tuple[int, ...],
        *,
        capacity_blocks: int = 1,
    ) -> TrialMetrics:
        simulator = PrefixKVCacheSimulator(
            capacity_blocks=capacity_blocks,
            block_size_tokens=4,
            prefill_cost_per_token=1.0,
            lookup_cost_per_block=0.0,
            eviction_cost_per_block=0.0,
        )
        return simulator.run(
            policy,
            requests(token_stream),
            split="train",
            workload="unit",
            seed=1,
        )

    admit_all = run(AdmitAllLRU(), (1, 2, 1))
    reject_all = run(baseline_no_cache(1, 4), (1, 1))
    threshold_and_ranking_error = run(
        AdmitAllLRU(),
        (1, 2, 2, 3, 2, 1, 1),
        capacity_blocks=2,
    )

    assert admit_all.avoidable_admission_count == 1
    assert admit_all.avoidable_admission_regret_tokens == 4.0
    assert admit_all.avoidable_rejection_count == 0
    assert admit_all.value_weighted_avoidable_eviction_regret_tokens == 0.0
    assert reject_all.avoidable_admission_count == 0
    assert reject_all.avoidable_rejection_count == 1
    assert reject_all.avoidable_rejection_regret_tokens == 4.0
    assert reject_all.value_weighted_avoidable_eviction_regret_tokens == 0.0
    assert threshold_and_ranking_error.avoidable_admission_regret_tokens == 4.0
    assert threshold_and_ranking_error.value_weighted_avoidable_eviction_regret_tokens == 4.0


def test_shadow_price_calibration_preserves_the_policy_zero_crossing() -> None:
    audit = _AdmissionAudit()
    block = SimpleNamespace(prefix_hash=1, depth=2, token_count=16)
    audit.record(
        now=0,
        request_index=0,
        block=block,
        score=1.0,
        accepted=True,
        feasible=True,
        incoming_value_tokens=32.0,
        displaced_value_tokens=16.0,
        capacity_weight_tokens=16,
    )
    audit.record(
        now=1,
        request_index=1,
        block=block,
        score=-1.0,
        accepted=False,
        feasible=True,
        incoming_value_tokens=0.0,
        displaced_value_tokens=16.0,
        capacity_weight_tokens=16,
    )

    metrics = audit.shadow_price_metrics()

    assert metrics["shadow_price_score_scale"] == pytest.approx(1.0)
    assert metrics["shadow_price_tracking_rmse"] == pytest.approx(0.0)
    assert metrics["shadow_price_tracking_mae"] == pytest.approx(0.0)
    assert metrics["oracle_shadow_price_mean"] == pytest.approx(1.0)


def test_temporal_and_tenant_tail_metrics_expose_service_collapse() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=1,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )
    requests = []
    for request_id in range(8):
        token = 1 if request_id < 4 else request_id
        tenant_id = 0 if request_id < 6 else 1
        requests.append(
            WorkloadRequest(
                info=RequestInfo(
                    request_id=request_id,
                    tenant_id=tenant_id,
                    session_id=0,
                    prompt_length=4,
                    priority=0,
                    request_type="unit",
                    prompt_tokens=tuple([token] * 4),
                ),
                true_output_length=1,
            )
        )

    metrics = simulator.run(
        AdmitAllLRU(),
        tuple(requests),
        split="train",
        workload="unit",
        seed=1,
    )

    assert metrics.token_hit_rate > metrics.worst_quarter_token_hit_rate
    assert metrics.final_quarter_token_hit_rate == 0.0
    assert metrics.quarter_token_hit_rate_stddev > 0.0
    assert metrics.tenant_count == 2
    assert metrics.tenant_token_hit_rate_p10 == 0.0
    assert metrics.tenant_jain_fairness < 1.0


def test_hidden_not_in_combined_score(monkeypatch) -> None:
    config_a = EvaluatorConfig(
        request_count=12,
        seeds=(3,),
        hidden_families=("adversarial_unique_prompts",),
    )
    config_b = EvaluatorConfig(
        request_count=12,
        seeds=(3,),
        hidden_families=("cross_family_mixture",),
    )
    monkeypatch.setattr(levi_evaluator, "DEFAULT_CONFIG", config_a)
    first = levi_evaluator.evaluate_factory(baseline_lru_blocks)
    monkeypatch.setattr(levi_evaluator, "DEFAULT_CONFIG", config_b)
    second = levi_evaluator.evaluate_factory(baseline_lru_blocks)

    assert first.metrics["combined_score"] == second.metrics["combined_score"]
    assert "hidden" not in first.artifacts["split_metrics"]
    assert all(not key.startswith("hidden/") for key in first.artifacts["workload_metrics"])


def test_structure_probe_not_in_combined_selection_score() -> None:
    config_a = EvaluatorConfig(
        request_count=12,
        seeds=(3,),
        validation_families=("hotset_cold_scan",),
        probe_families=("agent_trace_branching",),
    )
    config_b = config_a.with_updates(
        probe_families=("cyclic_working_set_pressure",),
    )

    first = PrefixKVCacheEvaluator(config_a, splits=("validation", "probe"))(baseline_lru_blocks)
    second = PrefixKVCacheEvaluator(config_b, splits=("validation", "probe"))(baseline_lru_blocks)

    assert first.combined_score == second.combined_score
    assert "probe/agent_trace_branching" in first.workload_metrics
    assert "probe/cyclic_working_set_pressure" in second.workload_metrics
    assert first.candidate_metadata["selection_invalid_fraction"] == 0.0


def test_structure_probe_invalidity_is_quarantined_from_selection() -> None:
    config = EvaluatorConfig(
        request_count=12,
        seeds=(3,),
        validation_families=("hotset_cold_scan",),
        probe_families=("agent_trace_branching",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation", "probe"))
    valid = evaluator(baseline_lru_blocks)
    trials = [
        (
            replace(
                trial,
                invalid=True,
                invalid_reason="probe-only failure",
            )
            if trial.split == "probe"
            else trial
        )
        for trial in valid.trials
    ]

    rescored = evaluator.rescore_trials(trials)

    assert rescored.combined_score == valid.combined_score
    assert rescored.success is True
    assert rescored.invalid_fraction == 0.0
    assert rescored.candidate_metadata["reporting_invalid_fraction"] > 0.0
    assert rescored.split_metrics["probe"]["invalid_fraction"] == 1.0


def test_candidate_program_can_be_compared_against_baselines(tmp_path, capsys, monkeypatch) -> None:
    candidate_path = tmp_path / "best_program.py"
    candidate_path.write_text(
        textwrap.dedent(
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
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        prefix_runner,
        "_evaluate_candidate_program",
        lambda *_args, **_kwargs: _report_result(12.0),
    )
    monkeypatch.setattr(
        prefix_runner,
        "_evaluate_baselines",
        lambda *_args, **_kwargs: {
            "lru": _report_result(10.0),
            "future_reuse_heuristic": _report_result(14.0),
            "oracle_future_reuse": _report_result(16.0),
        },
    )

    compare_baselines(
        quick=True,
        capacity_sweep_blocks=(8, 16),
        candidate_program=tmp_path,
    )

    output = capsys.readouterr().out
    assert "SMOKE-ONLY" in output
    assert "candidate: combined_score=" in output
    assert "capacity_8:" in output
    assert "capacity_16:" in output
    assert "lru: combined_score=" in output
    assert "sglang_radix_attention: combined_score=" not in output
    assert "[deployable]" in output
    assert "future_reuse_heuristic: combined_score=" in output
    assert "oracle_future_reuse: combined_score=" in output
    assert "[reporting-only/future-knowledge]" in output
    report = (tmp_path / "baseline_comparison.md").read_text(encoding="utf-8")
    assert "Candidate `scoring_fn_complexity`" in report
    assert "Capacity 8 token hit" in report
    assert "Capacity 16 token hit" in report
    assert "https://arxiv.org/html/2312.07104v1" in report
    assert "Smoke-only output; run the full panel before comparing policy rank." in report


def test_baseline_report_headline_does_not_overstate_candidate() -> None:
    headline = _baseline_report_headline(
        [
            ("oracle_future_reuse", score_record(90.0)),
            ("tinylfu_lru", score_record(70.0)),
            ("candidate", score_record(60.0)),
            ("lru", score_record(50.0)),
        ]
    )

    assert headline == (
        "The candidate ranking is shown against deployable and reporting-only baselines."
    )


def test_baseline_report_headline_states_mixed_future_knowledge_ordering() -> None:
    headline = _baseline_report_headline(
        [
            ("oracle_future_reuse", score_record(90.0)),
            ("candidate", score_record(80.0)),
            ("future_reuse_heuristic", score_record(70.0)),
            ("tinylfu_lru", score_record(60.0)),
        ]
    )

    assert headline == (
        "The candidate clears the deployable credibility baselines in this capacity "
        "sweep. It trails `oracle_future_reuse`. It beats `future_reuse_heuristic`."
    )


@pytest.mark.parametrize(
    ("identity_override", "error"),
    (
        ({"verifier_version": "2.0.0"}, "refuses mixed verifier versions"),
        (
            {"evaluation_context_sha256": "c" * 64},
            "refuses mixed evaluation contexts",
        ),
        ({"panel_sha256": "d" * 64}, "refuses mixed panels"),
    ),
)
def test_baseline_report_refuses_mixed_score_identity(
    identity_override: dict[str, str],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _baseline_report_headline(
            [
                ("candidate", score_record(80.0)),
                ("tinylfu_lru", score_record(70.0, **identity_override)),
            ]
        )


def test_complexity_counts_candidate_helper_methods() -> None:
    compact = """
class Policy:
    def score_admission(self, block, now):
        return 1.0

    def score_eviction(self, block, now):
        return 0.0


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return Policy()
"""
    helper_heavy = """
class Policy:
    def score_admission(self, block, now):
        return self._helper(block)

    def score_eviction(self, block, now):
        return 0.0

    def _helper(self, block):
        total = 0.0
        for index in range(8):
            total += index * 0.25
        return total


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return Policy()
"""

    assert scoring_fn_complexity(helper_heavy) > scoring_fn_complexity(compact)


def test_complexity_counts_nested_factory_policy_methods() -> None:
    nested_policy = """
from types import SimpleNamespace


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    def score_admission(block, now):
        total = 0.0
        for index in range(8):
            total += index * block.depth
        return total

    def score_eviction(block, now):
        return 0.0

    return SimpleNamespace(
        score_admission=score_admission,
        score_eviction=score_eviction,
    )
"""

    assert scoring_fn_complexity(nested_policy) > 0


def test_complexity_counts_policy_hidden_under_top_level_condition() -> None:
    direct = """
class Policy:
    def score_admission(self, block, now):
        return block.depth
"""
    wrapped = """
if True:
    class Policy:
        def score_admission(self, block, now):
            return block.depth
"""

    assert scoring_fn_complexity(wrapped) >= scoring_fn_complexity(direct)


def test_complexity_counts_exec_and_module_lambda_bodies() -> None:
    exec_source = 'exec("class Policy:\\n    pass")'
    lambda_source = "SCORE = lambda block, now: block.depth + now"
    disguised_all_source = "__all__ = lambda block, now: block.depth + now"

    assert scoring_fn_complexity(exec_source) > 0
    assert scoring_fn_complexity(lambda_source) > 0
    assert scoring_fn_complexity(disguised_all_source) > 0


def test_form_aware_complexity_subsidizes_only_canonical_primitives() -> None:
    primitive_composer = """
from prefix_cache_evolve.problems.prefix_kv_cache.primitives import MultiTimescaleDecay


class Policy:
    def __init__(self):
        self.decay = MultiTimescaleDecay((4.0,))

    def on_cache_hit(self, block, request, now):
        self.decay.observe(block.prefix_hash, 1.0, now)

    def score_admission(self, block, now):
        return self.decay.values(block.prefix_hash, now)[0]

    def score_eviction(self, block, now):
        return -self.decay.values(block.prefix_hash, now)[0]
"""
    hand_rolled = """
class Policy:
    def __init__(self):
        self.values_by_key = {}

    def on_cache_hit(self, block, request, now):
        value = self._value(block, now)
        self.values_by_key[block.prefix_hash] = (value + 1.0, now)

    def _value(self, block, now):
        value, observed_at = self.values_by_key.get(block.prefix_hash, (0.0, now))
        return value * 2.0 ** (-(now - observed_at) / 4.0)

    def score_admission(self, block, now):
        return self._value(block, now)

    def score_eviction(self, block, now):
        return -self._value(block, now)
"""

    primitive_raw = scoring_fn_complexity(primitive_composer)
    primitive_form_aware = scoring_fn_complexity(
        primitive_composer,
        form_aware=True,
    )
    hand_raw = scoring_fn_complexity(hand_rolled)
    hand_form_aware = scoring_fn_complexity(hand_rolled, form_aware=True)

    assert primitive_form_aware < primitive_raw
    assert hand_form_aware == hand_raw
    assert primitive_form_aware < hand_form_aware
    assert hand_raw <= math.ceil(1.6 * primitive_raw)
    assert primitive_form_aware >= math.ceil(0.75 * primitive_raw)


def test_form_aware_complexity_gives_small_credit_to_threshold_excess() -> None:
    source = """
from prefix_cache_evolve.problems.prefix_kv_cache.primitives import threshold_excess


class Policy:
    def score_admission(self, block, now):
        return threshold_excess(block.depth, 2.0)
"""

    raw = scoring_fn_complexity(source)
    form_aware = scoring_fn_complexity(source, form_aware=True)

    assert raw - form_aware == 1


def test_active_complexity_mode_is_configurable(monkeypatch) -> None:
    source = """
from prefix_cache_evolve.problems.prefix_kv_cache.primitives import MultiTimescaleDecay


class Policy:
    def __init__(self):
        self.decay = MultiTimescaleDecay((4.0,))

    def score_admission(self, block, now):
        return self.decay.combine(block.prefix_hash, now, (1.0,))
"""
    raw = scoring_fn_complexity(source)
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(form_aware_complexity=False),
    )
    assert levi_evaluator._source_complexity(source) == raw
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(form_aware_complexity=True),
    )
    assert levi_evaluator._source_complexity(source) < raw


def test_complexity_penalty_is_unbounded_and_concave() -> None:
    config = EvaluatorConfig(
        w_avg_tok=0.0,
        w_avg_blk=0.0,
        min_workload_weight=0.0,
        latency_weight=0.0,
        churn_weight=0.0,
        fairness_weight=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trials = [TrialMetrics(split="validation", workload="unit", seed=1)]
    penalties = [
        -evaluator._score_trials(trials, invalid_fraction=0.0, complexity=complexity)
        for complexity in (3_000, 4_000, 5_000)
    ]

    assert penalties[0] < penalties[1] < penalties[2]
    assert penalties[2] - penalties[1] < penalties[1] - penalties[0]


def test_invalid_score_is_below_large_representative_valid_complexity() -> None:
    config = EvaluatorConfig()
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trials = [TrialMetrics(split="validation", workload="unit", seed=1)]

    invalid_score = evaluator._score_trials(trials, invalid_fraction=1.0, complexity=0)
    valid_score = evaluator._score_trials(trials, invalid_fraction=0.0, complexity=100_000)

    assert invalid_score < valid_score


def test_score_combines_mean_and_min_workload_score() -> None:
    config = EvaluatorConfig(
        w_avg_tok=100.0,
        w_avg_blk=0.0,
        min_workload_weight=0.5,
        latency_weight=0.0,
        churn_weight=0.0,
        fairness_weight=0.0,
        k_complex=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trials = [
        TrialMetrics(
            split="validation",
            workload="strong",
            seed=1,
            token_hit_rate=0.8,
        ),
        TrialMetrics(
            split="validation",
            workload="weak",
            seed=1,
            token_hit_rate=0.2,
        ),
    ]

    assert evaluator._score_trials(trials, invalid_fraction=0.0, complexity=0) == 60.0


def test_score_penalizes_policy_caused_underfill_with_cap() -> None:
    config = EvaluatorConfig(
        w_avg_tok=0.0,
        w_avg_blk=0.0,
        min_workload_weight=0.0,
        request_tail_weight=0.0,
        worst_window_weight=0.0,
        priority_hit_weight=0.0,
        wasted_admission_weight=0.0,
        admission_utility_weight=0.0,
        avoidable_eviction_weight=0.0,
        latency_weight=0.0,
        churn_weight=0.0,
        underfill_weight=10.0,
        underfill_cap=3.0,
        fairness_weight=0.0,
        k_complex=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trial = TrialMetrics(
        split="validation",
        workload="unit",
        seed=1,
        policy_underfill_rate=0.4,
    )

    breakdown = evaluator._score_breakdown([trial], invalid_fraction=0.0, complexity=0)

    assert breakdown["policy_underfill_rate"] == 0.4
    assert breakdown["underfill_cost"] == 3.0
    assert breakdown["combined_score"] == -3.0


def test_score_blends_mean_with_worst_seed() -> None:
    config = EvaluatorConfig(
        w_avg_tok=100.0,
        w_avg_blk=0.0,
        min_workload_weight=0.0,
        min_seed_weight=0.25,
        request_tail_weight=0.0,
        worst_window_weight=0.0,
        priority_hit_weight=0.0,
        wasted_admission_weight=0.0,
        avoidable_eviction_weight=0.0,
        latency_weight=0.0,
        churn_weight=0.0,
        fairness_weight=0.0,
        k_complex=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trials = [
        TrialMetrics(
            split="validation",
            workload="unit",
            seed=1,
            token_hit_rate=0.8,
        ),
        TrialMetrics(
            split="validation",
            workload="unit",
            seed=2,
            token_hit_rate=0.4,
        ),
    ]

    assert math.isclose(
        evaluator._score_trials(trials, invalid_fraction=0.0, complexity=0),
        55.0,
    )


def test_score_rewards_temporal_tail_and_penalizes_avoidable_evictions() -> None:
    config = EvaluatorConfig(
        w_avg_tok=0.0,
        w_avg_blk=0.0,
        min_workload_weight=0.0,
        min_seed_weight=0.0,
        request_tail_weight=10.0,
        worst_window_weight=20.0,
        priority_hit_weight=0.0,
        wasted_admission_weight=6.0,
        avoidable_eviction_weight=8.0,
        latency_weight=0.0,
        churn_weight=0.0,
        fairness_weight=0.0,
        k_complex=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trial = TrialMetrics(
        split="validation",
        workload="unit",
        seed=1,
        request_token_hit_rate_p10=0.4,
        worst_quarter_token_hit_rate=0.5,
        wasted_admission_rate=0.75,
        wasted_admission_token_rate=0.25,
        avoidable_eviction_rate=0.125,
    )

    assert evaluator._score_trials([trial], invalid_fraction=0.0, complexity=0) == 11.5


def test_score_rewards_token_utility_concavely() -> None:
    config = EvaluatorConfig(
        w_avg_tok=0.0,
        w_avg_blk=0.0,
        min_workload_weight=0.0,
        min_seed_weight=0.0,
        request_tail_weight=0.0,
        worst_window_weight=0.0,
        priority_hit_weight=0.0,
        wasted_admission_weight=0.0,
        admission_utility_weight=2.0,
        avoidable_eviction_weight=0.0,
        latency_weight=0.0,
        churn_weight=0.0,
        fairness_weight=0.0,
        k_complex=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    full = TrialMetrics(
        split="validation",
        workload="unit",
        seed=1,
        admission_token_utility=1.0,
    )
    partial = TrialMetrics(
        split="validation",
        workload="unit",
        seed=1,
        admission_token_utility=0.25,
    )

    full_score = evaluator._score_trials([full], invalid_fraction=0.0, complexity=0)
    partial_score = evaluator._score_trials([partial], invalid_fraction=0.0, complexity=0)

    assert math.isclose(full_score, 2.0 * math.log1p(1.0))
    assert math.isclose(partial_score, 2.0 * math.log1p(0.25))
    assert full_score > partial_score


def test_auto_latency_normalization_is_scoped_per_workload() -> None:
    config = EvaluatorConfig(
        w_avg_tok=0.0,
        w_avg_blk=0.0,
        min_workload_weight=0.0,
        latency_weight=100.0,
        latency_cap=1_000.0,
        churn_weight=0.0,
        fairness_weight=0.0,
        k_complex=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trials = [
        TrialMetrics(
            split="validation",
            workload="short",
            seed=1,
            p95_latency_proxy=50.0,
            max_prefill_cost=100.0,
        ),
        TrialMetrics(
            split="validation",
            workload="long",
            seed=1,
            p95_latency_proxy=0.0,
            max_prefill_cost=10_000.0,
        ),
    ]

    assert evaluator._score_trials(trials, invalid_fraction=0.0, complexity=0) == -25.0


def test_capacity_sweep_reports_capacity_metrics() -> None:
    config = EvaluatorConfig(
        request_count=24,
        seeds=(3,),
        capacity_blocks=12,
        capacity_sweep_blocks=(8, 16),
        validation_families=("agent_trace_branching",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)

    assert set(result.capacity_metrics) == {"capacity_8", "capacity_16"}
    assert {trial.capacity_blocks for trial in result.trials} == {8, 16}
    assert result.candidate_metadata["capacity_sweep_blocks"] == "8,16"
    assert result.candidate_metadata["capacity_sweep_tokens"] == "128,256"
    assert result.candidate_metadata["block_size_tokens"] == 16
    assert result.candidate_metadata["workload_token_granularity"] == 8
    assert result.candidate_metadata["complexity_exponent"] == 0.75
    assert result.candidate_metadata["complexity_mode"] == "legacy_ast_nodes"


def test_aggregate_trials_preserves_peak_active_request_count() -> None:
    metrics = _aggregate_trials(
        [
            TrialMetrics(
                split="validation",
                workload="unit",
                seed=1,
                active_request_count_peak=3,
            ),
            TrialMetrics(
                split="validation",
                workload="unit",
                seed=2,
                active_request_count_peak=11,
            ),
        ]
    )

    assert metrics["active_request_count_peak"] == 11


def test_score_min_term_includes_capacity_variants() -> None:
    config = EvaluatorConfig(
        w_avg_tok=100.0,
        w_avg_blk=0.0,
        min_workload_weight=0.5,
        latency_weight=0.0,
        churn_weight=0.0,
        fairness_weight=0.0,
        k_complex=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trials = [
        TrialMetrics(
            split="validation",
            workload="agentic",
            seed=1,
            capacity_blocks=24,
            token_hit_rate=0.9,
        ),
        TrialMetrics(
            split="validation",
            workload="agentic",
            seed=1,
            capacity_blocks=48,
            token_hit_rate=0.1,
        ),
    ]

    assert evaluator._score_trials(trials, invalid_fraction=0.0, complexity=0) == 55.0


def test_complexity_penalty_orders(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(request_count=12, seeds=(3,)),
    )
    simple = levi_evaluator.evaluate_source(_minimal_policy_source("-1.0", "0.0"))
    complex_source = _minimal_policy_source(
        "-1.0 + 0.0 * (block.depth + block.hit_count + block.descendant_count)",
        "0.0 + 0.0 * (now + block.depth + block.hit_count + block.token_count)",
    )
    complex_result = levi_evaluator.evaluate_source(complex_source)

    assert complex_result.metrics["success"] is True
    assert simple.metrics["combined_score"] > complex_result.metrics["combined_score"]
    assert (
        simple.artifacts["candidate_metadata"]["scoring_fn_complexity"]
        < complex_result.artifacts["candidate_metadata"]["scoring_fn_complexity"]
    )


def test_existing_trials_can_be_rescored_without_rerunning_simulation() -> None:
    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        validation_families=("hotset_cold_scan",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)
    stricter = config.with_updates(churn_weight=1.0)

    rescored = PrefixKVCacheEvaluator(
        stricter,
        splits=("validation",),
    ).rescore_trials(result.trials)

    assert rescored.trials == result.trials
    assert rescored.combined_score < result.combined_score


def test_score_weight_sensitivity_uses_fixed_trials() -> None:
    config = EvaluatorConfig(
        request_count=36,
        seeds=(3,),
        capacity_blocks=8,
        validation_families=("hotset_cold_scan",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    results = {
        "candidate": evaluator(baseline_lru_blocks),
        "no_cache": evaluator(baseline_no_cache),
    }

    rows = _score_weight_sensitivity_rows(
        results,
        config,
        weights=("churn_weight",),
        factors=(0.0, 1.0, 2.0),
    )

    assert len(rows) == 3
    assert rows[1]["candidate_score"] == results["candidate"].combined_score
    assert rows[2]["candidate_score"] <= rows[1]["candidate_score"]


def test_runner_default_report_matches_levi_capacity_sweep() -> None:
    default_config = _config_from_args(
        quick=True,
        capacity_blocks=None,
        block_size_tokens=None,
    )
    explicit_config = _config_from_args(
        quick=True,
        capacity_blocks=12,
        block_size_tokens=None,
    )

    assert default_config.effective_capacity_blocks() == (24, 48)
    assert default_config.effective_capacity_tokens() == (384, 768)
    assert explicit_config.effective_capacity_blocks() == (12,)


def test_block_size_robustness_normalizes_capacity_and_replays_canonical_traffic(
    tmp_path,
    monkeypatch,
) -> None:
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text("def build_candidate(): pass\n", encoding="utf-8")
    observed_configs = []

    def fake_evaluate_candidate(config, path, *, splits):
        observed_configs.append(config)
        assert path == candidate_path
        assert splits == ("validation",)
        return _report_result(12.0, capacities=config.effective_capacity_blocks())

    fake_suite = SimpleNamespace(
        evaluate=lambda config, baselines, *, splits: {
            name: _report_result(10.0, capacities=config.effective_capacity_blocks())
            for name in baselines
        }
    )
    monkeypatch.setattr(prefix_runner, "_evaluate_candidate_program", fake_evaluate_candidate)
    monkeypatch.setattr(prefix_runner, "BASELINE_SUITE_EVALUATOR", fake_suite)

    output_path = tmp_path / "block_sizes.md"
    prefix_runner.write_block_size_robustness_report(
        output_path,
        candidate_program=candidate_path,
        quick=True,
    )

    assert [
        (
            config.block_size_tokens,
            config.effective_capacity_blocks(),
            config.effective_capacity_tokens(),
            config.effective_workload_token_granularity(),
        )
        for config in observed_configs
    ] == [
        (8, (48, 96), (384, 768), 8),
        (16, (24, 48), (384, 768), 8),
        (32, (12, 24), (384, 768), 8),
    ]
    report = output_path.read_text(encoding="utf-8")
    assert "identical synthetic token streams" in report
    assert "| 16 | 12.000 | 12.000 | 0.000 | 1 / 4 | `candidate` | 0.000 |" in report
    assert "| 16 | 24 / 48 | 384 / 768 | `candidate` |" in report


def test_block_size_robustness_rejects_inexact_token_capacity() -> None:
    with pytest.raises(ValueError, match="divisible"):
        prefix_runner._capacity_blocks_for_token_tiers(
            (385, 768),
            block_size_tokens=16,
        )


def test_candidate_prompt_names_only_supported_lifecycle_callbacks() -> None:
    config = prefix_runner._CONFIG_LOADER.load(Path("configs/prefix_kv_cache.yaml"))
    message = config.raw["prompt"]["system_message"]

    assert config.run_cost == {}
    assert "No other lifecycle callback fires." in message
    assert "session_id is request-only metadata" in message
    assert "now argument is a logical arrival step" in message
    assert "Priority is a deployable request signal, not proof of reuse" in message
    assert "Long-horizon tenant workloads repeatedly shift" in message
    assert "Do not hard-code workload-family names or use scrubbed request fields." in message
    assert 'request_type is normalized to "request"' in message
    assert "candidate factory receives a fixed policy seed" in message
    assert "Make exactly one semantic change per mutation." in message
    assert "effective complexity at or below 650 AST nodes" in message
    assert "targeted agentic and stochastic guidance" in message
    assert "agent_trace_branching probe remains reporting-only" in message
    for field in (
        "prev_last_accessed_at",
        "last_access_gap",
        "access_gap_mean",
        "access_gap_var",
        "subtree_hit_rate",
        "subtree_active_ref_count",
        "recent_admission_pressure",
        "recent_miss_rate",
        "MultiTimescaleDecay",
        "observe_vector",
        "decay_vector",
        "threshold_excess",
    ):
        assert field in message
    for callback in (
        "on_request_start",
        "on_cache_hit",
        "on_cache_miss",
        "on_request_end",
        "on_block_admitted",
        "on_block_evicted",
    ):
        assert callback in message


def test_candidate_config_matches_default_verifier_panel_and_score_weights() -> None:
    config = prefix_runner._CONFIG_LOADER.load(Path("configs/prefix_kv_cache.yaml"))
    settings = config.raw["problem"]["settings"]
    default = EvaluatorConfig()
    loaded = load_evaluator_config(Path("configs/prefix_kv_cache.yaml"))

    assert tuple(settings["train_families"]) == default.train_families
    assert tuple(settings["validation_families"]) == default.validation_families
    assert tuple(settings["probe_families"]) == default.probe_families
    assert tuple(settings["hidden_families"]) == default.hidden_families
    assert loaded.form_aware_complexity is True
    assert loaded.verifier_version == VERIFIER_VERSION
    assert loaded.family_request_multipliers == default.family_request_multipliers
    assert loaded.timeout_s == 90
    assert loaded.request_count == 96
    assert loaded.seeds == (11, 23, 37)
    assert loaded.policy_seed == 0
    assert loaded.effective_capacity_blocks() == (24, 48)
    assert loaded.effective_capacity_tokens() == (384, 768)
    assert loaded.block_size_tokens == 16
    assert loaded.workload_token_granularity == 8
    assert loaded.max_candidate_complexity is None
    assert loaded.promotion_max_candidate_complexity == 650
    assert loaded.surrogate_probe_tripwire_thresholds == {
        "agentic_branching": 0.12,
        "cyclic_working_set": 0.25,
    }
    assert loaded.reject_unsupported_source_patterns is True
    for field in (
        "w_avg_tok",
        "w_avg_blk",
        "min_workload_weight",
        "min_seed_weight",
        "request_tail_weight",
        "worst_window_weight",
        "priority_hit_weight",
        "wasted_admission_weight",
        "admission_utility_weight",
        "avoidable_eviction_weight",
        "latency_weight",
        "latency_cap",
        "churn_weight",
        "churn_cap",
        "underfill_weight",
        "underfill_cap",
        "fairness_weight",
        "fairness_cap",
        "k_complex",
        "complexity_exponent",
        "v_min",
        "invalid_surcharge",
    ):
        assert settings["scoring"][field] == getattr(default, field)


def test_evaluator_config_rejects_incomplete_or_unknown_tripwire_channels() -> None:
    with pytest.raises(ValueError, match=r"missing: cyclic_working_set"):
        EvaluatorConfig(
            surrogate_probe_tripwire_thresholds={
                "agentic_branching": 0.12,
            }
        )

    with pytest.raises(ValueError, match=r"unknown: unrelated_probe"):
        EvaluatorConfig(
            surrogate_probe_tripwire_thresholds={
                "agentic_branching": 0.12,
                "cyclic_working_set": 0.25,
                "unrelated_probe": 0.10,
            }
        )


def test_evaluator_config_rejects_invalid_robust_search_guidance() -> None:
    with pytest.raises(ValueError, match="requires at least one"):
        EvaluatorConfig(search_score_mode="robust_min")

    with pytest.raises(ValueError, match="configured train families"):
        EvaluatorConfig(
            search_score_mode="robust_min",
            search_guidance_families=("agent_trace_branching",),
        )

    with pytest.raises(ValueError, match="requires search_score_mode"):
        EvaluatorConfig(search_guidance_families=("agentic_tool_workflows",))


def test_candidate_search_configuration_is_forwarded_to_levi() -> None:
    config = prefix_runner._CONFIG_LOADER.load(Path("configs/prefix_kv_cache.yaml"))
    raw = config.raw
    kwargs = config.evolve_kwargs()

    for section in (
        "init",
        "cvt",
        "behavior",
        "meta_advice",
        "punctuated_equilibrium",
        "prompt_overrides",
    ):
        assert kwargs[section] == raw[section]
    assert kwargs["pipeline"]["output_mode"] == raw["pipeline"]["output_mode"]
    assert kwargs["pipeline"]["temperature"] == raw["llm"]["temperature"]
    assert kwargs["pipeline"]["max_tokens"] == raw["llm"]["max_tokens"]
    assert kwargs["pipeline"]["eval_timeout"] == raw["evaluator"]["timeout"]
    assert kwargs["pipeline"]["n_eval_processes"] == raw["evaluator"]["parallel_evaluations"]
    assert kwargs["cascade"]["enabled"] == raw["evaluator"]["cascade_evaluation"]
    assert config.search_seed == raw["search"]["seed"]
    assert {
        "train_workload_agentic_tool_workflows_token_hit_rate",
        "train_workload_agentic_tool_workflows_policy_underfill_rate",
        "validation_workload_stochastic_serving_mix_token_hit_rate",
        "validation_avoidable_eviction_rate",
        "validation_short_reuse_after_eviction_missed_token_rate",
        "validation_shadow_price_tracking_rmse",
    }.issubset(kwargs["behavior"]["score_keys"])


def test_eviction_specialist_config_fixes_admission_and_separates_promotion_cap() -> None:
    path = Path("configs/prefix_kv_cache_eviction_specialist.yaml")
    workflow_config = prefix_runner._CONFIG_LOADER.load(path)
    evaluator_config = load_evaluator_config(path)

    assert evaluator_config.fixed_admission_policy == "discovery_8tok_20260608"
    assert evaluator_config.candidate_policy_surface == "eviction_only"
    assert evaluator_config.search_score_mode == "raw_before_complexity"
    assert evaluator_config.max_candidate_complexity == 1000
    assert evaluator_config.promotion_max_candidate_complexity == 650
    assert evaluator_config.surrogate_probe_tripwire_thresholds == {
        "agentic_branching": 0.12,
        "cyclic_working_set": 0.25,
    }
    assert evaluator_config.effective_capacity_blocks() == (24, 48)
    assert evaluator_config.effective_capacity_tokens() == (384, 768)
    assert {
        "validation_avoidable_eviction_rate",
        "validation_short_reuse_after_eviction_missed_token_rate",
    }.issubset(workflow_config.behavior["score_keys"])
    assert "raw behavioral improvement" in workflow_config.problem_description
    assert "remain at most 650" in workflow_config.problem_description.lower()
    assert workflow_config.function_signature == (
        "def score_eviction(block, now, frequency, priority):"
    )
    assert workflow_config.init["n_diverse_seeds"] == 6
    assert workflow_config.cvt["n_centroids"] == 16


def test_prefix_evaluator_config_rejects_inactive_settings() -> None:
    with pytest.raises(ValueError, match=r"(?s)seed_count.*Extra inputs are not permitted"):
        evaluator_config_from_settings({"seed_count": 3})

    with pytest.raises(ValueError, match=r"(?s)complexity_cap.*Extra inputs are not permitted"):
        evaluator_config_from_settings({"scoring": {"complexity_cap": 500}})


def test_loaded_evaluator_config_requires_explicit_verifier_version(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
problem:
  settings: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must explicitly declare"):
        load_evaluator_config(config_path)


def test_evaluator_config_validates_direct_values_and_updates() -> None:
    with pytest.raises(ValueError, match=r"(?s)capacity_blocks.*greater than 0"):
        EvaluatorConfig(capacity_blocks=0)

    with pytest.raises(ValueError, match=r"(?s)min_seed_weight.*less than or equal to 1"):
        EvaluatorConfig(min_seed_weight=1.1)

    with pytest.raises(ValueError, match=r"(?s)kv_capacity_mode.*Input should be"):
        EvaluatorConfig(kv_capacity_mode="unknown")

    with pytest.raises(ValueError, match=r"verifier_version"):
        EvaluatorConfig(verifier_version="1.0")

    with pytest.raises(ValueError, match=r"implements verifier 1\.0\.0"):
        EvaluatorConfig(verifier_version="0.9.0")

    config = EvaluatorConfig().with_updates(scoring={"churn_weight": 0.5})

    assert config.churn_weight == 0.5


def test_active_evaluator_config_applies_quick_worker_override(monkeypatch) -> None:
    monkeypatch.setenv(PREFIX_KV_QUICK_ENV, "1")

    config = active_evaluator_config(EvaluatorConfig())

    assert config.request_count == 36
    assert config.seeds == (3,)
    assert config.family_request_multipliers == {}


def test_demo_run_evolution_uses_requested_seed_program(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best_program.py").write_text("chosen seed\n", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def execute(self, iterations):
            captured["iterations"] = iterations
            captured["config_path"] = os.environ[PREFIX_KV_CONFIG_ENV]
            captured["quick"] = os.environ[PREFIX_KV_QUICK_ENV]
            return SimpleNamespace()

    def fake_build_workflow(provider, *, program_source):
        captured["provider"] = provider
        captured["source"] = program_source.text()
        return FakeWorkflow()

    monkeypatch.setattr(prefix_runner, "_build_workflow", fake_build_workflow)

    prefix_runner.demo_run_evolution(
        iterations=7,
        quick=True,
        seed_program=run_dir,
        artifact_output=None,
    )

    assert captured["iterations"] == 7
    assert captured["source"] == "chosen seed\n"
    assert captured["config_path"] == str(Path("configs/prefix_kv_cache.yaml").resolve())
    assert captured["quick"] == "1"


def test_demo_run_evolution_uses_function_only_seed_for_eviction_specialist(monkeypatch) -> None:
    captured = {}

    class FakeWorkflow:
        def execute(self, iterations):
            return SimpleNamespace()

    def fake_build_workflow(provider, *, program_source):
        captured["source"] = program_source.text()
        return FakeWorkflow()

    monkeypatch.setattr(prefix_runner, "_build_workflow", fake_build_workflow)

    prefix_runner.demo_run_evolution(
        iterations=1,
        config_file="configs/prefix_kv_cache_eviction_specialist.yaml",
        artifact_output=None,
    )

    assert "def score_eviction(block, now, frequency, priority):" in captured["source"]
    assert "def score_admission" not in captured["source"]


def test_demo_run_evolution_defaults_to_production_incumbent(monkeypatch) -> None:
    captured = {}

    class FakeWorkflow:
        def execute(self, iterations):
            return SimpleNamespace()

    def fake_build_workflow(provider, *, program_source):
        captured["source"] = program_source.text()
        return FakeWorkflow()

    monkeypatch.setattr(prefix_runner, "_build_workflow", fake_build_workflow)

    prefix_runner.demo_run_evolution(iterations=1, quick=True, artifact_output=None)

    expected = current_incumbent("production").source_path.read_text(encoding="utf-8")
    assert captured["source"] == expected


def test_show_config_applies_model_and_search_seed_overrides(capsys, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "secret-value")
    prefix_runner._show_resolved_config(
        iterations=3,
        config_file="configs/prefix_kv_cache.yaml",
        quick=False,
        model="anthropic/example-model",
        primary_model=None,
        secondary_model=None,
        search_seed=17,
        api_base="http://localhost:8000/v1",
        api_key_env="LOCAL_MODEL_API_KEY",
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["models"] == {
        "model": "anthropic/example-model",
        "mutation_model": None,
        "paradigm_model": None,
    }
    assert payload["search_seed"] == 17
    assert payload["api_base"] == "http://localhost:8000/v1"
    assert payload["api_key_env"] == "LOCAL_MODEL_API_KEY"
    assert payload["api_key_env_set"] is True
    assert "secret-value" not in output
    assert payload["evaluator"]["workload_seeds"] == [11, 23, 37]


def test_hidden_report_evaluates_requested_candidate(tmp_path, monkeypatch, capsys) -> None:
    candidate_path = tmp_path / "best_program.py"
    candidate_path.write_text("def build_candidate(): pass\n", encoding="utf-8")
    captured = {}

    def fake_evaluate_candidate(config, path, *, splits):
        captured["path"] = path
        captured["splits"] = splits
        return score_record(12.5)

    monkeypatch.setattr(prefix_runner, "_evaluate_candidate_program", fake_evaluate_candidate)
    monkeypatch.setattr(prefix_runner, "REPORTING_BASELINES", {})

    prefix_runner.hidden_report(quick=True, candidate_program=candidate_path)

    assert captured == {"path": candidate_path, "splits": ("hidden",)}
    assert f"candidate={candidate_path}" in capsys.readouterr().out


def test_probe_report_evaluates_requested_candidate(tmp_path, monkeypatch, capsys) -> None:
    candidate_path = tmp_path / "best_program.py"
    candidate_path.write_text("def build_candidate(): pass\n", encoding="utf-8")
    output_path = tmp_path / "probe.json"
    captured = {}

    def fake_evaluate_candidate(config, path, *, splits):
        captured["path"] = path
        captured["splits"] = splits
        return score_record(
            12.5,
            success=True,
            invalid_fraction=0.0,
            split_metrics={"probe": {"token_hit_rate": 0.5}},
            workload_metrics={
                "probe/agent_trace_branching": {
                    "token_hit_rate": 0.5,
                    "block_hit_rate": 0.4,
                    "cache_churn_per_1k": 10.0,
                }
            },
            capacity_metrics={},
            candidate_metadata={},
            score_breakdown={"combined_score": 12.5},
        )

    monkeypatch.setattr(prefix_runner, "_evaluate_candidate_program", fake_evaluate_candidate)
    monkeypatch.setattr(prefix_runner, "REPORTING_BASELINES", {})

    payload = prefix_runner.probe_report(
        output_path=output_path,
        quick=True,
        candidate_program=candidate_path,
    )

    assert captured == {"path": candidate_path, "splits": ("probe",)}
    assert payload["selection_score_excludes_probe"] is True
    assert output_path.exists()
    assert f"structure_probe={output_path}" in capsys.readouterr().out


def test_workload_builder_uses_predicted_not_true_output_length() -> None:
    request = build_workload(
        "shared_system_prompt",
        request_count=1,
        block_size_tokens=8,
        seed=3,
    )[0]

    assert request.info.predicted_output_length is None
    assert request.info.prompt_tokens == ()
    assert request.prompt_tokens
    assert isinstance(request.true_output_length, int)


def test_session_continuation_growth_resumes_and_extends_prefix() -> None:
    requests = build_workload(
        "session_continuation_growth",
        request_count=8,
        block_size_tokens=8,
        seed=3,
    )

    first_turn = requests[0]
    resumed_session = requests[4]
    assert first_turn.info.session_id == resumed_session.info.session_id
    assert (
        resumed_session.prompt_tokens[: len(first_turn.prompt_tokens)] == first_turn.prompt_tokens
    )
    assert resumed_session.info.prompt_length == first_turn.info.prompt_length + 8


def test_agent_trace_branching_accumulates_tool_history_and_retries() -> None:
    requests = build_workload(
        "agent_trace_branching",
        request_count=48,
        block_size_tokens=8,
        seed=3,
    )

    prompt_lengths = [request.info.prompt_length for request in requests]
    request_types = {request.info.request_type for request in requests}
    assert max(prompt_lengths) > min(prompt_lengths) + 10 * 8
    assert request_types == {"agent_loop", "agent_retry"}


def test_agentic_tool_workflows_train_on_irregular_forks_and_replans() -> None:
    requests = build_workload(
        "agentic_tool_workflows",
        request_count=96,
        block_size_tokens=8,
        seed=3,
    )

    prompt_lengths = [request.info.prompt_length for request in requests]
    request_types = {request.info.request_type for request in requests}
    prompt_block_counts = [(length + 7) // 8 for length in prompt_lengths]
    assert max(prompt_block_counts) >= 48
    assert len({request.info.session_id for request in requests}) == 12
    assert request_types == {
        "agentic_step",
        "agentic_fork",
        "agentic_resume",
        "agentic_replan",
    }


def test_stochastic_serving_mix_interleaves_classes_in_bursts() -> None:
    requests = build_workload(
        "stochastic_serving_mix",
        request_count=96,
        block_size_tokens=8,
        seed=3,
    )

    request_classes = [request.info.request_type.split("_", maxsplit=2)[1] for request in requests]
    assert len(set(request_classes)) >= 4
    assert any(
        request_classes[index] == request_classes[index + 1] != request_classes[index + 2]
        for index in range(len(request_classes) - 2)
    )
    assert any(
        request_classes[index] != request_classes[index + 1]
        for index in range(len(request_classes) - 1)
    )
    arrival_steps = [request.arrival_step for request in requests]
    assert all(step is not None for step in arrival_steps)
    arrival_gaps = [right - left for left, right in zip(arrival_steps, arrival_steps[1:])]
    assert 0 in arrival_gaps
    assert max(arrival_gaps) > 1


def test_rolling_template_versions_models_canary_rollout_and_rollback() -> None:
    requests = build_workload(
        "rolling_template_versions",
        request_count=64,
        block_size_tokens=8,
        seed=3,
    )

    versions = [request.info.request_type for request in requests]
    assert set(versions[:16]) == {"rolling_template_v0"}
    assert set(versions[16:32]) == {"rolling_template_v0", "rolling_template_v1"}
    assert set(versions[32:48]) == {"rolling_template_v0", "rolling_template_v1"}
    assert versions[32:48].count("rolling_template_v1") > versions[32:48].count(
        "rolling_template_v0"
    )
    assert set(versions[48:]) == {"rolling_template_v0", "rolling_template_v1"}
    assert versions[48:].count("rolling_template_v0") > versions[48:].count("rolling_template_v1")


def test_heavy_tailed_prefix_lengths_include_expensive_outliers() -> None:
    requests = build_workload(
        "heavy_tailed_prefix_lengths",
        request_count=96,
        block_size_tokens=8,
        seed=3,
    )

    prompt_lengths = sorted(request.info.prompt_length for request in requests)
    median_prompt_length = prompt_lengths[len(prompt_lengths) // 2]
    assert len(set(prompt_lengths)) >= 8
    assert prompt_lengths[-1] >= 2 * median_prompt_length


def test_priority_burst_recovery_models_qos_pollution_and_recovery() -> None:
    requests = build_workload(
        "priority_burst_recovery",
        request_count=96,
        block_size_tokens=8,
        seed=3,
    )

    request_types = {request.info.request_type for request in requests}
    priorities = {request.info.priority for request in requests}
    warm_prompts = {
        request.prompt_tokens
        for request in requests[:24]
        if request.info.request_type == "priority_hot_warm"
    }
    assert request_types == {
        "priority_hot_warm",
        "priority_hot_during_burst",
        "priority_background_scan",
        "priority_medium_recovery",
        "priority_hot_recovery",
    }
    assert priorities == {0, 1, 3}
    assert any(
        request.info.request_type == "priority_background_scan" for request in requests[24:72]
    )
    assert any(
        request.info.request_type == "priority_hot_recovery"
        and request.prompt_tokens in warm_prompts
        for request in requests[72:]
    )

    config = EvaluatorConfig(
        request_count=96,
        seeds=(3,),
        capacity_blocks=24,
        validation_families=("priority_burst_recovery",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)
    metrics = result.workload_metrics["validation/priority_burst_recovery"]
    assert metrics["recovery_request_count"] > 0
    assert metrics["recovery_token_hit_rate"] > 0.0
    assert metrics["recovery_p95_latency_proxy"] > 0.0


def test_priority_aware_admission_protects_high_priority_hit_rate() -> None:
    class PriorityAwareAdmission(AdmitAllLRU):
        def __init__(self) -> None:
            self.priority = 0

        def on_request_start(self, request, now: int) -> None:
            self.priority = request.priority

        def score_admission(self, block, now: int) -> float:
            return 1.0 if self.priority > 0 else -1.0

    config = EvaluatorConfig(
        request_count=96,
        seeds=(3,),
        capacity_blocks=24,
        validation_families=("priority_burst_recovery",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    lru = evaluator(baseline_lru_blocks)
    priority_aware = evaluator(lambda *_: PriorityAwareAdmission())
    lru_metrics = lru.workload_metrics["validation/priority_burst_recovery"]
    priority_metrics = priority_aware.workload_metrics["validation/priority_burst_recovery"]

    assert (
        priority_metrics["high_priority_token_hit_rate"]
        > lru_metrics["high_priority_token_hit_rate"]
    )
    assert (
        priority_metrics["priority_weighted_token_hit_rate"]
        > lru_metrics["priority_weighted_token_hit_rate"]
    )
    assert priority_metrics["policy_bypass_token_rate"] > 0.0
    assert priority_metrics["cache_churn_per_1k"] < lru_metrics["cache_churn_per_1k"]


def test_cyclic_working_set_pressure_exposes_short_horizon_eviction_regret() -> None:
    requests = build_workload(
        "cyclic_working_set_pressure",
        request_count=96,
        block_size_tokens=8,
        seed=3,
    )

    assert {request.info.request_type for request in requests[:48]} == {"cyclic_working_set_small"}
    assert {request.info.request_type for request in requests[48:]} == {"cyclic_working_set_large"}
    assert len({request.prompt_tokens for request in requests[:48]}) == 9
    assert len({request.prompt_tokens for request in requests[48:]}) == 17

    config = EvaluatorConfig(
        request_count=96,
        seeds=(3,),
        capacity_blocks=24,
        validation_families=("cyclic_working_set_pressure",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    lru = evaluator(baseline_lru_blocks)
    tinylfu = evaluator(baseline_tinylfu_lru)
    lru_metrics = lru.workload_metrics["validation/cyclic_working_set_pressure"]
    tinylfu_metrics = tinylfu.workload_metrics["validation/cyclic_working_set_pressure"]

    assert lru_metrics["short_reuse_after_eviction_missed_token_rate"] > 0.0
    assert (
        tinylfu_metrics["short_reuse_after_eviction_missed_token_rate"]
        < lru_metrics["short_reuse_after_eviction_missed_token_rate"]
    )


def test_priority_one_off_noise_penalizes_blind_priority_admission() -> None:
    class PriorityOnlyAdmission(AdmitAllLRU):
        def __init__(self) -> None:
            self.priority = 0

        def on_request_start(self, request, now: int) -> None:
            self.priority = request.priority

        def score_admission(self, block, now: int) -> float:
            return 1.0 if self.priority > 0 else -1.0

    requests = build_workload(
        "priority_one_off_noise",
        request_count=50,
        block_size_tokens=8,
        seed=3,
    )
    high_priority = [
        request for request in requests if request.info.request_type == "priority_one_off_noise"
    ]
    normal = [
        request for request in requests if request.info.request_type == "priority_normal_recurring"
    ]
    assert len(high_priority) == 20
    assert len({request.prompt_tokens for request in high_priority}) == 20
    assert len({request.prompt_tokens for request in normal}) == 5

    config = EvaluatorConfig(
        request_count=96,
        seeds=(3,),
        capacity_blocks=24,
        validation_families=("priority_one_off_noise",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    priority_only = evaluator(lambda *_: PriorityOnlyAdmission())
    tinylfu = evaluator(baseline_tinylfu_lru)
    priority_metrics = priority_only.workload_metrics["validation/priority_one_off_noise"]
    tinylfu_metrics = tinylfu.workload_metrics["validation/priority_one_off_noise"]

    assert tinylfu_metrics["token_hit_rate"] > priority_metrics["token_hit_rate"]
    assert tinylfu_metrics["wasted_admission_rate"] < priority_metrics["wasted_admission_rate"]


def test_tenant_phase_shift_cycles_repeat_pollution_and_delayed_recovery() -> None:
    config = EvaluatorConfig(
        request_count=96,
        validation_families=("tenant_phase_shift_cycles",),
    )
    workload_config = config.workload_configs(("validation",))[0]
    requests = build_workload(
        workload_config.family,
        request_count=workload_config.request_count,
        block_size_tokens=config.block_size_tokens,
        seed=3,
    )
    request_types = [request.info.request_type for request in requests]
    warm_prompts = {
        (request.info.tenant_id, request.prompt_tokens)
        for request in requests
        if request.info.request_type == "tenant_cycle_warm"
    }
    recovery_prompts = {
        (request.info.tenant_id, request.prompt_tokens)
        for request in requests
        if request.info.request_type == "tenant_cycle_recovery"
    }

    assert workload_config.request_count == 288
    assert set(request_types) == {
        "tenant_cycle_warm",
        "tenant_cycle_pollution",
        "tenant_cycle_recovery",
    }
    assert recovery_prompts <= warm_prompts
    assert request_types.count("tenant_cycle_recovery") >= 60
    assert max(request.arrival_step or 0 for request in requests) > 200

    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)
    metrics = result.workload_metrics["validation/tenant_phase_shift_cycles"]
    assert metrics["recovery_phase_count"] == 6
    assert metrics["worst_recovery_phase_token_hit_rate"] <= metrics["recovery_token_hit_rate"]
    assert metrics["worst_recovery_phase_p95_latency_proxy"] > 0.0


def test_default_splits_include_production_shaped_workloads() -> None:
    config = EvaluatorConfig()

    assert "agentic_tool_workflows" in config.train_families
    assert {
        "stochastic_serving_mix",
        "rolling_template_versions",
        "heavy_tailed_prefix_lengths",
        "priority_burst_recovery",
        "priority_one_off_noise",
        "tenant_phase_shift_cycles",
    }.issubset(config.validation_families)
    assert {
        "agent_trace_branching",
        "cyclic_working_set_pressure",
    } == set(config.probe_families)
    assert set(config.probe_families).isdisjoint(config.validation_families)
    assert {
        "stochastic_serving_mix_shifted",
        "rolling_template_versions_shifted",
        "heavy_tailed_prefix_lengths_shifted",
        "priority_burst_recovery_shifted",
        "cyclic_working_set_pressure_shifted",
        "priority_one_off_noise_shifted",
        "tenant_phase_shift_cycles_shifted",
    }.issubset(config.hidden_families)


def test_production_shaped_workloads_reward_selective_admission() -> None:
    families = (
        "stochastic_serving_mix",
        "rolling_template_versions",
        "heavy_tailed_prefix_lengths",
    )
    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        validation_families=families,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    lru = evaluator(baseline_lru_blocks)
    tinylfu = evaluator(baseline_tinylfu_lru)

    for family in families:
        lru_metrics = lru.workload_metrics[f"validation/{family}"]
        tinylfu_metrics = tinylfu.workload_metrics[f"validation/{family}"]
        assert tinylfu_metrics["token_hit_rate"] > lru_metrics["token_hit_rate"]
        assert tinylfu_metrics["cache_churn_per_1k"] < lru_metrics["cache_churn_per_1k"]


def test_tenant_session_reentry_revisits_paused_context_with_new_tail() -> None:
    requests = build_workload(
        "tenant_session_reentry",
        request_count=40,
        block_size_tokens=8,
        seed=3,
    )

    first_visit = requests[0]
    resumed_session = requests[32]
    stable_prefix_tokens = 4 * 8
    assert first_visit.info.tenant_id == resumed_session.info.tenant_id
    assert first_visit.info.session_id == resumed_session.info.session_id
    assert (
        first_visit.prompt_tokens[:stable_prefix_tokens]
        == resumed_session.prompt_tokens[:stable_prefix_tokens]
    )
    assert first_visit.prompt_tokens != resumed_session.prompt_tokens


def test_tenant_session_reentry_rewards_selective_admission() -> None:
    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        block_size_tokens=8,
        hidden_families=("tenant_session_reentry",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("hidden",))
    lru = evaluator(baseline_lru_blocks)
    tinylfu = evaluator(baseline_tinylfu_lru)
    lru_metrics = lru.workload_metrics["hidden/tenant_session_reentry"]
    tinylfu_metrics = tinylfu.workload_metrics["hidden/tenant_session_reentry"]

    assert tinylfu_metrics["token_hit_rate"] > lru_metrics["token_hit_rate"]
    assert tinylfu_metrics["cache_churn_per_1k"] < lru_metrics["cache_churn_per_1k"]


def test_hotset_cold_scan_displaces_lru_and_rewards_scan_resistance() -> None:
    requests = build_workload(
        "hotset_cold_scan",
        request_count=24,
        block_size_tokens=8,
        seed=3,
    )
    assert requests[16].prompt_tokens == requests[0].prompt_tokens
    assert {request.info.request_type for request in requests[8:16]} == {"cold_scan"}

    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        block_size_tokens=8,
        validation_families=("hotset_cold_scan",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    lru = evaluator(baseline_lru_blocks)
    tinylfu = evaluator(baseline_tinylfu_lru)
    lru_metrics = lru.workload_metrics["validation/hotset_cold_scan"]
    tinylfu_metrics = tinylfu.workload_metrics["validation/hotset_cold_scan"]

    assert lru_metrics["reuse_after_eviction_missed_blocks"] > 0
    assert tinylfu_metrics["cache_churn_per_1k"] < lru_metrics["cache_churn_per_1k"]


def test_concurrent_long_generation_exercises_pinned_capacity_pressure() -> None:
    requests = build_workload(
        "concurrent_long_generation",
        request_count=24,
        block_size_tokens=8,
        seed=3,
    )
    assert all(request.info.predicted_output_length is not None for request in requests)
    assert min(request.true_output_length for request in requests) > 400
    assert [request.arrival_step for request in requests[:6]] == [0, 0, 1, 1, 2, 2]

    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=6,
        validation_families=("concurrent_long_generation",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)
    metrics = result.workload_metrics["validation/concurrent_long_generation"]

    assert metrics["forced_bypass_count"] > 0
    assert metrics["arrival_span_steps"] == 24
    assert metrics["active_request_count_peak"] > 2


def test_reasoning_burst_exposes_noisy_long_outputs_and_microbursts() -> None:
    requests = build_workload(
        "reasoning_burst",
        request_count=24,
        block_size_tokens=16,
        seed=3,
    )

    assert all(request.info.predicted_output_length is not None for request in requests)
    assert max(request.true_output_length for request in requests) >= 1024
    assert any(
        request.info.predicted_output_length != request.true_output_length for request in requests
    )
    assert [request.arrival_step for request in requests[:6]] == [0, 0, 0, 1, 1, 1]


def test_shared_kv_capacity_models_decode_pressure_without_changing_default() -> None:
    base = EvaluatorConfig(
        request_count=24,
        seeds=(3,),
        capacity_blocks=24,
        block_size_tokens=16,
        validation_families=("reasoning_burst",),
        family_request_multipliers={},
    )
    prefix_only = PrefixKVCacheEvaluator(base, splits=("validation",))(baseline_lru_blocks)
    shared = PrefixKVCacheEvaluator(
        base.with_updates(kv_capacity_mode="shared"),
        splits=("validation",),
    )(baseline_lru_blocks)
    shared_no_cache = PrefixKVCacheEvaluator(
        base.with_updates(kv_capacity_mode="shared"),
        splits=("validation",),
    )(baseline_no_cache)
    prefix_metrics = prefix_only.workload_metrics["validation/reasoning_burst"]
    shared_metrics = shared.workload_metrics["validation/reasoning_burst"]
    shared_no_cache_metrics = shared_no_cache.workload_metrics["validation/reasoning_burst"]

    assert prefix_metrics["decode_kv_blocks_requested"] == 0
    assert prefix_metrics["decode_kv_occupancy_peak"] == 0
    assert prefix_metrics["memory_occupancy_mean"] == prefix_metrics["prefix_kv_occupancy_mean"]
    assert shared_metrics["decode_kv_blocks_requested"] > 0
    assert shared_metrics["decode_kv_occupancy_peak"] > 0
    assert shared_metrics["decode_pressure_eviction_count"] > 0
    assert shared_metrics["decode_kv_allocation_failure_blocks"] > 0
    assert shared_metrics["prefix_kv_occupancy_mean"] < prefix_metrics["prefix_kv_occupancy_mean"]
    assert shared_no_cache_metrics["decode_kv_occupancy_mean"] > 0
    assert shared_no_cache_metrics["policy_underfill_rate"] == 1.0


def test_unknown_kv_capacity_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="kv capacity mode"):
        PrefixKVCacheSimulator(
            capacity_blocks=4,
            block_size_tokens=4,
            prefill_cost_per_token=1.0,
            lookup_cost_per_block=0.0,
            eviction_cost_per_block=0.0,
            kv_capacity_mode="unknown",
        )


def test_structural_prefix_metrics_are_reported() -> None:
    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        validation_families=("agent_trace_branching",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_prefix_fanout)
    metrics = result.workload_metrics["validation/agent_trace_branching"]

    assert "depth_1_2_block_hit_rate" in metrics
    assert "depth_3_4_token_hit_rate" in metrics
    assert "depth_5_8_recompute_tokens_saved" in metrics
    assert "high_descendant_eviction_rate" in metrics
    assert "cold_deep_admission_rate" in metrics
    assert "reuse_after_eviction_missed_tokens" in metrics
    assert "system_prefix_hit_contribution" in metrics
    assert "developer_prefix_hit_contribution" in metrics
    assert "user_prefix_hit_contribution" in metrics
    assert "priority_weighted_token_hit_rate" in metrics
    assert "high_priority_p95_latency_proxy" in metrics
    assert "request_token_hit_rate_p10" in metrics
    assert "p95_recompute_cost" in metrics
    assert "policy_bypass_token_rate" in metrics
    assert "policy_underfill_rate" in metrics
    assert "forced_bypass_token_rate" in metrics
    assert "useful_admission_rate" in metrics
    assert "wasted_admission_rate" in metrics
    assert "useful_admission_token_rate" in metrics
    assert "wasted_admission_token_rate" in metrics
    assert "admission_saved_tokens_per_admission" in metrics
    assert "admission_token_utility" in metrics
    assert "avoidable_admission_regret_token_rate" in metrics
    assert "avoidable_rejection_regret_token_rate" in metrics
    assert "short_reuse_after_eviction_missed_token_rate" in metrics
    assert "avoidable_eviction_rate" in metrics
    assert "avoidable_short_reuse_eviction_rate" in metrics
    assert "value_weighted_avoidable_eviction_regret_token_rate" in metrics
    assert "worst_quarter_token_hit_rate" in metrics
    assert "final_quarter_token_hit_rate" in metrics
    assert "worst_recovery_phase_token_hit_rate" in metrics
    assert "tenant_token_hit_rate_p10" in metrics
    assert "prefix_kv_occupancy_mean" in metrics
    assert "decode_kv_occupancy_mean" in metrics
    assert "decode_kv_allocation_failure_rate" in metrics
    assert "decode_pressure_eviction_rate" in metrics
    assert "token_hit_rate_worst_trial" in metrics
    assert "token_hit_rate_stddev_across_trials" in metrics
    assert metrics["token_hit_rate"] != metrics["block_hit_rate"]
    assert metrics["depth_1_2_token_hit_rate"] > 0.0
    assert metrics["developer_prefix_hit_tokens"] > 0.0


def test_shared_system_prompt_reports_role_hit_contributions() -> None:
    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        block_size_tokens=8,
        train_families=("shared_system_prompt",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("train",))(baseline_lru_blocks)
    metrics = result.workload_metrics["train/shared_system_prompt"]

    assert result.invalid_fraction == 0.0
    assert metrics["token_hit_rate"] > 0.25
    assert metrics["system_prefix_hit_tokens"] > 0.0
    assert metrics["developer_prefix_hit_tokens"] > 0.0
    assert "user_prefix_hit_tokens" in metrics
    assert (
        metrics["system_prefix_hit_contribution"]
        + metrics["developer_prefix_hit_contribution"]
        + metrics["user_prefix_hit_contribution"]
    ) <= 1.0


def test_recompute_cost_varies_with_depth() -> None:
    class CapturePolicy(AdmitAllLRU):
        def __init__(self) -> None:
            self.costs: list[float] = []

        def score_admission(self, block, now: int) -> float:
            self.costs.append(block.estimated_recompute_cost)
            return 1.0

    policy = CapturePolicy()
    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        train_families=("shared_system_prompt",),
    )
    PrefixKVCacheEvaluator(config, splits=("train",))(lambda *_: policy)

    assert len(set(policy.costs)) > 1
    assert policy.costs == sorted(policy.costs)


def test_admission_stays_prefix_contiguous() -> None:
    class RejectRootAdmitChildren(AdmitAllLRU):
        def score_admission(self, block, now: int) -> float:
            return -1.0 if block.depth == 1 else 1.0

    policy = RejectRootAdmitChildren()
    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        capacity_blocks=8,
        block_size_tokens=8,
        train_families=("shared_system_prompt",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("train",))(lambda *_: policy)
    metrics = result.workload_metrics["train/shared_system_prompt"]

    assert metrics["admission_count"] == 0
    assert metrics["memory_occupancy_peak"] == 0


def test_rejected_admission_still_observes_rest_of_missed_chain() -> None:
    class RejectRootCaptureMisses(AdmitAllLRU):
        def __init__(self) -> None:
            self.missed_depths: list[int] = []

        def score_admission(self, block, now: int) -> float:
            return -1.0 if block.depth == 1 else 1.0

        def on_cache_miss(self, block, request, now: int) -> None:
            self.missed_depths.append(block.depth)

    policy = RejectRootCaptureMisses()
    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        capacity_blocks=8,
        block_size_tokens=8,
        train_families=("shared_system_prompt",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("train",))(lambda *_: policy)
    metrics = result.workload_metrics["train/shared_system_prompt"]
    expected_tokens = build_workload(
        "shared_system_prompt",
        request_count=1,
        block_size_tokens=config.block_size_tokens,
        seed=1003,
    )[0].info.prompt_length

    assert policy.missed_depths == [1, 2, 3, 4]
    assert metrics["admission_count"] == 0
    assert metrics["recompute_tokens"] == expected_tokens


def test_future_reuse_metadata_is_live_after_current_request() -> None:
    class CaptureFutureReuse(AdmitAllLRU):
        def __init__(self) -> None:
            self.observed: list[tuple[int, int, float | None, float | None]] = []

        def on_cache_miss(self, block, request, now: int) -> None:
            self.observed.append(
                (
                    now,
                    block.depth,
                    block.estimated_future_reuse,
                    block.estimated_next_reuse_distance,
                )
            )

        def score_admission(self, block, now: int) -> float:
            return -1.0

    simulator = PrefixKVCacheSimulator(
        capacity_blocks=4,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
        expose_future_reuse=True,
    )
    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=8,
                priority=0,
                request_type="unit",
                prompt_tokens=tuple(range(8)),
            ),
            true_output_length=8,
        )
        for request_id in range(2)
    )
    policy = CaptureFutureReuse()

    simulator.run(policy, requests, split="train", workload="unit", seed=1)

    assert policy.observed[:2] == [(0, 1, 1.0, 1.0), (0, 2, 1.0, 1.0)]
    assert policy.observed[2:] == [(1, 1, 0.0, float("inf")), (1, 2, 0.0, float("inf"))]


def test_future_reuse_metadata_preserves_same_step_next_use() -> None:
    class CaptureFutureReuse(AdmitAllLRU):
        def __init__(self) -> None:
            self.observed: list[tuple[int, int, float | None, float | None]] = []

        def on_cache_miss(self, block, request, now: int) -> None:
            self.observed.append(
                (
                    now,
                    block.depth,
                    block.estimated_future_reuse,
                    block.estimated_next_reuse_distance,
                )
            )

        def score_admission(self, block, now: int) -> float:
            return -1.0

    simulator = PrefixKVCacheSimulator(
        capacity_blocks=4,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
        expose_future_reuse=True,
    )
    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=8,
                priority=0,
                request_type="unit",
                prompt_tokens=tuple(range(8)),
            ),
            true_output_length=8,
            arrival_step=0,
        )
        for request_id in range(2)
    )
    policy = CaptureFutureReuse()

    simulator.run(policy, requests, split="train", workload="unit", seed=1)

    assert policy.observed[:2] == [(0, 1, 1.0, 0.0), (0, 2, 1.0, 0.0)]
    assert policy.observed[2:] == [(0, 1, 0.0, float("inf")), (0, 2, 0.0, float("inf"))]


def test_write_baseline_plots_creates_svg_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        prefix_runner,
        "_evaluate_baselines",
        lambda *_args, **_kwargs: {
            "lru": _report_result(10.0),
            "tinylfu_lru": _report_result(12.0),
        },
    )

    paths = write_baseline_plots(tmp_path, quick=True)

    assert {path.name for path in paths} == {
        "baseline_combined_scores.svg",
        "validation_token_hit_heatmap.svg",
        "token_vs_block_hit.svg",
    }
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("<svg")
        assert "</svg>" in text


def test_agentic_surrogate_probe_tripwire_passes_within_threshold() -> None:
    tripwire = agentic_surrogate_probe_tripwire(
        {
            "train/agentic_tool_workflows": {"token_hit_rate": 0.48},
            "probe/agent_trace_branching": {"token_hit_rate": 0.38},
        },
        threshold=0.12,
    )

    assert tripwire["status"] == "pass"
    assert tripwire["flagged"] is False
    assert tripwire["surrogate_minus_probe"] == pytest.approx(0.10)
    assert tripwire["absolute_gap"] == pytest.approx(0.10)
    assert tripwire["selection_score_excludes_probe"] is True


def test_agentic_surrogate_probe_tripwire_flags_excessive_divergence() -> None:
    tripwire = agentic_surrogate_probe_tripwire(
        {
            "train/agentic_tool_workflows": {"token_hit_rate": 0.60},
            "probe/agent_trace_branching": {"token_hit_rate": 0.30},
        },
        threshold=0.12,
    )

    assert tripwire["status"] == "flagged"
    assert tripwire["flagged"] is True
    assert tripwire["flag_reason"] == "divergence_exceeds_threshold"
    assert tripwire["surrogate_minus_probe"] == pytest.approx(0.30)
    assert tripwire["absolute_gap"] == pytest.approx(0.30)


def test_agentic_surrogate_probe_tripwire_fails_closed_without_both_metrics() -> None:
    tripwire = agentic_surrogate_probe_tripwire(
        {"train/agentic_tool_workflows": {"token_hit_rate": 0.48}},
    )

    assert tripwire["status"] == "flagged"
    assert tripwire["flagged"] is True
    assert tripwire["flag_reason"] == "missing_or_invalid_metric"
    assert tripwire["absolute_gap"] is None


def test_surrogate_probe_tripwire_suite_passes_all_configured_channels() -> None:
    suite = prefix_runner._surrogate_probe_tripwire_suite(
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
    suite = prefix_runner._surrogate_probe_tripwire_suite(
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
    suite = prefix_runner._surrogate_probe_tripwire_suite(
        {
            "train/agentic_tool_workflows": _agentic_gate_metrics(0.48),
            "probe/agent_trace_branching": _agentic_gate_metrics(0.38),
        }
    )

    assert suite["status"] == "flagged"
    assert suite["flagged_channels"] == ["cyclic_working_set"]
    assert suite["channels"]["cyclic_working_set"]["flag_reason"] == ("missing_or_invalid_metric")


def test_surrogate_probe_tripwire_suite_accepts_per_family_thresholds() -> None:
    suite = prefix_runner._surrogate_probe_tripwire_suite(
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
    suite = prefix_runner._surrogate_probe_tripwire_suite(
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
    monkeypatch.setattr(
        prefix_runner,
        "_artifact_report_config",
        lambda: EvaluatorConfig(
            request_count=4,
            seeds=(3,),
            capacity_sweep_blocks=(8,),
        ),
    )
    monkeypatch.setattr(
        prefix_runner,
        "_evaluate_candidate_program",
        lambda *_args, **_kwargs: _report_result(12.0, capacities=(8,)),
    )
    monkeypatch.setattr(
        prefix_runner,
        "_evaluate_baselines",
        lambda *_args, **_kwargs: {
            "lru": _report_result(10.0, capacities=(8,)),
            "oracle_future_reuse": _report_result(16.0, capacities=(8,)),
        },
    )
    monkeypatch.setattr(
        prefix_runner,
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

    run_dir = save_run_artifacts(
        result,
        tmp_path,
        iterations=3,
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
    tmp_path, monkeypatch
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

    def fake_decomposition(_config, candidate_path):
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
            "hidden": {
                **score_identity(),
                "combined_score": -1.0,
            },
        }

    monkeypatch.setattr(
        prefix_runner,
        "_candidate_panel_decomposition",
        fake_decomposition,
    )

    prefix_runner._persist_best_generated_mutation(
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
    report = (tmp_path / "best_generated_mutation_decomposition.md").read_text(encoding="utf-8")
    assert "Best generated mutation" in report
    assert "Agent hit" in report


def test_specialist_promotion_adjudication_fails_over_complexity_limit(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "best_program.py").write_text("candidate\n", encoding="utf-8")

    def fake_decomposition(config, candidate_path):
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

    monkeypatch.setattr(prefix_runner, "_candidate_panel_decomposition", fake_decomposition)

    payload = prefix_runner._persist_specialist_promotion_adjudication(
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
    monkeypatch,
) -> None:
    (tmp_path / "best_program.py").write_text(
        "def score_eviction(block, now, frequency, priority):\n"
        "    return now - block.last_accessed_at - frequency - priority\n",
        encoding="utf-8",
    )
    captured_paths = []

    def fake_decomposition(config, candidate_path):
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

    monkeypatch.setattr(prefix_runner, "_candidate_panel_decomposition", fake_decomposition)

    payload = prefix_runner._persist_specialist_promotion_adjudication(
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


def test_compose_eviction_specialist_source_replaces_only_eviction_method() -> None:
    base_source = current_incumbent("discovery").source_path.read_text(encoding="utf-8")
    specialist_source = textwrap.dedent(
        """
        import statistics

        EVICTION_SCALE = 1.25

        def score_eviction(block, now, frequency, priority):
            age = statistics.fmean((0.0, now - block.last_accessed_at))
            return EVICTION_SCALE * age - frequency - priority
        """
    )

    composed = compose_eviction_specialist_source(specialist_source, base_source)

    assert "def score_admission(self, block, now):" in composed
    assert "frequency, priority = self._values(block.prefix_hash, now)" in composed
    assert "return EVICTION_SCALE * age - frequency - priority" in composed
    assert "def score_eviction(block, now, frequency, priority):" not in composed


@pytest.mark.parametrize(
    "source, expected_violation",
    [
        (
            """
            def score_eviction(block, now, frequency, priority):
                return 0.0

            def build_candidate(*args):
                return None
            """,
            "must expose score_eviction, not a policy factory",
        ),
        (
            """
            @staticmethod
            def score_eviction(block, now, frequency, priority):
                return 0.0
            """,
            "score_eviction must be undecorated so promotion composition preserves behavior",
        ),
        (
            """
            def score_eviction(block, now, frequency, priority=0):
                return 0.0
            """,
            "signature must be score_eviction(block, now, frequency, priority)",
        ),
    ],
)
def test_eviction_only_contract_rejects_uncomposable_score_surface(
    source: str,
    expected_violation: str,
) -> None:
    assert any(
        expected_violation in violation
        for violation in eviction_only_source_violations(textwrap.dedent(source))
    )


def _minimal_policy_source(admission_expr: str, eviction_expr: str) -> str:
    return textwrap.dedent(
        f"""
        class Policy:
            def on_request_start(self, request, now):
                pass

            def score_admission(self, block, now):
                return float({admission_expr})

            def score_eviction(self, block, now):
                return float({eviction_expr})

            def on_cache_hit(self, block, request, now):
                pass

            def on_cache_miss(self, block, request, now):
                pass

        def build_candidate(capacity_blocks, block_size_tokens, seed=None):
            return Policy()
        """
    )
