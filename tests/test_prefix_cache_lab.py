"""Functional tests for the local Prefix Cache Lab."""

from __future__ import annotations

import pytest

from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluatorConfig,
    PrefixKVCacheSimulator,
    baseline_lru_blocks,
    build_workload,
)
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents.registry import current_incumbent
from prefix_cache_evolve.problems.prefix_kv_cache.lab import SimulationLab


def _payload(**overrides):
    payload = {
        "policies": ["candidate", "vllm_apc", "lru"],
        "workload": "agent_trace_branching",
        "request_count": 16,
        "capacity_blocks": 12,
        "block_size_tokens": 8,
        "seed": 11,
    }
    payload.update(overrides)
    return payload


def test_lab_catalog_exposes_candidates_baselines_and_public_workloads() -> None:
    catalog = SimulationLab().catalog()

    policies = {policy["id"]: policy for policy in catalog["policies"]}
    workloads = {workload["id"] for workload in catalog["workloads"]}
    assert catalog["source"]["id"] == "synthetic"
    assert all(workload["description"] for workload in catalog["workloads"])
    assert policies["candidate"]["group"] == "candidate"
    assert policies["candidate"]["label"] == "Priority-aware pressure incumbent"
    assert policies["candidate"]["status"] == "promoted incumbent"
    assert policies["candidate"]["promoted"] is True
    assert (
        policies["candidate"]["benchmark_selection_score"]
        == current_incumbent("production").benchmark["selection_combined_score"]
    )
    assert (
        policies["candidate"]["benchmark_evaluation_context_sha256"]
        == current_incumbent("production").benchmark["evaluation_context_sha256"]
    )
    assert policies["candidate"]["benchmark_context"] == "production · 16-token verifier"
    assert "bounded multi-timescale reuse state" in policies["candidate"]["description"]
    assert policies["vllm_apc"]["group"] == "deployable"
    assert policies["vllm_apc"]["status"] == "deployable"
    assert policies["vllm_apc"]["promoted"] is False
    assert policies["vllm_apc"]["benchmark_selection_score"] is None
    assert policies["vllm_apc"]["benchmark_context"] is None
    assert policies["sglang_radix_attention"]["group"] == "deployable"
    assert "sglang_radix_attention" not in catalog["defaults"]["policies"]
    assert catalog["defaults"]["capacity_blocks"] == 24
    assert catalog["defaults"]["block_size_tokens"] == 16
    assert catalog["defaults"]["workload_token_granularity"] == 8
    assert catalog["defaults"]["workload"] == "agentic_tool_workflows"
    assert policies["oracle_future_reuse"]["group"] == ("reporting-only/future-knowledge")
    assert "agentic_tool_workflows" in workloads
    assert "agent_trace_branching" in workloads
    assert "adversarial_unique_prompts" not in workloads


def test_lab_simulation_emits_aligned_policy_snapshots() -> None:
    result = SimulationLab().simulate(_payload())

    assert result["source"] == "synthetic"
    assert result["config"]["request_count"] == 16
    assert result["config"]["capacity_tokens"] == 96
    assert result["config"]["workload_token_granularity"] == 8
    assert [policy["id"] for policy in result["policies"]] == [
        "candidate",
        "vllm_apc",
        "lru",
    ]
    assert result["policies"][0]["promoted"] is True
    assert (
        result["policies"][0]["benchmark_selection_score"]
        == current_incumbent("production").benchmark["selection_combined_score"]
    )
    assert result["policies"][1]["promoted"] is False
    for policy in result["policies"]:
        events = policy["events"]
        assert len(events) == 16
        assert [event["index"] for event in events] == list(range(16))
        assert all(event["resident_blocks"] <= event["capacity_blocks"] for event in events)
        assert policy["summary"]["token_hit_rate"] == events[-1]["cumulative_token_hit_rate"]
        assert "policy_underfill_rate" in policy["summary"]
        assert policy["summary"]["invalid"] is False

    first_event = result["policies"][0]["events"][0]
    assert first_event["prompt_blocks"] > 0
    assert first_event["prompt_tokens"] > 0
    events = result["policies"][0]["events"]
    assert any(event["admissions"] > 0 for event in events)
    cached_event = next(event for event in events if event["cache"])
    assert {
        "block_id",
        "parent_id",
        "depth",
        "hit_this_request",
        "in_request",
        "is_leaf",
    } <= cached_event["cache"][0].keys()


def test_lab_can_run_reporting_only_future_knowledge_baseline() -> None:
    result = SimulationLab().simulate(_payload(policies=["oracle_future_reuse"], request_count=8))

    policy = result["policies"][0]
    assert policy["group"] == "reporting-only/future-knowledge"
    assert len(policy["events"]) == 8
    assert policy["summary"]["invalid"] is False


def test_lab_workload_selection_changes_generated_traffic() -> None:
    lab = SimulationLab()

    agent = lab.simulate(_payload(policies=["lru"], request_count=16))
    hotset = lab.simulate(
        _payload(
            policies=["lru"],
            workload="hotset_cold_scan",
            request_count=16,
        )
    )

    agent_types = {event["request_type"] for event in agent["policies"][0]["events"]}
    hotset_types = {event["request_type"] for event in hotset["policies"][0]["events"]}
    assert agent["config"]["workload"] == "agent_trace_branching"
    assert hotset["config"]["workload"] == "hotset_cold_scan"
    assert agent_types <= {"agent_loop", "agent_retry"}
    assert hotset_types == {"hotset", "cold_scan"}
    assert agent_types.isdisjoint(hotset_types)
    assert [event["cumulative_token_hit_rate"] for event in agent["policies"][0]["events"]] != [
        event["cumulative_token_hit_rate"] for event in hotset["policies"][0]["events"]
    ]


def test_request_telemetry_does_not_change_simulation_metrics() -> None:
    class Collector:
        def __init__(self):
            self.events = []

        def on_request_complete(self, snapshot):
            self.events.append(snapshot)

    requests = build_workload(
        "agent_trace_branching",
        request_count=12,
        block_size_tokens=8,
        seed=11,
    )
    config = EvaluatorConfig(capacity_blocks=12, block_size_tokens=8)

    def run(observer=None):
        simulator = PrefixKVCacheSimulator(
            capacity_blocks=config.capacity_blocks,
            block_size_tokens=config.block_size_tokens,
            prefill_cost_per_token=config.prefill_cost_per_token,
            lookup_cost_per_block=config.lookup_cost_per_block,
            eviction_cost_per_block=config.eviction_cost_per_block,
            observer=observer,
        )
        return simulator.run(
            baseline_lru_blocks(config.capacity_blocks, config.block_size_tokens, 11),
            requests,
            split="lab",
            workload="agent_trace_branching",
            seed=11,
        )

    collector = Collector()
    assert run().as_dict() == run(collector).as_dict()
    assert len(collector.events) == len(requests)


def test_eviction_telemetry_exposes_legal_victims_without_changing_metrics() -> None:
    class Collector:
        def __init__(self):
            self.events = []

        def on_eviction_decision(self, snapshot):
            self.events.append(snapshot)

    requests = build_workload(
        "hotset_cold_scan",
        request_count=16,
        block_size_tokens=8,
        seed=11,
    )

    def run(observer=None):
        simulator = PrefixKVCacheSimulator(
            capacity_blocks=4,
            block_size_tokens=8,
            prefill_cost_per_token=1.0,
            lookup_cost_per_block=0.0,
            eviction_cost_per_block=0.0,
            eviction_decision_observer=observer,
        )
        return simulator.run(
            baseline_lru_blocks(4, 8, 11),
            requests,
            split="lab",
            workload="hotset_cold_scan",
            seed=11,
        )

    collector = Collector()
    assert run().as_dict() == run(collector).as_dict()
    assert collector.events
    for event in collector.events:
        candidate_hashes = {candidate.block.prefix_hash for candidate in event.candidates}
        assert event.victim_prefix_hash in candidate_hashes
        assert all(candidate.next_reuse_distance is not None for candidate in event.candidates)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"policies": []}, "non-empty"),
        ({"policies": ["missing"]}, "unknown policies"),
        ({"policies": ["lru", "lru"]}, "duplicates"),
        ({"workload": "hidden"}, "unknown workload"),
        ({"request_count": 0}, "between 1 and 200"),
        ({"capacity_blocks": True}, "must be an integer"),
    ],
)
def test_lab_rejects_invalid_simulation_inputs(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        SimulationLab().simulate(_payload(**overrides))
