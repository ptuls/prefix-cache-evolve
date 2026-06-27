"""Synthetic workload generation for prefix KV-cache evaluation."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from prefix_cache_evolve.evaluators.contracts import RequestInfo
from prefix_cache_evolve.evaluators.utilities import (
    prefix_role_from_label as _prefix_role_from_label,
)
from prefix_cache_evolve.evaluators.utilities import (
    stable_hash as _stable_hash,
)


@dataclass(frozen=True)
class WorkloadRequest:
    """Simulator-internal request, including true output length."""

    info: RequestInfo
    true_output_length: int
    prompt_tokens: tuple[int, ...] = ()
    prompt_token_roles: tuple[str, ...] = ()
    arrival_step: int | None = None


@dataclass(frozen=True)
class _PromptBlock:
    """Generated prompt block with explicit role metadata."""

    tokens: tuple[int, ...]
    role: str


def build_workload(
    family: str,
    *,
    request_count: int,
    block_size_tokens: int,
    seed: int,
) -> tuple[WorkloadRequest, ...]:
    """Build one deterministic synthetic workload family."""
    rng = random.Random(seed)
    builder = {
        "shared_system_prompt": _shared_system_prompt,
        "rag_template_reuse": _rag_template_reuse,
        "agentic_tool_workflows": _agentic_tool_workflows,
        "agent_trace_branching": _agent_trace_branching,
        "multi_tenant_skew": _multi_tenant_skew,
        "phase_shift_prompts": _phase_shift_prompts,
        "long_context_mixed": _long_context_mixed,
        "session_continuation_growth": _session_continuation_growth,
        "hotset_cold_scan": _hotset_cold_scan,
        "cyclic_working_set_pressure": _cyclic_working_set_pressure,
        "cyclic_working_set_pressure_shifted": _cyclic_working_set_pressure_shifted,
        "concurrent_long_generation": _concurrent_long_generation,
        "reasoning_burst": _reasoning_burst,
        "reasoning_burst_shifted": _reasoning_burst_shifted,
        "stochastic_serving_mix": _stochastic_serving_mix,
        "stochastic_serving_mix_shifted": _stochastic_serving_mix_shifted,
        "rolling_template_versions": _rolling_template_versions,
        "rolling_template_versions_shifted": _rolling_template_versions_shifted,
        "heavy_tailed_prefix_lengths": _heavy_tailed_prefix_lengths,
        "heavy_tailed_prefix_lengths_shifted": _heavy_tailed_prefix_lengths_shifted,
        "priority_burst_recovery": _priority_burst_recovery,
        "priority_burst_recovery_shifted": _priority_burst_recovery_shifted,
        "priority_one_off_noise": _priority_one_off_noise,
        "priority_one_off_noise_shifted": _priority_one_off_noise_shifted,
        "tenant_phase_shift_cycles": _tenant_phase_shift_cycles,
        "tenant_phase_shift_cycles_shifted": _tenant_phase_shift_cycles_shifted,
        "adversarial_unique_prompts": _adversarial_unique_prompts,
        "cross_family_mixture": _cross_family_mixture,
        "tenant_session_reentry": _tenant_session_reentry,
    }.get(family)
    if builder is None:
        raise ValueError(f"unknown workload family {family!r}")
    return tuple(builder(request_count, block_size_tokens, rng))


def _block(
    label: str,
    block_size_tokens: int,
    token_count: int | None = None,
) -> _PromptBlock:
    count = block_size_tokens if token_count is None else max(1, token_count)
    base = _stable_hash(label) % 1_000_000
    tokens = tuple(base + index for index in range(count))
    return _PromptBlock(tokens=tokens, role=_prefix_role_from_label(label))


def _partial_tail(label: str, block_size_tokens: int) -> _PromptBlock:
    token_count = 1 + (_stable_hash(label) % max(block_size_tokens - 1, 1))
    return _block(label, block_size_tokens, token_count=token_count)


def _request(
    *,
    request_id: int,
    tenant_id: int,
    session_id: int,
    blocks: Sequence[_PromptBlock],
    request_type: str,
    priority: int = 0,
    true_output_length: int = 96,
    predicted_output_length: int | None = None,
    arrival_step: int | None = None,
) -> WorkloadRequest:
    tokens = tuple(token for block in blocks for token in block.tokens)
    token_roles = tuple(block.role for block in blocks for _ in block.tokens)
    return WorkloadRequest(
        info=RequestInfo(
            request_id=request_id,
            tenant_id=tenant_id,
            session_id=session_id,
            prompt_length=len(tokens),
            priority=priority,
            request_type=request_type,
            prompt_tokens=(),
            predicted_output_length=predicted_output_length,
        ),
        true_output_length=true_output_length,
        prompt_tokens=tokens,
        prompt_token_roles=token_roles,
        arrival_step=arrival_step,
    )


def _reindex_request(
    request: WorkloadRequest,
    *,
    request_id: int,
    request_type: str,
    arrival_step: int | None = None,
) -> WorkloadRequest:
    """Copies a workload request with a new position and descriptive type."""
    info = request.info
    return WorkloadRequest(
        info=RequestInfo(
            request_id=request_id,
            tenant_id=info.tenant_id,
            session_id=info.session_id,
            prompt_length=info.prompt_length,
            priority=info.priority,
            request_type=request_type,
            prompt_tokens=info.prompt_tokens,
            predicted_output_length=info.predicted_output_length,
        ),
        true_output_length=request.true_output_length,
        prompt_tokens=request.prompt_tokens,
        prompt_token_roles=request.prompt_token_roles,
        arrival_step=request.arrival_step if arrival_step is None else arrival_step,
    )


def _shared_system_prompt(count: int, block_size: int, rng: random.Random) -> list[WorkloadRequest]:
    system = [
        _block("shared-system/a", block_size),
        _block("shared-system/b", block_size),
    ]
    tasks = [_block(f"shared-task/{idx}", block_size) for idx in range(5)]
    requests = []
    for request_id in range(count):
        task = tasks[request_id % len(tasks)]
        tail = _partial_tail(f"shared-tail/{request_id % 11}", block_size)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id % 8,
                blocks=[*system, task, tail],
                request_type="chat",
                true_output_length=64 + rng.randrange(96),
            )
        )
    return requests


def _rag_template_reuse(count: int, block_size: int, rng: random.Random) -> list[WorkloadRequest]:
    template = [
        _block("rag/template/a", block_size),
        _block("rag/template/b", block_size),
    ]
    chunks = [_block(f"rag/chunk/{idx}", block_size) for idx in range(8)]
    requests = []
    for request_id in range(count):
        chunk = chunks[(request_id // 2 + request_id) % len(chunks)]
        suffix = _partial_tail(f"rag/query/{request_id % 17}", block_size)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id % 13,
                blocks=[*template, chunk, suffix],
                request_type="rag",
                true_output_length=48 + rng.randrange(80),
            )
        )
    return requests


def _long_context_mixed(count: int, block_size: int, rng: random.Random) -> list[WorkloadRequest]:
    docs = [[_block(f"doc/{doc}/block/{idx}", block_size) for idx in range(6)] for doc in range(4)]
    requests = []
    for request_id in range(count):
        doc = docs[(request_id // 3) % len(docs)]
        length = 3 + (request_id % 4)
        tail = _partial_tail(f"doc/tail/{request_id % 19}", block_size)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id % 9,
                blocks=[*doc[:length], tail],
                request_type="long_context",
                true_output_length=96 + rng.randrange(160),
            )
        )
    return requests


def _session_continuation_growth(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    system = [
        _block("session/shared-system/a", block_size),
        _block("session/shared-system/b", block_size),
    ]
    session_count = 4
    session_roots = {
        session_id: _block(f"session/{session_id}/root", block_size)
        for session_id in range(session_count)
    }
    histories: dict[int, list[_PromptBlock]] = {
        session_id: [] for session_id in range(session_count)
    }
    requests = []
    for request_id in range(count):
        session_id = request_id % session_count
        turn = _block(
            f"session/{session_id}/turn/{len(histories[session_id])}",
            block_size,
        )
        histories[session_id].append(turn)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=session_id,
                blocks=[*system, session_roots[session_id], *histories[session_id]],
                request_type="session_continuation",
                true_output_length=64 + rng.randrange(128),
            )
        )
    return requests


def _agentic_tool_workflows(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    """Build interleaved tool workflows with forks, resumptions, and replans."""
    root = [
        _block("agentic/root/system", block_size),
        _block("agentic/root/instructions", block_size),
    ]
    schemas = [
        _block("agentic/schema/actions", block_size),
        _block("agentic/schema/observations", block_size),
    ]
    workflow_count = 6
    workflow_roots = {
        workflow_id: _block(f"agentic/workflow/{workflow_id}/branch/root", block_size)
        for workflow_id in range(workflow_count)
    }
    action_blocks = [
        _block(f"agentic/tool/action/{action_id}", block_size) for action_id in range(10)
    ]
    shared_observations = [
        _block(f"agentic/tool/observation/shared/{observation_id}", block_size)
        for observation_id in range(12)
    ]
    routes: dict[tuple[int, int], list[_PromptBlock]] = {
        (workflow_id, route_id): []
        for workflow_id in range(workflow_count)
        for route_id in range(2)
    }
    active_routes = {workflow_id: 0 for workflow_id in range(workflow_count)}
    workflow_visits = {workflow_id: 0 for workflow_id in range(workflow_count)}
    workflow_schedule = (0, 1, 2, 0, 3, 1, 4, 0, 5, 2, 1, 3)
    requests = []
    for request_id in range(count):
        workflow_id = workflow_schedule[request_id % len(workflow_schedule)]
        visit_index = workflow_visits[workflow_id]
        workflow_visits[workflow_id] += 1
        route_id = active_routes[workflow_id]
        history = list(routes[(workflow_id, route_id)])
        other_route_id = 1 - route_id
        other_history = routes[(workflow_id, other_route_id)]
        request_type = "agentic_step"

        if visit_index % 11 == 5:
            route_id = other_route_id
            history = list(history)
            active_routes[workflow_id] = route_id
            request_type = "agentic_fork"
        elif visit_index % 13 == 9 and len(history) >= 6:
            rollback_blocks = min(len(history), 2 + visit_index % 3)
            history = history[:-rollback_blocks]
            request_type = "agentic_replan"
        elif visit_index % 7 == 6 and other_history:
            route_id = other_route_id
            history = list(other_history)
            active_routes[workflow_id] = route_id
            request_type = "agentic_resume"

        step_count = 1 + int(request_id % 4 == 0) + int(request_id % 15 == 0)
        for step_index in range(step_count):
            action_id = (request_id * 3 + workflow_id * 5 + route_id + step_index) % len(
                action_blocks
            )
            history.append(action_blocks[action_id])
            if (request_id + workflow_id + step_index) % 3 == 0:
                observation_id = (action_id + request_id // 3 + route_id) % len(shared_observations)
                history.append(shared_observations[observation_id])
            else:
                history.append(
                    _block(
                        "agentic/tool/observation/unique/"
                        f"{workflow_id}/{route_id}/{request_id}/{step_index}",
                        block_size,
                    )
                )
        routes[(workflow_id, route_id)] = history
        tail = _partial_tail(
            f"agentic/tail/{workflow_id}/{route_id}/{request_id}",
            block_size,
        )
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=workflow_id * 10 + route_id,
                blocks=[
                    *root,
                    *schemas,
                    workflow_roots[workflow_id],
                    *history,
                    tail,
                ],
                request_type=request_type,
                true_output_length=80 + rng.randrange(224),
            )
        )
    return requests


def _agent_trace_branching(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    root = [_block("agent/root/a", block_size), _block("agent/root/b", block_size)]
    schema = [_block(f"agent/schema/{index}", block_size) for index in range(3)]
    branches = [_block(f"agent/branch/{idx}", block_size) for idx in range(4)]
    tool_calls = [_block(f"agent/tool-call/{idx}", block_size) for idx in range(6)]
    tool_results = [_block(f"agent/tool-result/shared/{idx}", block_size) for idx in range(8)]
    histories: dict[int, list[_PromptBlock]] = {
        branch_index: [] for branch_index in range(len(branches))
    }
    requests = []
    for request_id in range(count):
        branch_idx = (request_id // 2 + request_id) % len(branches)
        history = list(histories[branch_idx])
        request_type = "agent_loop"
        if request_id % 11 == 10 and len(history) >= 4:
            history = history[:-2]
            request_type = "agent_retry"

        loop_count = 1 + int(request_id % 3 == 0)
        for loop_index in range(loop_count):
            tool_index = (request_id + branch_idx * 3 + loop_index) % len(tool_calls)
            history.append(tool_calls[tool_index])
            if (request_id + loop_index) % 4 == 0:
                result_index = (tool_index + request_id // 4) % len(tool_results)
                history.append(tool_results[result_index])
            else:
                history.append(
                    _block(
                        f"agent/tool-result/unique/{branch_idx}/{request_id}/{loop_index}",
                        block_size,
                    )
                )
        histories[branch_idx] = history
        tail = _partial_tail(f"agent/tail/{branch_idx}/{request_id}", block_size)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=branch_idx,
                blocks=[*root, *schema, branches[branch_idx], *history, tail],
                request_type=request_type,
                true_output_length=96 + rng.randrange(192),
            )
        )
    return requests


def _multi_tenant_skew(count: int, block_size: int, rng: random.Random) -> list[WorkloadRequest]:
    tenant_roots = {
        tenant: [_block(f"tenant/{tenant}/root/{idx}", block_size) for idx in range(2)]
        for tenant in range(3)
    }
    requests = []
    for request_id in range(count):
        tenant = 0 if request_id % 6 in {0, 1, 2, 3} else (1 if request_id % 6 == 4 else 2)
        branch = _block(f"tenant/{tenant}/branch/{request_id % 5}", block_size)
        tail = _partial_tail(f"tenant/{tenant}/tail/{request_id % 13}", block_size)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=tenant,
                session_id=tenant * 100 + request_id % 9,
                blocks=[*tenant_roots[tenant], branch, tail],
                request_type="tenant",
                true_output_length=64 + rng.randrange(128),
            )
        )
    return requests


def _phase_shift_prompts(count: int, block_size: int, rng: random.Random) -> list[WorkloadRequest]:
    phases = [
        [_block(f"phase/{phase}/root/{idx}", block_size) for idx in range(2)] for phase in range(2)
    ]
    requests = []
    for request_id in range(count):
        phase = 0 if request_id < count // 2 else 1
        branch = _block(f"phase/{phase}/branch/{request_id % 6}", block_size)
        tail = _partial_tail(f"phase/{phase}/tail/{request_id % 11}", block_size)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id % 10,
                blocks=[*phases[phase], branch, tail],
                request_type="phase_shift",
                true_output_length=64 + rng.randrange(128),
            )
        )
    return requests


def _hotset_cold_scan(count: int, block_size: int, rng: random.Random) -> list[WorkloadRequest]:
    hot_root = [
        _block("hotset/root/a", block_size),
        _block("hotset/root/b", block_size),
    ]
    hot_prompts = [
        [
            *hot_root,
            _block(f"hotset/branch/{index}", block_size),
            _partial_tail(f"hotset/tail/{index}", block_size),
        ]
        for index in range(4)
    ]
    warm_count = count // 3
    scan_end = 2 * warm_count
    requests = []
    for request_id in range(count):
        if warm_count <= request_id < scan_end:
            blocks = [_block(f"scan/{request_id}/block/{index}", block_size) for index in range(4)]
            blocks[-1] = _partial_tail(f"scan/{request_id}/tail", block_size)
            request_type = "cold_scan"
        else:
            hot_index = request_id % len(hot_prompts)
            blocks = hot_prompts[hot_index]
            request_type = "hotset"
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id % len(hot_prompts),
                blocks=blocks,
                request_type=request_type,
                true_output_length=32 + rng.randrange(64),
            )
        )
    return requests


def _cyclic_working_set_pressure(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _cyclic_working_set_pressure_workload(
        count,
        block_size,
        rng,
        shifted=False,
    )


def _cyclic_working_set_pressure_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _cyclic_working_set_pressure_workload(
        count,
        block_size,
        rng,
        shifted=True,
    )


def _cyclic_working_set_pressure_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    """Builds cyclic working sets slightly larger than common cache capacities."""
    root = [
        _block("cyclic/root/system", block_size),
        _block("cyclic/root/instructions", block_size),
    ]
    small_set_size = 12 if shifted else 9
    large_set_size = 22 if shifted else 17
    prompt_count = large_set_size
    prompts = [
        [
            *root,
            _block(f"cyclic/prompt/{index}/branch", block_size),
            _block(f"cyclic/prompt/{index}/context", block_size),
            _partial_tail(f"cyclic/prompt/{index}/tail", block_size),
        ]
        for index in range(prompt_count)
    ]
    requests = []
    for request_id in range(count):
        in_large_phase = request_id >= count // 2
        working_set_size = large_set_size if in_large_phase else small_set_size
        cycle_position = request_id if not shifted else request_id * 5
        prompt_index = cycle_position % working_set_size
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=prompt_index,
                blocks=prompts[prompt_index],
                request_type=(
                    "cyclic_working_set_large" if in_large_phase else "cyclic_working_set_small"
                ),
                true_output_length=48 + rng.randrange(96),
            )
        )
    return requests


def _concurrent_long_generation(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    root = [
        _block("concurrent/root/a", block_size),
        _block("concurrent/root/b", block_size),
    ]
    branches = [_block(f"concurrent/branch/{index}", block_size) for index in range(12)]
    requests = []
    for request_id in range(count):
        branch_index = request_id % len(branches)
        predicted_output_length = 512 + 64 * (request_id % 4)
        true_output_length = predicted_output_length + rng.randrange(-64, 65)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=branch_index,
                blocks=[
                    *root,
                    branches[branch_index],
                    _partial_tail(f"concurrent/tail/{request_id % 24}", block_size),
                ],
                request_type="long_generation",
                true_output_length=true_output_length,
                predicted_output_length=predicted_output_length,
                arrival_step=request_id // 2,
            )
        )
    return requests


def _reasoning_burst(count: int, block_size: int, rng: random.Random) -> list[WorkloadRequest]:
    return _reasoning_burst_workload(count, block_size, rng, shifted=False)


def _reasoning_burst_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _reasoning_burst_workload(count, block_size, rng, shifted=True)


def _reasoning_burst_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    """Builds microbursts with noisy, heavy-tailed reasoning outputs."""
    root = [
        _block("reasoning/system", block_size),
        _block("reasoning/developer", block_size),
        _block("reasoning/shared-task", block_size),
    ]
    branches = [_block(f"reasoning/branch/{index}", block_size) for index in range(10)]
    tools = [_block(f"reasoning/schema/{index}", block_size) for index in range(4)]
    long_lengths = (768, 1280, 2048, 3072) if shifted else (384, 640, 1024, 1536)
    burst_width = 4 if shifted else 3
    requests = []
    for request_id in range(count):
        is_reasoning = request_id % 4 != 3
        branch_index = (request_id * (3 if shifted else 1)) % len(branches)
        if is_reasoning:
            true_output_length = long_lengths[request_id % len(long_lengths)]
            prediction_scale = rng.choice((0.60, 0.75, 0.90, 1.10, 1.25))
            request_type = "reasoning_generation_shifted" if shifted else "reasoning_generation"
        else:
            true_output_length = 32 + 16 * (request_id % 4)
            prediction_scale = rng.choice((0.80, 1.00, 1.20))
            request_type = "reasoning_followup_shifted" if shifted else "reasoning_followup"
        predicted_output_length = max(1, round(true_output_length * prediction_scale))
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=request_id % 2,
                session_id=branch_index,
                blocks=[
                    *root,
                    branches[branch_index],
                    tools[branch_index % len(tools)],
                    _partial_tail(
                        f"reasoning/tail/{branch_index}/{request_id % 3}",
                        block_size,
                    ),
                ],
                request_type=request_type,
                priority=int(request_id % 7 == 0),
                true_output_length=true_output_length,
                predicted_output_length=predicted_output_length,
                arrival_step=request_id // burst_width,
            )
        )
    return requests


def _stochastic_serving_mix(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _stochastic_serving_mix_workload(count, block_size, rng, shifted=False)


def _stochastic_serving_mix_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _stochastic_serving_mix_workload(count, block_size, rng, shifted=True)


def _stochastic_serving_mix_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    source_requests = {
        "chat": _shared_system_prompt(count, block_size, rng),
        "rag": _rag_template_reuse(count, block_size, rng),
        "agent": _agentic_tool_workflows(count, block_size, rng),
        "long": _concurrent_long_generation(count, block_size, rng),
        "oneoff": _adversarial_unique_prompts(count, block_size, rng),
    }
    source_indices = {source_name: 0 for source_name in source_requests}
    regimes: tuple[tuple[tuple[str, float], ...], ...]
    if shifted:
        regimes = (
            (
                ("chat", 0.20),
                ("rag", 0.15),
                ("agent", 0.35),
                ("long", 0.20),
                ("oneoff", 0.10),
            ),
            (
                ("chat", 0.10),
                ("rag", 0.10),
                ("agent", 0.25),
                ("long", 0.35),
                ("oneoff", 0.20),
            ),
            (
                ("chat", 0.10),
                ("rag", 0.10),
                ("agent", 0.15),
                ("long", 0.20),
                ("oneoff", 0.45),
            ),
        )
        burst_probability = 0.70
        max_burst_length = 7
        arrival_gaps = (0, 1, 2, 5)
        arrival_gap_weights = (0.55, 0.30, 0.10, 0.05)
    else:
        regimes = (
            (
                ("chat", 0.45),
                ("rag", 0.25),
                ("agent", 0.15),
                ("long", 0.10),
                ("oneoff", 0.05),
            ),
            (
                ("chat", 0.15),
                ("rag", 0.20),
                ("agent", 0.35),
                ("long", 0.20),
                ("oneoff", 0.10),
            ),
            (
                ("chat", 0.25),
                ("rag", 0.15),
                ("agent", 0.10),
                ("long", 0.15),
                ("oneoff", 0.35),
            ),
        )
        burst_probability = 0.55
        max_burst_length = 5
        arrival_gaps = (0, 1, 2, 5)
        arrival_gap_weights = (0.35, 0.45, 0.15, 0.05)

    requests = []
    active_source = ""
    remaining_burst = 0
    arrival_step = 0
    for request_id in range(count):
        if request_id:
            arrival_step += rng.choices(arrival_gaps, weights=arrival_gap_weights, k=1)[0]
        regime_index = min(2, request_id * len(regimes) // max(1, count))
        regime = regimes[regime_index]
        choices = tuple(choice for choice, _ in regime)
        weights = tuple(weight for _, weight in regime)
        if remaining_burst <= 0:
            active_source = rng.choices(choices, weights=weights, k=1)[0]
            if rng.random() < burst_probability:
                remaining_burst = rng.randrange(2, max_burst_length + 1) - 1
        else:
            remaining_burst -= 1

        source_index = source_indices[active_source]
        source_request = source_requests[active_source][source_index % count]
        source_indices[active_source] += 1
        requests.append(
            _reindex_request(
                source_request,
                request_id=request_id,
                request_type=f"mix_{active_source}_{source_request.info.request_type}",
                arrival_step=arrival_step,
            )
        )
    return requests


def _rolling_template_versions(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _rolling_template_versions_workload(count, block_size, rng, shifted=False)


def _rolling_template_versions_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _rolling_template_versions_workload(count, block_size, rng, shifted=True)


def _rolling_template_versions_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    shared_root = _block("rolling/root/shared", block_size)
    templates = {
        0: [
            shared_root,
            _block("rolling/version/0/instructions", block_size),
            _block("rolling/version/0/schema", block_size),
        ],
        1: [
            shared_root,
            _block("rolling/version/1/instructions", block_size),
            _block("rolling/version/1/schema", block_size),
        ],
        2: [
            _block("rolling/root/revised", block_size),
            _block("rolling/version/2/instructions", block_size),
            _block("rolling/version/2/schema", block_size),
        ],
    }
    tasks = [_block(f"rolling/task/{index}", block_size) for index in range(6)]
    requests = []
    for request_id in range(count):
        phase = min(3, request_id * 4 // max(1, count))
        if shifted:
            if phase == 0:
                version = 0
            elif phase == 1:
                version = int(request_id % 3 == 0)
            elif phase == 2:
                version = 1
            else:
                version = 2 if request_id % 3 else 1
        elif phase == 0:
            version = 0
        elif phase == 1:
            version = int(request_id % 4 == 0)
        elif phase == 2:
            version = int(request_id % 4 != 0)
        else:
            version = int(request_id % 5 == 0)

        task_index = (request_id + version * 2) % len(tasks)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id % 12,
                blocks=[
                    *templates[version],
                    tasks[task_index],
                    _partial_tail(f"rolling/tail/{request_id}", block_size),
                ],
                request_type=f"rolling_template_v{version}",
                true_output_length=80 + rng.randrange(112),
            )
        )
    return requests


def _heavy_tailed_prefix_lengths(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _heavy_tailed_prefix_lengths_workload(count, block_size, rng, shifted=False)


def _heavy_tailed_prefix_lengths_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _heavy_tailed_prefix_lengths_workload(count, block_size, rng, shifted=True)


def _heavy_tailed_prefix_lengths_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    alpha = 1.25 if shifted else 1.55
    max_depth = 32 if shifted else 20
    doc_count = 6 if shifted else 4
    docs = [
        [_block(f"heavy/doc/{doc_index}/chunk/{depth}", block_size) for depth in range(max_depth)]
        for doc_index in range(doc_count)
    ]
    doc_weights = list(range(doc_count, 0, -1))
    root = [
        _block("heavy/root/system", block_size),
        _block("heavy/root/instructions", block_size),
    ]
    requests = []
    for request_id in range(count):
        doc_index = rng.choices(range(doc_count), weights=doc_weights, k=1)[0]
        body_depth = min(
            max_depth,
            max(2, int(1 + rng.paretovariate(alpha) * (3 if shifted else 2))),
        )
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=request_id % 3,
                session_id=request_id % 16,
                blocks=[
                    *root,
                    *docs[doc_index][:body_depth],
                    _partial_tail(f"heavy/tail/{request_id}", block_size),
                ],
                request_type="heavy_tailed_prefix",
                true_output_length=64 + rng.randrange(224),
            )
        )
    return requests


def _priority_burst_recovery(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _priority_burst_recovery_workload(count, block_size, rng, shifted=False)


def _priority_burst_recovery_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _priority_burst_recovery_workload(count, block_size, rng, shifted=True)


def _priority_burst_recovery_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    high_priority = 5 if shifted else 3
    medium_priority = 2 if shifted else 1
    hot_prompt_count = 6 if shifted else 4
    scan_depth = 8 if shifted else 6
    high_during_burst_interval = 7 if shifted else 5
    root = [
        _block("priority/root/system", block_size),
        _block("priority/root/instructions", block_size),
    ]
    hot_prompts = [
        [
            *root,
            _block(f"priority/hot/{index}/branch", block_size),
            _block(f"priority/hot/{index}/context", block_size),
            _partial_tail(f"priority/hot/{index}/tail", block_size),
        ]
        for index in range(hot_prompt_count)
    ]
    medium_prompts = [
        [
            *root,
            _block(f"priority/medium/{index}/branch", block_size),
            _partial_tail(f"priority/medium/{index}/tail", block_size),
        ]
        for index in range(3)
    ]
    warm_end = max(1, count // 4)
    burst_end = max(warm_end + 1, 3 * count // 4)
    requests = []
    arrival_step = 0
    for request_id in range(count):
        if request_id < warm_end:
            hot_index = request_id % len(hot_prompts)
            blocks = hot_prompts[hot_index]
            priority = high_priority
            request_type = "priority_hot_warm"
        elif request_id < burst_end:
            burst_index = request_id - warm_end
            if burst_index % high_during_burst_interval == 0:
                hot_index = (request_id + burst_index // 2) % len(hot_prompts)
                blocks = hot_prompts[hot_index]
                priority = high_priority
                request_type = "priority_hot_during_burst"
            else:
                blocks = [
                    _block(
                        f"priority/background/{request_id}/block/{depth}",
                        block_size,
                    )
                    for depth in range(scan_depth)
                ]
                blocks[-1] = _partial_tail(f"priority/background/{request_id}/tail", block_size)
                priority = 0
                request_type = "priority_background_scan"
        elif request_id % 5 == 0:
            medium_index = request_id % len(medium_prompts)
            blocks = medium_prompts[medium_index]
            priority = medium_priority
            request_type = "priority_medium_recovery"
        else:
            hot_index = request_id % len(hot_prompts)
            blocks = hot_prompts[hot_index]
            priority = high_priority
            request_type = "priority_hot_recovery"

        if request_id:
            arrival_step += (
                rng.choice((0, 0, 1))
                if warm_end <= request_id < burst_end
                else rng.choice((1, 1, 2))
            )
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=priority * 100 + request_id % max(1, hot_prompt_count),
                blocks=blocks,
                request_type=request_type,
                priority=priority,
                true_output_length=(
                    128 + rng.randrange(160) if priority > 0 else 24 + rng.randrange(48)
                ),
                arrival_step=arrival_step,
            )
        )
    return requests


def _priority_one_off_noise(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _priority_one_off_noise_workload(count, block_size, rng, shifted=False)


def _priority_one_off_noise_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _priority_one_off_noise_workload(count, block_size, rng, shifted=True)


def _priority_one_off_noise_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    """Mixes reusable normal traffic with high-priority one-off requests."""
    normal_root = [
        _block("priority-noise/normal/root/system", block_size),
        _block("priority-noise/normal/root/instructions", block_size),
    ]
    normal_prompt_count = 7 if shifted else 5
    normal_prompts = [
        [
            *normal_root,
            _block(f"priority-noise/normal/{index}/branch", block_size),
            _block(f"priority-noise/normal/{index}/context", block_size),
            _partial_tail(f"priority-noise/normal/{index}/tail", block_size),
        ]
        for index in range(normal_prompt_count)
    ]
    high_priority_root = _block("priority-noise/high/root/shared", block_size)
    sequence_length = 5
    high_priority_positions = {2, 3, 4} if shifted else {3, 4}
    high_priority = 6 if shifted else 4
    unique_depth = 8 if shifted else 6
    requests = []
    for request_id in range(count):
        if request_id % sequence_length in high_priority_positions:
            blocks = [
                high_priority_root,
                *[
                    _block(
                        f"priority-noise/high/{request_id}/unique/{depth}",
                        block_size,
                    )
                    for depth in range(unique_depth - 1)
                ],
            ]
            blocks[-1] = _partial_tail(
                f"priority-noise/high/{request_id}/tail",
                block_size,
            )
            priority = high_priority
            request_type = "priority_one_off_noise"
            session_id = 10_000 + request_id
        else:
            normal_index = (request_id // sequence_length + request_id) % len(normal_prompts)
            blocks = normal_prompts[normal_index]
            priority = 0
            request_type = "priority_normal_recurring"
            session_id = normal_index
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=session_id,
                blocks=blocks,
                request_type=request_type,
                priority=priority,
                true_output_length=(
                    160 + rng.randrange(192) if priority > 0 else 48 + rng.randrange(80)
                ),
            )
        )
    return requests


def _tenant_phase_shift_cycles(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _tenant_phase_shift_cycles_workload(count, block_size, rng, shifted=False)


def _tenant_phase_shift_cycles_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _tenant_phase_shift_cycles_workload(count, block_size, rng, shifted=True)


def _tenant_phase_shift_cycles_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    """Builds repeated tenant phase shifts with pollution and delayed recovery."""
    tenant_count = 5 if shifted else 4
    hot_prompt_count = 5 if shifted else 4
    pollution_depth = 10 if shifted else 7
    cycle_count = 8 if shifted else 6
    tenant_roots = {
        tenant: [
            _block(f"tenant-cycle/{tenant}/root/system", block_size),
            _block(f"tenant-cycle/{tenant}/root/instructions", block_size),
        ]
        for tenant in range(tenant_count)
    }
    hot_prompts = {
        tenant: [
            [
                *tenant_roots[tenant],
                _block(f"tenant-cycle/{tenant}/hot/{index}/branch", block_size),
                _block(f"tenant-cycle/{tenant}/hot/{index}/context", block_size),
                _partial_tail(f"tenant-cycle/{tenant}/hot/{index}/tail", block_size),
            ]
            for index in range(hot_prompt_count)
        ]
        for tenant in range(tenant_count)
    }
    cycle_length = max(12, math.ceil(count / cycle_count))
    warm_length = max(3, cycle_length // 4)
    pollution_end = max(warm_length + 3, 3 * cycle_length // 4)
    requests = []
    arrival_step = 0
    for request_id in range(count):
        cycle = min(cycle_count - 1, request_id // cycle_length)
        cycle_offset = request_id - cycle * cycle_length
        active_tenant = cycle % tenant_count
        if cycle_offset < warm_length:
            hot_index = (request_id + cycle) % hot_prompt_count
            tenant = active_tenant
            blocks = hot_prompts[tenant][hot_index]
            request_type = "tenant_cycle_warm"
            output_length = 96 + rng.randrange(160)
        elif cycle_offset < pollution_end:
            tenant = (active_tenant + 1 + cycle_offset) % tenant_count
            blocks = [
                *tenant_roots[tenant],
                *[
                    _block(
                        f"tenant-cycle/{cycle}/pollution/{request_id}/{depth}",
                        block_size,
                    )
                    for depth in range(pollution_depth - 2)
                ],
            ]
            blocks[-1] = _partial_tail(
                f"tenant-cycle/{cycle}/pollution/{request_id}/tail",
                block_size,
            )
            request_type = "tenant_cycle_pollution"
            output_length = 24 + rng.randrange(64)
        else:
            hot_index = (request_id + cycle) % hot_prompt_count
            tenant = active_tenant
            blocks = hot_prompts[tenant][hot_index]
            request_type = "tenant_cycle_recovery"
            output_length = 96 + rng.randrange(160)

        if request_id:
            arrival_step += (
                rng.choice((0, 0, 1))
                if request_type == "tenant_cycle_pollution"
                else rng.choice((1, 1, 2, 3))
            )
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=tenant,
                session_id=tenant * 100 + request_id % hot_prompt_count,
                blocks=blocks,
                request_type=request_type,
                true_output_length=output_length,
                arrival_step=arrival_step,
            )
        )
    return requests


def _adversarial_unique_prompts(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    requests = []
    for request_id in range(count):
        blocks = [
            _block(f"unique/{request_id}/block/{idx}/{rng.randrange(10_000)}", block_size)
            for idx in range(4)
        ]
        blocks[-1] = _partial_tail(
            f"unique/{request_id}/block/partial/{rng.randrange(10_000)}",
            block_size,
        )
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=request_id % 4,
                session_id=request_id,
                blocks=blocks,
                request_type="adversarial",
                true_output_length=32 + rng.randrange(64),
            )
        )
    return requests


def _tenant_session_reentry(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    tenant_roots = {
        tenant: [_block(f"reentry/tenant/{tenant}/root/{index}", block_size) for index in range(2)]
        for tenant in range(3)
    }
    session_contexts = {
        (tenant, session): [
            _block(
                f"reentry/tenant/{tenant}/session/{session}/context/{index}",
                block_size,
            )
            for index in range(3)
        ]
        for tenant in range(3)
        for session in range(4)
    }
    tenant_pattern = (0, 1, 0, 2, 0, 1, 2, 0)
    visits = {key: 0 for key in session_contexts}
    requests = []
    for request_id in range(count):
        tenant = tenant_pattern[request_id % len(tenant_pattern)]
        session = (request_id // len(tenant_pattern) + 3 * tenant) % 4
        visits[(tenant, session)] += 1
        stable_context = session_contexts[(tenant, session)]
        stable_depth = 2 if visits[(tenant, session)] == 1 else 3
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=tenant,
                session_id=tenant * 100 + session,
                blocks=[
                    *tenant_roots[tenant],
                    *stable_context[:stable_depth],
                    _partial_tail(
                        f"reentry/tail/{tenant}/{session}/{request_id}",
                        block_size,
                    ),
                ],
                request_type="tenant_session_reentry",
                true_output_length=48 + rng.randrange(96),
            )
        )
    return requests


def _cross_family_mixture(count: int, block_size: int, rng: random.Random) -> list[WorkloadRequest]:
    shared = _shared_system_prompt(count // 3, block_size, rng)
    phase = _phase_shift_prompts(count // 3, block_size, rng)
    unique = _adversarial_unique_prompts(count - len(shared) - len(phase), block_size, rng)
    requests = []
    for request_id, request in enumerate([*shared, *phase, *unique]):
        info = request.info
        requests.append(
            WorkloadRequest(
                info=RequestInfo(
                    request_id=request_id,
                    tenant_id=info.tenant_id,
                    session_id=info.session_id,
                    prompt_length=info.prompt_length,
                    priority=info.priority,
                    request_type=f"hidden_{info.request_type}",
                    prompt_tokens=info.prompt_tokens,
                    predicted_output_length=info.predicted_output_length,
                ),
                true_output_length=request.true_output_length,
                prompt_tokens=request.prompt_tokens,
                prompt_token_roles=request.prompt_token_roles,
                arrival_step=request.arrival_step,
            )
        )
    return requests
