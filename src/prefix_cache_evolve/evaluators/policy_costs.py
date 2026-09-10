"""Opt-in policy cost measurements, kept outside deterministic selection scores."""

from __future__ import annotations

import ast
import gc
import inspect
import json
import math
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import CodeType, FunctionType, MethodDescriptorType, ModuleType
from typing import Any

from prefix_cache_evolve.evaluators.baselines import BASELINE_REGISTRY
from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.evaluators.contracts import PrefixBlockInfo, PrefixKVPolicy, RequestInfo
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluationResult,
    PrefixKVCacheEvaluator,
    PrefixKVCacheSimulator,
    TrialMetrics,
    WorkloadRequest,
)

PROFILE_CONTRACT = "prefix-kv-cache-policy-costs-v3"
CALLBACKS = (
    "on_request_start",
    "on_cache_hit",
    "on_cache_miss",
    "score_admission",
    "score_eviction",
)


def baseline_source(name: str) -> str:
    """Extract baseline classes, bases, and module helpers without factory wrappers.

    Both baselines and candidates use the existing uncredited implementation AST
    counter. Imports and thin entry wrappers are excluded; annotations and
    docstrings are retained. Shared library implementations are excluded for both.
    This diagnostic convention never changes historical baseline score charges.
    """
    factory = BASELINE_REGISTRY.factories()[name]
    path = inspect.getsourcefile(factory)
    if path is None:
        raise ValueError(f"source is unavailable for baseline {name}")
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    definitions = {
        node.name: node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    pending = [factory.__name__]
    visited: set[str] = set()
    while pending:
        key = pending.pop()
        if key in visited:
            continue
        visited.add(key)
        pending.extend(
            node.id
            for node in ast.walk(definitions[key])
            if isinstance(node, ast.Name) and node.id in definitions
        )
    visited.remove(factory.__name__)
    return "\n\n".join(
        ast.get_source_segment(source, node) or ""
        for key, node in definitions.items()
        if key in visited
    )


def _native_size(value: object) -> int:
    """Measure native object storage without calling a candidate's size hook.

    Bypass metaclass hooks too. Native base descriptors understand variable-size
    storage (e.g. list capacity); object.__sizeof__ is the final fallback.
    Interpreter GC/preheaders are intentionally excluded from this estimate.
    """
    for cls in type.__dict__["__mro__"].__get__(type(value)):
        descriptor = type.__dict__["__dict__"].__get__(cls).get("__sizeof__")
        if type(descriptor) is MethodDescriptorType:
            return descriptor(value)
    raise TypeError("object has no native size descriptor")


def retained_size(value: object) -> int:
    """Estimate the reachable instance-state graph, counting shared objects once.

    Follows native GC references, including container-subclass attributes, slots,
    callback owners and captured/default callback state. Does not execute Python
    inspection hooks. Excludes code, module globals, classes, GC/preheaders and
    unmaterialized instance-dictionary bookkeeping. This is an object-storage
    estimate, not RSS, allocator capacity, KV tensors or transient peaks.
    """
    seen: set[int] = set()
    pending = [value]
    total = 0
    while pending:
        item = pending.pop()
        if id(item) in seen or issubclass(type(item), (type, ModuleType, CodeType)):
            continue
        seen.add(id(item))
        if type(item) is FunctionType:
            # A function stored on an instance can own state without keeping it
            # in that instance's dictionary. Do not follow __globals__ or code.
            pending.extend(
                value
                for value in (
                    item.__closure__,
                    item.__defaults__,
                    item.__kwdefaults__,
                    item.__dict__,
                )
                if value is not None
            )
            continue
        total += _native_size(item)
        pending.extend(gc.get_referents(item))
    return total


class Distribution:
    """Bounded logarithmic histogram with exact count, total and maximum.

    Quantiles are upper bucket bounds (about 1% resolution), not retained samples.
    The profiler's own storage therefore does not grow with callback count.
    """

    def __init__(self) -> None:
        self.count = 0
        self.total = 0
        self.maximum = 0
        self.buckets: dict[int, int] = {}

    def add(self, value: int) -> None:
        """Record one nonnegative observation."""
        value = max(0, value)
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        bucket = math.ceil(math.log1p(value) / math.log(1.01))
        self.buckets[bucket] = self.buckets.get(bucket, 0) + 1

    def summary(self) -> dict[str, int | float]:
        """Return measured totals and approximate p50/p95/p99 upper bounds."""
        result: dict[str, int | float] = {
            "count": self.count,
            "total": self.total,
            "mean": self.total / self.count if self.count else 0.0,
            "max": self.maximum,
        }
        for percentile in (50, 95, 99):
            threshold = math.ceil(self.count * percentile / 100)
            cumulative = 0
            result[f"p{percentile}_upper"] = 0
            for bucket, count in sorted(self.buckets.items()):
                cumulative += count
                if cumulative >= threshold:
                    result[f"p{percentile}_upper"] = min(
                        self.maximum, math.ceil(math.expm1(bucket * math.log(1.01)))
                    )
                    break
        return result


class PolicyProfiler:
    """Wrap only policy calls; simulator and state-walk time stay outside timers."""

    on_request_start: Callable[[RequestInfo, int], None]
    on_cache_hit: Callable[[PrefixBlockInfo, RequestInfo, int], None]
    on_cache_miss: Callable[[PrefixBlockInfo, RequestInfo, int], None]
    score_admission: Callable[[PrefixBlockInfo, int], float]
    score_eviction: Callable[[PrefixBlockInfo, int], float]

    def __init__(self, policy: PrefixKVPolicy, simulator: PrefixKVCacheSimulator) -> None:
        self.policy = policy
        self.simulator = simulator
        self.wall = {name: Distribution() for name in CALLBACKS}
        self.cpu = {name: Distribution() for name in CALLBACKS}
        self.scan_wall = Distribution()
        self.scan_cpu = Distribution()
        self.scan_width = Distribution()
        self.pending_wall = 0
        self.pending_cpu = 0
        self.pending_width = 0
        self.sessions: set[tuple[int, int]] = set()
        self.state_samples: list[dict[str, int]] = []
        self.request_count = 0
        # Wrappers must not hide a missing/noncallable initial policy hook.
        simulator._validate_policy(policy)
        for name in CALLBACKS:
            setattr(self, name, self._wrap(name))

    def _wrap(self, name: str) -> Callable[..., Any]:
        def measured(*args: Any) -> Any:
            # The simulator resolves hooks on every call; a policy may replace
            # them during a request. Attribute lookup stays outside the timer.
            callback = getattr(self.policy, name)
            if name == "on_request_start":
                request = args[0]
                if request.session_id is not None:
                    self.sessions.add((request.tenant_id, request.session_id))
            cpu_start = time.thread_time_ns()
            wall_start = time.perf_counter_ns()
            try:
                return callback(*args)
            finally:
                wall = time.perf_counter_ns() - wall_start
                cpu = time.thread_time_ns() - cpu_start
                self.wall[name].add(wall)
                self.cpu[name].add(cpu)
                if name == "score_eviction":
                    self.pending_wall += wall
                    self.pending_cpu += cpu
                    self.pending_width += 1

        return measured

    def on_eviction_decision(self, snapshot: Any) -> None:
        """Record total policy-ranking time per scan, excluding simulator work."""
        if self.pending_width != len(snapshot.candidates):
            raise ValueError("eviction callback count disagrees with scan width")
        self.scan_wall.add(self.pending_wall)
        self.scan_cpu.add(self.pending_cpu)
        self.scan_width.add(self.pending_width)
        self.pending_wall = self.pending_cpu = self.pending_width = 0

    def on_request_complete(self, snapshot: Any) -> None:
        """Sample retained state at geometrically spaced completed requests."""
        self.request_count = snapshot.index + 1
        if self.request_count & (self.request_count - 1) == 0:
            self.sample_state()

    def sample_state(self) -> None:
        """Capture state against observed unique prefixes and known sessions."""
        if self.state_samples and self.state_samples[-1]["requests"] == self.request_count:
            return
        self.state_samples.append(
            {
                "requests": self.request_count,
                "unique_prefixes": len(self.simulator.blocks),
                "known_sessions": len(self.sessions),
                "resident_blocks": self.simulator.resident_count,
                "retained_bytes": retained_size(self.policy),
            }
        )

    def summary(self) -> dict[str, Any]:
        """Return distributions and a sampled state trajectory for one trial."""
        self.sample_state()
        return {
            "callbacks": {
                name: {"wall_ns": self.wall[name].summary(), "cpu_ns": self.cpu[name].summary()}
                for name in CALLBACKS
            },
            "eviction_scans": {
                "wall_ns": self.scan_wall.summary(),
                "cpu_ns": self.scan_cpu.summary(),
                "candidates": self.scan_width.summary(),
            },
            "state_samples": self.state_samples,
        }


def profile_policy(
    factory: Callable[..., PrefixKVPolicy],
    config: EvaluatorConfig,
    *,
    complexity: int,
    splits: tuple[str, ...],
) -> EvaluationResult:
    """Run an instrumented diagnostic replay without changing score formulas.

    The ordinary evaluator starts tracemalloc, which distorts callback timings.
    This simulator adapter uses its normal workload/aggregation path but bypasses
    tracemalloc for these explicitly nondeterministic diagnostic measurements.
    Container limits still bound the entire profiling process.
    """
    if config.candidate_policy_surface != "full" or config.fixed_admission_policy is not None:
        raise ValueError("policy profiling requires the full policy surface")
    profiles: list[dict[str, Any]] = []

    class ProfiledEvaluator(PrefixKVCacheEvaluator):
        def _run_trial(
            self,
            factory: Callable[..., PrefixKVPolicy],
            requests: tuple[WorkloadRequest, ...],
            **kwargs: Any,
        ) -> TrialMetrics:
            simulator = PrefixKVCacheSimulator(
                capacity_blocks=kwargs["capacity_blocks"],
                block_size_tokens=kwargs["block_size_tokens"],
                prefill_cost_per_token=config.prefill_cost_per_token,
                lookup_cost_per_block=config.lookup_cost_per_block,
                eviction_cost_per_block=config.eviction_cost_per_block,
                active_tokens_per_step=config.active_tokens_per_step,
                kv_capacity_mode=config.kv_capacity_mode,
                max_memory_bytes=config.max_memory_bytes,
            )
            policy = factory(
                kwargs["capacity_blocks"], kwargs["block_size_tokens"], config.policy_seed
            )
            profiler = PolicyProfiler(policy, simulator)
            simulator.observer = profiler
            simulator.eviction_decision_observer = profiler
            profiler.sample_state()
            trial = simulator.run(
                profiler,
                requests,
                split=kwargs["split"],
                workload=kwargs["workload"],
                seed=kwargs["seed"],
                scoring_fn_complexity=complexity,
            )
            profiles.append(
                {
                    **{
                        key: kwargs[key]
                        for key in (
                            "split",
                            "workload",
                            "seed",
                            "capacity_blocks",
                            "block_size_tokens",
                        )
                    },
                    **profiler.summary(),
                }
            )
            return trial

    result = ProfiledEvaluator(config, splits=splits)(factory, scoring_fn_complexity=complexity)
    result.candidate_metadata["policy_cost_profile"] = json.dumps(
        {
            "contract": PROFILE_CONTRACT,
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "trials": profiles,
        }
    )
    return result


def equivalent_behavior(reference: EvaluationResult, candidate: EvaluationResult) -> bool:
    """Require identical deterministic trial metrics apart from source complexity.

    This is the predeclared, zero-tolerance simplification criterion on supplied
    visible trials. It neither opens hidden data nor establishes generalization.
    """
    if (
        not reference.success
        or not candidate.success
        or not reference.trials
        or reference.invalid_fraction
        or candidate.invalid_fraction
        or any(trial.invalid for trial in (*reference.trials, *candidate.trials))
    ):
        return False
    if (reference.evaluation_context_sha256, reference.panel_sha256) != (
        candidate.evaluation_context_sha256,
        candidate.panel_sha256,
    ):
        return False

    def behavior(result: EvaluationResult) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in asdict(trial).items() if key != "scoring_fn_complexity"}
            for trial in result.trials
        ]

    return behavior(reference) == behavior(candidate)
