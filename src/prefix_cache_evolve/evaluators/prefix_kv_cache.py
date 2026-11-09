"""Compatibility exports for the original evaluator module.

New code imports simulation, evaluation, and contracts from their owning modules.
"""

from prefix_cache_evolve.evaluators.baselines import BASELINE_REGISTRY as BASELINE_REGISTRY
from prefix_cache_evolve.evaluators.baselines import BASELINES as BASELINES
from prefix_cache_evolve.evaluators.baselines import REPORTING_BASELINES as REPORTING_BASELINES
from prefix_cache_evolve.evaluators.baselines import (
    baseline_cost_aware_lru as baseline_cost_aware_lru,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_depth_prefer_shallow as baseline_depth_prefer_shallow,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_future_reuse_heuristic as baseline_future_reuse_heuristic,
)
from prefix_cache_evolve.evaluators.baselines import baseline_lfu_blocks as baseline_lfu_blocks
from prefix_cache_evolve.evaluators.baselines import baseline_lru_blocks as baseline_lru_blocks
from prefix_cache_evolve.evaluators.baselines import baseline_no_cache as baseline_no_cache
from prefix_cache_evolve.evaluators.baselines import (
    baseline_oracle_future_reuse as baseline_oracle_future_reuse,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_prefix_anchor as baseline_prefix_anchor,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_prefix_fanout as baseline_prefix_fanout,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_recompute_cost_greedy as baseline_recompute_cost_greedy,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_sglang_radix_attention as baseline_sglang_radix_attention,
)
from prefix_cache_evolve.evaluators.baselines import (
    baseline_tenant_fair_lru as baseline_tenant_fair_lru,
)
from prefix_cache_evolve.evaluators.baselines import baseline_tinylfu_lru as baseline_tinylfu_lru
from prefix_cache_evolve.evaluators.baselines import baseline_vllm_apc as baseline_vllm_apc
from prefix_cache_evolve.evaluators.complexity import scoring_fn_complexity as scoring_fn_complexity
from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig as EvaluatorConfig
from prefix_cache_evolve.evaluators.configuration import WorkloadConfig as WorkloadConfig
from prefix_cache_evolve.evaluators.contracts import PolicyFactory as PolicyFactory
from prefix_cache_evolve.evaluators.contracts import PrefixBlockInfo as PrefixBlockInfo
from prefix_cache_evolve.evaluators.contracts import PrefixKVPolicy as PrefixKVPolicy
from prefix_cache_evolve.evaluators.contracts import RequestInfo as RequestInfo
from prefix_cache_evolve.evaluators.evaluation import (
    PrefixKVCacheEvaluator as PrefixKVCacheEvaluator,
)
from prefix_cache_evolve.evaluators.fingerprints import (
    evaluation_context_sha256 as evaluation_context_sha256,
)
from prefix_cache_evolve.evaluators.fingerprints import panel_sha256 as panel_sha256
from prefix_cache_evolve.evaluators.fingerprints import (
    request_stream_fingerprint_record as request_stream_fingerprint_record,
)
from prefix_cache_evolve.evaluators.panels import PreparedWorkload as PreparedWorkload
from prefix_cache_evolve.evaluators.panels import prepare_workloads as prepare_workloads
from prefix_cache_evolve.evaluators.panels import (
    trace_geometry_overrides as trace_geometry_overrides,
)
from prefix_cache_evolve.evaluators.results import (
    AdmissionDecisionDiagnostic as AdmissionDecisionDiagnostic,
)
from prefix_cache_evolve.evaluators.results import EvaluationResult as EvaluationResult
from prefix_cache_evolve.evaluators.results import TrialMetrics as TrialMetrics
from prefix_cache_evolve.evaluators.simulator import InvalidCandidateError as InvalidCandidateError
from prefix_cache_evolve.evaluators.simulator import (
    PrefixKVCacheSimulator as PrefixKVCacheSimulator,
)
from prefix_cache_evolve.evaluators.simulator_support import _ActiveDecode as _ActiveDecode
from prefix_cache_evolve.evaluators.simulator_support import (
    _AdmissionAccounting as _AdmissionAccounting,
)
from prefix_cache_evolve.evaluators.simulator_support import _AdmissionAudit as _AdmissionAudit
from prefix_cache_evolve.evaluators.simulator_support import _AdmissionOutcome as _AdmissionOutcome
from prefix_cache_evolve.evaluators.simulator_support import _BlockState as _BlockState
from prefix_cache_evolve.evaluators.simulator_support import _CapacityOutcome as _CapacityOutcome
from prefix_cache_evolve.evaluators.simulator_support import _correlation as _correlation
from prefix_cache_evolve.evaluators.simulator_support import (
    _FutureReuseTracker as _FutureReuseTracker,
)
from prefix_cache_evolve.evaluators.simulator_support import (
    _request_arrival_steps as _request_arrival_steps,
)
from prefix_cache_evolve.evaluators.simulator_support import _window_mean as _window_mean
from prefix_cache_evolve.evaluators.telemetry import CacheBlockSnapshot as CacheBlockSnapshot
from prefix_cache_evolve.evaluators.telemetry import (
    EvictionCandidateSnapshot as EvictionCandidateSnapshot,
)
from prefix_cache_evolve.evaluators.telemetry import (
    EvictionDecisionObserver as EvictionDecisionObserver,
)
from prefix_cache_evolve.evaluators.telemetry import (
    EvictionDecisionSnapshot as EvictionDecisionSnapshot,
)
from prefix_cache_evolve.evaluators.telemetry import RequestSnapshot as RequestSnapshot
from prefix_cache_evolve.evaluators.telemetry import SimulatorObserver as SimulatorObserver
from prefix_cache_evolve.evaluators.trace_replay import (
    TRACE_REPLAY_CONTRACT as TRACE_REPLAY_CONTRACT,
)
from prefix_cache_evolve.evaluators.workloads import WorkloadRequest as WorkloadRequest
from prefix_cache_evolve.evaluators.workloads import build_workload as build_workload
