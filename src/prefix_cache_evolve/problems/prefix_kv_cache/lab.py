"""Local web lab for visual prefix-cache policy comparisons."""

from __future__ import annotations

import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Mapping
from urllib.parse import urlsplit

import click

from prefix_cache_evolve.evaluators.baselines import (
    ALL_REPORTING_BASELINES,
    BASELINE_REGISTRY,
)
from prefix_cache_evolve.evaluators.contracts import PolicyFactory
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluatorConfig,
    PrefixKVCacheSimulator,
    TrialMetrics,
    WorkloadRequest,
    build_workload,
)
from prefix_cache_evolve.evaluators.telemetry import RequestSnapshot
from prefix_cache_evolve.problems.prefix_kv_cache.production_incumbent import (
    build_candidate,
)

_DEFAULT_POLICIES = (
    "candidate",
    "vllm_apc",
    "tinylfu_lru",
    "lru",
)
_INCUMBENT_ID = "candidate"
_INCUMBENT_BENCHMARK_SELECTION_SCORE = 65.649
_POLICY_DESCRIPTIONS = {
    "candidate": (
        "Promoted pressure-aware policy with bounded multi-timescale reuse state. "
        "It filters low-evidence admissions while preserving recurring and priority reuse."
    ),
    "no_cache": "Control policy that bypasses every cache admission.",
    "lru": "Admit all blocks and evict the least recently used legal leaf.",
    "sglang_radix_attention": (
        "SGLang RadixAttention's default admit-all, zero-reference leaf-LRU replacement policy."
    ),
    "vllm_apc": "vLLM automatic prefix caching model with full-block admission and LRU.",
    "lfu": "Admit all blocks and preserve the most frequently reused blocks.",
    "depth_prefer_shallow": "Protect shallow shared prefix blocks before deeper leaves.",
    "recompute_greedy": "Protect blocks with the highest estimated recompute cost.",
    "cost_aware_lru": "Balance recency against estimated recompute cost.",
    "prefix_fanout": "Protect prefix anchors with many known descendants.",
    "prefix_anchor": "Protect shallow, high-fanout anchors using an age penalty.",
    "tinylfu_lru": "Filter deeper one-off blocks before applying LRU eviction.",
    "tenant_fair_lru": "Bias LRU eviction toward tenants already receiving more hits.",
    "future_reuse_heuristic": "Reporting-only policy with future-reuse metadata.",
    "oracle_future_reuse": "Reporting-only constrained next-use oracle.",
}
_WORKLOAD_DESCRIPTIONS = {
    "shared_system_prompt": "Repeated requests share a stable system prefix.",
    "rag_template_reuse": "Retrieval prompts reuse templates and branch into documents.",
    "long_context_mixed": "Mixed short and long contexts stress depth and capacity.",
    "session_continuation_growth": "Sessions repeatedly extend their existing prompts.",
    "agentic_tool_workflows": (
        "Interleaved tool workflows fork, resume prior routes, and replan from shared checkpoints."
    ),
    "phase_shift_prompts": "The hot prompt population changes over time.",
    "multi_tenant_skew": "Tenants have uneven traffic volume and reuse.",
    "hotset_cold_scan": "A reusable hot set competes with a stream of one-off prompts.",
    "concurrent_long_generation": "Long generations pin blocks while new requests arrive.",
    "stochastic_serving_mix": "A mixed stochastic serving workload.",
    "rolling_template_versions": "Prompt templates evolve through rolling versions.",
    "heavy_tailed_prefix_lengths": "Prefix lengths follow a heavy-tailed distribution.",
    "priority_burst_recovery": "Priority bursts test cache recovery after disruption.",
    "priority_one_off_noise": "High-priority one-off prompts compete with reusable traffic.",
    "tenant_phase_shift_cycles": "Tenant hot sets shift in repeated cycles.",
    "agent_trace_branching": "Agent traces repeatedly branch from shared tool histories.",
    "cyclic_working_set_pressure": "Working sets cycle through a cache smaller than demand.",
}
_STATIC_CONTENT_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
}


def _display_name(name: str) -> str:
    """Convert an internal identifier to a concise UI label."""
    special_names = {
        "candidate": "Priority-aware pressure incumbent",
        "lru": "LRU",
        "lfu": "LFU",
        "tinylfu_lru": "TinyLFU + LRU",
        "vllm_apc": "vLLM APC",
        "sglang_radix_attention": "SGLang RadixAttention",
    }
    return special_names.get(name, name.replace("_", " ").title())


def _policy_metadata(name: str) -> dict[str, Any]:
    """Return stable UI metadata for a policy."""
    group = "candidate" if name == _INCUMBENT_ID else BASELINE_REGISTRY.group(name)
    promoted = name == _INCUMBENT_ID
    return {
        "id": name,
        "label": _display_name(name),
        "description": _POLICY_DESCRIPTIONS[name],
        "group": group,
        "status": "promoted incumbent" if promoted else group,
        "promoted": promoted,
        "benchmark_selection_score": (_INCUMBENT_BENCHMARK_SELECTION_SCORE if promoted else None),
        "benchmark_context": "production · 16-token verifier" if promoted else None,
    }


class _SnapshotCollector:
    """Collects simulator snapshots for JSON transport."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def on_request_complete(self, snapshot: RequestSnapshot) -> None:
        self.events.append(asdict(snapshot))


class SimulationLab:
    """Runs bounded, comparable policy simulations for the browser lab."""

    def __init__(self) -> None:
        config = EvaluatorConfig()
        self._workload_token_granularity = config.effective_workload_token_granularity()
        self._workloads = tuple(
            dict.fromkeys(
                config.train_families + config.validation_families + config.probe_families
            )
        )
        self._factories: dict[str, PolicyFactory] = {
            "candidate": build_candidate,
            **ALL_REPORTING_BASELINES,
        }

    def catalog(self) -> dict[str, Any]:
        """Return policies, workloads, and defaults supported by the lab."""
        return {
            "source": {
                "id": "synthetic",
                "label": "Synthetic traffic",
                "description": (
                    "Deterministic generated requests with fixed token streams, so "
                    "cache block-size changes remain directly comparable."
                ),
            },
            "policies": [
                {
                    **_policy_metadata(name),
                    "default_selected": name in _DEFAULT_POLICIES,
                }
                for name in self._factories
            ],
            "workloads": [
                {
                    "id": name,
                    "label": _display_name(name),
                    "description": _WORKLOAD_DESCRIPTIONS[name],
                }
                for name in self._workloads
            ],
            "defaults": {
                "policies": list(_DEFAULT_POLICIES),
                "workload": "agentic_tool_workflows",
                "request_count": 64,
                "capacity_blocks": 24,
                "block_size_tokens": 16,
                "workload_token_granularity": self._workload_token_granularity,
                "seed": 11,
            },
            "limits": {
                "max_policies": 6,
                "request_count": [1, 200],
                "capacity_blocks": [2, 256],
                "block_size_tokens": [2, 32],
                "seed": [0, 1_000_000],
            },
        }

    def simulate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Run selected policies over one identical generated workload."""
        policies = self._policies(payload.get("policies"))
        workload = str(payload.get("workload", ""))
        if workload not in self._workloads:
            raise ValueError(f"unknown workload {workload!r}")
        request_count = _bounded_int(payload, "request_count", 1, 200)
        capacity_blocks = _bounded_int(payload, "capacity_blocks", 2, 256)
        block_size_tokens = _bounded_int(payload, "block_size_tokens", 2, 32)
        seed = _bounded_int(payload, "seed", 0, 1_000_000)
        requests = build_workload(
            workload,
            request_count=request_count,
            block_size_tokens=self._workload_token_granularity,
            seed=seed,
        )
        config = EvaluatorConfig(
            capacity_blocks=capacity_blocks,
            block_size_tokens=block_size_tokens,
        )

        policy_results = [
            self._run_policy(
                name,
                requests=requests,
                workload=workload,
                seed=seed,
                config=config,
            )
            for name in policies
        ]
        return {
            "source": "synthetic",
            "config": {
                "workload": workload,
                "request_count": request_count,
                "capacity_blocks": capacity_blocks,
                "block_size_tokens": block_size_tokens,
                "capacity_tokens": capacity_blocks * block_size_tokens,
                "workload_token_granularity": self._workload_token_granularity,
                "seed": seed,
            },
            "policies": policy_results,
        }

    def _policies(self, value: Any) -> tuple[str, ...]:
        if not isinstance(value, list) or not value:
            raise ValueError("policies must be a non-empty list")
        if len(value) > 6:
            raise ValueError("at most 6 policies can be compared")
        policies = tuple(str(name) for name in value)
        unknown = [name for name in policies if name not in self._factories]
        if unknown:
            raise ValueError(f"unknown policies: {', '.join(unknown)}")
        if len(set(policies)) != len(policies):
            raise ValueError("policies must not contain duplicates")
        return policies

    def _run_policy(
        self,
        name: str,
        *,
        requests: tuple[WorkloadRequest, ...],
        workload: str,
        seed: int,
        config: EvaluatorConfig,
    ) -> dict[str, Any]:
        collector = _SnapshotCollector()
        policy = self._factories[name](
            config.capacity_blocks,
            config.block_size_tokens,
            seed,
        )
        simulator = PrefixKVCacheSimulator(
            capacity_blocks=config.capacity_blocks,
            block_size_tokens=config.block_size_tokens,
            prefill_cost_per_token=config.prefill_cost_per_token,
            lookup_cost_per_block=config.lookup_cost_per_block,
            eviction_cost_per_block=config.eviction_cost_per_block,
            active_tokens_per_step=config.active_tokens_per_step,
            expose_future_reuse=BASELINE_REGISTRY.requires_future_reuse(name),
            observer=collector,
        )
        metrics = simulator.run(
            policy,
            requests,
            split="lab",
            workload=workload,
            seed=seed,
        )
        return {
            **_policy_metadata(name),
            "summary": _summary(metrics),
            "events": collector.events,
        }


def _summary(metrics: TrialMetrics) -> dict[str, Any]:
    """Return the metrics most useful during interactive comparisons."""
    return {
        "token_hit_rate": metrics.token_hit_rate,
        "block_hit_rate": metrics.block_hit_rate,
        "p95_latency_proxy": metrics.p95_latency_proxy,
        "eviction_count": metrics.eviction_count,
        "admission_count": metrics.admission_count,
        "admission_rejection_count": metrics.admission_rejection_count,
        "cache_churn_per_1k": metrics.cache_churn_per_1k,
        "policy_underfill_rate": metrics.policy_underfill_rate,
        "memory_occupancy_mean": metrics.memory_occupancy_mean,
        "memory_occupancy_peak": metrics.memory_occupancy_peak,
        "tenant_jain_fairness": metrics.tenant_jain_fairness,
        "invalid": metrics.invalid,
        "invalid_reason": metrics.invalid_reason,
    }


def _bounded_int(
    payload: Mapping[str, Any],
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


class LabRequestHandler(BaseHTTPRequestHandler):
    """Serve the lab SPA and its small JSON API."""

    lab = SimulationLab()

    def do_GET(self) -> None:
        """Serve lab assets and read-only API endpoints."""
        path = urlsplit(self.path).path
        if path == "/api/catalog":
            self._send_json(self.lab.catalog())
            return
        if path == "/api/health":
            self._send_json({"status": "ok"})
            return
        filename = "index.html" if path == "/" else path.removeprefix("/")
        if filename not in _STATIC_CONTENT_TYPES:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        asset = resources.files(__package__).joinpath("lab_static", filename)
        self._send_bytes(asset.read_bytes(), _STATIC_CONTENT_TYPES[filename])

    def do_POST(self) -> None:
        """Serve the simulation API endpoint."""
        if urlsplit(self.path).path != "/api/simulate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > 1_000_000:
                raise ValueError("request body is too large")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            self._send_json(self.lab.simulate(payload))
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format_string: str, *args: Any) -> None:
        """Use concise server logs."""
        print(f"{self.address_string()} - {format_string % args}")

    def _send_json(
        self,
        value: Mapping[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(value, separators=(",", ":")).encode()
        self._send_bytes(body, "application/json; charset=utf-8", status=status)

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=click.IntRange(min=1, max=65535), default=8765, show_default=True)
def main(host: str, port: int) -> None:
    """Run the local Prefix Cache Lab server."""
    server = ThreadingHTTPServer((host, port), LabRequestHandler)
    click.echo(f"Prefix Cache Lab listening at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
