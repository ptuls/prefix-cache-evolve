"""Prepare fixed synthetic and trace workloads for evaluation and manifests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig, TraceWorkloadConfig
from prefix_cache_evolve.evaluators.fingerprints import request_stream_fingerprint_record
from prefix_cache_evolve.evaluators.workloads import WorkloadRequest, build_workload
from prefix_cache_evolve.problems.prefix_kv_cache.trace_replay import (
    TRACE_REPLAY_CONTRACT,
    load_anonymized_trace,
)


@dataclass(frozen=True)
class PreparedWorkload:
    """One ordered request stream shared by every policy and cache capacity."""

    split: str
    family: str
    base_seed: int
    seed_offset: int
    requests: tuple[WorkloadRequest, ...]
    block_size_tokens: int
    capacity_blocks: tuple[int, ...]
    trace: TraceWorkloadConfig | None = None

    @property
    def actual_seed(self) -> int:
        """Return the synthetic generator seed, or zero for a fixed trace."""
        return self.base_seed + self.seed_offset

    def fingerprint(self) -> dict[str, object]:
        """Describe the request stream and any pinned trace provenance."""
        record = request_stream_fingerprint_record(
            self.requests,
            split=self.split,
            family=self.family,
            base_seed=self.base_seed,
            seed_offset=self.seed_offset,
            actual_seed=self.actual_seed,
        )
        if self.trace is not None:
            record["source"] = {
                "kind": "anonymized_trace",
                "replay_contract": TRACE_REPLAY_CONTRACT,
                "sha256": self.trace.sha256,
                "block_size_tokens": self.trace.block_size_tokens,
                "arrival_bucket_ms": self.trace.arrival_bucket_ms,
            }
            if self.trace.capacity_sweep_blocks:
                record["source"]["capacity_sweep_blocks"] = list(self.capacity_blocks)
                record["source"]["capacity_sweep_tokens"] = [
                    capacity * self.block_size_tokens for capacity in self.capacity_blocks
                ]
            if self.trace.time_window is not None:
                record["source"]["time_window"] = self.trace.time_window.model_dump(mode="json")
        return record


def prepare_workloads(
    config: EvaluatorConfig,
    *,
    splits: tuple[str, ...],
    workload_builder: Callable[..., tuple[WorkloadRequest, ...]] = build_workload,
) -> tuple[PreparedWorkload, ...]:
    """Prepare requested splits without opening quarantined trace files.

    Synthetic seeds generate independent streams. A trace is already a fixed
    stream and is evaluated once per capacity, regardless of those seeds.
    """
    prepared = []
    for workload in config.workload_configs(splits):
        for seed in config.seeds:
            prepared.append(
                PreparedWorkload(
                    split=workload.split,
                    family=workload.family,
                    base_seed=seed,
                    seed_offset=workload.seed_offset,
                    requests=workload_builder(
                        workload.family,
                        request_count=workload.request_count,
                        block_size_tokens=config.effective_workload_token_granularity(),
                        seed=seed + workload.seed_offset,
                    ),
                    block_size_tokens=config.block_size_tokens,
                    capacity_blocks=config.effective_capacity_blocks(),
                )
            )

    session_splits: dict[int, set[tuple[str, str | None]]] = {}
    for trace in config.trace_workloads:
        if trace.split not in splits:
            continue
        requests = load_anonymized_trace(
            Path(trace.path),
            block_size_tokens=trace.block_size_tokens,
            arrival_bucket_ms=trace.arrival_bucket_ms,
            expected_sha256=trace.sha256,
            timestamp_range_ms=(
                (trace.time_window.start_ms, trace.time_window.end_ms)
                if trace.time_window is not None
                else None
            ),
        )
        if len(requests) != trace.request_count:
            raise ValueError(f"{trace.path}: request count does not match the pinned trace")
        origin = trace.time_window.source_sha256 if trace.time_window is not None else None
        for request in requests:
            if request.info.session_id is None:
                if trace.time_window is None:
                    raise ValueError(
                        "traces without session identities require chronological time windows"
                    )
                continue
            previous = session_splits.setdefault(request.info.session_id, set())
            # A declared chronological split permits recurring sessions across
            # separated windows in the same capture. It never claims group isolation.
            # Every origin must qualify, including multiple origins within a split.
            if any(
                previous_split != trace.split and (origin is None or previous_origin != origin)
                for previous_split, previous_origin in previous
            ):
                raise ValueError("trace sessions must not overlap across evaluation splits")
            previous.add((trace.split, origin))
        prepared.append(
            PreparedWorkload(
                split=trace.split,
                family=trace.family,
                base_seed=0,
                seed_offset=0,
                requests=requests,
                block_size_tokens=trace.block_size_tokens,
                capacity_blocks=trace.effective_capacity_blocks(config.effective_capacity_blocks()),
                trace=trace,
            )
        )
    return tuple(prepared)


def trace_geometry_overrides(
    config: EvaluatorConfig,
    *,
    splits: tuple[str, ...],
) -> list[dict[str, object]]:
    """Describe trace geometry that overrides the evaluator defaults."""
    overrides = []
    default_capacities = config.effective_capacity_blocks()
    for trace in config.trace_workloads:
        if trace.split not in splits or not trace.capacity_sweep_blocks:
            continue
        capacities = trace.effective_capacity_blocks(default_capacities)
        overrides.append(
            {
                "split": trace.split,
                "family": trace.family,
                "block_size_tokens": trace.block_size_tokens,
                "capacity_blocks": list(capacities),
                "capacity_tokens": [capacity * trace.block_size_tokens for capacity in capacities],
            }
        )
    return overrides
