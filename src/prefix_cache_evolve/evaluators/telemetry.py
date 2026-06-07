"""Observability contracts for prefix KV-cache simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CacheBlockSnapshot:
    """Immutable, transport-friendly view of one resident cache block."""

    block_id: str
    parent_id: str | None
    depth: int
    token_count: int
    prefix_role: str
    tenant_id: int
    hit_count: int
    active_ref_count: int
    descendant_count: int
    last_accessed_at: int
    is_leaf: bool
    in_request: bool
    hit_this_request: bool


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    """State and outcome emitted after one simulated request completes."""

    index: int
    now: int
    request_id: int
    tenant_id: int
    priority: int
    request_type: str
    prompt_blocks: int
    prompt_tokens: int
    matched_blocks: int
    hit_tokens: int
    admissions: int
    evictions: int
    bypassed_tokens: int
    resident_blocks: int
    capacity_blocks: int
    latency_proxy: float
    cumulative_token_hit_rate: float
    cumulative_evictions: int
    cache: tuple[CacheBlockSnapshot, ...]


class SimulatorObserver(Protocol):
    """Consumes request-complete snapshots without changing simulator behavior."""

    def on_request_complete(self, snapshot: RequestSnapshot) -> None:
        """Record or stream one completed request."""

        ...
