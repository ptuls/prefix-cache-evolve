"""Candidate-visible contracts for prefix KV-cache policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class RequestInfo:
    """Candidate-visible request metadata."""

    request_id: int
    tenant_id: int
    session_id: int
    prompt_length: int
    priority: int
    request_type: str
    prompt_tokens: tuple[int, ...]
    predicted_output_length: int | None = None
    recent_admission_pressure: float = 0.0
    recent_miss_rate: float = 0.0


@dataclass(frozen=True)
class PrefixBlockInfo:
    """Candidate-visible prefix block metadata."""

    block_id: int
    prefix_hash: int
    parent_hash: int | None
    depth: int
    start_token: int
    end_token: int
    token_count: int
    tenant_id: int
    created_at: int
    last_accessed_at: int
    hit_count: int
    descendant_count: int
    active_ref_count: int
    estimated_recompute_cost: float
    prev_last_accessed_at: int | None = None
    last_access_gap: int | None = None
    access_gap_mean: float | None = None
    access_gap_var: float | None = None
    subtree_hit_rate: float = 0.0
    subtree_active_ref_count: int = 0
    estimated_future_reuse: float | None = None
    estimated_next_reuse_distance: float | None = None


class PrefixKVPolicy(Protocol):
    """Scoring-only policy interface used by the simulator."""

    def on_request_start(self, request: RequestInfo, now: int) -> None:
        """Observe the start of a request."""
        ...

    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        """Return the desirability of admitting a block."""
        ...

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        """Return the desirability of evicting a resident block."""
        ...

    def on_cache_hit(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        """Observe a cache hit."""
        ...

    def on_cache_miss(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        """Observe a cache miss."""
        ...


PolicyFactory = Callable[[int, int, int | None], PrefixKVPolicy]
