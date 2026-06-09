"""Utility helpers for deterministic prefix KV-cache evaluation."""

from __future__ import annotations

import hashlib
import math
from statistics import median
from typing import Any

DEPTH_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("depth_1_2", 1, 2),
    ("depth_3_4", 3, 4),
    ("depth_5_8", 5, 8),
    ("depth_9_plus", 9, None),
)
PREFIX_ROLES = ("system", "developer", "user")
TOKEN_PREFIX_ROLES: dict[int, str] = {}


def percentile(values: list[float], percentile_value: int) -> float:
    """Return a nearest-rank percentile for finite metric lists."""
    if not values:
        return 0.0
    if percentile_value == 50:
        return float(median(values))
    values = sorted(values)
    index = math.ceil((percentile_value / 100.0) * len(values)) - 1
    return float(values[max(0, min(index, len(values) - 1))])


def window_token_hit_rates(
    request_hit_records: list[tuple[int, int]],
    *,
    window_count: int,
) -> list[float]:
    """Return token-weighted hit rates for contiguous request windows."""
    if not request_hit_records or window_count <= 0:
        return []
    effective_window_count = min(window_count, len(request_hit_records))
    window_hits = [0] * effective_window_count
    window_tokens = [0] * effective_window_count
    for index, (hit_tokens, total_tokens) in enumerate(request_hit_records):
        window = min(
            effective_window_count - 1,
            index * effective_window_count // len(request_hit_records),
        )
        window_hits[window] += hit_tokens
        window_tokens[window] += total_tokens
    return [
        hits / tokens if tokens else 0.0
        for hits, tokens in zip(window_hits, window_tokens, strict=True)
    ]


def jain_fairness(values: list[float]) -> float:
    """Return Jain's fairness index, treating all-zero service as equal."""
    if not values:
        return 1.0
    squared_sum = sum(value * value for value in values)
    if squared_sum == 0.0:
        return 1.0
    return sum(values) ** 2 / (len(values) * squared_sum)


def stable_hash(value: object) -> int:
    """Return a deterministic unsigned 64-bit hash for simulator identifiers."""
    digest = hashlib.blake2b(repr(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def request_prefix_hashes(
    request: Any,
    block_size_tokens: int,
) -> list[int]:
    """Return stable prefix hashes for a workload request."""
    prefix_hashes: list[int] = []
    prefix_tokens: list[int] = []
    tokens = request.prompt_tokens or request.info.prompt_tokens
    for start in range(0, len(tokens), block_size_tokens):
        chunk = tokens[start : start + block_size_tokens]
        prefix_tokens.extend(chunk)
        prefix_hashes.append(stable_hash((request.info.tenant_id, tuple(prefix_tokens))))
    return prefix_hashes


def depth_band(depth: int) -> str:
    """Return the reporting band for a prefix depth."""
    for name, low, high in DEPTH_BANDS:
        if depth >= low and (high is None or depth <= high):
            return name
    return DEPTH_BANDS[0][0]


def structural_metrics(
    *,
    depth_total_blocks: dict[str, int],
    depth_hit_blocks: dict[str, int],
    depth_total_tokens: dict[str, int],
    depth_hit_tokens: dict[str, int],
    prefix_role_hit_tokens: dict[str, int],
    total_hit_tokens: int,
    high_descendant_evictions: int,
    eviction_count: int,
    cold_deep_admission_opportunities: int,
    cold_deep_admissions: int,
    reuse_after_eviction_missed_blocks: int,
    reuse_after_eviction_missed_tokens: int,
    recompute_tokens: int,
) -> dict[str, float]:
    """Build structural diagnostic metrics from one simulator trial."""
    metrics: dict[str, float] = {}
    for band, _, _ in DEPTH_BANDS:
        total_blocks = depth_total_blocks.get(band, 0)
        total_tokens = depth_total_tokens.get(band, 0)
        hit_blocks = depth_hit_blocks.get(band, 0)
        hit_tokens = depth_hit_tokens.get(band, 0)
        metrics[f"{band}_block_hit_rate"] = hit_blocks / total_blocks if total_blocks else 0.0
        metrics[f"{band}_token_hit_rate"] = hit_tokens / total_tokens if total_tokens else 0.0
        metrics[f"{band}_recompute_tokens_saved"] = float(hit_tokens)

    metrics["high_descendant_eviction_count"] = float(high_descendant_evictions)
    metrics["high_descendant_eviction_rate"] = (
        high_descendant_evictions / eviction_count if eviction_count else 0.0
    )
    metrics["cold_deep_admission_opportunities"] = float(cold_deep_admission_opportunities)
    metrics["cold_deep_admission_count"] = float(cold_deep_admissions)
    metrics["cold_deep_admission_rate"] = (
        cold_deep_admissions / cold_deep_admission_opportunities
        if cold_deep_admission_opportunities
        else 0.0
    )
    metrics["reuse_after_eviction_missed_blocks"] = float(reuse_after_eviction_missed_blocks)
    metrics["reuse_after_eviction_missed_tokens"] = float(reuse_after_eviction_missed_tokens)
    metrics["reuse_after_eviction_missed_token_rate"] = (
        reuse_after_eviction_missed_tokens / recompute_tokens if recompute_tokens else 0.0
    )
    for role in PREFIX_ROLES:
        hit_tokens = prefix_role_hit_tokens.get(role, 0)
        metrics[f"{role}_prefix_hit_tokens"] = float(hit_tokens)
        metrics[f"{role}_prefix_hit_contribution"] = (
            hit_tokens / total_hit_tokens if total_hit_tokens else 0.0
        )
    return metrics


def prefix_role(tokens: tuple[int, ...]) -> str:
    """Resolve a generated prompt block's role from registered token metadata."""
    roles = {TOKEN_PREFIX_ROLES[token] for token in tokens if token in TOKEN_PREFIX_ROLES}
    if len(roles) == 1:
        return roles.pop()
    return "unknown"


def prefix_role_from_label(label: str) -> str:
    """Infer a synthetic prompt block role from its generator label."""
    if any(
        marker in label
        for marker in (
            "tail",
            "query",
            "tool",
            "retry",
            "turn",
            "scan",
            "unique",
        )
    ):
        return "user"
    if any(
        marker in label
        for marker in (
            "shared-system",
            "rag/template",
            "agent/root",
            "/root/",
        )
    ):
        return "system"
    if any(
        marker in label
        for marker in (
            "shared-task",
            "rag/chunk",
            "agent/branch",
            "schema",
            "/branch/",
            "doc/",
        )
    ):
        return "developer"
    return "unknown"
