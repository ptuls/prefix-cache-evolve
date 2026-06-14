"""Immutable promoted prefix KV-cache incumbents."""

from __future__ import annotations

from typing import Any

from .registry import current_incumbent_factory
from .registry import incumbent_record as incumbent_record


def build_current_incumbent(
    capacity_blocks: int,
    block_size_tokens: int,
    seed: int | None = None,
) -> Any:
    """Build the current production incumbent."""
    return current_incumbent_factory("production")(capacity_blocks, block_size_tokens, seed)


def build_discovery_incumbent(
    capacity_blocks: int,
    block_size_tokens: int,
    seed: int | None = None,
) -> Any:
    """Build the retained discovery incumbent."""
    return current_incumbent_factory("discovery")(capacity_blocks, block_size_tokens, seed)
