"""Standalone prefix KV-cache simulation and policy-evolution toolkit."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prefix_cache_evolve.problems.prefix_kv_cache.primitives import (
        MultiTimescaleDecay,
        decay_vector,
        threshold_excess,
    )

__all__ = ["MultiTimescaleDecay", "decay_vector", "threshold_excess"]


def __getattr__(name: str) -> Any:
    """Load historical primitive exports without coupling core imports to the app."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    primitives = import_module("prefix_cache_evolve.problems.prefix_kv_cache.primitives")
    value = getattr(primitives, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Keep lazy public exports discoverable."""
    return sorted(set(globals()) | set(__all__))
