"""Shared builders for concise score-bearing test records."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from prefix_cache_evolve.evaluators.verifier import VERIFIER_VERSION

TEST_CONTEXT_SHA = "b" * 64
TEST_PANEL_SHA = "a" * 64


def score_identity(
    *,
    verifier_version: str = VERIFIER_VERSION,
    evaluation_context_sha256: str = TEST_CONTEXT_SHA,
    panel_sha256: str = TEST_PANEL_SHA,
) -> dict[str, str]:
    """Return a complete synthetic score identity."""
    return {
        "verifier_version": verifier_version,
        "evaluation_context_sha256": evaluation_context_sha256,
        "panel_sha256": panel_sha256,
    }


def score_record(
    combined_score: float,
    *,
    verifier_version: str = VERIFIER_VERSION,
    evaluation_context_sha256: str = TEST_CONTEXT_SHA,
    panel_sha256: str = TEST_PANEL_SHA,
    **fields: Any,
) -> SimpleNamespace:
    """Return a namespace representing one score-bearing result."""
    return SimpleNamespace(
        **score_identity(
            verifier_version=verifier_version,
            evaluation_context_sha256=evaluation_context_sha256,
            panel_sha256=panel_sha256,
        ),
        combined_score=combined_score,
        **fields,
    )
