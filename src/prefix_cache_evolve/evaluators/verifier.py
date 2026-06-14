"""Version contract for score-producing verifier outputs."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

VERIFIER_VERSION = "1.0.0"
VERIFIER_VERSION_PATTERN = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
_VERIFIER_VERSION_RE = re.compile(VERIFIER_VERSION_PATTERN)
_SHA256_RE = re.compile(SHA256_PATTERN)


@dataclass(frozen=True)
class ScoreIdentity:
    """Machine-readable identity for a comparable verifier score."""

    verifier_version: str
    evaluation_context_sha256: str
    panel_sha256: str


def validate_verifier_version(version: str) -> str:
    """Return a valid strict semantic verifier version."""
    if not _VERIFIER_VERSION_RE.fullmatch(version):
        raise ValueError("verifier_version must use strict MAJOR.MINOR.PATCH semantic versioning")
    return version


def verifier_version_of(value: Any) -> str:
    """Extract and validate a verifier version from a result or serialized record."""
    if isinstance(value, Mapping):
        version = value.get("verifier_version")
    else:
        version = getattr(value, "verifier_version", None)
    if not isinstance(version, str):
        raise ValueError("score record is missing verifier_version")
    return validate_verifier_version(version)


def evaluation_context_of(value: Any) -> str:
    """Extract a valid evaluation-context SHA-256 from a score record."""
    if isinstance(value, Mapping):
        context_sha = value.get("evaluation_context_sha256")
    else:
        context_sha = getattr(value, "evaluation_context_sha256", None)
    if not isinstance(context_sha, str) or not _SHA256_RE.fullmatch(context_sha):
        raise ValueError("score record is missing a valid evaluation_context_sha256")
    return context_sha


def panel_sha_of(value: Any) -> str:
    """Extract a valid panel SHA-256 from a score record."""
    if isinstance(value, Mapping):
        panel_sha = value.get("panel_sha256")
    else:
        panel_sha = getattr(value, "panel_sha256", None)
    if not isinstance(panel_sha, str) or not _SHA256_RE.fullmatch(panel_sha):
        raise ValueError("score record is missing a valid panel_sha256")
    return panel_sha


def require_single_verifier_version(
    values: Iterable[Any],
    *,
    context: str,
) -> str:
    """Return one verifier version after rejecting missing or mixed versions."""
    records = tuple(values)
    versions = {verifier_version_of(value) for value in records}
    if not versions:
        raise ValueError(f"{context} has no versioned score records")
    if len(versions) != 1:
        raise ValueError(
            f"{context} refuses mixed verifier versions: {', '.join(sorted(versions))}"
        )
    return next(iter(versions))


def require_single_score_identity(
    values: Iterable[Any],
    *,
    context: str,
) -> ScoreIdentity:
    """Return one identity after rejecting incomparable score records."""
    records = tuple(values)
    verifier_version = require_single_verifier_version(records, context=context)
    contexts = {evaluation_context_of(value) for value in records}
    if len(contexts) != 1:
        raise ValueError(
            f"{context} refuses mixed evaluation contexts: {', '.join(sorted(contexts))}"
        )
    panels = {panel_sha_of(value) for value in records}
    if len(panels) != 1:
        raise ValueError(f"{context} refuses mixed panels: {', '.join(sorted(panels))}")
    return ScoreIdentity(
        verifier_version=verifier_version,
        evaluation_context_sha256=next(iter(contexts)),
        panel_sha256=next(iter(panels)),
    )
