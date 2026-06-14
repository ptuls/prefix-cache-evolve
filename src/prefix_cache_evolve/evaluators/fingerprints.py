"""Canonical fingerprints for evaluator configurations and request panels."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def canonical_sha256(payload: object) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible payload."""
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def canonical_json(payload: object) -> bytes:
    """Return deterministic ASCII JSON bytes."""
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def request_stream_sha256(requests: Iterable[Any]) -> str:
    """Return a stable hash of all simulator-relevant request fields."""
    digest = hashlib.sha256()
    for request in requests:
        payload = {
            "request_id": request.info.request_id,
            "tenant_id": request.info.tenant_id,
            "session_id": request.info.session_id,
            "prompt_length": request.info.prompt_length,
            "priority": request.info.priority,
            "request_type": request.info.request_type,
            "predicted_output_length": request.info.predicted_output_length,
            "true_output_length": request.true_output_length,
            "arrival_step": request.arrival_step,
            "prompt_tokens": list(request.prompt_tokens or request.info.prompt_tokens),
        }
        digest.update(canonical_json(payload))
        digest.update(b"\n")
    return digest.hexdigest()


def request_stream_fingerprint_record(
    requests: Iterable[Any],
    *,
    split: str,
    family: str,
    base_seed: int,
    seed_offset: int,
    actual_seed: int,
) -> dict[str, Any]:
    """Return the canonical manifest entry for one ordered request stream."""
    request_tuple = tuple(requests)
    arrivals = [
        index if request.arrival_step is None else request.arrival_step
        for index, request in enumerate(request_tuple)
    ]
    return {
        "split": split,
        "family": family,
        "base_seed": base_seed,
        "seed_offset": seed_offset,
        "actual_seed": actual_seed,
        "request_count": len(request_tuple),
        "request_stream_sha256": request_stream_sha256(request_tuple),
        "total_prompt_tokens": sum(request.info.prompt_length for request in request_tuple),
        "total_output_tokens": sum(request.true_output_length for request in request_tuple),
        "unique_prompt_count": len(
            {request.prompt_tokens or request.info.prompt_tokens for request in request_tuple}
        ),
        "arrival_span_steps": arrivals[-1] - arrivals[0] + 1 if arrivals else 0,
        "request_type_counts": dict(
            sorted(Counter(request.info.request_type for request in request_tuple).items())
        ),
    }


def panel_sha256(
    *,
    evaluation: Mapping[str, Any],
    streams: Iterable[Mapping[str, Any]],
) -> str:
    """Return the stable identity of an evaluated request panel."""
    return canonical_sha256(
        {
            "evaluation": dict(evaluation),
            "streams": list(streams),
        }
    )


def evaluation_context_sha256(
    *,
    verifier_version: str,
    evaluator_config: Mapping[str, Any],
    panel_sha: str,
) -> str:
    """Return the identity of one verifier, config, and request-panel contract."""
    return canonical_sha256(
        {
            "schema": "prefix-kv-cache-evaluation-context-v1",
            "verifier_version": verifier_version,
            "evaluator_config": dict(evaluator_config),
            "panel_sha256": panel_sha,
        }
    )
