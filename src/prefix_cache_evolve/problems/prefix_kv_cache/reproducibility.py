"""Reproducibility manifests for synthetic and trace-derived request streams."""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluatorConfig,
    WorkloadRequest,
    build_workload,
)


def build_workload_manifest(
    config: EvaluatorConfig,
    *,
    splits: tuple[str, ...] = ("train", "validation", "probe", "hidden"),
) -> dict[str, object]:
    """Describe and fingerprint every synthetic request stream in an evaluation panel."""
    streams = []
    for workload in config.workload_configs(splits):
        for base_seed in config.seeds:
            actual_seed = base_seed + workload.seed_offset
            requests = build_workload(
                workload.family,
                request_count=workload.request_count,
                block_size_tokens=config.effective_workload_token_granularity(),
                seed=actual_seed,
            )
            streams.append(
                {
                    "split": workload.split,
                    "family": workload.family,
                    "base_seed": base_seed,
                    "seed_offset": workload.seed_offset,
                    "actual_seed": actual_seed,
                    "request_count": len(requests),
                    "request_stream_sha256": request_stream_sha256(requests),
                    "total_prompt_tokens": sum(request.info.prompt_length for request in requests),
                    "total_output_tokens": sum(request.true_output_length for request in requests),
                    "unique_prompt_count": len(
                        {
                            request.prompt_tokens or request.info.prompt_tokens
                            for request in requests
                        }
                    ),
                    "arrival_span_steps": _arrival_span_steps(requests),
                    "request_type_counts": dict(
                        sorted(Counter(request.info.request_type for request in requests).items())
                    ),
                }
            )

    manifest_core = {
        "schema": "prefix-kv-cache-workload-manifest-v1",
        "determinism_scope": (
            "Deterministic for the same source, Python environment, evaluator config, "
            "and deterministic candidate policy."
        ),
        "generator": _generator_metadata(),
        "evaluation": {
            "splits": list(splits),
            "capacity_blocks": list(config.effective_capacity_blocks()),
            "capacity_tokens": list(config.effective_capacity_tokens()),
            "physical_block_size_tokens": config.block_size_tokens,
            "workload_token_granularity": config.effective_workload_token_granularity(),
            "request_count": config.request_count,
            "base_seeds": list(config.seeds),
            "family_request_multipliers": dict(sorted(config.family_request_multipliers.items())),
            "stream_count": len(streams),
        },
        "streams": streams,
    }
    return {
        **manifest_core,
        "panel_sha256": _canonical_sha256(
            {
                "evaluation": manifest_core["evaluation"],
                "streams": streams,
            }
        ),
    }


def request_stream_sha256(requests: Iterable[WorkloadRequest]) -> str:
    """Return a stable hash of all simulator-relevant fields in request order."""
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
        digest.update(_canonical_json(payload))
        digest.update(b"\n")
    return digest.hexdigest()


def stable_workload_manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the environment-independent fields used for reproducibility checks."""
    return {
        "schema": manifest.get("schema"),
        "panel_sha256": manifest.get("panel_sha256"),
        "evaluation": manifest.get("evaluation"),
        "streams": manifest.get("streams"),
    }


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hash of one input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generator_metadata() -> dict[str, str]:
    source_path_value = inspect.getsourcefile(build_workload)
    source_path = Path(source_path_value) if source_path_value else None
    metadata = {
        "module": build_workload.__module__,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }
    if source_path is not None and source_path.is_file():
        metadata["source_file"] = source_path.name
        metadata["source_sha256"] = file_sha256(source_path)
    return metadata


def _arrival_span_steps(requests: tuple[WorkloadRequest, ...]) -> int:
    if not requests:
        return 0
    arrivals = [
        index if request.arrival_step is None else request.arrival_step
        for index, request in enumerate(requests)
    ]
    return arrivals[-1] - arrivals[0] + 1


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
