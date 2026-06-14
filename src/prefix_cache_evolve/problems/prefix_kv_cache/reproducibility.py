"""Reproducibility manifests for synthetic and trace-derived request streams."""

from __future__ import annotations

import hashlib
import inspect
import platform
from pathlib import Path
from typing import Any, Mapping

from prefix_cache_evolve.evaluators.fingerprints import (
    evaluation_context_sha256,
    panel_sha256,
    request_stream_fingerprint_record,
)
from prefix_cache_evolve.evaluators.fingerprints import (
    request_stream_sha256 as request_stream_sha256,
)
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluatorConfig,
    build_workload,
)
from prefix_cache_evolve.evaluators.verifier import require_single_score_identity


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
                request_stream_fingerprint_record(
                    requests,
                    split=workload.split,
                    family=workload.family,
                    base_seed=base_seed,
                    seed_offset=workload.seed_offset,
                    actual_seed=actual_seed,
                )
            )

    evaluation = {
        "verifier_version": config.verifier_version,
        "splits": list(splits),
        "capacity_blocks": list(config.effective_capacity_blocks()),
        "capacity_tokens": list(config.effective_capacity_tokens()),
        "physical_block_size_tokens": config.block_size_tokens,
        "workload_token_granularity": config.effective_workload_token_granularity(),
        "request_count": config.request_count,
        "base_seeds": list(config.seeds),
        "family_request_multipliers": dict(sorted(config.family_request_multipliers.items())),
        "stream_count": len(streams),
    }
    panel_sha = panel_sha256(
        evaluation={key: value for key, value in evaluation.items() if key != "verifier_version"},
        streams=streams,
    )
    context_sha = evaluation_context_sha256(
        verifier_version=config.verifier_version,
        evaluator_config=config.model_dump(mode="json"),
        panel_sha=panel_sha,
    )
    manifest_core = {
        "schema": "prefix-kv-cache-workload-manifest-v1",
        "verifier_version": config.verifier_version,
        "evaluation_context_sha256": context_sha,
        "panel_sha256": panel_sha,
        "determinism_scope": (
            "Deterministic for the same source, Python environment, evaluator config, "
            "and deterministic candidate policy."
        ),
        "generator": _generator_metadata(),
        "evaluation": {
            **evaluation,
            "evaluation_context_sha256": context_sha,
            "panel_sha256": panel_sha,
        },
        "streams": streams,
    }
    return manifest_core


def stable_workload_manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the environment-independent fields used for reproducibility checks."""
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("workload manifest is missing evaluation metadata")
    identity = require_single_score_identity(
        (manifest, evaluation),
        context="workload manifest",
    )
    return {
        "schema": manifest.get("schema"),
        "verifier_version": identity.verifier_version,
        "evaluation_context_sha256": identity.evaluation_context_sha256,
        "panel_sha256": identity.panel_sha256,
        "evaluation": evaluation,
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
