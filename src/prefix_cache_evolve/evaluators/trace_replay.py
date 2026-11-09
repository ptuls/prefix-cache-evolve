"""Metadata-only production-trace calibration and replay support."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from prefix_cache_evolve.evaluators.contracts import RequestInfo
from prefix_cache_evolve.evaluators.workloads import WorkloadRequest

TRACE_REPLAY_CONTRACT = "prefix-kv-cache-trace-replay-v2"

_FORBIDDEN_CONTENT_KEYS = {
    "content",
    "input",
    "messages",
    "prompt",
    "prompt_text",
    "prompt_tokens",
    "raw_prompt",
    "text",
    "tokens",
}
_ALLOWED_FIELDS = {
    "output_length",
    "predicted_output_length",
    "prefix_hashes",
    "prefix_path",
    "priority",
    "prompt_length",
    "request_type",
    "session_hash",
    "session_id",
    "tenant_hash",
    "tenant_id",
    "timestamp_ms",
}


@dataclass(frozen=True)
class TraceRecord:
    """Validated opaque metadata for one request, before token expansion."""

    timestamp_ms: float
    tenant_key: str
    session_key: str | None
    request_type: str
    priority: int
    prompt_length: int
    output_length: int
    predicted_output_length: int | None
    prefix_path: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical metadata-only JSONL representation."""
        record = {
            "timestamp_ms": self.timestamp_ms,
            "tenant_hash": self.tenant_key,
            "session_hash": self.session_key,
            "request_type": self.request_type,
            "priority": self.priority,
            "prompt_length": self.prompt_length,
            "output_length": self.output_length,
            "prefix_path": list(self.prefix_path),
        }
        if self.predicted_output_length is not None:
            record["predicted_output_length"] = self.predicted_output_length
        return record


def load_anonymized_trace(
    path: Path,
    *,
    block_size_tokens: int,
    arrival_bucket_ms: int = 100,
    request_limit: int | None = None,
    expected_sha256: str | None = None,
    timestamp_range_ms: tuple[float, float] | None = None,
) -> tuple[WorkloadRequest, ...]:
    """Load metadata-only JSONL into simulator requests.

    Prefix path entries must be opaque identifiers for each prefix-cache block.
    They are deterministically expanded into private simulator tokens; raw prompt
    content is rejected and candidate-visible ``prompt_tokens`` remains empty.
    """
    if block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be positive")
    if arrival_bucket_ms <= 0:
        raise ValueError("arrival_bucket_ms must be positive")
    records = tuple(
        iter_trace_records(
            path,
            request_limit=request_limit,
            expected_sha256=expected_sha256,
            block_size_tokens=block_size_tokens,
            timestamp_range_ms=timestamp_range_ms,
        )
    )
    first_timestamp = records[0].timestamp_ms
    requests = []
    for request_id, record in enumerate(records):
        prompt_tokens = _prefix_path_tokens(
            record.prefix_path,
            prompt_length=record.prompt_length,
            block_size_tokens=block_size_tokens,
        )
        requests.append(
            WorkloadRequest(
                info=RequestInfo(
                    request_id=request_id,
                    tenant_id=_stable_int(record.tenant_key),
                    session_id=(
                        _stable_int(record.session_key) if record.session_key is not None else None
                    ),
                    prompt_length=record.prompt_length,
                    priority=record.priority,
                    request_type=record.request_type,
                    prompt_tokens=(),
                    predicted_output_length=record.predicted_output_length,
                ),
                true_output_length=record.output_length,
                prompt_tokens=prompt_tokens,
                arrival_step=int((record.timestamp_ms - first_timestamp) // arrival_bucket_ms),
            )
        )
    return tuple(requests)


def calibrate_anonymized_trace(
    path: Path,
    *,
    arrival_bucket_ms: int = 100,
    request_limit: int | None = None,
) -> dict[str, Any]:
    """Summarize trace-derived workload mix, depth, burst, and length targets."""
    if arrival_bucket_ms <= 0:
        raise ValueError("arrival_bucket_ms must be positive")
    records = tuple(iter_trace_records(path, request_limit=request_limit))
    request_types = Counter(record.request_type for record in records)
    timestamps = [record.timestamp_ms for record in records]
    arrival_gaps = [right - left for left, right in zip(timestamps, timestamps[1:], strict=False)]
    first_timestamp = timestamps[0]
    arrival_buckets = Counter(
        int((timestamp - first_timestamp) // arrival_bucket_ms) for timestamp in timestamps
    )
    predicted_lengths = [
        record.predicted_output_length
        for record in records
        if record.predicted_output_length is not None
    ]
    return {
        "schema": "prefix-kv-cache-trace-calibration-v1",
        "request_count": len(records),
        "tenant_count": len({record.tenant_key for record in records}),
        "session_count": len(
            {record.session_key for record in records if record.session_key is not None}
        ),
        "requests_without_session": sum(record.session_key is None for record in records),
        "arrival_bucket_ms": arrival_bucket_ms,
        "workload_mix": {
            request_type: {
                "count": count,
                "fraction": count / len(records),
            }
            for request_type, count in sorted(request_types.items())
        },
        "prefix_depth_blocks": _distribution([len(record.prefix_path) for record in records]),
        "prompt_length_tokens": _distribution([record.prompt_length for record in records]),
        "output_length_tokens": _distribution([record.output_length for record in records]),
        "predicted_output_length_tokens": _distribution(predicted_lengths),
        "arrival_gap_ms": _distribution(arrival_gaps),
        "arrival_bursts": {
            "same_bucket_request_fraction": (
                sum(count for count in arrival_buckets.values() if count > 1) / len(records)
            ),
            "max_requests_per_bucket": max(arrival_buckets.values()),
            "occupied_bucket_count": len(arrival_buckets),
        },
    }


def iter_trace_records(
    path: Path,
    *,
    request_limit: int | None = None,
    expected_sha256: str | None = None,
    block_size_tokens: int | None = None,
    timestamp_range_ms: tuple[float, float] | None = None,
) -> Iterator[TraceRecord]:
    """Yield validated records and verify the complete file when a hash is supplied.

    Callers must exhaust the iterator to complete checksum verification. Records
    outside a requested prefix are hashed but not parsed.
    """
    if request_limit is not None and request_limit <= 0:
        raise ValueError("request_limit must be positive")
    if block_size_tokens is not None and block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be positive")
    if timestamp_range_ms is not None:
        start_ms, end_ms = timestamp_range_ms
        if not (math.isfinite(start_ms) and math.isfinite(end_ms) and 0 <= start_ms < end_ms):
            raise ValueError("timestamp range must have finite 0 <= start_ms < end_ms")
    digest = hashlib.sha256()
    record_count = 0
    previous_timestamp = 0.0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if expected_sha256 is not None:
                digest.update(line)
            if request_limit is not None and record_count >= request_limit:
                if expected_sha256 is None:
                    break
                continue
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            record = _parse_record(payload, path=path, line_number=line_number)
            if timestamp_range_ms is not None and not (
                timestamp_range_ms[0] <= record.timestamp_ms < timestamp_range_ms[1]
            ):
                raise ValueError(f"{path}: record lies outside its declared time window")
            if record.timestamp_ms < previous_timestamp:
                raise ValueError(f"{path}: timestamp_ms values must be nondecreasing")
            if block_size_tokens is not None and len(record.prefix_path) != math.ceil(
                record.prompt_length / block_size_tokens
            ):
                raise ValueError(
                    "prefix_path depth must equal ceil(prompt_length / block_size_tokens)"
                )
            previous_timestamp = record.timestamp_ms
            record_count += 1
            yield record
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise ValueError(f"{path}: trace SHA-256 does not match the pinned content")
    if not record_count:
        raise ValueError(f"{path}: trace contains no requests")


def _parse_record(
    payload: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
) -> TraceRecord:
    forbidden = sorted(_find_forbidden_content_keys(payload))
    if forbidden:
        raise ValueError(
            f"{path}:{line_number}: raw-content fields are forbidden: " + ", ".join(forbidden)
        )
    unknown = sorted(set(payload) - _ALLOWED_FIELDS)
    if unknown:
        raise ValueError(f"{path}:{line_number}: unsupported trace fields: " + ", ".join(unknown))
    timestamp_ms = _number(payload, "timestamp_ms", path, line_number)
    if timestamp_ms < 0:
        raise ValueError(f"{path}:{line_number}: timestamp_ms must be nonnegative")
    tenant_key = _opaque_key(
        payload,
        ("tenant_hash", "tenant_id"),
        path=path,
        line_number=line_number,
    )
    assert tenant_key is not None
    session_key = _opaque_key(
        payload,
        ("session_hash", "session_id"),
        path=path,
        line_number=line_number,
        allow_null=True,
    )
    # Compatibility for the documented v1 Mooncake converter sentinel. Do not
    # infer sessions from arbitrary labels, request types, or request order.
    if session_key == "mooncake:unknown":
        session_key = None
    prefix_fields = [name for name in ("prefix_path", "prefix_hashes") if name in payload]
    if len(prefix_fields) != 1:
        raise ValueError(
            f"{path}:{line_number}: provide exactly one of prefix_path or prefix_hashes"
        )
    prefix_path = payload[prefix_fields[0]]
    if not isinstance(prefix_path, list) or not prefix_path:
        raise ValueError(f"{path}:{line_number}: prefix_path must be a non-empty list")
    opaque_path = tuple(_opaque_value(value, path, line_number) for value in prefix_path)
    prompt_length = _integer(payload, "prompt_length", path, line_number, minimum=1)
    output_length = _integer(payload, "output_length", path, line_number, minimum=0)
    predicted = payload.get("predicted_output_length")
    if predicted is not None:
        predicted = _integer(
            payload,
            "predicted_output_length",
            path,
            line_number,
            minimum=0,
        )
    request_type = payload.get("request_type", "trace_replay")
    if not isinstance(request_type, str) or not request_type:
        raise ValueError(f"{path}:{line_number}: request_type must be a string")
    return TraceRecord(
        timestamp_ms=timestamp_ms,
        tenant_key=tenant_key,
        session_key=session_key,
        request_type=request_type,
        priority=(
            _integer(payload, "priority", path, line_number, minimum=0)
            if "priority" in payload
            else 0
        ),
        prompt_length=prompt_length,
        output_length=output_length,
        predicted_output_length=predicted,
        prefix_path=opaque_path,
    )


def _find_forbidden_content_keys(value: Any) -> set[str]:
    forbidden = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _FORBIDDEN_CONTENT_KEYS:
                forbidden.add(str(key))
            forbidden.update(_find_forbidden_content_keys(child))
    elif isinstance(value, list):
        for child in value:
            forbidden.update(_find_forbidden_content_keys(child))
    return forbidden


def _prefix_path_tokens(
    prefix_path: tuple[str, ...],
    *,
    prompt_length: int,
    block_size_tokens: int,
) -> tuple[int, ...]:
    expected_depth = math.ceil(prompt_length / block_size_tokens)
    if len(prefix_path) != expected_depth:
        raise ValueError("prefix_path depth must equal ceil(prompt_length / block_size_tokens)")
    tokens: list[int] = []
    for depth in range(1, len(prefix_path) + 1):
        tokens.extend(_opaque_block_tokens(prefix_path[:depth], block_size_tokens))
    return tuple(tokens[:prompt_length])


def _opaque_block_tokens(path: tuple[str, ...], count: int) -> tuple[int, ...]:
    label = json.dumps(path, separators=(",", ":"), ensure_ascii=True).encode()
    return tuple(
        int.from_bytes(
            hashlib.sha256(label + b":" + str(index).encode()).digest()[:8],
            "big",
        )
        for index in range(count)
    )


def _stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def _opaque_key(
    payload: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    path: Path,
    line_number: int,
    allow_null: bool = False,
) -> str | None:
    present = [name for name in names if name in payload]
    if len(present) != 1:
        raise ValueError(f"{path}:{line_number}: provide exactly one of " + " or ".join(names))
    value = payload[present[0]]
    if value is None and allow_null:
        return None
    return _opaque_value(value, path, line_number)


def _opaque_value(value: Any, path: Path, line_number: int) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{path}:{line_number}: opaque identifiers must be strings or integers")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{path}:{line_number}: opaque identifiers must not be empty")
    return str(value)


def _number(
    payload: Mapping[str, Any],
    name: str,
    path: Path,
    line_number: int,
) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}:{line_number}: {name} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{path}:{line_number}: {name} must be finite")
    return float(value)


def _integer(
    payload: Mapping[str, Any],
    name: str,
    path: Path,
    line_number: int,
    *,
    minimum: int,
) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path}:{line_number}: {name} must be an integer >= {minimum}")
    return value


def _distribution(values: Iterable[int | float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "max": ordered[-1],
    }


def _percentile(values: list[float], percentile: int) -> float:
    index = math.ceil(percentile / 100.0 * len(values)) - 1
    return values[max(0, min(index, len(values) - 1))]
