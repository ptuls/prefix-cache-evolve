"""Convert LMCache agentic sessions into metadata-only replay traces."""

from __future__ import annotations

import hashlib
import heapq
import hmac
import json
import math
import os
import struct
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

_SUPPORTED_ROLES = frozenset({"assistant", "system", "tool", "user"})


class _ReplayRecord(TypedDict):
    timestamp_ms: float
    tenant_hash: str
    session_hash: str
    request_type: str
    priority: int
    prompt_length: int
    output_length: int
    prefix_path: list[str]


@dataclass(frozen=True)
class _ConvertedRequest:
    record: _ReplayRecord
    pre_gap_ms: float
    source_ordinal: int


@dataclass(frozen=True)
class LmcacheAgenticConversionConfig:
    """Controls conversion and deterministic closed-loop replay scheduling."""

    block_size_tokens: int = 512
    arrival_bucket_ms: int = 100
    active_tokens_per_step: int = 64
    concurrency: int = 20
    session_limit: int | None = None
    tokenizer_name: str = "cl100k_base"
    skip_invalid: bool = False

    def validate(self) -> None:
        """Reject invalid conversion settings."""
        if self.block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive")
        if self.arrival_bucket_ms <= 0:
            raise ValueError("arrival_bucket_ms must be positive")
        if self.active_tokens_per_step <= 0:
            raise ValueError("active_tokens_per_step must be positive")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if self.session_limit is not None and self.session_limit <= 0:
            raise ValueError("session_limit must be positive")
        if not self.tokenizer_name:
            raise ValueError("tokenizer_name must not be empty")


def convert_lmcache_agentic_rows(
    rows: Iterable[Mapping[str, Any]],
    output_path: Path,
    manifest_path: Path,
    *,
    encode: Callable[[str], Sequence[int]],
    hash_key: bytes,
    source: Mapping[str, object],
    config: LmcacheAgenticConversionConfig = LmcacheAgenticConversionConfig(),
) -> dict[str, object]:
    """Write replay-safe JSONL from LMCache agentic rows.

    Raw messages are used only while producing opaque block hashes. Replay
    output contains no prompt text or original identifiers.
    """
    config.validate()
    if len(hash_key) < 32:
        raise ValueError("hash_key must contain at least 32 bytes of key material")
    if output_path.resolve() == manifest_path.resolve():
        raise ValueError("output_path and manifest_path must be different")
    if output_path.exists() or manifest_path.exists():
        raise ValueError("output and manifest paths must not already exist")

    sessions: dict[str, list[_ConvertedRequest]] = defaultdict(list)
    previous_inputs: dict[str, tuple[Mapping[str, Any], ...]] = {}
    session_models: dict[str, str] = {}
    rejected_sessions: set[str] = set()
    skipped_reasons: Counter[str] = Counter()
    discarded_requests = 0
    rows_scanned = 0
    source_ordinal = 0

    for row_number, row in enumerate(rows, start=1):
        rows_scanned += 1
        session_id = None
        try:
            session_id = _required_string(row, "session_id")
            if session_id in rejected_sessions:
                discarded_requests += 1
                continue
            if (
                config.session_limit is not None
                and session_id not in sessions
                and len(sessions) >= config.session_limit
            ):
                continue
            model = _required_string(row, "model")
            previous_model = session_models.get(session_id, model)
            if previous_model != model:
                raise ValueError("a session must use exactly one model")
            messages = _messages(row.get("input"))
            previous = previous_inputs.get(session_id)
            if previous is not None and not (
                len(messages) > len(previous) and messages[: len(previous)] == previous
            ):
                raise ValueError("session input must be a strict cumulative message prefix")
            pre_gap_ms = _nonnegative_number(row.get("pre_gap"), "pre_gap") * 1_000.0
            if not math.isfinite(pre_gap_ms):
                raise ValueError("pre_gap milliseconds must be finite")
            if previous is None and pre_gap_ms != 0:
                raise ValueError("the first request in each session must have pre_gap 0")
            output_length = _nonnegative_integer(row.get("output_length"), "output_length")
            prompt_tokens = _validated_tokens(encode(_render_prompt(messages)))
            if not prompt_tokens:
                raise ValueError("tokenized prompt must not be empty")
            session_hash = _hmac_hex(hash_key, "session-v1", session_id.encode())
            # The source identifies sessions and models, not tenants. A shared
            # accounting placeholder must not fabricate per-session fairness.
            tenant_hash = "lmcache:unknown"
            sessions[session_id].append(
                _ConvertedRequest(
                    record={
                        "timestamp_ms": 0.0,
                        "tenant_hash": tenant_hash,
                        "session_hash": session_hash,
                        "request_type": _request_type(session_id),
                        "priority": 0,
                        "prompt_length": len(prompt_tokens),
                        "output_length": output_length,
                        "prefix_path": _prefix_path(
                            prompt_tokens,
                            block_size_tokens=config.block_size_tokens,
                            tokenizer_name=config.tokenizer_name,
                            model=model,
                            hash_key=hash_key,
                        ),
                    },
                    pre_gap_ms=pre_gap_ms,
                    source_ordinal=source_ordinal,
                )
            )
            previous_inputs[session_id] = messages
            session_models[session_id] = model
            source_ordinal += 1
        except ValueError as exc:
            # Without a session identity there is no safe way to identify and
            # discard the affected timeline, even in skip-invalid mode.
            if not config.skip_invalid or session_id is None:
                raise ValueError(f"LMCache row {row_number}: {exc}") from exc
            skipped_reasons[str(exc)] += 1
            discarded_requests += 1
            if session_id is not None:
                # pre_gap is relative to the immediately preceding response.
                # Splicing around an invalid row would invent timing/history.
                rejected_sessions.add(session_id)
                discarded_requests += len(sessions.get(session_id, ()))
                sessions.pop(session_id, None)
                previous_inputs.pop(session_id, None)
                session_models.pop(session_id, None)

    if not sessions or not source_ordinal:
        raise ValueError("LMCache conversion produced no replay requests")
    for requests in sessions.values():
        if requests[0].pre_gap_ms != 0:
            raise ValueError("the first request in each session must have pre_gap 0")

    scheduled = _schedule_sessions(sessions, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_records(output_path, scheduled)
    manifest: dict[str, object] = {
        "schema": "prefix-kv-cache-lmcache-agentic-conversion-v2",
        "source": dict(source),
        "output_path": str(output_path),
        "output_sha256": _file_sha256(output_path),
        "rows_scanned": rows_scanned,
        "requests_written": len(scheduled),
        "sessions_written": len(sessions),
        "models": dict(sorted(Counter(session_models.values()).items())),
        "skipped_rows": discarded_requests,
        "invalid_rows": sum(skipped_reasons.values()),
        "discarded_sessions": len(rejected_sessions),
        "invalid_row_handling": (
            "discard the entire affected session; fail if its identity is unavailable"
        ),
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
        "tokenizer": {
            "name": config.tokenizer_name,
            "prompt_serialization": "prefix-cache-lmcache-agentic-v1",
        },
        "block_size_tokens": config.block_size_tokens,
        "scheduling": {
            "algorithm": "deterministic-closed-loop-lanes-v2",
            "concurrency": config.concurrency,
            "arrival_bucket_ms": config.arrival_bucket_ms,
            "active_tokens_per_step": config.active_tokens_per_step,
            "intra_session_pre_gap_source": "dataset",
            "decode_duration": (
                "max(1, ceil(previous_output_length / active_tokens_per_step)) buckets"
            ),
        },
        "session_limit": config.session_limit,
        "hash": {
            "algorithm": "HMAC-SHA256",
            "key_fingerprint_sha256": hashlib.sha256(hash_key).hexdigest(),
            "model_namespaced_prefixes": True,
        },
        "privacy": {"raw_content_written": False, "raw_identifiers_written": False},
        "limitations": [
            "The source contains collected agent trajectories, not production serving logs.",
            "Tenant identities are unavailable; one accounting pool does not measure "
            "tenant fairness.",
            "Models have disjoint prefix namespaces but share a modeled pool; "
            "this is not a single-model GPU cache.",
            "Cross-session start times are synthesized with deterministic closed-loop lanes.",
            "Tokenization and canonical prompt serialization approximate provider-specific "
            "chat templates.",
            "Dataset pre_gap excludes inference time; replay adds a deterministic "
            "decode-duration proxy.",
        ],
    }
    _write_json(manifest_path, manifest)
    return manifest


def _schedule_sessions(
    sessions: Mapping[str, Sequence[_ConvertedRequest]],
    config: LmcacheAgenticConversionConfig,
) -> list[_ReplayRecord]:
    lanes = [(0.0, lane) for lane in range(config.concurrency)]
    heapq.heapify(lanes)
    scheduled: list[tuple[float, int, _ReplayRecord]] = []
    for session_id in sorted(sessions, key=lambda value: hashlib.sha256(value.encode()).digest()):
        lane_start, lane = heapq.heappop(lanes)
        timestamp = lane_start
        requests = sessions[session_id]
        for index, converted in enumerate(requests):
            if index:
                previous_output = requests[index - 1].record["output_length"]
                decode_steps = max(1, math.ceil(previous_output / config.active_tokens_per_step))
                timestamp += decode_steps * config.arrival_bucket_ms + converted.pre_gap_ms
            if not math.isfinite(timestamp):
                raise ValueError("scheduled timestamps must be finite")
            record = cast(_ReplayRecord, dict(converted.record))
            record["timestamp_ms"] = timestamp
            scheduled.append((timestamp, converted.source_ordinal, record))
        final_steps = max(
            1, math.ceil(requests[-1].record["output_length"] / config.active_tokens_per_step)
        )
        heapq.heappush(lanes, (timestamp + final_steps * config.arrival_bucket_ms, lane))
    return [record for _, _, record in sorted(scheduled)]


def _messages(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("input must be a non-empty message list")
    messages = []
    for message in value:
        if not isinstance(message, Mapping):
            raise ValueError("input messages must be objects")
        role = _required_string(message, "role").lower()
        if role not in _SUPPORTED_ROLES:
            raise ValueError("unsupported message role")
        try:
            json.dumps(message, ensure_ascii=True, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("input messages must contain JSON values") from exc
        messages.append(dict(message))
    return tuple(messages)


def _render_prompt(messages: Sequence[Mapping[str, Any]]) -> str:
    pieces = []
    for message in messages:
        role = str(message["role"]).lower()
        payload = {
            key: value for key, value in message.items() if key != "role" and value is not None
        }
        pieces.append(f"<|{role}|>\n")
        pieces.append(json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
        pieces.append("\n")
    pieces.append("<|assistant|>\n")
    return "".join(pieces)


def _prefix_path(
    tokens: Sequence[int],
    *,
    block_size_tokens: int,
    tokenizer_name: str,
    model: str,
    hash_key: bytes,
) -> list[str]:
    prefix_path = []
    namespace = json.dumps([tokenizer_name, model], separators=(",", ":")).encode() + b"\0"
    for start in range(0, len(tokens), block_size_tokens):
        packed = bytearray()
        for token in tokens[start : start + block_size_tokens]:
            packed.extend(struct.pack(">Q", token))
        prefix_path.append(_hmac_hex(hash_key, "prefix-block-v1", namespace + bytes(packed)))
    return prefix_path


def _validated_tokens(tokens: Sequence[int]) -> tuple[int, ...]:
    validated = []
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < 2**64:
            raise ValueError("tokenizer must return unsigned 64-bit integer token IDs")
        validated.append(token)
    return tuple(validated)


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _request_type(session_id: str) -> str:
    source = session_id.split("__", maxsplit=1)[0].lower()
    if source not in {"gaia", "swebench", "wildclaw"}:
        source = "other"
    return f"lmcache_{source}"


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _nonnegative_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a nonnegative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be a nonnegative finite number")
    return result


def _hmac_hex(key: bytes, domain: str, payload: bytes) -> str:
    return hmac.new(key, domain.encode() + b"\0" + payload, hashlib.sha256).hexdigest()


def _write_records(path: Path, records: Sequence[_ReplayRecord]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            )
            handle.write("\n")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
