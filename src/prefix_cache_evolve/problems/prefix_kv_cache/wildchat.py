"""Convert WildChat conversations into metadata-only prefix-cache traces."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import sqlite3
import struct
import tempfile
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

_SUPPORTED_ROLES = frozenset({"assistant", "developer", "system", "tool", "user"})


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
    request_hash: str
    parent_request_hash: str | None
    reply_hash: str
    timestamp_source: str


@dataclass(frozen=True)
class WildChatConversionConfig:
    """Controls conversion from WildChat rows to replay records."""

    block_size_tokens: int = 16
    turn_spacing_ms: int = 1_000
    conversation_limit: int | None = None
    minimum_requests_per_conversation: int = 1
    skip_invalid: bool = False
    tokenizer_name: str = "cl100k_base"
    timestamp_mode: str = "message"

    def validate(self) -> None:
        """Reject invalid conversion settings."""
        if self.block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive")
        if self.turn_spacing_ms < 0:
            raise ValueError("turn_spacing_ms must be nonnegative")
        if self.conversation_limit is not None and self.conversation_limit <= 0:
            raise ValueError("conversation_limit must be positive")
        if self.minimum_requests_per_conversation <= 0:
            raise ValueError("minimum_requests_per_conversation must be positive")
        if not self.tokenizer_name:
            raise ValueError("tokenizer_name must not be empty")
        if self.timestamp_mode not in {"message", "synthetic"}:
            raise ValueError("timestamp_mode must be message or synthetic")


def convert_wildchat_rows(
    rows: Iterable[Mapping[str, Any]],
    output_path: Path,
    manifest_path: Path,
    *,
    encode: Callable[[str], Sequence[int]],
    hash_key: bytes,
    source: Mapping[str, object],
    config: WildChatConversionConfig = WildChatConversionConfig(),
) -> dict[str, object]:
    """Write replay-safe JSONL from WildChat rows and return its manifest.

    Raw message content is used only in memory for tokenization. The output and
    temporary SQLite database contain HMAC identifiers, lengths, and timestamps.
    """
    config.validate()
    if len(hash_key) < 32:
        raise ValueError("hash_key must contain at least 32 bytes of key material")
    if output_path.resolve() == manifest_path.resolve():
        raise ValueError("output_path and manifest_path must be different")
    if output_path.exists() or manifest_path.exists():
        raise ValueError("output and manifest paths must not already exist")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_reasons: Counter[str] = Counter()
    rows_scanned = 0
    conversations_converted = 0
    duplicate_requests = 0
    ordinal = 0

    with tempfile.TemporaryDirectory(prefix="prefix-cache-wildchat-") as temp_dir:
        database_path = Path(temp_dir) / "requests.sqlite3"
        with sqlite3.connect(database_path) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE requests (request_hash TEXT PRIMARY KEY, "
                "session_hash TEXT, parent_hash TEXT, timestamp_ms REAL, "
                "timestamp_source TEXT, reply_hash TEXT, ordinal INTEGER, payload TEXT)"
            )
            connection.execute("CREATE TABLE inferred_sessions (session_hash TEXT PRIMARY KEY)")

            for row_number, row in enumerate(rows, start=1):
                if (
                    config.conversation_limit is not None
                    and conversations_converted >= config.conversation_limit
                ):
                    break
                rows_scanned += 1
                try:
                    records = _convert_wildchat_row(
                        row,
                        encode=encode,
                        hash_key=hash_key,
                        config=config,
                    )
                except ValueError as exc:
                    if not config.skip_invalid:
                        raise ValueError(f"WildChat row {row_number}: {exc}") from exc
                    skipped_reasons[str(exc)] += 1
                    continue
                if len(records) < config.minimum_requests_per_conversation:
                    skipped_reasons["too few assistant requests"] += 1
                    continue

                for index, converted in enumerate(records):
                    duplicate_requests += _store_request(
                        connection, converted, ordinal=ordinal + index, row_number=row_number
                    )
                ordinal += len(records)
                conversations_converted += 1

            requests_written = connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            if not requests_written:
                raise ValueError("WildChat conversion produced no replay requests")
            _reconcile_inferred_timestamps(connection)
            timestamp_sources = dict(
                connection.execute(
                    "SELECT timestamp_source, COUNT(*) FROM requests GROUP BY timestamp_source"
                )
            )
            connection.commit()
            _write_sorted_requests(connection, output_path)

    manifest: dict[str, object] = {
        "schema": "prefix-kv-cache-wildchat-conversion-v3",
        "source": dict(source),
        "output_path": str(output_path),
        "output_sha256": _file_sha256(output_path),
        "rows_scanned": rows_scanned,
        "conversations_converted": conversations_converted,
        "requests_written": requests_written,
        "duplicate_requests_skipped": duplicate_requests,
        "skipped_conversations": sum(skipped_reasons.values()),
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
        "tokenizer": {
            "name": config.tokenizer_name,
            "prompt_serialization": "prefix-cache-wildchat-chat-v2",
        },
        "block_size_tokens": config.block_size_tokens,
        "turn_spacing_ms": config.turn_spacing_ms,
        "timestamps": {
            "mode": config.timestamp_mode,
            "source_counts": dict(sorted(timestamp_sources.items())),
            "synthetic_anchor": "last conversation response",
            "missing_timestamp_bounds": "neighboring recorded responses",
            "overlap_resolution": "recorded times override inferred times; preserve turn order",
        },
        "session_identity": "first-turn-identifier-or-content-tenant-time-fallback-v2",
        "conversation_limit": config.conversation_limit,
        "minimum_requests_per_conversation": config.minimum_requests_per_conversation,
        "skip_invalid": config.skip_invalid,
        "hash": {
            "algorithm": "HMAC-SHA256",
            "key_fingerprint_sha256": hashlib.sha256(hash_key).hexdigest(),
            "model_namespaced_prefixes": True,
            "unknown_model_scope": "session",
        },
        "privacy": {
            "raw_content_written": False,
            "raw_identifiers_written": False,
        },
        "limitations": [
            "WildChat assistant timestamps mark response completion, not request arrival; "
            "replay uses these as timing proxies. Missing timestamps use synthetic spacing "
            "bounded by recorded responses and anchored at the last response when needed. "
            "Explicit synthetic mode ignores message timestamps.",
            "Overlapping snapshots keep the first inferred time per turn unless recorded "
            "timestamps or conversation order require an adjustment; inferred timing "
            "depends on the pinned source row order.",
            "Without turn identifiers, session identity falls back to the content hash, "
            "tenant, and last-response timestamp; duplicate detection is then approximate.",
            "The canonical prompt serialization is deterministic but does not reproduce "
            "the original provider's private system prompt or exact chat template.",
            "This is conversation-derived replay, not a production serving trace.",
            "Known models use separate prefix namespaces; unknown models share "
            "only within a session.",
        ],
    }
    _write_json(manifest_path, manifest)
    return manifest


def _convert_wildchat_row(
    row: Mapping[str, Any],
    *,
    encode: Callable[[str], Sequence[int]],
    hash_key: bytes,
    config: WildChatConversionConfig,
) -> list[_ConvertedRequest]:
    conversation_hash = _required_string(row, "conversation_hash")
    last_timestamp_ms = _timestamp_ms(row.get("timestamp"))
    messages = row.get("conversation")
    if not isinstance(messages, list) or not messages:
        raise ValueError("conversation must be a non-empty list")
    if any(not isinstance(message, Mapping) for message in messages):
        raise ValueError("conversation messages must be objects")
    roles = [_required_string(message, "role").lower() for message in messages]
    if any(role not in _SUPPORTED_ROLES for role in roles):
        raise ValueError("unsupported conversation role")

    first_turn_id = next(
        (
            identifier
            for message in messages
            if (identifier := _turn_identifier(message.get("turn_identifier"))) is not None
        ),
        None,
    )
    tenant_source = row.get("hashed_ip")
    if not isinstance(tenant_source, str) or not tenant_source:
        tenant_source = (
            f"first-turn:{first_turn_id}"
            if first_turn_id is not None
            else f"conversation:{conversation_hash}"
        )
    tenant_hash = _hmac_hex(hash_key, "tenant-v1", tenant_source.encode())
    session_source = (
        f"first-turn:{first_turn_id}"
        if first_turn_id is not None
        else json.dumps(
            [conversation_hash, tenant_source, last_timestamp_ms], separators=(",", ":")
        )
    )
    session_hash = _hmac_hex(hash_key, "session-v2", session_source.encode())
    model = row.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model must be a non-empty string when provided")
    model_namespace = json.dumps(
        ["model", model] if model is not None else ["unknown-session", session_hash]
    )
    timestamps = _request_timestamps(messages, roles, last_timestamp_ms, config)
    context: list[tuple[str, str]] = []
    records: list[_ConvertedRequest] = []
    assistant_index = 0
    for message, role in zip(messages, roles, strict=True):
        content = _required_string(message, "content", allow_empty=True)
        if role == "assistant":
            if not context:
                raise ValueError("assistant response is missing prompt context")
            prompt_tokens = _validated_tokens(encode(_render_prompt(context)))
            output_tokens = _validated_tokens(encode(content))
            if not prompt_tokens:
                raise ValueError("tokenized prompt must not be empty")
            turn_id = _turn_identifier(message.get("turn_identifier"))
            request_source = (
                f"turn:{turn_id}"
                if turn_id is not None
                else f"session:{session_hash}:{assistant_index}"
            )
            timestamp_ms, timestamp_source = timestamps[assistant_index]
            records.append(
                _ConvertedRequest(
                    record={
                        "timestamp_ms": timestamp_ms,
                        "tenant_hash": tenant_hash,
                        "session_hash": session_hash,
                        "request_type": "wildchat",
                        "priority": 0,
                        "prompt_length": len(prompt_tokens),
                        "output_length": len(output_tokens),
                        "prefix_path": _prefix_path(
                            prompt_tokens,
                            block_size_tokens=config.block_size_tokens,
                            tokenizer_name=config.tokenizer_name,
                            model_namespace=model_namespace,
                            hash_key=hash_key,
                        ),
                    },
                    request_hash=_hmac_hex(hash_key, "request-v1", request_source.encode()),
                    parent_request_hash=records[-1].request_hash if records else None,
                    reply_hash=_hmac_hex(hash_key, "reply-content-v1", content.encode()),
                    timestamp_source=timestamp_source,
                )
            )
            assistant_index += 1
        context.append((role, content))
    return records


def _turn_identifier(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value) == "":
        raise ValueError("turn_identifier must be a non-empty string or integer")
    return str(value)


def _request_timestamps(
    messages: Sequence[Mapping[str, Any]],
    roles: Sequence[str],
    last_timestamp_ms: float,
    config: WildChatConversionConfig,
) -> list[tuple[float, str]]:
    assistants = [
        message for message, role in zip(messages, roles, strict=True) if role == "assistant"
    ]
    if not assistants:
        return []
    timestamps = [(0.0, "synthetic")] * len(assistants)
    anchors = [(-1, 0.0)]
    for index, message in enumerate(assistants):
        if config.timestamp_mode == "message" and message.get("timestamp") is not None:
            timestamp = _timestamp_ms(message["timestamp"])
            timestamps[index] = (timestamp, "assistant_response")
            anchors.append((index, timestamp))
    if anchors[-1][0] != len(assistants) - 1:
        anchors.append((len(assistants) - 1, last_timestamp_ms))
        timestamps[-1] = (last_timestamp_ms, "synthetic")
    for (left_index, left_time), (right_index, right_time) in zip(anchors, anchors[1:]):
        if right_time < left_time:
            raise ValueError("assistant timestamps must be nonnegative and nondecreasing")
        spacing = min(config.turn_spacing_ms, (right_time - left_time) / (right_index - left_index))
        for index in range(left_index + 1, right_index):
            timestamps[index] = (right_time - (right_index - index) * spacing, "synthetic")
    return timestamps


def _store_request(
    connection: sqlite3.Connection,
    converted: _ConvertedRequest,
    *,
    ordinal: int,
    row_number: int,
) -> bool:
    """Insert one turn, reconciling duplicate timing evidence without losing content checks."""
    if converted.timestamp_source == "synthetic":
        connection.execute(
            "INSERT OR IGNORE INTO inferred_sessions VALUES (?)",
            (converted.record["session_hash"],),
        )
    previous = connection.execute(
        "SELECT parent_hash, timestamp_ms, timestamp_source, reply_hash, payload "
        "FROM requests WHERE request_hash = ?",
        (converted.request_hash,),
    ).fetchone()
    timestamp = converted.record["timestamp_ms"]
    if previous is not None:
        parent_hash, previous_timestamp, previous_source, reply_hash, payload = previous
        previous_record = json.loads(payload)
        previous_record["timestamp_ms"] = timestamp
        conflicting_time = (
            previous_source == converted.timestamp_source == "assistant_response"
            and previous_timestamp != timestamp
        )
        if (
            previous_record != converted.record
            or parent_hash != converted.parent_request_hash
            or reply_hash != converted.reply_hash
            or conflicting_time
        ):
            raise ValueError(f"WildChat row {row_number}: conflicting duplicate turn identifier")
        if previous_source == "synthetic" and converted.timestamp_source == "assistant_response":
            connection.execute(
                "UPDATE requests SET timestamp_ms = ?, timestamp_source = ? WHERE request_hash = ?",
                (timestamp, converted.timestamp_source, converted.request_hash),
            )
        return True
    connection.execute(
        "INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            converted.request_hash,
            converted.record["session_hash"],
            converted.parent_request_hash,
            timestamp,
            converted.timestamp_source,
            converted.reply_hash,
            ordinal,
            json.dumps(converted.record, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        ),
    )
    return False


def _reconcile_inferred_timestamps(connection: sqlite3.Connection) -> None:
    """Constrain inferred times by recorded ancestors and descendants across snapshots.

    A session can branch, so ordering uses preceding assistant turn identifiers
    instead of treating every response in a session as one linear conversation.
    Only one session's metadata is loaded at a time.
    """
    if not connection.execute("SELECT EXISTS(SELECT 1 FROM inferred_sessions)").fetchone()[0]:
        return
    connection.execute("CREATE INDEX requests_session ON requests (session_hash)")
    sessions = connection.execute("SELECT session_hash FROM inferred_sessions")
    for (session_hash,) in sessions:
        nodes = {
            request_hash: (parent_hash, timestamp, source)
            for request_hash, parent_hash, timestamp, source in connection.execute(
                "SELECT request_hash, parent_hash, timestamp_ms, timestamp_source "
                "FROM requests WHERE session_hash = ? ORDER BY ordinal",
                (session_hash,),
            )
        }
        children: dict[str, list[str]] = defaultdict(list)
        pending: deque[str] = deque()
        for request_hash, (parent_hash, _, _) in nodes.items():
            if parent_hash is None:
                pending.append(request_hash)
            elif parent_hash in nodes:
                children[parent_hash].append(request_hash)
            else:
                raise ValueError("overlapping snapshots have a missing preceding turn")
        ordered = []
        while pending:
            request_hash = pending.popleft()
            ordered.append(request_hash)
            pending.extend(children[request_hash])
        if len(ordered) != len(nodes):
            raise ValueError("overlapping snapshots have cyclic turn identifiers")
        latest: dict[str, float] = {}
        for request_hash in reversed(ordered):
            _, timestamp, source = nodes[request_hash]
            latest[request_hash] = (
                timestamp
                if source == "assistant_response"
                else min((latest[child] for child in children[request_hash]), default=math.inf)
            )
        resolved: dict[str, float] = {}
        for request_hash in ordered:
            parent_hash, timestamp, source = nodes[request_hash]
            earliest = resolved[parent_hash] if parent_hash is not None else 0.0
            if latest[request_hash] < earliest:
                raise ValueError("overlapping snapshots have reversed recorded response timestamps")
            resolved[request_hash] = (
                timestamp
                if source == "assistant_response"
                else min(max(timestamp, earliest), latest[request_hash])
            )
            if resolved[request_hash] != timestamp:
                connection.execute(
                    "UPDATE requests SET timestamp_ms = ? WHERE request_hash = ?",
                    (resolved[request_hash], request_hash),
                )


def _render_prompt(messages: Sequence[tuple[str, str]]) -> str:
    # JSON escaping keeps literal role delimiters/newlines in content from
    # impersonating message boundaries in the canonical prompt.
    pieces = [
        f"<|{role}|>\n{json.dumps(content, ensure_ascii=True)}\n" for role, content in messages
    ]
    pieces.append("<|assistant|>\n")
    return "".join(pieces)


def _prefix_path(
    tokens: Sequence[int],
    *,
    block_size_tokens: int,
    tokenizer_name: str,
    model_namespace: str,
    hash_key: bytes,
) -> list[str]:
    prefix_path = []
    for start in range(0, len(tokens), block_size_tokens):
        block = tokens[start : start + block_size_tokens]
        packed = bytearray()
        for token in block:
            packed.extend(struct.pack(">Q", token))
        payload = (
            json.dumps([tokenizer_name, model_namespace], separators=(",", ":")).encode()
            + b"\0"
            + bytes(packed)
        )
        prefix_path.append(_hmac_hex(hash_key, "prefix-block-v1", payload))
    return prefix_path


def _validated_tokens(tokens: Sequence[int]) -> tuple[int, ...]:
    validated = []
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < 2**64:
            raise ValueError("tokenizer must return unsigned 64-bit integer token IDs")
        validated.append(token)
    return tuple(validated)


def _timestamp_ms(value: Any) -> float:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be ISO-8601") from exc
    else:
        raise ValueError("timestamp must be an ISO-8601 string or datetime")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    timestamp = parsed.timestamp() * 1_000.0
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp must be finite and nonnegative")
    return timestamp


def _required_string(
    payload: Mapping[str, Any],
    field: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{field} must be {qualifier}")
    return value


def _hmac_hex(key: bytes, domain: str, payload: bytes) -> str:
    return hmac.new(key, domain.encode() + b"\0" + payload, hashlib.sha256).hexdigest()


def _write_sorted_requests(connection: sqlite3.Connection, output_path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        for timestamp, payload in connection.execute(
            "SELECT timestamp_ms, payload FROM requests ORDER BY timestamp_ms, ordinal"
        ):
            record = json.loads(payload)
            record["timestamp_ms"] = timestamp
            handle.write(
                json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            )
            handle.write("\n")
    os.replace(temporary_path, output_path)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
