"""Convert WildChat conversations into metadata-only prefix-cache traces."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import struct
import tempfile
from collections import Counter
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
class WildChatConversionConfig:
    """Controls conversion from WildChat rows to replay records."""

    block_size_tokens: int = 16
    turn_spacing_ms: int = 1_000
    conversation_limit: int | None = None
    minimum_requests_per_conversation: int = 1
    skip_invalid: bool = False
    tokenizer_name: str = "cl100k_base"

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_reasons: Counter[str] = Counter()
    rows_scanned = 0
    conversations_converted = 0
    requests_written = 0
    ordinal = 0

    with tempfile.TemporaryDirectory(prefix="prefix-cache-wildchat-") as temp_dir:
        database_path = Path(temp_dir) / "requests.sqlite3"
        with sqlite3.connect(database_path) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE requests (timestamp_ms REAL, ordinal INTEGER, payload TEXT)"
            )

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

                connection.executemany(
                    "INSERT INTO requests VALUES (?, ?, ?)",
                    (
                        (
                            float(record["timestamp_ms"]),
                            ordinal + index,
                            json.dumps(
                                record,
                                ensure_ascii=True,
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                        )
                        for index, record in enumerate(records)
                    ),
                )
                ordinal += len(records)
                requests_written += len(records)
                conversations_converted += 1
            connection.commit()

            if not requests_written:
                raise ValueError("WildChat conversion produced no replay requests")
            _write_sorted_requests(connection, output_path)

    manifest: dict[str, object] = {
        "schema": "prefix-kv-cache-wildchat-conversion-v1",
        "source": dict(source),
        "output_path": str(output_path),
        "output_sha256": _file_sha256(output_path),
        "rows_scanned": rows_scanned,
        "conversations_converted": conversations_converted,
        "requests_written": requests_written,
        "skipped_conversations": sum(skipped_reasons.values()),
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
        "tokenizer": {
            "name": config.tokenizer_name,
            "prompt_serialization": "prefix-cache-wildchat-chat-v1",
        },
        "block_size_tokens": config.block_size_tokens,
        "turn_spacing_ms": config.turn_spacing_ms,
        "conversation_limit": config.conversation_limit,
        "minimum_requests_per_conversation": config.minimum_requests_per_conversation,
        "skip_invalid": config.skip_invalid,
        "hash": {
            "algorithm": "HMAC-SHA256",
            "key_fingerprint_sha256": hashlib.sha256(hash_key).hexdigest(),
        },
        "privacy": {
            "raw_content_written": False,
            "raw_identifiers_written": False,
        },
        "limitations": [
            "WildChat provides one timestamp per conversation, so intra-conversation "
            "request spacing is synthetic.",
            "The canonical prompt serialization is deterministic but does not reproduce "
            "the original provider's private system prompt or exact chat template.",
            "This is conversation-derived replay, not a production serving trace.",
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
) -> list[_ReplayRecord]:
    conversation_hash = _required_string(row, "conversation_hash")
    tenant_source = row.get("hashed_ip")
    if not isinstance(tenant_source, str) or not tenant_source:
        tenant_source = f"conversation:{conversation_hash}"
    timestamp_ms = _timestamp_ms(row.get("timestamp"))
    messages = row.get("conversation")
    if not isinstance(messages, list) or not messages:
        raise ValueError("conversation must be a non-empty list")

    tenant_hash = _hmac_hex(hash_key, "tenant-v1", tenant_source.encode())
    session_hash = _hmac_hex(hash_key, "session-v1", conversation_hash.encode())
    context: list[tuple[str, str]] = []
    records: list[_ReplayRecord] = []
    assistant_index = 0
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("conversation messages must be objects")
        role = _required_string(message, "role").lower()
        if role not in _SUPPORTED_ROLES:
            raise ValueError(f"unsupported conversation role {role!r}")
        content = _required_string(message, "content", allow_empty=True)
        if role == "assistant":
            if not context:
                raise ValueError("assistant response is missing prompt context")
            prompt_tokens = _validated_tokens(encode(_render_prompt(context)))
            output_tokens = _validated_tokens(encode(content))
            if not prompt_tokens:
                raise ValueError("tokenized prompt must not be empty")
            records.append(
                {
                    "timestamp_ms": timestamp_ms + assistant_index * config.turn_spacing_ms,
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
                        hash_key=hash_key,
                    ),
                }
            )
            assistant_index += 1
        context.append((role, content))
    return records


def _render_prompt(messages: Sequence[tuple[str, str]]) -> str:
    pieces = [f"<|{role}|>\n{content}\n" for role, content in messages]
    pieces.append("<|assistant|>\n")
    return "".join(pieces)


def _prefix_path(
    tokens: Sequence[int],
    *,
    block_size_tokens: int,
    tokenizer_name: str,
    hash_key: bytes,
) -> list[str]:
    prefix_path = []
    for start in range(0, len(tokens), block_size_tokens):
        block = tokens[start : start + block_size_tokens]
        packed = bytearray()
        for token in block:
            packed.extend(struct.pack(">Q", token))
        payload = tokenizer_name.encode() + b"\0" + bytes(packed)
        prefix_path.append(_hmac_hex(hash_key, "prefix-block-v1", payload))
    return prefix_path


def _validated_tokens(tokens: Sequence[int]) -> tuple[int, ...]:
    validated = []
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError("tokenizer must return nonnegative integer token IDs")
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
    return parsed.timestamp() * 1_000.0


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
        for (payload,) in connection.execute(
            "SELECT payload FROM requests ORDER BY timestamp_ms, ordinal"
        ):
            handle.write(str(payload))
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
