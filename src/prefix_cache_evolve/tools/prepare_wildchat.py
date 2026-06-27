"""Prepare WildChat conversations for metadata-only prefix-cache replay."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import click

from prefix_cache_evolve.problems.prefix_kv_cache.wildchat import (
    WildChatConversionConfig,
    convert_wildchat_rows,
)

_DEFAULT_DATASET_ID = "allenai/WildChat-1M"
_DEFAULT_OUTPUT = Path("artifacts/traces/wildchat.jsonl")
_DEFAULT_HASH_KEY_ENV = "PREFIX_CACHE_TRACE_HASH_KEY"


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Local WildChat JSON, JSONL, or Parquet file; otherwise stream from Hugging Face.",
)
@click.option("--dataset-id", default=_DEFAULT_DATASET_ID, show_default=True)
@click.option(
    "--dataset-revision",
    default="main",
    show_default=True,
    help="Requested Hugging Face revision; the resolved commit is recorded in the manifest.",
)
@click.option("--split", default="train", show_default=True)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_OUTPUT,
    show_default=True,
)
@click.option(
    "--manifest-output",
    type=click.Path(path_type=Path),
    help="Conversion manifest; defaults to OUTPUT.manifest.json.",
)
@click.option("--block-size-tokens", type=click.IntRange(min=1), default=16, show_default=True)
@click.option(
    "--turn-spacing-ms",
    type=click.IntRange(min=0),
    default=1_000,
    show_default=True,
    help="Synthetic spacing between assistant requests in one conversation.",
)
@click.option(
    "--conversation-limit",
    type=click.IntRange(min=1),
    help="Optional number of successfully converted conversations.",
)
@click.option(
    "--minimum-requests-per-conversation",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Use 2 or more to retain only multi-turn conversations.",
)
@click.option(
    "--encoding",
    "encoding_name",
    default="cl100k_base",
    show_default=True,
    help="tiktoken encoding used for deterministic canonical prompts.",
)
@click.option(
    "--hash-key-env",
    default=_DEFAULT_HASH_KEY_ENV,
    show_default=True,
    help="Environment variable containing at least 32 bytes of HMAC key material.",
)
@click.option(
    "--skip-invalid",
    is_flag=True,
    help="Skip malformed conversations and record aggregate reasons in the manifest.",
)
def main(
    input_path: Path | None,
    dataset_id: str,
    dataset_revision: str,
    split: str,
    output_path: Path,
    manifest_output: Path | None,
    block_size_tokens: int,
    turn_spacing_ms: int,
    conversation_limit: int | None,
    minimum_requests_per_conversation: int,
    encoding_name: str,
    hash_key_env: str,
    skip_invalid: bool,
) -> None:
    """Convert WildChat into the repository's replay-safe JSONL format."""
    hash_key_value = os.environ.get(hash_key_env)
    if hash_key_value is None:
        raise click.ClickException(
            f"{hash_key_env} is not set; generate a private key with `openssl rand -hex 32`"
        )
    hash_key = hash_key_value.encode()
    if len(hash_key) < 32:
        raise click.ClickException(f"{hash_key_env} must contain at least 32 bytes")

    try:
        encode = _build_encoder(encoding_name)
        rows, source = _load_source(
            input_path=input_path,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            split=split,
        )
        effective_manifest_output = manifest_output or output_path.with_suffix(
            output_path.suffix + ".manifest.json"
        )
        manifest = convert_wildchat_rows(
            rows,
            output_path,
            effective_manifest_output,
            encode=encode,
            hash_key=hash_key,
            source=source,
            config=WildChatConversionConfig(
                block_size_tokens=block_size_tokens,
                turn_spacing_ms=turn_spacing_ms,
                conversation_limit=conversation_limit,
                minimum_requests_per_conversation=minimum_requests_per_conversation,
                skip_invalid=skip_invalid,
                tokenizer_name=encoding_name,
            ),
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"wildchat_trace={output_path}")
    click.echo(f"wildchat_manifest={effective_manifest_output}")
    click.echo(
        json.dumps(
            {
                "conversations_converted": manifest["conversations_converted"],
                "requests_written": manifest["requests_written"],
                "skipped_conversations": manifest["skipped_conversations"],
            },
            sort_keys=True,
        )
    )


def _build_encoder(encoding_name: str):
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError(
            "WildChat conversion requires the `wildchat` extra: uv sync --extra wildchat"
        ) from exc
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except ValueError as exc:
        raise ValueError(f"unknown tiktoken encoding {encoding_name!r}") from exc

    def encode(text: str) -> Sequence[int]:
        return encoding.encode(text, disallowed_special=())

    return encode


def _load_source(
    *,
    input_path: Path | None,
    dataset_id: str,
    dataset_revision: str,
    split: str,
) -> tuple[Iterable[Mapping[str, Any]], dict[str, object]]:
    if input_path is not None:
        suffix = input_path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            return _iter_jsonl(input_path), {
                "kind": "local_jsonl",
                "path": str(input_path),
                "sha256": _file_sha256(input_path),
            }
        if suffix == ".json":
            return _iter_json(input_path), {
                "kind": "local_json",
                "path": str(input_path),
                "sha256": _file_sha256(input_path),
            }
        if suffix == ".parquet":
            load_dataset = _datasets_loader()
            rows = load_dataset(
                "parquet",
                data_files=str(input_path),
                split="train",
                streaming=True,
            )
            return rows, {
                "kind": "local_parquet",
                "path": str(input_path),
                "sha256": _file_sha256(input_path),
            }
        raise ValueError("local WildChat input must be JSON, JSONL, NDJSON, or Parquet")

    load_dataset = _datasets_loader()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "WildChat conversion requires the `wildchat` extra: uv sync --extra wildchat"
        ) from exc
    resolved_revision = (
        HfApi()
        .dataset_info(
            dataset_id,
            revision=dataset_revision,
        )
        .sha
    )
    if not resolved_revision:
        raise RuntimeError(f"unable to resolve Hugging Face revision {dataset_revision!r}")
    rows = load_dataset(
        dataset_id,
        split=split,
        revision=resolved_revision,
        streaming=True,
    )
    source: dict[str, object] = {
        "kind": "huggingface",
        "dataset_id": dataset_id,
        "requested_revision": dataset_revision,
        "resolved_revision": resolved_revision,
        "split": split,
        "url": f"https://huggingface.co/datasets/{dataset_id}",
    }
    if dataset_id == _DEFAULT_DATASET_ID:
        source["license"] = "odc-by"
    return rows, source


def _datasets_loader():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Parquet and Hugging Face streaming require the `wildchat` extra: "
            "uv sync --extra wildchat"
        ) from exc
    return load_dataset


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield payload


def _iter_json(path: Path) -> Iterable[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: JSON input must contain a list of rows")
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}: row {index} must be an object")
        yield row


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
