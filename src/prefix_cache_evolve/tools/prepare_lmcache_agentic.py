"""Prepare LMCache agentic sessions for metadata-only cache replay."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import click

from prefix_cache_evolve.problems.prefix_kv_cache.lmcache_agentic import (
    LmcacheAgenticConversionConfig,
    convert_lmcache_agentic_rows,
)

_DEFAULT_DATASET_ID = "sammshen/lmcache-agentic-traces"
_DEFAULT_OUTPUT = Path("artifacts/traces/lmcache-agentic.jsonl")
_DEFAULT_HASH_KEY_ENV = "PREFIX_CACHE_TRACE_HASH_KEY"


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Local JSON, JSONL, or Parquet rows; otherwise stream from Hugging Face.",
)
@click.option("--dataset-id", default=_DEFAULT_DATASET_ID, show_default=True)
@click.option("--dataset-revision", default="main", show_default=True)
@click.option("--split", default="train", show_default=True)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_OUTPUT,
    show_default=True,
)
@click.option("--manifest-output", type=click.Path(path_type=Path))
@click.option("--block-size-tokens", type=click.IntRange(min=1), default=512, show_default=True)
@click.option("--arrival-bucket-ms", type=click.IntRange(min=1), default=100, show_default=True)
@click.option("--active-tokens-per-step", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--concurrency", type=click.IntRange(min=1), default=20, show_default=True)
@click.option("--session-limit", type=click.IntRange(min=1))
@click.option("--encoding", "encoding_name", default="cl100k_base", show_default=True)
@click.option("--hash-key-env", default=_DEFAULT_HASH_KEY_ENV, show_default=True)
@click.option("--skip-invalid", is_flag=True)
def main(
    input_path: Path | None,
    dataset_id: str,
    dataset_revision: str,
    split: str,
    output_path: Path,
    manifest_output: Path | None,
    block_size_tokens: int,
    arrival_bucket_ms: int,
    active_tokens_per_step: int,
    concurrency: int,
    session_limit: int | None,
    encoding_name: str,
    hash_key_env: str,
    skip_invalid: bool,
) -> None:
    """Convert LMCache agentic sessions into replay-safe JSONL."""
    hash_key_value = os.environ.get(hash_key_env)
    if hash_key_value is None:
        raise click.ClickException(f"{hash_key_env} is not set")
    hash_key = hash_key_value.encode()
    if len(hash_key) < 32:
        raise click.ClickException(f"{hash_key_env} must contain at least 32 bytes")
    effective_manifest = manifest_output or output_path.with_suffix(
        output_path.suffix + ".manifest.json"
    )
    try:
        rows, source = _load_source(
            input_path=input_path,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            split=split,
        )
        manifest = convert_lmcache_agentic_rows(
            rows,
            output_path,
            effective_manifest,
            encode=_build_encoder(encoding_name),
            hash_key=hash_key,
            source=source,
            config=LmcacheAgenticConversionConfig(
                block_size_tokens=block_size_tokens,
                arrival_bucket_ms=arrival_bucket_ms,
                active_tokens_per_step=active_tokens_per_step,
                concurrency=concurrency,
                session_limit=session_limit,
                tokenizer_name=encoding_name,
                skip_invalid=skip_invalid,
            ),
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"lmcache_agentic_trace={output_path}")
    click.echo(f"lmcache_agentic_manifest={effective_manifest}")
    click.echo(
        json.dumps(
            {
                "requests_written": manifest["requests_written"],
                "sessions_written": manifest["sessions_written"],
                "skipped_rows": manifest["skipped_rows"],
            },
            sort_keys=True,
        )
    )


def _build_encoder(encoding_name: str):
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError("LMCache conversion requires the `wildchat` extra") from exc
    encoding = tiktoken.get_encoding(encoding_name)

    def encode(text: str) -> Sequence[int]:
        return encoding.encode(text, disallowed_special=())

    return encode


def _load_source(
    *, input_path: Path | None, dataset_id: str, dataset_revision: str, split: str
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
            return _iter_parquet((str(input_path),)), {
                "kind": "local_parquet",
                "path": str(input_path),
                "sha256": _file_sha256(input_path),
            }
        raise ValueError("local LMCache input must be JSON, JSONL, NDJSON, or Parquet")
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("LMCache conversion requires the `wildchat` extra") from exc
    resolved_revision = HfApi().dataset_info(dataset_id, revision=dataset_revision).sha
    if not resolved_revision:
        raise RuntimeError(f"unable to resolve Hugging Face revision {dataset_revision!r}")
    rows = load_dataset(dataset_id, split=split, revision=resolved_revision, streaming=True)
    return rows, {
        "kind": "huggingface",
        "dataset_id": dataset_id,
        "requested_revision": dataset_revision,
        "resolved_revision": resolved_revision,
        "split": split,
        "url": f"https://huggingface.co/datasets/{dataset_id}",
        "license": "cc-by-4.0" if dataset_id == _DEFAULT_DATASET_ID else None,
    }


def _iter_parquet(paths: Iterable[str]) -> Iterable[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ImportError("Parquet conversion requires the `wildchat` extra") from exc
    for path in paths:
        with parquet.ParquetFile(path) as source:
            columns = [
                name
                for name in ("session_id", "model", "input", "output_length", "pre_gap")
                if name in source.schema_arrow.names
            ]
            for batch in source.iter_batches(batch_size=64, columns=columns, use_threads=False):
                yield from batch.to_pylist()


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield row


def _iter_json(path: Path) -> Iterable[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: JSON input must contain a list")
    for row in payload:
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}: rows must be objects")
        yield row


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
