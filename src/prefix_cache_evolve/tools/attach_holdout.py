"""Attach an independent dataset to final evaluation without exposing it to search."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import click
import yaml

from prefix_cache_evolve.evaluators.configuration import TraceWorkloadConfig
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import file_sha256
from prefix_cache_evolve.problems.prefix_kv_cache.trace_replay import iter_trace_records
from prefix_cache_evolve.workflow.config import WorkflowFileConfig, load_yaml_document


@click.command()
@click.option(
    "--trace",
    "trace_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Converted trace with its OUTPUT.manifest.json sidecar.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Existing search configuration; its workload files are retained.",
)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--family", default="agentx", show_default=True)
@click.option(
    "--capacity-tokens",
    type=click.IntRange(min=1),
    multiple=True,
    required=True,
    help="Repeat for each cache capacity; must divide by the source block size.",
)
@click.option("--arrival-bucket-ms", type=click.IntRange(min=1), default=100, show_default=True)
def main(
    trace_path: Path,
    config_path: Path,
    output_dir: Path,
    family: str,
    capacity_tokens: tuple[int, ...],
    arrival_bucket_ms: int,
) -> None:
    """Pin a complete independent trace to hidden, excluded from normal evolution."""
    try:
        attach_holdout(
            trace_path,
            config_path,
            output_dir,
            family=family,
            capacity_tokens=capacity_tokens,
            arrival_bucket_ms=arrival_bucket_ms,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"evolution_config={output_dir / 'evolution.yaml'}")


def attach_holdout(
    trace_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    family: str,
    capacity_tokens: tuple[int, ...],
    arrival_bucket_ms: int = 100,
) -> None:
    """Validate provenance and group separation, then add only a hidden workload."""
    if output_dir.exists():
        raise ValueError(f"{output_dir} already exists; choose a new panel directory")
    base = load_evaluator_config(config_path)
    source_manifest = trace_path.with_suffix(trace_path.suffix + ".manifest.json")
    provenance = json.loads(source_manifest.read_text())
    trace_sha = file_sha256(trace_path)
    if not isinstance(provenance, dict) or provenance.get("output_sha256") != trace_sha:
        raise ValueError("source manifest output_sha256 does not match the trace")
    block_size = provenance.get("block_size_tokens")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("source manifest must declare a positive block_size_tokens")
    if not capacity_tokens or any(
        type(tokens) is not int or tokens <= 0 or tokens % block_size for tokens in capacity_tokens
    ):
        raise ValueError("capacity tokens must be positive multiples of the native block size")
    sessions = set()
    count = 0
    for record in iter_trace_records(
        trace_path, block_size_tokens=block_size, expected_sha256=trace_sha
    ):
        if record.session_key is None:
            raise ValueError("an independent holdout requires known session identities")
        sessions.add(record.session_key)
        count += 1
    if provenance.get("request_count") != count:
        raise ValueError("source manifest request_count does not match the trace")
    for trace in base.trace_workloads:
        if trace.split == "hidden":
            continue
        for record in iter_trace_records(
            Path(trace.path),
            block_size_tokens=trace.block_size_tokens,
            expected_sha256=trace.sha256,
        ):
            if record.session_key in sessions:
                raise ValueError("holdout sessions overlap a search or probe trace")
    holdout = TraceWorkloadConfig(
        path="hidden.jsonl",
        sha256=trace_sha,
        family=family,
        split="hidden",
        block_size_tokens=block_size,
        request_count=count,
        arrival_bucket_ms=arrival_bucket_ms,
        capacity_sweep_blocks=tuple(tokens // block_size for tokens in capacity_tokens),
    )
    config = base.with_updates(trace_workloads=(*base.trace_workloads, holdout))
    document = WorkflowFileConfig.model_validate(load_yaml_document(config_path)).model_dump(
        exclude_none=True, exclude_unset=True
    )
    document["problem"]["settings"] = config.model_dump(mode="json")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".holdout-", dir=output_dir.parent) as temporary:
        bundle = Path(temporary) / "panel"
        bundle.mkdir()
        shutil.copyfile(trace_path, bundle / "hidden.jsonl")
        shutil.copyfile(source_manifest, bundle / "source.manifest.json")
        if file_sha256(bundle / "hidden.jsonl") != trace_sha:
            raise ValueError("trace changed while attaching holdout")
        if json.loads((bundle / "source.manifest.json").read_text()) != provenance:
            raise ValueError("source manifest changed while attaching holdout")
        (bundle / "evolution.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
        (bundle / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "prefix-kv-cache-external-holdout-v1",
                    "base_config": {
                        "path": str(config_path.resolve()),
                        "sha256": file_sha256(config_path),
                    },
                    "holdout": holdout.model_dump(mode="json"),
                    "source_manifest_sha256": file_sha256(bundle / "source.manifest.json"),
                    "selection_rule": "Freeze selection before hidden evaluation.",
                    "limitations": [
                        "Retain original search panels at their resolved paths.",
                        "Tuning after reading hidden results makes this development data.",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        load_evaluator_config(bundle / "evolution.yaml")
        bundle.rename(output_dir)
