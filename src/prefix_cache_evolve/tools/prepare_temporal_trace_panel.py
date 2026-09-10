"""Prepare ordered time windows without inventing missing session identities."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import click
import yaml

from prefix_cache_evolve.evaluators.configuration import TraceTimeWindow, TraceWorkloadConfig
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import file_sha256
from prefix_cache_evolve.problems.prefix_kv_cache.trace_replay import iter_trace_records
from prefix_cache_evolve.workflow.config import WorkflowFileConfig, load_yaml_document

_SPLITS = ("train", "validation", "hidden")


@click.command()
@click.option(
    "--trace",
    "trace_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Base workflow with a prompt, cache geometry, and budgets suitable for this capture.",
)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--family", default="production_trace", show_default=True)
@click.option("--arrival-bucket-ms", type=click.IntRange(min=1), default=100, show_default=True)
@click.option(
    "--window",
    "windows",
    type=(click.Choice(_SPLITS), float, float),
    multiple=True,
    required=True,
    help="Repeat SPLIT START_MS END_MS for disjoint half-open source-clock windows.",
)
def main(
    trace_path: Path,
    config_path: Path,
    output_dir: Path,
    family: str,
    arrival_bucket_ms: int,
    windows: tuple[tuple[str, float, float], ...],
) -> None:
    """Build cold-start chronological search and hidden trace panels.

    Every request in each window is retained in source order. Session and tenant
    identifiers are unchanged; chronological splits do not assert group isolation.
    Synthetic families are omitted and can be evaluated separately as regressions.
    """
    try:
        _prepare_panel(trace_path, config_path, output_dir, family, arrival_bucket_ms, windows)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"trace_panel={output_dir / 'manifest.json'}")
    click.echo(f"evolution_config={output_dir / 'evolution.yaml'}")


def _prepare_panel(
    trace_path: Path,
    config_path: Path,
    output_dir: Path,
    family: str,
    arrival_bucket_ms: int,
    windows: tuple[tuple[str, float, float], ...],
) -> None:
    if output_dir.exists():
        raise ValueError(f"{output_dir} already exists; choose a new panel directory")
    if {split for split, _, _ in windows} != set(_SPLITS):
        raise ValueError("provide at least one window for each of train, validation, and hidden")
    base = load_evaluator_config(config_path)
    if base.trace_workloads:
        raise ValueError("the base configuration must not already contain trace_workloads")
    source_sha = file_sha256(trace_path)
    ordered = sorted(windows, key=lambda window: window[1])
    bounds = [
        TraceTimeWindow(source_sha256=source_sha, start_ms=start, end_ms=end)
        for _, start, end in ordered
    ]
    for index in range(1, len(ordered)):
        if bounds[index - 1].end_ms > bounds[index].start_ms:
            raise ValueError("time windows must not overlap")
        if _SPLITS.index(ordered[index - 1][0]) > _SPLITS.index(ordered[index][0]):
            raise ValueError("time splits must follow train/validation/hidden order")
    source_manifest = trace_path.with_suffix(trace_path.suffix + ".manifest.json")
    provenance_sha = None
    if source_manifest.is_file():
        provenance = json.loads(source_manifest.read_text(encoding="utf-8"))
        if not isinstance(provenance, dict) or provenance.get("output_sha256") != source_sha:
            raise ValueError("source manifest output_sha256 does not match the input trace")
        if provenance.get("block_size_tokens") != base.block_size_tokens:
            raise ValueError("source manifest block size does not match the evaluator")
        provenance_sha = file_sha256(source_manifest)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".temporal-panel-", dir=output_dir.parent) as temporary:
        bundle = Path(temporary) / "panel"
        bundle.mkdir()
        indices: Counter[str] = Counter()
        traces: list[dict[str, Any]] = []
        for (split, _, _), window in zip(ordered, bounds, strict=True):
            indices[split] += 1
            traces.append(
                {
                    "split": split,
                    "family": f"{family}_{indices[split]}",
                    "path": f"{split}-{indices[split]}.jsonl",
                    "time_window": window.model_dump(mode="json"),
                    "request_count": 0,
                    "total_prompt_tokens": 0,
                    "source_record_start": None,
                    "source_record_stop": None,
                    "first_timestamp_ms": None,
                    "last_timestamp_ms": None,
                }
            )
        source_count = 0
        window_index = 0
        with ExitStack() as stack:
            handles = [
                stack.enter_context((bundle / trace["path"]).open("w", encoding="utf-8"))
                for trace in traces
            ]
            for record_index, record in enumerate(
                iter_trace_records(
                    trace_path,
                    expected_sha256=source_sha,
                    block_size_tokens=base.block_size_tokens,
                )
            ):
                source_count += 1
                while (
                    window_index < len(bounds)
                    and record.timestamp_ms >= bounds[window_index].end_ms
                ):
                    window_index += 1
                if (
                    window_index == len(bounds)
                    or record.timestamp_ms < bounds[window_index].start_ms
                ):
                    continue
                trace = traces[window_index]
                handles[window_index].write(
                    json.dumps(record.as_dict(), sort_keys=True, separators=(",", ":")) + "\n"
                )
                if trace["request_count"] == 0:
                    trace["source_record_start"] = record_index
                    trace["first_timestamp_ms"] = record.timestamp_ms
                trace["source_record_stop"] = record_index + 1
                trace["last_timestamp_ms"] = record.timestamp_ms
                trace["request_count"] += 1
                trace["total_prompt_tokens"] += record.prompt_length
        if any(trace["request_count"] == 0 for trace in traces):
            raise ValueError("every requested time window must contain at least one request")
        workloads = []
        for trace in traces:
            trace["sha256"] = file_sha256(bundle / trace["path"])
            workloads.append(
                TraceWorkloadConfig(
                    **{
                        key: trace[key]
                        for key in (
                            "path",
                            "sha256",
                            "family",
                            "split",
                            "request_count",
                            "time_window",
                        )
                    },
                    block_size_tokens=base.block_size_tokens,
                    arrival_bucket_ms=arrival_bucket_ms,
                )
            )
        config = base.with_updates(
            trace_workloads=workloads,
            train_families=(),
            validation_families=(),
            probe_families=(),
            hidden_families=(),
            family_request_multipliers={},
            search_score_mode="combined",
            search_guidance_families=(),
        )
        document = WorkflowFileConfig.model_validate(load_yaml_document(config_path)).model_dump(
            exclude_none=True, exclude_unset=True
        )
        document.setdefault("problem", {})["settings"] = config.model_dump(mode="json")
        (bundle / "evolution.yaml").write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
        manifest = {
            "schema": "prefix-kv-cache-temporal-trace-panel-v1",
            "source": {
                "path": str(trace_path),
                "sha256": source_sha,
                "request_count": source_count,
            },
            "source_manifest_sha256": provenance_sha,
            "base_config_sha256": file_sha256(config_path),
            "block_size_tokens": base.block_size_tokens,
            "arrival_bucket_ms": arrival_bucket_ms,
            "partition": {"algorithm": "half-open-time-windows-v1", "cold_start_per_window": True},
            "excluded_request_count": source_count
            - sum(trace["request_count"] for trace in traces),
            "traces": traces,
            "limitations": [
                "Chronological separation does not establish session or tenant independence.",
                "Prefixes and sessions can recur across windows; policy/cache state resets.",
                "Validation is visible to evolution; hidden is reserved for final evaluation.",
                "Omitted gaps and cold starts change the serving state of the original capture.",
                "Arrival buckets and output-length pinning are proxies, not measured latency.",
            ],
        }
        (bundle / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if provenance_sha is not None:
            shutil.copyfile(source_manifest, bundle / "source.manifest.json")
            if file_sha256(bundle / "source.manifest.json") != provenance_sha:
                raise ValueError("source manifest changed while preparing the panel")
        load_evaluator_config(bundle / "evolution.yaml")
        bundle.rename(output_dir)


if __name__ == "__main__":
    main()
