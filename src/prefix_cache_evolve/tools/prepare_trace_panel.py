"""Prepare disjoint trace splits and a pinned evolution configuration."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import click
import yaml

from prefix_cache_evolve.evaluators.configuration import TraceWorkloadConfig
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import (
    DEFAULT_CONFIG_PATH,
    load_evaluator_config,
)
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
    help="Metadata-only JSONL produced by a dataset converter.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/traces/trace-panel"),
    show_default=True,
    help="New directory for split traces, provenance, and evolution.yaml.",
)
@click.option("--family", default="real_trace", show_default=True)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=DEFAULT_CONFIG_PATH,
    help="Base workflow configuration supplying the policy contract and score settings.",
)
@click.option("--block-size-tokens", type=click.IntRange(min=1))
@click.option(
    "--capacity-tokens",
    type=click.IntRange(min=1),
    multiple=True,
    help="Optional repeated trace cache capacities, divisible by the native block size.",
)
@click.option("--arrival-bucket-ms", type=click.IntRange(min=1), default=100, show_default=True)
@click.option(
    "--group-by",
    type=click.Choice(("tenant", "session")),
    default="tenant",
    show_default=True,
    help="Keep every group in one split; tenant grouping also holds out users.",
)
@click.option("--split-seed", type=int, default=0, show_default=True)
@click.option(
    "--validation-fraction",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    default=0.2,
    show_default=True,
)
@click.option(
    "--hidden-fraction",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    default=0.2,
    show_default=True,
)
@click.option(
    "--keep-synthetic",
    is_flag=True,
    help="Add traces to the base synthetic train/validation/hidden families. Probes are retained.",
)
@click.option(
    "--source-manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Converter provenance; automatically uses TRACE.manifest.json when present.",
)
def main(
    trace_path: Path,
    output_dir: Path,
    family: str,
    config_path: Path,
    block_size_tokens: int | None,
    capacity_tokens: tuple[int, ...],
    arrival_bucket_ms: int,
    group_by: str,
    split_seed: int,
    validation_fraction: float,
    hidden_fraction: float,
    keep_synthetic: bool,
    source_manifest: Path | None,
) -> None:
    """Split a trace into reproducible search and held-out evaluation panels."""
    try:
        _prepare_panel(
            trace_path=trace_path,
            output_dir=output_dir,
            family=family,
            config_path=config_path,
            block_size_tokens=block_size_tokens,
            capacity_tokens=capacity_tokens,
            arrival_bucket_ms=arrival_bucket_ms,
            group_by=group_by,
            split_seed=split_seed,
            validation_fraction=validation_fraction,
            hidden_fraction=hidden_fraction,
            keep_synthetic=keep_synthetic,
            source_manifest=source_manifest,
        )
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"trace_panel={output_dir / 'manifest.json'}")
    click.echo(f"evolution_config={output_dir / 'evolution.yaml'}")


def _prepare_panel(
    *,
    trace_path: Path,
    output_dir: Path,
    family: str,
    config_path: Path,
    block_size_tokens: int | None,
    arrival_bucket_ms: int,
    group_by: str,
    split_seed: int,
    validation_fraction: float,
    hidden_fraction: float,
    keep_synthetic: bool,
    source_manifest: Path | None,
    capacity_tokens: tuple[int, ...] = (),
) -> None:
    if output_dir.exists():
        raise ValueError(f"{output_dir} already exists; choose a new panel directory")
    if validation_fraction + hidden_fraction >= 1:
        raise ValueError("validation and hidden fractions must leave a nonempty train split")
    base = load_evaluator_config(config_path)
    if base.trace_workloads:
        raise ValueError("the base configuration must not already contain trace_workloads")
    block_size = block_size_tokens or base.block_size_tokens
    if any(tokens % block_size for tokens in capacity_tokens):
        raise ValueError("capacity tokens must be multiples of the native block size")
    trace_sha = file_sha256(trace_path)
    automatic_manifest = trace_path.with_suffix(trace_path.suffix + ".manifest.json")
    if source_manifest is None and automatic_manifest.is_file():
        source_manifest = automatic_manifest
    source_manifest_sha = None
    if source_manifest is not None:
        provenance = json.loads(source_manifest.read_text(encoding="utf-8"))
        if not isinstance(provenance, dict) or provenance.get("output_sha256") != trace_sha:
            raise ValueError("source manifest output_sha256 does not match the input trace")
        if provenance.get("block_size_tokens") != block_size:
            raise ValueError("source manifest block size does not match --block-size-tokens")
        source_manifest_sha = file_sha256(source_manifest)

    groups: set[str] = set()
    session_groups: dict[str, str] = {}
    for record in iter_trace_records(
        trace_path, expected_sha256=trace_sha, block_size_tokens=block_size
    ):
        group = record.tenant_key if group_by == "tenant" else record.session_key
        if record.session_key is None or group in {
            "mooncake:unknown",
            "lmcache:unknown",
            "qwen:unknown",
            "agentx:unknown",
        }:
            raise ValueError(
                "disjoint groups require known identities; use chronological windows "
                "for unknown sessions or group known sessions with --group-by session"
            )
        assert group is not None
        previous = session_groups.setdefault(record.session_key, group)
        if previous != group:
            raise ValueError("a session spans multiple tenants; use --group-by session")
        groups.add(group)
    assignments = _assign_groups(groups, split_seed, validation_fraction, hidden_fraction)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".trace-panel-", dir=output_dir.parent) as temporary:
        bundle = Path(temporary) / "panel"
        bundle.mkdir()
        traces = _write_splits(trace_path, bundle, assignments, group_by, trace_sha, block_size)
        workloads = tuple(
            TraceWorkloadConfig(
                path=trace["path"],
                sha256=trace["sha256"],
                family=family,
                split=trace["split"],
                block_size_tokens=block_size,
                request_count=trace["request_count"],
                arrival_bucket_ms=arrival_bucket_ms,
                capacity_sweep_blocks=tuple(tokens // block_size for tokens in capacity_tokens),
            )
            for trace in traces
        )
        updates: dict[str, Any] = {
            "trace_workloads": workloads,
            "block_size_tokens": block_size,
        }
        if not keep_synthetic:
            updates.update(
                train_families=(),
                validation_families=(),
                hidden_families=(),
                family_request_multipliers={},
                search_score_mode="combined",
                search_guidance_families=(),
            )
        config = base.with_updates(**updates)
        document = WorkflowFileConfig.model_validate(load_yaml_document(config_path)).model_dump(
            exclude_none=True, exclude_unset=True
        )
        document["problem"]["settings"] = config.model_dump(mode="json")
        document.setdefault("search", {})["notes"] = (
            "Optimize the configured trace train/validation panels. Validation feedback is "
            "visible during search; hidden traces are reserved for final adjudication. "
            "Synthetic probes remain reporting-only. Traces contain opaque prefix identities "
            "and use the configured timing proxies, not measured serving latency."
        )
        if not keep_synthetic:
            document["behavior"] = {
                "score_keys": [
                    "validation_token_hit_rate",
                    "validation_cache_churn_per_1k",
                    "validation_policy_underfill_rate",
                ]
            }
        (bundle / "evolution.yaml").write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
        manifest = {
            "schema": "prefix-kv-cache-trace-panel-v1",
            "source": {"path": str(trace_path), "sha256": trace_sha},
            "source_manifest_sha256": source_manifest_sha,
            "base_config_sha256": file_sha256(config_path),
            "block_size_tokens": block_size,
            "arrival_bucket_ms": arrival_bucket_ms,
            "capacity_tokens": list(capacity_tokens),
            "partition": {
                "algorithm": "sorted-sha256-groups-v1",
                "group_by": group_by,
                "seed": split_seed,
                "validation_fraction": validation_fraction,
                "hidden_fraction": hidden_fraction,
                "group_count": len(groups),
                "split_group_counts": dict(Counter(assignments.values())),
            },
            "keep_synthetic": keep_synthetic,
            "family": family,
            "traces": traces,
            "limitations": [
                "Random group holdouts do not establish transfer to later time periods.",
                "Validation is visible to evolution; only hidden is held out from search.",
                "Filtering groups changes the arrival mix; replay timings are simulator proxies.",
            ],
        }
        (bundle / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if source_manifest is not None:
            shutil.copyfile(source_manifest, bundle / "source.manifest.json")
            if file_sha256(bundle / "source.manifest.json") != source_manifest_sha:
                raise ValueError("source manifest changed while preparing the panel")
        # Resolve the relative trace paths exactly as an evolution worker will.
        load_evaluator_config(bundle / "evolution.yaml")
        bundle.rename(output_dir)


def _assign_groups(
    groups: set[str], seed: int, validation_fraction: float, hidden_fraction: float
) -> dict[str, str]:
    if len(groups) < 3:
        raise ValueError("at least three distinct groups are required for train/validation/hidden")
    ordered = sorted(
        groups,
        key=lambda group: (
            hashlib.sha256(f"prefix-trace-split-v1\0{seed}\0{group}".encode()).digest(),
            group,
        ),
    )
    validation_count = max(1, int(len(ordered) * validation_fraction))
    hidden_count = max(1, int(len(ordered) * hidden_fraction))
    train_count = len(ordered) - validation_count - hidden_count
    if train_count < 1:
        raise ValueError("too few groups for these split fractions; increase the input sample")
    return {
        group: (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "hidden"
        )
        for index, group in enumerate(ordered)
    }


def _write_splits(
    source: Path,
    output_dir: Path,
    assignments: dict[str, str],
    group_by: str,
    source_sha: str,
    block_size: int,
) -> list[dict[str, Any]]:
    request_counts: Counter[str] = Counter()
    prompt_tokens: Counter[str] = Counter()
    sessions: dict[str, set[str]] = {split: set() for split in _SPLITS}
    tenants: dict[str, set[str]] = {split: set() for split in _SPLITS}
    with ExitStack() as stack:
        handles = {
            split: stack.enter_context((output_dir / f"{split}.jsonl").open("w", encoding="utf-8"))
            for split in _SPLITS
        }
        for record in iter_trace_records(
            source, expected_sha256=source_sha, block_size_tokens=block_size
        ):
            group = record.tenant_key if group_by == "tenant" else record.session_key
            assert group is not None and record.session_key is not None
            split = assignments[group]
            handles[split].write(
                json.dumps(record.as_dict(), sort_keys=True, separators=(",", ":")) + "\n"
            )
            request_counts[split] += 1
            prompt_tokens[split] += record.prompt_length
            sessions[split].add(record.session_key)
            tenants[split].add(record.tenant_key)
    return [
        {
            "split": split,
            "path": f"{split}.jsonl",
            "sha256": file_sha256(output_dir / f"{split}.jsonl"),
            "request_count": request_counts[split],
            "session_count": len(sessions[split]),
            "tenant_count": len(tenants[split]),
            "total_prompt_tokens": prompt_tokens[split],
        }
        for split in _SPLITS
    ]


if __name__ == "__main__":
    main()
