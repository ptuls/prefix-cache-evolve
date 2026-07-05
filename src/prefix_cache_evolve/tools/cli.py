"""Consolidated command-line interface for repository analysis tools."""

from __future__ import annotations

import json

import click

from prefix_cache_evolve.problems.prefix_kv_cache.incumbents.registry import (
    current_incumbents,
    incumbent_records,
    validate_incumbent_registry,
)
from prefix_cache_evolve.tools.lazy_group import LazyCommand, LazyGroup

_ANALYZE_COMMANDS = {
    "eviction": LazyCommand(
        "prefix_cache_evolve.tools.analyze_eviction:main",
        "Analyze eviction choice, regret, and specialist distillations.",
    ),
    "reasoning-kv": LazyCommand(
        "prefix_cache_evolve.tools.analyze_reasoning_kv:main",
        "Compare policies under shared reasoning decode KV pressure.",
    ),
    "rediscovery": LazyCommand(
        "prefix_cache_evolve.tools.analyze_rediscovery:main",
        "Adjudicate weak-seed evolution runs against the incumbent.",
    ),
    "regret": LazyCommand(
        "prefix_cache_evolve.tools.analyze_regret:main",
        "Audit admission and eviction regret.",
    ),
}
_VERIFY_COMMANDS = {
    "significance": LazyCommand(
        "prefix_cache_evolve.tools.verify_significance:main",
        "Test whether the headline score gap exceeds seed noise.",
    ),
}
_ABLATE_COMMANDS = {
    "structured": LazyCommand(
        "prefix_cache_evolve.tools.ablate_structured:main",
        "Ablate structured policy terms.",
    ),
}
_TUNE_COMMANDS = {
    "compact": LazyCommand(
        "prefix_cache_evolve.tools.tune_compact:main",
        "Tune the compact deployable policy.",
    ),
}
_DATASET_COMMANDS = {
    "wildchat": LazyCommand(
        "prefix_cache_evolve.tools.prepare_wildchat:main",
        "Prepare deterministic WildChat trace-replay data.",
    ),
}


@click.group()
def main() -> None:
    """Run prefix-cache analyses, ablations, and tuning tools."""


@main.group(cls=LazyGroup, lazy_subcommands=_ANALYZE_COMMANDS)
def analyze() -> None:
    """Run diagnostic and causal analyses."""


@main.group(cls=LazyGroup, lazy_subcommands=_VERIFY_COMMANDS)
def verify() -> None:
    """Run statistical verification of headline claims."""


@main.group(cls=LazyGroup, lazy_subcommands=_ABLATE_COMMANDS)
def ablate() -> None:
    """Run controlled policy ablations."""


@main.group(cls=LazyGroup, lazy_subcommands=_TUNE_COMMANDS)
def tune() -> None:
    """Run deterministic policy tuning."""


@main.group()
def incumbents() -> None:
    """Inspect and validate immutable incumbent bundles."""


@main.group(cls=LazyGroup, lazy_subcommands=_DATASET_COMMANDS)
def datasets() -> None:
    """Prepare public datasets for replay-safe evaluation."""


@incumbents.command("list")
def list_incumbents() -> None:
    """Print registered incumbent identities and headline benchmarks."""
    current_by_role = {role: record.incumbent_id for role, record in current_incumbents().items()}
    payload = [
        {
            "id": record.incumbent_id,
            "role": record.role,
            "status": record.payload["status"],
            "current_roles": sorted(
                role
                for role, incumbent_id in current_by_role.items()
                if incumbent_id == record.incumbent_id
            ),
            "source_path": str(record.source_path),
            "source_sha256": record.source_sha256,
            "effective_complexity": record.effective_complexity,
            "benchmark": dict(record.benchmark),
        }
        for record in incumbent_records()
    ]
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@incumbents.command("validate")
def validate_incumbents() -> None:
    """Fail closed if any incumbent source or manifest has drifted."""
    records = validate_incumbent_registry()
    click.echo(f"validated_incumbents={len(records)}")


if __name__ == "__main__":
    main()
