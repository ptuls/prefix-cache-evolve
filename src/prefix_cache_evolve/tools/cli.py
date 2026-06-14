"""Consolidated command-line interface for repository analysis tools."""

from __future__ import annotations

import json

import click

from prefix_cache_evolve.problems.prefix_kv_cache.incumbents.registry import (
    current_incumbents,
    incumbent_records,
    validate_incumbent_registry,
)
from prefix_cache_evolve.tools.ablate_structured import main as structured_ablation
from prefix_cache_evolve.tools.analyze_eviction import main as eviction_analysis
from prefix_cache_evolve.tools.analyze_reasoning_kv import main as reasoning_kv_analysis
from prefix_cache_evolve.tools.analyze_rediscovery import main as rediscovery_analysis
from prefix_cache_evolve.tools.analyze_regret import main as regret_analysis
from prefix_cache_evolve.tools.tune_compact import main as compact_tuning


@click.group()
def main() -> None:
    """Run prefix-cache analyses, ablations, and tuning tools."""


@main.group()
def analyze() -> None:
    """Run diagnostic and causal analyses."""


@main.group()
def ablate() -> None:
    """Run controlled policy ablations."""


@main.group()
def tune() -> None:
    """Run deterministic policy tuning."""


@main.group()
def incumbents() -> None:
    """Inspect and validate immutable incumbent bundles."""


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


analyze.add_command(eviction_analysis, name="eviction")
analyze.add_command(rediscovery_analysis, name="rediscovery")
analyze.add_command(regret_analysis, name="regret")
analyze.add_command(reasoning_kv_analysis, name="reasoning-kv")
ablate.add_command(structured_ablation, name="structured")
tune.add_command(compact_tuning, name="compact")


if __name__ == "__main__":
    main()
