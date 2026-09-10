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
from prefix_cache_evolve.tools.analyze_policy_costs import main as policy_cost_analysis
from prefix_cache_evolve.tools.analyze_reasoning_kv import main as reasoning_kv_analysis
from prefix_cache_evolve.tools.analyze_rediscovery import main as rediscovery_analysis
from prefix_cache_evolve.tools.analyze_regret import main as regret_analysis
from prefix_cache_evolve.tools.attach_holdout import main as attach_holdout
from prefix_cache_evolve.tools.prepare_agentx import main as prepare_agentx
from prefix_cache_evolve.tools.prepare_lmcache_agentic import main as prepare_lmcache_agentic
from prefix_cache_evolve.tools.prepare_mooncake import main as prepare_mooncake
from prefix_cache_evolve.tools.prepare_qwen import main as prepare_qwen
from prefix_cache_evolve.tools.prepare_temporal_trace_panel import (
    main as prepare_temporal_trace_panel,
)
from prefix_cache_evolve.tools.prepare_trace_panel import main as prepare_trace_panel
from prefix_cache_evolve.tools.prepare_wildchat import main as prepare_wildchat
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


@main.group()
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


analyze.add_command(eviction_analysis, name="eviction")
analyze.add_command(rediscovery_analysis, name="rediscovery")
analyze.add_command(regret_analysis, name="regret")
analyze.add_command(reasoning_kv_analysis, name="reasoning-kv")
analyze.add_command(policy_cost_analysis, name="policy-costs")
ablate.add_command(structured_ablation, name="structured")
datasets.add_command(prepare_wildchat, name="wildchat")
datasets.add_command(prepare_lmcache_agentic, name="lmcache-agentic")
datasets.add_command(prepare_mooncake, name="mooncake")
datasets.add_command(prepare_qwen, name="qwen")
datasets.add_command(prepare_agentx, name="agentx")
datasets.add_command(attach_holdout, name="attach-holdout")
datasets.add_command(prepare_trace_panel, name="trace-panel")
datasets.add_command(prepare_temporal_trace_panel, name="temporal-trace-panel")
tune.add_command(compact_tuning, name="compact")


if __name__ == "__main__":
    main()
