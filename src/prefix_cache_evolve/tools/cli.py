"""Consolidated command-line interface for repository analysis tools."""

from __future__ import annotations

import click

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


analyze.add_command(eviction_analysis, name="eviction")
analyze.add_command(rediscovery_analysis, name="rediscovery")
analyze.add_command(regret_analysis, name="regret")
analyze.add_command(reasoning_kv_analysis, name="reasoning-kv")
ablate.add_command(structured_ablation, name="structured")
tune.add_command(compact_tuning, name="compact")


if __name__ == "__main__":
    main()
