"""Functional tests for repository Click commands."""

import runpy
from pathlib import Path
from typing import cast

import click
import pytest
from click.testing import CliRunner

from prefix_cache_evolve.problems.prefix_kv_cache.lab import main as lab_main
from prefix_cache_evolve.problems.prefix_kv_cache.runner import main as runner_main
from prefix_cache_evolve.tools.ablate_structured import main as ablate_main
from prefix_cache_evolve.tools.analyze_eviction import main as eviction_main
from prefix_cache_evolve.tools.analyze_reasoning_kv import main as reasoning_main
from prefix_cache_evolve.tools.analyze_regret import main as regret_main
from prefix_cache_evolve.tools.cli import main as tools_main
from prefix_cache_evolve.tools.tune_compact import main as tune_main

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
plot_main = cast(
    click.Command,
    runpy.run_path(str(_REPOSITORY_ROOT / "scripts/plot_prefix_kv_eval_trajectory.py"))["main"],
)
sweep_main = cast(
    click.Command,
    runpy.run_path(str(_REPOSITORY_ROOT / "scripts/sweep_prefix_kv_baselines.py"))["main"],
)

_COMMANDS: tuple[tuple[str, click.Command], ...] = (
    ("runner", runner_main),
    ("lab", lab_main),
    ("ablate", ablate_main),
    ("eviction", eviction_main),
    ("reasoning", reasoning_main),
    ("regret", regret_main),
    ("tools", tools_main),
    ("tune", tune_main),
    ("plot", plot_main),
    ("sweep", sweep_main),
)


@pytest.mark.parametrize(
    "command",
    [command for _, command in _COMMANDS],
    ids=[name for name, _ in _COMMANDS],
)
def test_click_commands_expose_help(command: click.Command) -> None:
    result = CliRunner().invoke(command, ["--help"])

    assert result.exit_code == 0
    assert "Options:" in result.output
    assert "--help" in result.output


def test_runner_show_config_does_not_start_evolution() -> None:
    result = CliRunner().invoke(runner_main, ["--show-config", "--quick"])

    assert result.exit_code == 0
    assert '"iterations": 25' in result.output
    assert '"search_seed"' in result.output


@pytest.mark.parametrize(
    "arguments",
    (
        ["analyze", "--help"],
        ["analyze", "eviction", "--help"],
        ["analyze", "rediscovery", "--help"],
        ["analyze", "regret", "--help"],
        ["analyze", "reasoning-kv", "--help"],
        ["ablate", "structured", "--help"],
        ["tune", "compact", "--help"],
    ),
)
def test_consolidated_tools_expose_subcommand_help(arguments: list[str]) -> None:
    result = CliRunner().invoke(tools_main, arguments)

    assert result.exit_code == 0
    assert "Options:" in result.output


@pytest.mark.parametrize("mode", ("--shadow-price", "--causal-components"))
def test_mechanism_diagnostics_default_to_json_only(mode: str) -> None:
    with CliRunner().isolated_filesystem():
        result = CliRunner().invoke(
            regret_main,
            [
                mode,
                "--config",
                str(_REPOSITORY_ROOT / "configs/prefix_kv_cache.yaml"),
                "--request-count",
                "4",
                "--seeds",
                "3",
                "--splits",
                "validation",
                "--workloads",
                "priority_burst_recovery",
                "--capacity-blocks",
                "8",
            ],
        )

        assert result.exit_code == 0
        output_paths = [line for line in result.output.splitlines() if line.strip()]
        assert len(output_paths) == 1
        assert output_paths[0].endswith(".json")
        assert not Path("docs/results").exists()


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (
            ["--model", "openai/test", "--primary-model", "openai/other"],
            "--model cannot be combined",
        ),
        (
            ["--sensitivity-report"],
            "--sensitivity-report requires --candidate-program",
        ),
    ),
)
def test_runner_rejects_invalid_option_combinations(
    arguments: list[str],
    message: str,
) -> None:
    result = CliRunner().invoke(runner_main, arguments)

    assert result.exit_code != 0
    assert message in result.output
