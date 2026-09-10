"""Functional tests for repository Click commands."""

import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from prefix_cache_evolve.problems.prefix_kv_cache.lab import main as lab_main
from prefix_cache_evolve.problems.prefix_kv_cache.runner import main as runner_main
from prefix_cache_evolve.tools.analyze_regret import main as regret_main
from prefix_cache_evolve.tools.cli import main as tools_main

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("command", (lab_main, tools_main))
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
        ["analyze", "policy-costs", "--help"],
        ["ablate", "structured", "--help"],
        ["datasets", "wildchat", "--help"],
        ["tune", "compact", "--help"],
    ),
)
def test_consolidated_tools_expose_subcommand_help(arguments: list[str]) -> None:
    result = CliRunner().invoke(tools_main, arguments)

    assert result.exit_code == 0
    assert "Options:" in result.output


def test_incumbent_validation_command_passes() -> None:
    result = CliRunner().invoke(tools_main, ["incumbents", "validate"])

    assert result.exit_code == 0
    assert result.output == "validated_incumbents=4\n"


def test_incumbent_list_distinguishes_history_from_current_assignments() -> None:
    result = CliRunner().invoke(tools_main, ["incumbents", "list"])

    assert result.exit_code == 0
    records = {record["id"]: record for record in json.loads(result.output)}
    assert records["historical_compact_20260607"]["current_roles"] == []
    assert records["production_16tok_20260609"]["status"] == "promoted"
    assert records["production_16tok_20260609"]["current_roles"] == ["production"]


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
