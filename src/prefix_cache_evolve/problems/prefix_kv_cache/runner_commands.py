"""Typed command selection and dispatch for the prefix KV-cache runner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import click


class RunnerAction(Enum):
    """One mutually exclusive runner operation."""

    EVOLVE = "evolve"
    SHOW_CONFIG = "show-config"
    CALIBRATE_TRACE = "calibrate-trace"
    REPLAY_TRACE = "replay-trace"
    WORKLOAD_MANIFEST = "workload-manifest"
    SENSITIVITY_REPORT = "sensitivity-report"
    BLOCK_SIZE_REPORT = "block-size-report"
    BASELINE_REPORT = "baseline-report"
    HIDDEN_REPORT = "hidden-report"
    PROBE_REPORT = "probe-report"
    PLOT_REPORT = "plot-report"


@dataclass(frozen=True)
class RunnerOptions:
    """Validated values accepted by the compatibility runner command."""

    iterations: int
    quick: bool
    workload_preset: str
    capacity_blocks: int | None
    capacity_sweep_blocks: str
    block_size_tokens: int | None
    baseline_report: bool
    candidate_program: Path | None
    hidden_report: bool
    probe_report: bool
    probe_output: Path
    plot_report: bool
    plot_output: Path
    artifact_output: Path
    seed_program: Path | None
    no_save_artifacts: bool
    config: str
    model: str | None
    primary_model: str | None
    secondary_model: str | None
    search_seed: int | None
    api_base: str | None
    api_key_env: str | None
    show_config: bool
    calibrate_trace: Path | None
    replay_trace: Path | None
    trace_output: Path
    trace_arrival_bucket_ms: int
    trace_request_limit: int | None
    workload_manifest: bool
    workload_manifest_output: Path
    workload_manifest_reference: Path | None
    sensitivity_report: bool
    sensitivity_output: Path
    block_size_report: bool
    block_size_sweep: str
    block_size_output: Path

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RunnerOptions:
        """Construct typed runner options from Click keyword arguments."""
        return cls(**dict(values))

    def action(self) -> RunnerAction:
        """Return the selected action and reject ambiguous invocations."""
        selected = [
            (action, option)
            for action, option, enabled in (
                (RunnerAction.SHOW_CONFIG, "--show-config", self.show_config),
                (
                    RunnerAction.CALIBRATE_TRACE,
                    "--calibrate-trace",
                    self.calibrate_trace is not None,
                ),
                (
                    RunnerAction.REPLAY_TRACE,
                    "--replay-trace",
                    self.replay_trace is not None,
                ),
                (
                    RunnerAction.WORKLOAD_MANIFEST,
                    "--workload-manifest",
                    self.workload_manifest,
                ),
                (
                    RunnerAction.SENSITIVITY_REPORT,
                    "--sensitivity-report",
                    self.sensitivity_report,
                ),
                (
                    RunnerAction.BLOCK_SIZE_REPORT,
                    "--block-size-report",
                    self.block_size_report,
                ),
                (RunnerAction.BASELINE_REPORT, "--baseline-report", self.baseline_report),
                (RunnerAction.HIDDEN_REPORT, "--hidden-report", self.hidden_report),
                (RunnerAction.PROBE_REPORT, "--probe-report", self.probe_report),
                (RunnerAction.PLOT_REPORT, "--plot-report", self.plot_report),
            )
            if enabled
        ]
        if len(selected) > 1:
            options = ", ".join(option for _, option in selected)
            raise click.UsageError(f"runner actions are mutually exclusive: {options}")
        return selected[0][0] if selected else RunnerAction.EVOLVE

    def validate(self) -> RunnerAction:
        """Validate cross-option constraints and return the selected action."""
        action = self.action()
        if self.model and (self.primary_model or self.secondary_model):
            raise click.UsageError(
                "--model cannot be combined with --primary-model or --secondary-model"
            )
        if action is RunnerAction.SENSITIVITY_REPORT and self.candidate_program is None:
            raise click.UsageError("--sensitivity-report requires --candidate-program")
        return action


def dispatch(options: RunnerOptions) -> None:
    """Dispatch one validated runner command."""
    from prefix_cache_evolve.problems.prefix_kv_cache import runner

    action = options.validate()
    try:
        capacity_sweep_blocks = runner._parse_capacity_sweep(options.capacity_sweep_blocks)
        block_size_sweep = runner._parse_block_size_sweep(options.block_size_sweep)
    except ValueError as error:
        raise click.BadParameter(str(error)) from error
    quick = options.quick or options.workload_preset == "small"

    if action is RunnerAction.SHOW_CONFIG:
        runner._show_resolved_config(
            iterations=options.iterations,
            config_file=options.config,
            quick=quick,
            model=options.model,
            primary_model=options.primary_model,
            secondary_model=options.secondary_model,
            search_seed=options.search_seed,
            api_base=options.api_base,
            api_key_env=options.api_key_env,
        )
    elif action is RunnerAction.CALIBRATE_TRACE:
        assert options.calibrate_trace is not None
        runner.calibrate_trace_report(
            options.calibrate_trace,
            output_path=options.trace_output,
            arrival_bucket_ms=options.trace_arrival_bucket_ms,
            request_limit=options.trace_request_limit,
        )
    elif action is RunnerAction.REPLAY_TRACE:
        assert options.replay_trace is not None
        runner.replay_trace_report(
            options.replay_trace,
            output_path=options.trace_output,
            candidate_program=options.candidate_program,
            arrival_bucket_ms=options.trace_arrival_bucket_ms,
            request_limit=options.trace_request_limit,
            config_file=options.config,
            capacity_blocks=options.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=options.block_size_tokens,
        )
    elif action is RunnerAction.WORKLOAD_MANIFEST:
        runner.write_workload_manifest_report(
            options.workload_manifest_output,
            reference_path=options.workload_manifest_reference,
            quick=quick,
            capacity_blocks=options.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=options.block_size_tokens,
            config_file=options.config,
        )
    elif action is RunnerAction.SENSITIVITY_REPORT:
        assert options.candidate_program is not None
        runner.write_score_weight_sensitivity_report(
            options.sensitivity_output,
            candidate_program=options.candidate_program,
            config_file=options.config,
            capacity_blocks=options.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=options.block_size_tokens,
        )
    elif action is RunnerAction.BLOCK_SIZE_REPORT:
        runner.write_block_size_robustness_report(
            options.block_size_output,
            candidate_program=options.candidate_program,
            quick=quick,
            config_file=options.config,
            block_sizes=block_size_sweep,
        )
    elif action is RunnerAction.BASELINE_REPORT:
        runner.compare_baselines(
            quick=quick,
            capacity_blocks=options.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=options.block_size_tokens,
            candidate_program=options.candidate_program,
            config_file=options.config,
        )
    elif action is RunnerAction.HIDDEN_REPORT:
        runner.hidden_report(
            quick=quick,
            capacity_blocks=options.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=options.block_size_tokens,
            candidate_program=options.candidate_program,
            config_file=options.config,
        )
    elif action is RunnerAction.PROBE_REPORT:
        runner.probe_report(
            output_path=options.probe_output,
            quick=quick,
            capacity_blocks=options.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=options.block_size_tokens,
            candidate_program=options.candidate_program,
            config_file=options.config,
        )
    elif action is RunnerAction.PLOT_REPORT:
        paths = runner.write_baseline_plots(
            options.plot_output,
            quick=quick,
            capacity_blocks=options.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=options.block_size_tokens,
            config_file=options.config,
        )
        for path in paths:
            click.echo(path)
    else:
        runner.demo_run_evolution(
            iterations=options.iterations,
            config_file=options.config,
            quick=quick,
            seed_program=options.seed_program,
            artifact_output=None if options.no_save_artifacts else options.artifact_output,
            model=options.model,
            primary_model=options.primary_model,
            secondary_model=options.secondary_model,
            search_seed=options.search_seed,
            api_base=options.api_base,
            api_key_env=options.api_key_env,
        )
