"""Command dispatch for the prefix KV-cache runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import click


def dispatch(args: SimpleNamespace) -> None:
    """Dispatch one validated runner command."""
    from prefix_cache_evolve.problems.prefix_kv_cache import runner

    if args.model and (args.primary_model or args.secondary_model):
        raise click.UsageError(
            "--model cannot be combined with --primary-model or --secondary-model"
        )
    try:
        capacity_sweep_blocks = runner._parse_capacity_sweep(args.capacity_sweep_blocks)
        block_size_sweep = runner._parse_block_size_sweep(args.block_size_sweep)
    except ValueError as error:
        raise click.BadParameter(str(error)) from error
    quick = args.quick or args.workload_preset == "small"
    if args.show_config:
        runner._show_resolved_config(
            iterations=args.iterations,
            config_file=args.config,
            quick=quick,
            model=args.model,
            primary_model=args.primary_model,
            secondary_model=args.secondary_model,
            search_seed=args.search_seed,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
        )
        return
    if args.calibrate_trace is not None:
        runner.calibrate_trace_report(
            args.calibrate_trace,
            output_path=args.trace_output,
            arrival_bucket_ms=args.trace_arrival_bucket_ms,
            request_limit=args.trace_request_limit,
        )
        return
    if args.replay_trace is not None:
        runner.replay_trace_report(
            args.replay_trace,
            output_path=args.trace_output,
            candidate_program=args.candidate_program,
            arrival_bucket_ms=args.trace_arrival_bucket_ms,
            request_limit=args.trace_request_limit,
            config_file=args.config,
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
        )
        return
    if args.workload_manifest:
        runner.write_workload_manifest_report(
            args.workload_manifest_output,
            reference_path=args.workload_manifest_reference,
            quick=quick,
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
            config_file=args.config,
        )
        return
    if args.sensitivity_report:
        if args.candidate_program is None:
            raise click.UsageError("--sensitivity-report requires --candidate-program")
        runner.write_score_weight_sensitivity_report(
            args.sensitivity_output,
            candidate_program=args.candidate_program,
            config_file=args.config,
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
        )
        return
    if args.block_size_report:
        runner.write_block_size_robustness_report(
            args.block_size_output,
            candidate_program=args.candidate_program,
            quick=quick,
            config_file=args.config,
            block_sizes=block_size_sweep,
        )
        return
    if args.baseline_report:
        runner.compare_baselines(
            quick=quick,
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
            candidate_program=args.candidate_program,
            config_file=args.config,
        )
        return
    if args.hidden_report:
        runner.hidden_report(
            quick=quick,
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
            candidate_program=args.candidate_program,
            config_file=args.config,
        )
        return
    if args.probe_report:
        runner.probe_report(
            output_path=args.probe_output,
            quick=quick,
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
            candidate_program=args.candidate_program,
            config_file=args.config,
        )
        return
    if args.plot_report:
        paths = runner.write_baseline_plots(
            Path(args.plot_output),
            quick=quick,
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
            config_file=args.config,
        )
        for path in paths:
            click.echo(path)
        return
    runner.demo_run_evolution(
        iterations=args.iterations,
        config_file=args.config,
        quick=quick,
        seed_program=args.seed_program,
        artifact_output=None if args.no_save_artifacts else Path(args.artifact_output),
        model=args.model,
        primary_model=args.primary_model,
        secondary_model=args.secondary_model,
        search_seed=args.search_seed,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
    )
