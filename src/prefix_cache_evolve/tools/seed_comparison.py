"""Freeze paired seed experiments without starting model calls."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import click
import yaml

from prefix_cache_evolve.artifacts import file_sha256, write_json
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import build_workload_manifest
from prefix_cache_evolve.workflow.config import ConfigLoader

_POLICIES = Path(__file__).resolve().parents[1] / "problems/prefix_kv_cache/seeds"
_PROMPT = """Design a compact online admission and eviction policy for an LLM prefix KV-cache.
The supplied program is a starting condition. Preserve, simplify, or replace any part when
visible evaluator feedback supports the change. No particular architecture is required.

Expose build_candidate(capacity_blocks, block_size_tokens, seed=None) or candidate_factory
and return an object with score_admission(block, now), score_eviction(block, now), and
on_request_start(request, now), on_cache_hit(block, request, now), on_cache_miss(block,
request, now). All three callbacks must exist; they may be no-ops. No other callback fires.
The simulator owns cache state. Admission requires a positive score. Eviction chooses the
highest score among inactive resident leaves. Block depths are one-based. now is a logical
arrival step and may repeat. Use prefix_hash or block_id as a stable key, not object identity.

Available block fields: block_id, prefix_hash, parent_hash, depth, start_token, end_token,
token_count, tenant_id, created_at, last_accessed_at, hit_count, descendant_count,
active_ref_count, estimated_recompute_cost, prev_last_accessed_at, last_access_gap,
access_gap_mean, access_gap_var, subtree_hit_rate, subtree_active_ref_count.
Future-reuse fields are unavailable and must not be referenced.
Available request fields: request_id, tenant_id, session_id, prompt_length, priority,
request_type, prompt_tokens, predicted_output_length, recent_admission_pressure,
recent_miss_rate. request_id is opaque, request_type is normalized, and prompt_tokens is
empty. Do not use these three fields as policy signals. Missing session or tenant identity
does not indicate recurrence. Block objects have no session_id. Do not guess extra fields.

Optional primitives in prefix_cache_evolve.problems.prefix_kv_cache.primitives are
MultiTimescaleDecay (observe, observe_vector, values, combine), decay_vector, and
threshold_excess. Use bounded state only when measurements justify it. Candidate imports
are limited to math and that primitives module. Keep imports at module scope. Do not use
dynamic execution, introspection, decorators, broad exception handlers, or guessed fallbacks.

Maximize the supplied raw_before_complexity search score, which retains cache-quality
costs while removing the source-size charge. Use visible feedback to improve admission
utility, hit rate and tail service while controlling churn, underfill, waste and fairness.
Do not hard-code workloads, identifiers, timestamps, capacities, block sizes or coefficients
from an unavailable policy. Evaluation allows 750 effective AST nodes; final selection
requires at most 650. Prefer complete proposals below 550 nodes during initialization.
For an over-650 parent, simplify before adding behavior. Hidden and probe panels are
unavailable during search. Produce complete valid Python using only the documented API.
"""
_DIVERSITY_PROMPT = """{problem_description}

Required signature: {function_signature}
Existing evaluated programs:
{existing_seeds}

Propose one compact alternative using a different online decision rule. Keep the complete
policy below 550 effective AST nodes. Output only Python in a fenced python block.
"""
_RUNTIME_PACKAGES = ("levi", "litellm", "numpy", "scikit-learn", "pydantic", "pyyaml")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _RUNTIME_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _shared_config(config_path: Path, evaluations: int, dollars: float, seconds: int) -> dict:
    """Keep the supplied evaluator panel while replacing seed-specific search advice."""
    original = ConfigLoader().load(config_path)
    evaluator = load_evaluator_config(config_path)
    if evaluator.candidate_policy_surface != "full":
        raise ValueError("seed comparison requires the joint admission/eviction surface")
    evaluator = evaluator.with_updates(
        search_score_mode="raw_before_complexity",
        search_guidance_families=(),
        max_candidate_complexity=750,
        promotion_max_candidate_complexity=650,
        probe_families=(),
        hidden_families=(),
        trace_workloads=tuple(
            trace for trace in evaluator.trace_workloads if trace.split in {"train", "validation"}
        ),
    )
    # Freeze resolved models; reject an ambient override rather than silently changing arms.
    if os.environ.get("LEVI_MODEL"):
        raise ValueError("unset LEVI_MODEL before freezing a seed comparison")
    llm = dict(original.raw.get("llm", {}))
    llm.update(primary_model=original.mutation_model, secondary_model=original.paradigm_model)
    llm.setdefault("max_tokens", 6000)
    return {
        "max_iterations": evaluations,
        "budget_dollars": dollars,
        "budget_seconds": seconds,
        "llm": llm,
        "function_signature": (
            "def build_candidate(capacity_blocks, block_size_tokens, seed=None):"
        ),
        "prompt": {"system_message": _PROMPT},
        "problem": {
            "description": "Optimize an online prefix-cache policy using visible measurements.",
            "settings": evaluator.model_dump(mode="json"),
        },
        "search": {"seed": 0},
        "evaluator": {
            "timeout": evaluator.timeout_s,
            "cascade_evaluation": False,
            "parallel_evaluations": 1,
        },
        "init": {
            "enabled": True,
            "n_diverse_seeds": 2,
            "n_variants_per_seed": 1,
            "diversity_model": original.paradigm_model,
            "variant_models": [original.mutation_model],
            "diversity_prompt": _DIVERSITY_PROMPT,
            "diversity_llm_kwargs": {"max_tokens": 6000},
        },
        "cvt": {"n_centroids": 8, "data_driven_centroids": False},
        "behavior": original.behavior,
        "pipeline": {
            "n_llm_workers": 1,
            "n_parents": 1,
            "n_inspirations": 0,
            "output_mode": "diff",
        },
        "meta_advice": {"enabled": False},
        "punctuated_equilibrium": {"enabled": False},
    }


def prepare_campaign(
    config_path: Path,
    output: Path,
    *,
    minimal: Path,
    strong: Path,
    search_seeds: tuple[int, ...],
    evaluations: int,
    dollars_per_run: float,
    seconds_per_run: int,
    order_seed: int = 20260911,
) -> dict[str, Any]:
    """Write an immutable plan with matched configurations and hashed inputs.

    This does not evaluate policies or contact a model. Fresh holdouts are a
    separate preregistration; search configurations exclude all hidden/probe inputs.
    """
    if len(search_seeds) < 2 or len(set(search_seeds)) != len(search_seeds):
        raise ValueError("at least two distinct search seeds are required")
    if any(seed < 0 or seed >= 2**32 for seed in search_seeds):
        raise ValueError("search seeds must fit NumPy's unsigned 32-bit seed range")
    if (
        evaluations < 16
        or not math.isfinite(dollars_per_run)
        or dollars_per_run <= 0
        or seconds_per_run <= 0
    ):
        raise ValueError("require at least 16 evaluations and positive cost/time guardrails")
    sources = {"minimal": minimal.read_bytes(), "strong": strong.read_bytes()}
    if sources["minimal"] == sources["strong"]:
        raise ValueError("minimal and strong seeds must have different source")
    shared = _shared_config(config_path, evaluations, dollars_per_run, seconds_per_run)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    common_path = output / "common.yaml"
    common_path.write_text(yaml.safe_dump(shared, sort_keys=False), encoding="utf-8")
    evaluator = load_evaluator_config(common_path)
    panel = build_workload_manifest(evaluator, splits=("train", "validation"))
    write_json(output / "visible_panel.json", panel)
    if not panel["streams"]:
        raise ValueError("the visible search panel is empty")
    frozen_files = {
        "common.yaml": file_sha256(common_path),
        "visible_panel.json": file_sha256(output / "visible_panel.json"),
    }
    trace_files = {trace.path: file_sha256(Path(trace.path)) for trace in evaluator.trace_workloads}
    rng = random.Random(order_seed)
    runs = []
    for pair, seed in enumerate(search_seeds, start=1):
        arms = ["minimal", "strong"]
        rng.shuffle(arms)
        for arm in arms:
            run_id = f"pair-{pair:02d}-{arm}"
            run_dir = output / run_id
            run_dir.mkdir()
            (run_dir / "seed_program.py").write_bytes(sources[arm])
            config = copy.deepcopy(shared)
            config["search"] = {"seed": seed, "seed_program": "seed_program.py"}
            config["output_dir"] = str(run_dir / "search")
            run_config = run_dir / "evolution.yaml"
            run_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            for name in ("seed_program.py", "evolution.yaml"):
                frozen_files[f"{run_id}/{name}"] = file_sha256(run_dir / name)
            runs.append({"id": run_id, "pair": pair, "arm": arm, "search_seed": seed})
    package_root = Path(__file__).resolve().parents[1]
    runtime_files = {str(path): file_sha256(path) for path in sorted(package_root.rglob("*.py"))}
    lock = package_root.parents[1] / "uv.lock"
    if lock.is_file():
        runtime_files[str(lock)] = file_sha256(lock)
    manifest = {
        "schema": "prefix-cache-seed-comparison-v1",
        "status": "prepared_not_run",
        "runs": runs,
        "order_seed": order_seed,
        "primary_unit": "one independent search; paired by search RNG seed",
        "evaluation_budget_per_run": evaluations,
        "dollar_guardrail_per_run": dollars_per_run,
        "planned_dollar_guardrails_total": 2 * len(search_seeds) * dollars_per_run,
        "seconds_guardrail_per_run": seconds_per_run,
        "shared_config_sha256": _digest(shared),
        "prompt_sha256": hashlib.sha256(
            ConfigLoader().load(common_path).problem_description.encode()
        ).hexdigest(),
        "visible_panel_sha256": panel["panel_sha256"],
        "frozen_files": frozen_files,
        "trace_files": trace_files,
        "runtime_files": runtime_files,
        "python": sys.version,
        "packages": _package_versions(),
        "selection_rule": "highest visible raw score among valid archived candidates <=650 AST "
        "nodes; ties: smaller AST, then lexicographic source SHA-256; include seed",
        "launch_gate": "offline tests, both seed replays, excluded paired initialization pilot, "
        "fresh final holdout preregistration, and model-spend authorization",
        "limitations": [
            "No searches or policy evaluations have been performed by preparation.",
            "Provider seeds and asynchronous ordering do not guarantee identical randomness.",
            "Dollar guardrails are accounting limits; an in-flight model call can overshoot.",
            "This experiment fixes CVT cells and disables meta-advice and periodic paradigms.",
            "Results apply to these two sources and this search protocol, not all seed quality.",
        ],
    }
    write_json(output / "campaign.json", manifest)
    return manifest


def validate_campaign(output: Path) -> dict[str, Any]:
    """Fail if any frozen search, trace or implementation input has changed."""
    manifest = json.loads((output / "campaign.json").read_text(encoding="utf-8"))
    for group in ("frozen_files", "trace_files", "runtime_files"):
        for name, expected in manifest[group].items():
            path = output / name if group == "frozen_files" else Path(name)
            if file_sha256(path) != expected:
                raise ValueError(f"frozen campaign input changed: {path}")
    if (
        os.environ.get("LEVI_MODEL")
        or sys.version != manifest["python"]
        or _package_versions() != manifest["packages"]
    ):
        raise ValueError("campaign runtime differs: Python/packages or LEVI_MODEL override")
    return manifest


@click.group()
def main() -> None:
    """Prepare and validate controlled, repeated seed comparisons (no model calls)."""


@main.command("prepare")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option(
    "--minimal", type=click.Path(path_type=Path, exists=True), default=_POLICIES / "minimal_lru.py"
)
@click.option(
    "--strong",
    type=click.Path(path_type=Path, exists=True),
    default=_POLICIES / "qwen_mixed_20260909.py",
)
@click.option(
    "--search-seed",
    "search_seeds",
    multiple=True,
    type=click.IntRange(0, 2**32 - 1),
    default=(104729, 130363, 155921, 181081, 206369),
)
@click.option("--evaluations", type=click.IntRange(min=16), default=64, show_default=True)
@click.option("--dollars-per-run", type=click.FloatRange(min=0, min_open=True), required=True)
@click.option("--seconds-per-run", type=click.IntRange(min=1), default=14400, show_default=True)
def prepare(**kwargs: Any) -> None:
    """Freeze five paired repetitions by default, with a shared neutral prompt."""
    try:
        manifest = prepare_campaign(**kwargs)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"prepared_runs={len(manifest['runs'])}; model_calls=0")


@main.command("validate")
@click.argument("output", type=click.Path(path_type=Path, exists=True))
def validate(output: Path) -> None:
    """Check frozen input hashes before a separately authorized campaign launch."""
    try:
        manifest = validate_campaign(output)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"validated_runs={len(manifest['runs'])}")


if __name__ == "__main__":
    main()
