"""Compare existing prefix-cache policies under shared reasoning decode KV pressure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prefix_cache_evolve.evaluators.baselines import BASELINE_REGISTRY, REPORTING_BASELINES
from prefix_cache_evolve.evaluators.complexity import scoring_fn_complexity
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluationResult,
    PrefixKVCacheEvaluator,
)
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.pressure_aware_incumbent import (
    build_candidate as build_incumbent,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_INCUMBENT_PATH = (
    _REPOSITORY_ROOT
    / "src/prefix_cache_evolve/problems/prefix_kv_cache/pressure_aware_incumbent.py"
)
_WORKLOADS = (
    "concurrent_long_generation",
    "reasoning_burst",
    "reasoning_burst_shifted",
    "stochastic_serving_mix",
)
_SUMMARY_METRICS = (
    "token_hit_rate",
    "request_token_hit_rate_p10",
    "cache_churn_per_1k",
    "prefix_kv_occupancy_mean",
    "decode_kv_occupancy_mean",
    "decode_kv_occupancy_peak",
    "decode_kv_blocks_requested",
    "decode_kv_blocks_allocated",
    "decode_kv_allocation_failure_rate",
    "decode_pressure_eviction_count",
    "decode_pressure_eviction_rate",
)


def _raw_score(result: EvaluationResult) -> float:
    return result.combined_score + result.score_breakdown.get("complexity_cost", 0.0)


def _summarize_result(result: EvaluationResult) -> dict[str, object]:
    metrics = result.split_metrics["validation"]
    summary: dict[str, object] = {
        "raw_score": _raw_score(result),
        "charged_score": result.combined_score,
        "complexity_cost": result.score_breakdown.get("complexity_cost", 0.0),
    }
    summary.update({name: metrics[name] for name in _SUMMARY_METRICS})
    summary["workloads"] = {
        key.removeprefix("validation/"): {
            name: value[name]
            for name in (
                "token_hit_rate",
                "cache_churn_per_1k",
                "decode_kv_occupancy_mean",
                "decode_kv_allocation_failure_rate",
                "decode_pressure_eviction_count",
            )
        }
        for key, value in result.workload_metrics.items()
        if key.startswith("validation/")
    }
    return summary


def run_analysis(
    config_path: Path,
    *,
    request_count: int | None = None,
    seeds: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Run the policy panel under prefix-only and shared-KV capacity models."""

    base = load_evaluator_config(config_path)
    if request_count is not None:
        base = base.with_updates(request_count=request_count)
    if seeds is not None:
        base = base.with_updates(seeds=seeds)
    base = base.with_updates(
        validation_families=_WORKLOADS,
        family_request_multipliers={},
    )
    incumbent_complexity = scoring_fn_complexity(
        _INCUMBENT_PATH.read_text(encoding="utf-8"),
        form_aware=base.form_aware_complexity,
    )
    policies = {"incumbent": build_incumbent, **REPORTING_BASELINES}
    modes: dict[str, dict[str, object]] = {}
    for mode in ("prefix_only", "shared"):
        mode_config = base.with_updates(kv_capacity_mode=mode)
        policy_results = {}
        for name, factory in policies.items():
            evaluator = PrefixKVCacheEvaluator(
                mode_config,
                splits=("validation",),
                expose_future_reuse=(
                    name != "incumbent" and BASELINE_REGISTRY.requires_future_reuse(name)
                ),
            )
            complexity = incumbent_complexity if name == "incumbent" else 0
            policy_results[name] = _summarize_result(
                evaluator(factory, scoring_fn_complexity=complexity)
            )
        ranked = sorted(
            policy_results,
            key=lambda name: (-float(policy_results[name]["raw_score"]), name),
        )
        for rank, name in enumerate(ranked, start=1):
            policy_results[name]["raw_rank"] = rank
        modes[mode] = policy_results
    return {
        "schema": "prefix-kv-cache-reasoning-kv-analysis-v1",
        "config": str(config_path),
        "capacity_blocks": list(base.effective_capacity_blocks()),
        "block_size_tokens": base.block_size_tokens,
        "active_tokens_per_step": base.active_tokens_per_step,
        "request_count": base.request_count,
        "seeds": list(base.seeds),
        "workloads": list(_WORKLOADS),
        "incumbent_complexity": incumbent_complexity,
        "modes": modes,
    }


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    modes = payload["modes"]
    prefix_only = modes["prefix_only"]
    shared = modes["shared"]
    shared_ranked = sorted(shared, key=lambda name: shared[name]["raw_rank"])
    lines = [
        "# Reasoning Decode-KV Robustness",
        "",
        "This panel keeps the operative prefix-only verifier unchanged and replays existing",
        "algorithms with an opt-in shared-capacity model. In shared mode, generated decode KV",
        "grows over logical time, is non-evictable, and forces inactive prefix-leaf eviction.",
        "When pinned prefixes plus decode KV exhaust capacity, the simulator records failed",
        "decode-block allocations.",
        "",
        f"Panel: capacities `{payload['capacity_blocks']}`, block size "
        f"`{payload['block_size_tokens']}`, `{payload['request_count']}` requests per workload,",
        f"seeds `{payload['seeds']}`, and workloads `{payload['workloads']}`.",
        "",
        "Decode allocation failure is reported but is not yet a score term. Raw-score ranking",
        "therefore measures prefix-policy quality under pressure, not end-to-end serving",
        "feasibility.",
    ]
    for mode in ("prefix_only", "shared"):
        rows = modes[mode]
        ranked = sorted(rows, key=lambda name: rows[name]["raw_rank"])
        lines.extend(
            [
                "",
                f"## {mode.replace('_', ' ').title()}",
                "",
                "| Rank | Policy | Raw score | Token hit | Request p10 | Churn/1k | "
                "Prefix KV | Decode KV | Decode fail | Decode-pressure evictions |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in ranked:
            row = rows[name]
            lines.append(
                f"| {row['raw_rank']} | `{name}` | {row['raw_score']:.3f} | "
                f"{row['token_hit_rate']:.4f} | {row['request_token_hit_rate_p10']:.4f} | "
                f"{row['cache_churn_per_1k']:.1f} | {row['prefix_kv_occupancy_mean']:.1f} | "
                f"{row['decode_kv_occupancy_mean']:.1f} | "
                f"{row['decode_kv_allocation_failure_rate']:.1%} | "
                f"{row['decode_pressure_eviction_count']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Rank and Metric Shift",
            "",
            "| Policy | Prefix rank | Shared rank | Raw delta | Hit delta | Churn delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in shared_ranked:
        prefix_row = prefix_only[name]
        shared_row = shared[name]
        lines.append(
            f"| `{name}` | {prefix_row['raw_rank']} | {shared_row['raw_rank']} | "
            f"{shared_row['raw_score'] - prefix_row['raw_score']:+.3f} | "
            f"{shared_row['token_hit_rate'] - prefix_row['token_hit_rate']:+.4f} | "
            f"{shared_row['cache_churn_per_1k'] - prefix_row['cache_churn_per_1k']:+.1f} |"
        )
    top_shared = shared_ranked[0]
    top_deployable = next(
        name
        for name in shared_ranked
        if name not in {"future_reuse_heuristic", "oracle_future_reuse"}
    )
    incumbent = shared["incumbent"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"`{top_shared}` has the strongest raw prefix-policy score in shared mode;",
            f"`{top_deployable}` is the strongest deployable policy by raw behavior. The",
            f"incumbent ranks `{incumbent['raw_rank']}` with token hit "
            f"`{incumbent['token_hit_rate']:.4f}` and decode allocation failure "
            f"`{incumbent['decode_kv_allocation_failure_rate']:.1%}`.",
            f"Its charged score is `{incumbent['charged_score']:.3f}` after the incumbent-only",
            "complexity charge; baseline implementations are not complexity-charged in this",
            "report, so raw behavior is the meaningful policy comparison.",
            "",
            "The failure rate is the main systems result: these prefix-cache-sized capacities",
            "cannot sustain the synthetic reasoning bursts when prompt and decode KV share the",
            "same pool. Eviction-policy differences still change which reusable prefixes survive,",
            "but no prefix policy can recover capacity occupied by active decode state. A production",
            "extension should add scheduler actions such as admission control, preemption, or",
            "separate prefix/decode budgets before using this mode as an optimization objective.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prefix_kv_cache.yaml"),
    )
    parser.add_argument("--request-count", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/prefix_kv_cache_reasoning_kv_analysis.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/results/reasoning_kv_robustness.md"),
    )
    args = parser.parse_args()

    payload = run_analysis(
        args.config,
        request_count=args.request_count,
        seeds=tuple(args.seeds) if args.seeds else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(args.markdown, payload)
    print(args.output)
    print(args.markdown)


if __name__ == "__main__":
    main()
