"""Baseline comparison report rendering."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Protocol

from prefix_cache_evolve.evaluators.baselines import BASELINE_REGISTRY
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluationResult,
    EvaluatorConfig,
)

QUICK_REPORT_WARNING = (
    "SMOKE-ONLY: `--quick` uses `request_count=36` and one seed. "
    "Do not use this table for policy ranking decisions; rerun without `--quick`."
)


class BaselineMetadata(Protocol):
    """Minimal baseline metadata needed by report rendering."""

    def group(self, name: str) -> str: ...


def baseline_group(name: str, metadata: BaselineMetadata = BASELINE_REGISTRY) -> str:
    """Return the report group for a baseline or candidate."""

    return metadata.group(name)


def baseline_report_headline(
    ranked: list[tuple[str, EvaluationResult]],
    metadata: BaselineMetadata = BASELINE_REGISTRY,
) -> str:
    """Summarize candidate rank without mixing deployable and oracle claims."""

    names = [name for name, _ in ranked]
    if "candidate" not in names:
        return "Reporting baselines ranked by combined score."
    scores = {name: result.combined_score for name, result in ranked}
    candidate_score = scores["candidate"]
    deployable_scores = [
        score
        for name, score in scores.items()
        if name != "candidate" and metadata.group(name) == "deployable"
    ]
    reporting_scores = {
        name: score for name, score in scores.items() if metadata.group(name) != "deployable"
    }
    clears_deployable = not deployable_scores or candidate_score > max(deployable_scores)
    if not clears_deployable:
        return "The candidate ranking is shown against deployable and reporting-only baselines."

    above = [
        name
        for name in names
        if name in reporting_scores and reporting_scores[name] > candidate_score
    ]
    below = [
        name
        for name in names
        if name in reporting_scores and reporting_scores[name] < candidate_score
    ]
    headline = "The candidate clears the deployable credibility baselines in this capacity sweep."
    if above:
        headline += " It trails " + _format_policy_names(above) + "."
    if below:
        headline += " It beats " + _format_policy_names(below) + "."
    return headline


def write_baseline_comparison_report(
    path: Path,
    results: dict[str, EvaluationResult],
    *,
    candidate_path: Path,
    command: str,
    quick: bool,
    config: EvaluatorConfig,
    metadata: BaselineMetadata = BASELINE_REGISTRY,
) -> Path:
    """Write a Markdown comparison of the candidate and reporting baselines."""

    ranked = sorted(results.items(), key=lambda item: item[1].combined_score, reverse=True)
    lines = [
        "# Prefix KV-Cache Best Program Baseline Comparison",
        "",
        f"Candidate: `{candidate_path}`",
        "",
        "Command:",
        "",
        "```bash",
        command,
        "```",
        "",
    ]
    if quick:
        lines.extend([f"> **{QUICK_REPORT_WARNING}**", ""])
    capacities = config.effective_capacity_blocks()
    capacity_headers = "".join(f" Capacity {capacity} token hit |" for capacity in capacities)
    lines.extend(
        [
            "## Headline",
            "",
            (
                "Smoke-only output; run the full panel before comparing policy rank."
                if quick
                else baseline_report_headline(ranked, metadata)
            ),
            "",
            (
                "| Rank | Policy | Group | Combined score |"
                f"{capacity_headers} Worst-quarter hit | Request p10 hit | "
                "Token-wtd admission waste | Admission token utility | "
                "Avoidable eviction | Priority-burst weighted hit | "
                "Priority-noise token hit | Policy underfill | Churn per 1k |"
            ),
            "|---:|---|---|---:|" + "---:|" * (len(capacities) + 9),
        ]
    )
    lines.extend(_summary_rows(ranked, metadata, capacities))
    lines.extend(_workload_detail(ranked, results, split="validation"))
    if _split_workloads(results, "probe"):
        lines.extend(_workload_detail(ranked, results, split="probe"))
    lines.extend(_report_notes(results, config, quick=quick))

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_baseline_plot_files(
    output_dir: Path,
    results: dict[str, EvaluationResult],
) -> tuple[Path, ...]:
    """Render baseline comparison SVG files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "baseline_combined_scores.svg",
        output_dir / "validation_token_hit_heatmap.svg",
        output_dir / "token_vs_block_hit.svg",
    )
    paths[0].write_text(_combined_score_svg(results), encoding="utf-8")
    paths[1].write_text(_validation_heatmap_svg(results), encoding="utf-8")
    paths[2].write_text(_token_vs_block_svg(results), encoding="utf-8")
    return paths


def _summary_rows(
    ranked: list[tuple[str, EvaluationResult]],
    metadata: BaselineMetadata,
    capacities: tuple[int, ...],
) -> list[str]:
    rows = []
    for rank, (name, result) in enumerate(ranked, start=1):
        capacity_cells = "".join(
            f"{float(result.capacity_metrics.get(f'capacity_{capacity}', {}).get('token_hit_rate', 0.0)):.3f} | "
            for capacity in capacities
        )
        priority = result.workload_metrics["validation/priority_burst_recovery"]
        priority_noise = result.workload_metrics["validation/priority_one_off_noise"]
        validation = result.split_metrics["validation"]
        rows.append(
            f"| {rank} | `{name}` | {metadata.group(name)} | "
            f"{result.combined_score:.3f} | "
            f"{capacity_cells}"
            f"{float(validation['worst_quarter_token_hit_rate']):.3f} | "
            f"{float(validation['request_token_hit_rate_p10']):.3f} | "
            f"{float(validation['wasted_admission_token_rate']):.3f} | "
            f"{float(validation['admission_token_utility']):.3f} | "
            f"{float(validation['avoidable_eviction_rate']):.3f} | "
            f"{float(priority['priority_weighted_token_hit_rate']):.3f} | "
            f"{float(priority_noise['token_hit_rate']):.3f} | "
            f"{float(validation['policy_underfill_rate']):.3f} | "
            f"{float(validation['cache_churn_per_1k']):.1f} |"
        )
    return rows


def _workload_detail(
    ranked: list[tuple[str, EvaluationResult]],
    results: dict[str, EvaluationResult],
    *,
    split: str,
) -> list[str]:
    workloads = _split_workloads(results, split)
    if split == "validation":
        title = "## Validation Workload Detail"
        suffix = "Validation"
        introduction: list[str] = []
    else:
        title = "## Held-Out Structure-Generalization Probe"
        suffix = "Probe"
        introduction = [
            (
                "These recurrence-heavy families are evaluated and reported but "
                "excluded from the candidate-selection combined score."
            ),
            "",
        ]
    header = "| Policy | " + " | ".join(
        f"{workload.split('/', 1)[1]} token hit" for workload in workloads
    )
    header += f" | {suffix} block hit | {suffix} churn per 1k |"
    separator = "|---|" + "---:|" * (len(workloads) + 2)
    lines = ["", title, "", *introduction, header, separator]
    for name, result in ranked:
        split_metrics = result.split_metrics[split]
        workload_cells = "".join(
            f" {float(result.workload_metrics[workload]['token_hit_rate']):.3f} |"
            for workload in workloads
        )
        lines.append(
            f"| `{name}` |{workload_cells} "
            f"{float(split_metrics['block_hit_rate']):.3f} | "
            f"{float(split_metrics['cache_churn_per_1k']):.1f} |"
        )
    return lines


def _report_notes(
    results: dict[str, EvaluationResult],
    config: EvaluatorConfig,
    *,
    quick: bool,
) -> list[str]:
    candidate = results["candidate"]
    breakdown = candidate.score_breakdown
    lines = [
        "",
        "## Notes",
        "",
        (
            "- Candidate `scoring_fn_complexity` in this report is "
            f"`{candidate.candidate_metadata.get('scoring_fn_complexity')}`; "
            "the combined score includes that penalty."
        ),
        (
            "- Candidate score breakdown: mean workload "
            f"`{breakdown.get('mean_workload_score', 0.0):.3f}`, minimum-workload "
            f"contribution `{breakdown.get('min_workload_contribution', 0.0):.3f}`, "
            f"churn cost `{breakdown.get('churn_cost', 0.0):.3f}`, underfill cost "
            f"`{breakdown.get('underfill_cost', 0.0):.3f}`, fairness cost "
            f"`{breakdown.get('fairness_cost', 0.0):.3f}`, and complexity cost "
            f"`{breakdown.get('complexity_cost', 0.0):.3f}`."
        ),
        (
            "- `policy_underfill_rate` is policy bypass multiplied by unused mean "
            "capacity. It penalizes deliberate bypass while cache space remains idle, "
            "without charging natural underfill when the policy admits every miss."
        ),
        (
            "- `future_reuse_heuristic` and `oracle_future_reuse` use "
            "simulator-provided future knowledge and are not deployable. The former "
            "is count-weighted; the latter is a Belady-style next-use oracle "
            "constrained by the simulator's leaf-only eviction model."
        ),
        (
            "- `tinylfu_lru` admits only shallow or repeated blocks, so it often "
            "trades lower hit rate for lower churn."
        ),
        (
            "- `vllm_apc` models vLLM automatic prefix caching: it admits only full "
            "blocks and uses LRU eviction with deepest-prefix tie-breaking. The "
            "simulator supplies active-reference pinning and legal leaf filtering."
        ),
        (
            "- `sglang_radix_attention` models SGLang RadixAttention's default "
            "radix-cache replacement behavior: retain prefixes at cache-page "
            "boundaries and recursively evict the least-recently-used zero-reference "
            "leaf. The simulator treats every modeled block-tree node as a cacheable "
            "radix unit, making it behaviorally equivalent to `lru`; capacity remains "
            "fixed-block-counted rather than token/page-counted, and cache-aware "
            "scheduling and attention kernels are out of scope. It remains registered "
            "as a selectable reference but is excluded from default comparisons. See "
            "https://arxiv.org/html/2312.07104v1 and the pinned SGLang source at "
            "https://github.com/sgl-project/sglang/tree/"
            "52f221cce088abc998fa9d3812416a45ee0e2e25/python/sglang/srt/mem_cache."
        ),
        (
            "- `prefix_anchor` is a deployable structural anchor baseline; "
            "`prefix_fanout` is a simpler descendant-count protection baseline."
        ),
        (
            "- Priority-burst weighted hit is reported from `priority_burst_recovery`; "
            "priority-noise token hit checks the opposite failure mode, where high "
            "priority does not imply reuse."
        ),
        (
            "- Request p10, worst-quarter hit, token-weighted admission waste, "
            "admission token utility, and avoidable eviction are aggregated across "
            "the validation panel."
        ),
        (
            f"- This report uses `request_count={config.request_count}`, seeds "
            f"`{config.seeds}`, block size `{config.block_size_tokens}`, block-capacity "
            f"sweep `{config.effective_capacity_blocks()}`, token-capacity sweep "
            f"`{config.effective_capacity_tokens()}`, and canonical synthetic workload "
            f"token granularity `{config.effective_workload_token_granularity()}`."
        ),
    ]
    if quick:
        lines.append("- This is a smoke-only single-seed report, not a policy-ranking report.")
    lines.append("")
    return lines


def _split_workloads(
    results: dict[str, EvaluationResult],
    split: str,
) -> list[str]:
    sample = next(iter(results.values()))
    return [workload for workload in sample.workload_metrics if workload.startswith(f"{split}/")]


def _format_policy_names(names: list[str]) -> str:
    return " and ".join(f"`{name}`" for name in names)


def _combined_score_svg(results: dict[str, EvaluationResult]) -> str:
    width = 920
    row_height = 34
    left = 190
    top = 54
    right = 40
    bottom = 36
    height = top + row_height * len(results) + bottom
    min_score = min(result.combined_score for result in results.values())
    max_score = max(result.combined_score for result in results.values())
    span = max(max_score - min_score, 1.0)
    plot_width = width - left - right
    lines = [_svg_header(width, height, "Prefix KV-cache Baseline Scores")]
    lines.append(_text(24, 30, "Baseline combined scores", size=20, weight="700"))
    for index, (name, result) in enumerate(
        sorted(results.items(), key=lambda item: item[1].combined_score, reverse=True)
    ):
        y = top + index * row_height
        score = result.combined_score
        bar_width = max(2.0, (score - min_score) / span * plot_width)
        lines.append(_text(16, y + 21, name, size=13))
        lines.append(
            f'<rect x="{left}" y="{y + 5}" width="{bar_width:.2f}" '
            f'height="20" fill="#2563eb" rx="2" />'
        )
        lines.append(_text(left + bar_width + 8, y + 21, f"{score:.1f}", size=12))
    lines.append("</svg>")
    return "\n".join(lines)


def _validation_heatmap_svg(results: dict[str, EvaluationResult]) -> str:
    workloads = _split_workloads(results, "validation")
    cell_w = 120
    cell_h = 32
    left = 190
    top = 86
    width = left + cell_w * len(workloads) + 36
    height = top + cell_h * len(results) + 42
    lines = [_svg_header(width, height, "Validation Token Hit Rate Heatmap")]
    lines.append(_text(24, 30, "Validation token hit rate", size=20, weight="700"))
    for col, workload in enumerate(workloads):
        label = workload.split("/", 1)[1]
        lines.append(_text(left + col * cell_w + 6, 68, label, size=11))
    for row, (name, result) in enumerate(results.items()):
        y = top + row * cell_h
        lines.append(_text(16, y + 21, name, size=13))
        for col, workload in enumerate(workloads):
            x = left + col * cell_w
            rate = float(result.workload_metrics[workload]["token_hit_rate"])
            fill = _blue_scale(rate)
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 4}" '
                f'height="{cell_h - 4}" fill="{fill}" rx="2" />'
            )
            lines.append(_text(x + 8, y + 20, f"{rate:.3f}", size=12))
    lines.append("</svg>")
    return "\n".join(lines)


def _token_vs_block_svg(results: dict[str, EvaluationResult]) -> str:
    width = 720
    height = 520
    left = 72
    top = 48
    plot_w = 450
    plot_h = 380
    points = []
    for name, result in results.items():
        validation = result.split_metrics["validation"]
        points.append(
            (
                name,
                float(validation["block_hit_rate"]),
                float(validation["token_hit_rate"]),
                result.combined_score,
            )
        )
    lines = [_svg_header(width, height, "Token vs Block Hit Rate")]
    lines.append(_text(24, 30, "Validation token vs block hit rate", size=20, weight="700"))
    lines.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
        'fill="#f8fafc" stroke="#cbd5e1" />'
    )
    for tick in range(0, 6):
        value = tick / 5
        x = left + value * plot_w
        y = top + plot_h - value * plot_h
        lines.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e2e8f0" />'
        )
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e2e8f0" />'
        )
        lines.append(_text(x - 10, top + plot_h + 20, f"{value:.1f}", size=11))
        lines.append(_text(28, y + 4, f"{value:.1f}", size=11))
    lines.append(_text(left + 165, height - 26, "block hit rate", size=13))
    lines.append(_text(14, top + 190, "token", size=13))
    for index, (name, block_rate, token_rate, score) in enumerate(points):
        x = left + block_rate * plot_w
        y = top + plot_h - token_rate * plot_h
        color = _palette(index)
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}">'
            f"<title>{html.escape(name)} score={score:.1f}</title></circle>"
        )
        lines.append(_text(left + plot_w + 24, top + 24 + index * 24, name, size=12, fill=color))
    lines.append("</svg>")
    return "\n".join(lines)


def _svg_header(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        '<rect width="100%" height="100%" fill="white" />'
    )


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 12,
    weight: str = "400",
    fill: str = "#0f172a",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">'
        f"{html.escape(value)}</text>"
    )


def _blue_scale(value: float) -> str:
    value = max(0.0, min(1.0, value))
    lightness = int(94 - value * 46)
    return f"hsl(214, 78%, {lightness}%)"


def _palette(index: int) -> str:
    colors = (
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
    )
    return colors[index % len(colors)]
