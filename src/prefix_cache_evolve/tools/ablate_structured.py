"""Ablate structured prefix KV-cache policy terms on full evaluation panels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import click

from prefix_cache_evolve.artifacts import write_json
from prefix_cache_evolve.evaluators.prefix_kv_cache import PrefixKVCacheEvaluator
from prefix_cache_evolve.evaluators.results import EvaluationResult
from prefix_cache_evolve.evaluators.verifier import (
    require_single_score_identity,
    require_single_verifier_version,
)
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import (
    DEFAULT_CONFIG_PATH,
    load_evaluator_config,
)
from prefix_cache_evolve.problems.prefix_kv_cache.seeds.structured_recurrence import (
    StructuredRecurrencePolicy,
)


@dataclass(frozen=True)
class Ablation:
    """One behavioral feature deletion."""

    name: str
    disabled: frozenset[str]


ABLATIONS = (
    Ablation("all_terms", frozenset()),
    Ablation("without_recurrence", frozenset({"recurrence"})),
    Ablation("without_subtree", frozenset({"subtree"})),
    Ablation("without_regime", frozenset({"regime"})),
    Ablation("without_miss_state", frozenset({"miss_state"})),
    Ablation("without_priority_state", frozenset({"priority_state"})),
    Ablation(
        "without_recurrence_and_priority_state",
        frozenset({"recurrence", "priority_state"}),
    ),
    Ablation(
        "without_recurrence_priority_and_miss_state",
        frozenset({"recurrence", "priority_state", "miss_state"}),
    ),
)


class AblationStructuredPolicy(StructuredRecurrencePolicy):
    """Structured policy with independently removable score components."""

    def __init__(
        self,
        capacity_blocks,
        block_size_tokens,
        seed=None,
        *,
        disabled: frozenset[str] = frozenset(),
    ):
        super().__init__(capacity_blocks, block_size_tokens, seed)
        self._disabled = disabled

    def score_admission(self, block, now):
        """Score admission with the selected components disabled."""
        fast, slow, priority, _ = self._state.values(block.prefix_hash, now)
        reuse = math.log1p(fast + 0.6 * slow)
        structure = 0.15 * math.log1p(block.descendant_count)
        structure += 0.08 * math.log1p(block.active_ref_count)
        if "subtree" not in self._disabled:
            structure += 0.08 * math.log1p(block.subtree_active_ref_count)
            structure += 0.12 * math.log1p(1.0 + block.subtree_hit_rate)
        recurrence = 0.0
        if "recurrence" not in self._disabled:
            recurrence = 0.1 * math.log1p(1.0 + (block.access_gap_mean or 0.0))
            recurrence -= 0.05 * math.log1p(1.0 + (block.access_gap_var or 0.0))
        regime = 0.0
        if "regime" not in self._disabled:
            regime = 0.2 * self._pressure + 0.12 * self._miss_rate
        priority_value = 0.0 if "priority_state" in self._disabled else priority
        return (
            0.52 * reuse
            + 0.1 * priority_value
            + 0.18 * math.log1p(block.estimated_recompute_cost / 64.0)
            + structure
            + recurrence
            + 0.12 * self._priority
            - 0.1 * block.depth
            - 0.08 * math.log1p(max(1.0, block.token_count) / self._block_size_tokens)
            - regime
            - 0.12
        )

    def score_eviction(self, block, now):
        """Score eviction with the selected components disabled."""
        fast, slow, priority, misses = self._state.values(block.prefix_hash, now)
        reuse = math.log1p(fast + 0.6 * slow)
        structure = math.log1p(block.descendant_count + block.active_ref_count)
        if "subtree" not in self._disabled:
            structure = math.log1p(
                block.descendant_count + block.subtree_active_ref_count + block.active_ref_count
            )
        recurrence = 0.0
        if "recurrence" not in self._disabled:
            recurrence = 0.04 * math.log1p(1.0 + (block.access_gap_mean or 0.0))
            recurrence -= 0.02 * math.log1p(1.0 + (block.access_gap_var or 0.0))
        priority_value = 0.0 if "priority_state" in self._disabled else priority
        miss_value = 0.0 if "miss_state" in self._disabled else misses
        return (
            0.92 * math.log1p(max(0, now - block.last_accessed_at))
            - reuse
            - 0.18 * structure
            - 0.28 * math.log1p(block.estimated_recompute_cost / 64.0)
            - 0.18 * priority_value
            + 0.24 * miss_value
            - 0.08 * block.depth
            + recurrence
        )


def _factory(disabled: frozenset[str]):
    def build_candidate(capacity_blocks, block_size_tokens, seed=None):
        return AblationStructuredPolicy(
            capacity_blocks,
            block_size_tokens,
            seed,
            disabled=disabled,
        )

    return build_candidate


def _summary(result: EvaluationResult, split: str) -> dict[str, float | str]:
    metrics = result.split_metrics[split]
    return {
        "verifier_version": result.verifier_version,
        "evaluation_context_sha256": result.evaluation_context_sha256,
        "panel_sha256": result.panel_sha256,
        "combined_score_without_complexity": result.combined_score,
        "mean_workload_score": result.score_breakdown["mean_workload_score"],
        "min_workload_contribution": result.score_breakdown["min_workload_contribution"],
        "churn_cost": result.score_breakdown["churn_cost"],
        "fairness_cost": result.score_breakdown["fairness_cost"],
        "token_hit_rate": float(metrics["token_hit_rate"]),
        "worst_quarter_token_hit_rate": float(metrics["worst_quarter_token_hit_rate"]),
        "wasted_admission_token_rate": float(metrics["wasted_admission_token_rate"]),
        "avoidable_eviction_rate": float(metrics["avoidable_eviction_rate"]),
        "cache_churn_per_1k": float(metrics["cache_churn_per_1k"]),
    }


def run_ablation(config_path: Path) -> dict[str, object]:
    """Evaluate every structured feature deletion on validation and probe."""
    config = load_evaluator_config(config_path)
    rows = []
    for ablation in ABLATIONS:
        factory = _factory(ablation.disabled)
        selection = PrefixKVCacheEvaluator(
            config,
            splits=("train", "validation", "probe"),
        )(factory)
        probe = PrefixKVCacheEvaluator(config, splits=("probe",))(factory)
        rows.append(
            {
                "variant": ablation.name,
                "disabled": sorted(ablation.disabled),
                "selection": _summary(selection, "validation"),
                "probe": _summary(probe, "probe"),
                "probe_families": {
                    family.removeprefix("probe/"): {
                        "token_hit_rate": float(metrics["token_hit_rate"]),
                        "cache_churn_per_1k": float(metrics["cache_churn_per_1k"]),
                    }
                    for family, metrics in probe.workload_metrics.items()
                    if family.startswith("probe/")
                },
            }
        )
    identities = {
        panel: require_single_score_identity(
            (row[panel] for row in rows),
            context=f"structured ablation {panel} comparison",
        )
        for panel in ("selection", "probe")
    }
    return {
        "schema": "prefix-kv-cache-structured-ablation-v1",
        "verifier_version": config.verifier_version,
        "evaluation_contexts": {
            panel: identity.evaluation_context_sha256 for panel, identity in identities.items()
        },
        "panel_sha256s": {panel: identity.panel_sha256 for panel, identity in identities.items()},
        "config": str(config_path),
        "complexity_note": (
            "Behavior-only ablations are evaluated with complexity zero; source "
            "deletion and charged evaluation follow after selecting useful terms."
        ),
        "variants": rows,
    }


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    variants = payload["variants"]
    verifier_version = require_single_verifier_version(
        (row[panel] for row in variants for panel in ("selection", "probe")),
        context="structured ablation report",
    )
    identities = {
        panel: require_single_score_identity(
            (row[panel] for row in variants),
            context=f"structured ablation {panel} comparison",
        )
        for panel in ("selection", "probe")
    }
    if verifier_version != payload.get("verifier_version"):
        raise ValueError("structured ablation report version does not match its score rows")
    lines = [
        "# Structured Prefix KV-Cache Ablation",
        "",
        f"Verifier: `{verifier_version}`",
        "",
        "Evaluation contexts: "
        + ", ".join(
            f"`{panel}={identity.evaluation_context_sha256}`"
            for panel, identity in identities.items()
        ),
        "",
        "Panels: "
        + ", ".join(f"`{panel}={identity.panel_sha256}`" for panel, identity in identities.items()),
        "",
        str(payload["complexity_note"]),
        "",
        "| Variant | Selection raw | Mean | Min contrib. | Churn cost | "
        "Probe raw | Agent hit | Cyclic hit | Probe churn/1k |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in variants:  # type: ignore[assignment]
        selection = row["selection"]
        probe = row["probe"]
        families = row["probe_families"]
        lines.append(
            f"| `{row['variant']}` | "
            f"{selection['combined_score_without_complexity']:.3f} | "
            f"{selection['mean_workload_score']:.3f} | "
            f"{selection['min_workload_contribution']:.3f} | "
            f"{selection['churn_cost']:.3f} | "
            f"{probe['combined_score_without_complexity']:.3f} | "
            f"{families['agent_trace_branching']['token_hit_rate']:.4f} | "
            f"{families['cyclic_working_set_pressure']['token_hit_rate']:.4f} | "
            f"{probe['cache_churn_per_1k']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@click.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    default=DEFAULT_CONFIG_PATH,
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("artifacts/prefix_kv_cache_structured_ablation.json"),
    show_default=True,
)
def main(config: Path, output: Path) -> None:
    """Ablate structured prefix KV-cache policy terms."""
    payload = run_ablation(config)
    write_json(output, payload)
    markdown_path = output.with_suffix(".md")
    _write_markdown(markdown_path, payload)
    click.echo(output)
    click.echo(markdown_path)


if __name__ == "__main__":
    main()
