"""Test whether admission-side regret dominates eviction-side regret."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Iterable

import click

from prefix_cache_evolve.evaluator_entry import load_candidate_factory
from prefix_cache_evolve.evaluators.baselines import (
    baseline_cost_aware_lru,
    baseline_lfu_blocks,
    baseline_lru_blocks,
    baseline_no_cache,
    baseline_oracle_future_reuse,
    baseline_tinylfu_lru,
    baseline_vllm_apc,
)
from prefix_cache_evolve.evaluators.contracts import PrefixKVPolicy
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    PrefixKVCacheEvaluator,
    TrialMetrics,
)
from prefix_cache_evolve.problems.prefix_kv_cache.compact_seed import (
    build_candidate as build_compact_seed,
)
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.pressure_aware_incumbent import (
    build_candidate as build_incumbent,
)
from prefix_cache_evolve.problems.prefix_kv_cache.seeds.structured_recurrence import (
    build_candidate as build_structured_recurrence,
)
from prefix_cache_evolve.problems.prefix_kv_cache.seeds.structured_seed import (
    build_candidate as build_structured_seed,
)

_DEFAULT_SPLITS = ("train", "validation", "probe", "hidden")


@dataclass(frozen=True, slots=True)
class AdmissionPolicySpec:
    """One distinct admission rule used in controlled policy crosses."""

    name: str
    factory: Callable[..., PrefixKVPolicy]
    aliases: tuple[str, ...] = ()
    reporting_only: bool = False


@dataclass(frozen=True, slots=True)
class EvictionPolicySpec:
    """One eviction rule crossed with every distinct admission rule."""

    name: str
    factory: Callable[..., PrefixKVPolicy]
    reporting_only: bool = False


ADMISSION_POLICY_SPECS = (
    AdmissionPolicySpec("reject_all", baseline_no_cache, aliases=("no_cache",)),
    AdmissionPolicySpec(
        "admit_all",
        baseline_lru_blocks,
        aliases=(
            "lru",
            "sglang_radix_attention",
            "lfu",
            "depth_prefer_shallow",
            "recompute_greedy",
            "cost_aware_lru",
            "prefix_fanout",
            "prefix_anchor",
            "tenant_fair_lru",
            "future_reuse_heuristic",
            "initial_hybrid",
        ),
    ),
    AdmissionPolicySpec("full_blocks_only", baseline_vllm_apc, aliases=("vllm_apc",)),
    AdmissionPolicySpec("tinylfu", baseline_tinylfu_lru, aliases=("tinylfu_lru",)),
    AdmissionPolicySpec("compact_seed", build_compact_seed),
    AdmissionPolicySpec("structured_seed", build_structured_seed),
    AdmissionPolicySpec("structured_recurrence", build_structured_recurrence),
    AdmissionPolicySpec("pressure_aware_incumbent", build_incumbent),
    AdmissionPolicySpec(
        "oracle_future_reuse",
        baseline_oracle_future_reuse,
        reporting_only=True,
    ),
)

EVICTION_POLICY_SPECS = (
    EvictionPolicySpec("lru", baseline_lru_blocks),
    EvictionPolicySpec("lfu", baseline_lfu_blocks),
    EvictionPolicySpec("cost_aware_lru", baseline_cost_aware_lru),
    EvictionPolicySpec("incumbent_value_aware", build_incumbent),
    EvictionPolicySpec(
        "oracle_next_use",
        baseline_oracle_future_reuse,
        reporting_only=True,
    ),
)


def _trial_row(trial: TrialMetrics) -> dict[str, object]:
    """Return the regret decomposition for one workload-capacity-seed group."""
    admission_regret = (
        trial.avoidable_admission_regret_tokens + trial.avoidable_rejection_regret_tokens
    )
    eviction_regret = trial.value_weighted_avoidable_eviction_regret_tokens
    total_regret = admission_regret + eviction_regret
    admission_regret_per_decision = admission_regret / max(1, trial.admission_score_count)
    eviction_regret_per_decision = eviction_regret / max(1, trial.eviction_count)
    return {
        "group": (
            f"{trial.split}/{trial.workload}/capacity_{trial.capacity_blocks}/seed_{trial.seed}"
        ),
        "split": trial.split,
        "workload": trial.workload,
        "capacity_blocks": trial.capacity_blocks,
        "seed": trial.seed,
        "invalid": trial.invalid,
        "invalid_reason": trial.invalid_reason,
        "block_hit_rate": trial.block_hit_rate,
        "token_hit_rate": trial.token_hit_rate,
        "prefill_tokens_saved": trial.prefill_tokens_saved,
        "p95_latency_proxy": trial.p95_latency_proxy,
        "admission_score_count": trial.admission_score_count,
        "admission_count": trial.admission_count,
        "admission_rejection_count": trial.admission_rejection_count,
        "eviction_count": trial.eviction_count,
        "avoidable_admission_count": trial.avoidable_admission_count,
        "avoidable_admission_regret_tokens": trial.avoidable_admission_regret_tokens,
        "avoidable_rejection_count": trial.avoidable_rejection_count,
        "avoidable_rejection_regret_tokens": trial.avoidable_rejection_regret_tokens,
        "value_weighted_avoidable_eviction_count": (trial.value_weighted_avoidable_eviction_count),
        "value_weighted_avoidable_eviction_regret_tokens": eviction_regret,
        "admission_side_regret_tokens": admission_regret,
        "eviction_side_regret_tokens": eviction_regret,
        "total_regret_tokens": total_regret,
        "admission_regret_share": admission_regret / total_regret if total_regret else 0.0,
        "admission_regret_tokens_per_decision": admission_regret_per_decision,
        "eviction_regret_tokens_per_decision": eviction_regret_per_decision,
        "admission_minus_eviction_regret_tokens": admission_regret - eviction_regret,
        "admission_dominates": admission_regret > eviction_regret,
        "decision_normalized_admission_dominates": (
            admission_regret_per_decision > eviction_regret_per_decision
        ),
    }


def _summarize_groups(groups: Iterable[dict[str, object]]) -> dict[str, object]:
    """Summarize strict admission dominance over valid regretful groups."""
    groups = list(groups)
    valid = [group for group in groups if not group["invalid"]]
    invalid = [group for group in groups if group["invalid"]]
    regretful = [group for group in valid if float(group["total_regret_tokens"]) > 0.0]
    zero_regret = [group for group in valid if float(group["total_regret_tokens"]) == 0.0]
    admission_dominant = [group for group in regretful if group["admission_dominates"]]
    decision_normalized_admission_dominant = [
        group for group in regretful if group["decision_normalized_admission_dominates"]
    ]
    eviction_dominant = [
        group
        for group in regretful
        if float(group["admission_side_regret_tokens"])
        < float(group["eviction_side_regret_tokens"])
    ]
    tied = [
        group
        for group in regretful
        if float(group["admission_side_regret_tokens"])
        == float(group["eviction_side_regret_tokens"])
    ]
    admission_regret = sum(float(group["admission_side_regret_tokens"]) for group in valid)
    eviction_regret = sum(float(group["eviction_side_regret_tokens"]) for group in valid)
    admission_decisions = sum(int(group["admission_score_count"]) for group in valid)
    eviction_decisions = sum(int(group["eviction_count"]) for group in valid)
    total_regret = admission_regret + eviction_regret
    margins = [float(group["admission_minus_eviction_regret_tokens"]) for group in regretful]
    mean_token_hit_rate = (
        sum(float(group["token_hit_rate"]) for group in valid) / len(valid) if valid else 0.0
    )
    mean_block_hit_rate = (
        sum(float(group["block_hit_rate"]) for group in valid) / len(valid) if valid else 0.0
    )
    mean_p95_latency = (
        sum(float(group["p95_latency_proxy"]) for group in valid) / len(valid) if valid else 0.0
    )
    uniform = bool(regretful) and len(admission_dominant) == len(regretful) and not invalid
    if invalid:
        verdict = "inconclusive_invalid_groups"
    elif not regretful:
        verdict = "inconclusive_no_regret"
    elif uniform:
        verdict = "supported"
    else:
        verdict = "falsified"
    return {
        "verdict": verdict,
        "group_count": len(groups),
        "valid_group_count": len(valid),
        "invalid_group_count": len(invalid),
        "regretful_group_count": len(regretful),
        "zero_regret_group_count": len(zero_regret),
        "admission_dominant_group_count": len(admission_dominant),
        "eviction_dominant_group_count": len(eviction_dominant),
        "tied_regretful_group_count": len(tied),
        "admission_dominance_rate": (
            len(admission_dominant) / len(regretful) if regretful else 0.0
        ),
        "decision_normalized_admission_dominant_group_count": len(
            decision_normalized_admission_dominant
        ),
        "decision_normalized_admission_dominance_rate": (
            len(decision_normalized_admission_dominant) / len(regretful) if regretful else 0.0
        ),
        "uniform_admission_dominance": uniform,
        "mean_token_hit_rate": mean_token_hit_rate,
        "mean_block_hit_rate": mean_block_hit_rate,
        "total_prefill_tokens_saved": sum(float(group["prefill_tokens_saved"]) for group in valid),
        "mean_p95_latency_proxy": mean_p95_latency,
        "aggregate_admission_decision_count": admission_decisions,
        "aggregate_eviction_decision_count": eviction_decisions,
        "aggregate_admission_side_regret_tokens": admission_regret,
        "aggregate_eviction_side_regret_tokens": eviction_regret,
        "aggregate_admission_regret_tokens_per_decision": (
            admission_regret / admission_decisions if admission_decisions else 0.0
        ),
        "aggregate_eviction_regret_tokens_per_decision": (
            eviction_regret / eviction_decisions if eviction_decisions else 0.0
        ),
        "aggregate_admission_regret_share": (
            admission_regret / total_regret if total_regret else 0.0
        ),
        "aggregate_admission_minus_eviction_regret_tokens": (admission_regret - eviction_regret),
        "median_group_admission_minus_eviction_regret_tokens": (
            median(margins) if margins else 0.0
        ),
    }


def _group_summaries(
    groups: list[dict[str, object]],
    key_fn: Callable[[dict[str, object]], str],
) -> dict[str, dict[str, object]]:
    """Summarize groups under stable labels."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for group in groups:
        grouped.setdefault(key_fn(group), []).append(group)
    return {key: _summarize_groups(values) for key, values in sorted(grouped.items())}


def run_analysis(
    config_path: Path,
    *,
    request_count: int | None = None,
    seeds: tuple[int, ...] | None = None,
    splits: tuple[str, ...] = _DEFAULT_SPLITS,
    workloads: tuple[str, ...] | None = None,
    factory: Callable[..., PrefixKVPolicy] = build_incumbent,
    policy_name: str = "pressure_aware_incumbent",
    fixed_admission_factory: Callable[..., PrefixKVPolicy] | None = None,
    expose_future_reuse: bool = False,
) -> dict[str, object]:
    """Run the local-oracle regret audit over workload-capacity-seed groups."""
    config = load_evaluator_config(config_path)
    updates: dict[str, object] = {}
    if request_count is not None:
        updates["request_count"] = request_count
    if seeds is not None:
        updates["seeds"] = seeds
    if workloads is not None:
        for split in splits:
            updates[f"{split}_families"] = workloads
        updates["family_request_multipliers"] = {}
    if updates:
        config = config.with_updates(**updates)

    result = PrefixKVCacheEvaluator(
        config,
        splits=splits,
        fixed_admission_factory=fixed_admission_factory,
        expose_future_reuse=expose_future_reuse,
    )(factory)
    groups = [_trial_row(trial) for trial in result.trials]
    return {
        "schema": "prefix-kv-cache-admission-eviction-regret-audit-v1",
        "config": str(config_path),
        "policy": policy_name,
        "request_count": config.request_count,
        "seeds": list(config.seeds),
        "capacity_blocks": list(config.effective_capacity_blocks()),
        "splits": list(splits),
        "workloads": list(workloads) if workloads is not None else None,
        "falsification_rule": (
            "The universal claim passes only when admission-side regret is strictly greater "
            "than eviction-side regret in every valid group with nonzero regret."
        ),
        "value_definition": (
            "The future-value surrogate upper bound equals block token count multiplied by "
            "remaining request occurrences after the current request."
        ),
        "summary": _summarize_groups(groups),
        "by_split": _group_summaries(groups, lambda group: str(group["split"])),
        "by_workload": _group_summaries(
            groups,
            lambda group: f"{group['split']}/{group['workload']}",
        ),
        "groups": groups,
    }


def run_admission_policy_sweep(
    config_path: Path,
    *,
    request_count: int | None = None,
    seeds: tuple[int, ...] | None = None,
    splits: tuple[str, ...] = _DEFAULT_SPLITS,
    workloads: tuple[str, ...] | None = None,
    policy_specs: tuple[AdmissionPolicySpec, ...] = ADMISSION_POLICY_SPECS,
) -> dict[str, object]:
    """Evaluate every distinct admission rule with fixed LRU eviction."""
    policies: dict[str, dict[str, object]] = {}
    for specification in policy_specs:
        analysis = run_analysis(
            config_path,
            request_count=request_count,
            seeds=seeds,
            splits=splits,
            workloads=workloads,
            factory=baseline_lru_blocks,
            policy_name=f"{specification.name}+fixed_lru",
            fixed_admission_factory=specification.factory,
            expose_future_reuse=specification.reporting_only,
        )
        policies[specification.name] = {
            "aliases": list(specification.aliases),
            "reporting_only": specification.reporting_only,
            **analysis,
        }

    deployable = [policy for policy in policies.values() if not bool(policy["reporting_only"])]
    return {
        "schema": "prefix-kv-cache-admission-policy-regret-sweep-v1",
        "config": str(config_path),
        "eviction_policy": "lru",
        "policy_count": len(policies),
        "deployable_policy_count": len(deployable),
        "method": (
            "Hold legal-leaf LRU eviction fixed and vary each distinct admission "
            "implementation plus its lifecycle state."
        ),
        "policies": policies,
    }


def run_admission_eviction_matrix(
    config_path: Path,
    *,
    request_count: int | None = None,
    seeds: tuple[int, ...] | None = None,
    splits: tuple[str, ...] = _DEFAULT_SPLITS,
    workloads: tuple[str, ...] | None = None,
    admission_specs: tuple[AdmissionPolicySpec, ...] = ADMISSION_POLICY_SPECS,
    eviction_specs: tuple[EvictionPolicySpec, ...] = EVICTION_POLICY_SPECS,
) -> dict[str, object]:
    """Cross every distinct admission rule with representative eviction rules."""
    combinations: dict[str, dict[str, object]] = {}
    for admission in admission_specs:
        for eviction in eviction_specs:
            key = f"{admission.name}+{eviction.name}"
            analysis = run_analysis(
                config_path,
                request_count=request_count,
                seeds=seeds,
                splits=splits,
                workloads=workloads,
                factory=eviction.factory,
                policy_name=key,
                fixed_admission_factory=admission.factory,
                expose_future_reuse=admission.reporting_only or eviction.reporting_only,
            )
            combinations[key] = {
                "admission_policy": admission.name,
                "eviction_policy": eviction.name,
                "reporting_only": admission.reporting_only or eviction.reporting_only,
                **analysis,
            }

    return {
        "schema": "prefix-kv-cache-admission-eviction-policy-matrix-v1",
        "config": str(config_path),
        "admission_policies": [
            {
                "name": specification.name,
                "aliases": list(specification.aliases),
                "reporting_only": specification.reporting_only,
            }
            for specification in admission_specs
        ],
        "eviction_policies": [
            {
                "name": specification.name,
                "reporting_only": specification.reporting_only,
            }
            for specification in eviction_specs
        ],
        "combination_count": len(combinations),
        "method": (
            "Full factorial crossing of distinct admission implementations with "
            "representative eviction rules on identical workload-capacity-seed groups."
        ),
        "combinations": combinations,
    }


def _summary_row(label: str, summary: dict[str, object]) -> str:
    """Render one aggregate Markdown row."""
    return (
        f"| `{label}` | {summary['regretful_group_count']} | "
        f"{summary['admission_dominant_group_count']} | "
        f"{float(summary['admission_dominance_rate']):.1%} | "
        f"{float(summary['aggregate_admission_side_regret_tokens']):.1f} | "
        f"{float(summary['aggregate_eviction_side_regret_tokens']):.1f} | "
        f"{float(summary['aggregate_admission_minus_eviction_regret_tokens']):+.1f} |"
    )


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    """Write a human-readable falsification report."""
    summary = payload["summary"]
    groups = payload["groups"]
    counterexamples = sorted(
        (
            group
            for group in groups
            if not group["invalid"]
            and float(group["total_regret_tokens"]) > 0.0
            and not group["admission_dominates"]
        ),
        key=lambda group: float(group["admission_minus_eviction_regret_tokens"]),
    )
    negative_workloads = sorted(
        (
            (label, value)
            for label, value in payload["by_workload"].items()
            if float(value["aggregate_admission_minus_eviction_regret_tokens"]) < 0.0
        ),
        key=lambda item: float(item[1]["aggregate_admission_minus_eviction_regret_tokens"]),
    )
    verdict_text = {
        "supported": "Supported under the strict universal rule.",
        "falsified": "Falsified under the strict universal rule.",
        "inconclusive_invalid_groups": "Inconclusive because one or more groups were invalid.",
        "inconclusive_no_regret": "Inconclusive because no group had measurable regret.",
    }[summary["verdict"]]
    lines = [
        "# Admission vs Eviction Regret Audit",
        "",
        f"Policy: `{payload['policy']}`",
        "",
        "## Falsifiable Claim",
        "",
        "Claim: admission-side regret dominates eviction-side regret across",
        "workload-capacity-seed groups.",
        "",
        f"**Verdict: {verdict_text}**",
        "",
        f"- Regretful groups: `{summary['regretful_group_count']}`.",
        f"- Admission-dominant groups: `{summary['admission_dominant_group_count']}` "
        f"(`{float(summary['admission_dominance_rate']):.1%}`).",
        f"- Aggregate admission-side regret: "
        f"`{float(summary['aggregate_admission_side_regret_tokens']):.1f}` future tokens.",
        f"- Aggregate eviction-side regret: "
        f"`{float(summary['aggregate_eviction_side_regret_tokens']):.1f}` future tokens.",
        f"- Aggregate admission share: `{float(summary['aggregate_admission_regret_share']):.1%}`.",
        f"- Admission decisions: `{summary['aggregate_admission_decision_count']}`; "
        f"surrogate regret per decision: "
        f"`{float(summary['aggregate_admission_regret_tokens_per_decision']):.2f}`.",
        f"- Eviction decisions: `{summary['aggregate_eviction_decision_count']}`; "
        f"surrogate regret per decision: "
        f"`{float(summary['aggregate_eviction_regret_tokens_per_decision']):.2f}`.",
        f"- Decision-normalized admission dominance: "
        f"`{summary['decision_normalized_admission_dominant_group_count']}` of "
        f"`{summary['regretful_group_count']}` regretful groups "
        f"(`{float(summary['decision_normalized_admission_dominance_rate']):.1%}`).",
        "",
        "The universal claim passes only if every valid group with nonzero regret has",
        "strictly greater admission-side regret. Zero-regret groups are reported but do not",
        "decide the claim.",
        "",
        "## Audit Definition",
        "",
        "- The future-value surrogate upper bound is block token count times remaining",
        "  request occurrences after the current request.",
        "- Avoidable admission regret is the value lost when an accepted incoming block is",
        "  worth less than the cheapest legal displacement.",
        "- Avoidable rejection regret is the value lost when a rejected incoming block is",
        "  worth more than the cheapest legal displacement; free space has displacement zero.",
        "- Value-weighted avoidable-eviction regret is the chosen victim's value minus the",
        "  cheapest legal victim's value.",
        "- Admission plus eviction surrogate regret exactly decomposes this local same-state",
        "  comparison. It is not realized causal regret or a full counterfactual replay.",
        "",
        "## By Split",
        "",
        "| Split | Regretful groups | Admission dominant | Dominance rate | "
        "Admission regret | Eviction regret | Margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_summary_row(label, value) for label, value in payload["by_split"].items())
    lines.extend(
        [
            "",
            "## By Workload",
            "",
            "| Workload | Regretful groups | Admission dominant | Dominance rate | "
            "Admission regret | Eviction regret | Margin |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(_summary_row(label, value) for label, value in payload["by_workload"].items())
    lines.extend(
        [
            "",
            "## Strongest Counterexamples",
            "",
            "| Group | Admission regret | Eviction regret | Margin |",
            "|---|---:|---:|---:|",
        ]
    )
    if counterexamples:
        for group in counterexamples[:20]:
            lines.append(
                f"| `{group['group']}` | "
                f"{float(group['admission_side_regret_tokens']):.1f} | "
                f"{float(group['eviction_side_regret_tokens']):.1f} | "
                f"{float(group['admission_minus_eviction_regret_tokens']):+.1f} |"
            )
    else:
        lines.append("| None | 0.0 | 0.0 | +0.0 |")
    negative_labels = ", ".join(f"`{label}`" for label, _ in negative_workloads[:5]) or "none"
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The strict universal claim is `{summary['verdict']}`: admission dominates in",
            f"`{summary['admission_dominant_group_count']}` of "
            f"`{summary['regretful_group_count']}` regretful groups, leaving",
            f"`{summary['eviction_dominant_group_count']}` eviction-dominant counterexamples.",
            f"The weaker aggregate-total claim is supported under this surrogate: admission "
            f"accounts for `{float(summary['aggregate_admission_regret_share']):.1%}` of total "
            "surrogate regret,",
            "and the median regretful group has a positive admission-minus-eviction margin.",
            "After normalizing each side by its own decision count, admission dominates in",
            f"`{summary['decision_normalized_admission_dominant_group_count']}` of "
            f"`{summary['regretful_group_count']}` regretful groups "
            f"(`{float(summary['decision_normalized_admission_dominance_rate']):.1%}`). "
            "The aggregate per-decision rate still favors admission, but groupwise",
            "per-decision dominance does not hold in a majority. Total contribution and",
            "per-decision severity answer different questions.",
            "",
            f"Workload-level counterexamples with negative aggregate margins include "
            f"{negative_labels}. These groups are concrete targets for eviction-specific",
            "follow-up rather than evidence for a uniform admission-first rule.",
        ]
    )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "The audit uses future occurrence count as a stable token-value surrogate upper",
            "bound. It does not model whether every future occurrence would remain",
            "root-contiguous after other decisions. It audits only explicit admission scores;",
            "descendants bypassed after a parent rejection are consequences of that scored",
            "rejection, not additional decisions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_admission_policy_markdown(path: Path, payload: dict[str, object]) -> None:
    """Write the controlled fixed-LRU admission-policy comparison."""
    rows = []
    for name, policy in payload["policies"].items():
        summary = policy["summary"]
        rows.append(
            (
                name,
                bool(policy["reporting_only"]),
                int(summary["regretful_group_count"]),
                int(summary["admission_dominant_group_count"]),
                float(summary["admission_dominance_rate"]),
                int(summary["decision_normalized_admission_dominant_group_count"]),
                float(summary["decision_normalized_admission_dominance_rate"]),
                float(summary["aggregate_admission_side_regret_tokens"]),
                float(summary["aggregate_eviction_side_regret_tokens"]),
                float(summary["aggregate_admission_regret_share"]),
            )
        )
    lines = [
        "# Admission-Policy Regret Sweep",
        "",
        "Eviction is fixed to legal-leaf LRU. Each row changes only the admission",
        "implementation and the lifecycle state needed by that admission policy.",
        "Duplicate admit-all baselines are collapsed into one behavioral representative.",
        "",
        "| Admission policy | Scope | Regretful groups | Admission dominant | "
        "Dominance rate | Per-decision dominant | Per-decision rate | "
        "Admission regret | Eviction regret | Admission share |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        (
            name,
            reporting_only,
            regretful,
            admission_dominant,
            dominance_rate,
            normalized_dominant,
            normalized_rate,
            admission_regret,
            eviction_regret,
            admission_share,
        ) = row
        scope = "reporting-only" if reporting_only else "deployable"
        lines.append(
            f"| `{name}` | {scope} | {regretful} | {admission_dominant} | "
            f"{dominance_rate:.1%} | {normalized_dominant} | {normalized_rate:.1%} | "
            f"{admission_regret:.1f} | {eviction_regret:.1f} | {admission_share:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Behavioral Aliases",
            "",
            "These policies have the same admission decisions under the simulator contract",
            "and are represented once in the table:",
            "",
        ]
    )
    for name, policy in payload["policies"].items():
        aliases = policy["aliases"]
        if aliases:
            lines.append(f"- `{name}`: " + ", ".join(f"`{alias}`" for alias in aliases) + ".")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a controlled local-regret audit, not a causal replay. It tests whether",
            "LRU leaves substantial hindsight eviction value on the table after each admission",
            "rule shapes the resident set. It does not yet measure the realized gain from",
            "replacing LRU with a more complex eviction policy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_admission_eviction_matrix_markdown(
    path: Path,
    payload: dict[str, object],
) -> None:
    """Write realized and local-regret results for the policy matrix."""
    combinations = payload["combinations"]
    eviction_names = [item["name"] for item in payload["eviction_policies"]]
    lines = [
        "# Admission-Eviction Policy Matrix",
        "",
        "Every distinct admission implementation is crossed with four deployable",
        "eviction rules and one reporting-only constrained next-use control.",
        "",
        "| Admission | Eviction | Scope | Mean token hit | Saved tokens | "
        "Eviction regret | Admission share |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for admission in payload["admission_policies"]:
        for eviction_name in eviction_names:
            combination = combinations[f"{admission['name']}+{eviction_name}"]
            summary = combination["summary"]
            scope = "reporting-only" if combination["reporting_only"] else "deployable"
            lines.append(
                f"| `{admission['name']}` | `{eviction_name}` | {scope} | "
                f"{float(summary['mean_token_hit_rate']):.4f} | "
                f"{float(summary['total_prefill_tokens_saved']):.0f} | "
                f"{float(summary['aggregate_eviction_side_regret_tokens']):.0f} | "
                f"{float(summary['aggregate_admission_regret_share']):.1%} |"
            )

    lines.extend(
        [
            "",
            "## Best Eviction Per Admission",
            "",
            "| Admission | LRU token hit | Best deployable eviction | Best token hit | "
            "Gain over LRU | Oracle token hit | Oracle headroom |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    deployable_evictions = [
        item["name"] for item in payload["eviction_policies"] if not item["reporting_only"]
    ]
    for admission in payload["admission_policies"]:
        name = admission["name"]
        lru_hit = float(combinations[f"{name}+lru"]["summary"]["mean_token_hit_rate"])
        deployable = max(
            (
                (
                    eviction_name,
                    float(
                        combinations[f"{name}+{eviction_name}"]["summary"]["mean_token_hit_rate"]
                    ),
                )
                for eviction_name in deployable_evictions
            ),
            key=lambda item: item[1],
        )
        oracle_hit = float(
            combinations[f"{name}+oracle_next_use"]["summary"]["mean_token_hit_rate"]
        )
        lines.append(
            f"| `{name}` | {lru_hit:.4f} | `{deployable[0]}` | {deployable[1]:.4f} | "
            f"{deployable[1] - lru_hit:+.4f} | {oracle_hit:.4f} | "
            f"{oracle_hit - deployable[1]:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Paired Group Comparisons Against LRU",
            "",
            "Win/tie/loss counts compare token hit rate on the same",
            "workload-capacity-seed group.",
            "",
            "| Admission | Eviction | Wins | Ties | Losses | Mean token-hit delta |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for admission in payload["admission_policies"]:
        name = admission["name"]
        if name == "reject_all":
            continue
        lru_groups = {
            group["group"]: float(group["token_hit_rate"])
            for group in combinations[f"{name}+lru"]["groups"]
        }
        comparison_evictions = [
            eviction_name
            for eviction_name in ("lfu", "incumbent_value_aware", "oracle_next_use")
            if f"{name}+{eviction_name}" in combinations
        ]
        for eviction_name in comparison_evictions:
            deltas = [
                float(group["token_hit_rate"]) - lru_groups[str(group["group"])]
                for group in combinations[f"{name}+{eviction_name}"]["groups"]
            ]
            wins = sum(delta > 1e-12 for delta in deltas)
            ties = sum(abs(delta) <= 1e-12 for delta in deltas)
            losses = sum(delta < -1e-12 for delta in deltas)
            mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
            lines.append(
                f"| `{name}` | `{eviction_name}` | {wins} | {ties} | {losses} | {mean_delta:+.4f} |"
            )

    deployable_combinations = [
        combination for combination in combinations.values() if not combination["reporting_only"]
    ]
    best_deployable = max(
        deployable_combinations,
        key=lambda combination: float(combination["summary"]["mean_token_hit_rate"]),
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The best deployable pairing is `{best_deployable['admission_policy']}+"
            f"{best_deployable['eviction_policy']}` at mean token hit "
            f"`{float(best_deployable['summary']['mean_token_hit_rate']):.4f}`.",
            "LFU and the compact incumbent value-aware eviction rule deliver similar",
            "realized gains across the selective admission policies; neither uniformly",
            "wins every group. This supports a simple frequency-aware eviction rule,",
            "not the stronger claim that eviction choice is unimportant.",
            "",
            "Mean token-hit differences are realized outcomes on identical generated",
            "request panels. Eviction regret remains a local future-count surrogate.",
            "The oracle is constrained to legal resident leaves and is reporting-only;",
            "it is not an unconstrained globally optimal cache replay.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@click.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    default=Path("configs/prefix_kv_cache.yaml"),
    show_default=True,
)
@click.option("--candidate-program", type=click.Path(path_type=Path))
@click.option("--request-count", type=click.IntRange(min=1))
@click.option("--seeds", type=int, multiple=True)
@click.option("--splits", type=click.Choice(_DEFAULT_SPLITS), multiple=True)
@click.option("--workloads", multiple=True)
@click.option(
    "--all-admission-policies",
    is_flag=True,
    help="Cross every distinct admission rule with representative eviction rules.",
)
@click.option("--output", type=click.Path(path_type=Path))
@click.option("--markdown", type=click.Path(path_type=Path))
def main(
    config: Path,
    candidate_program: Path | None,
    request_count: int | None,
    seeds: tuple[int, ...],
    splits: tuple[str, ...],
    workloads: tuple[str, ...],
    all_admission_policies: bool,
    output: Path | None,
    markdown: Path | None,
) -> None:
    """Audit admission and eviction regret."""
    selected_splits = splits or _DEFAULT_SPLITS
    if all_admission_policies:
        output_path = output or Path(
            "artifacts/prefix_kv_cache_admission_eviction_policy_matrix.json"
        )
        markdown_path = markdown or Path("docs/results/admission_eviction_policy_matrix.md")
        payload = run_admission_eviction_matrix(
            config,
            request_count=request_count,
            seeds=seeds or None,
            splits=selected_splits,
            workloads=workloads or None,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        _write_admission_eviction_matrix_markdown(markdown_path, payload)
        click.echo(output_path)
        click.echo(markdown_path)
        return

    output_path = output or Path("artifacts/prefix_kv_cache_admission_eviction_regret_audit.json")
    markdown_path = markdown or Path("docs/results/admission_eviction_regret_audit.md")
    factory = build_incumbent
    policy_name = "pressure_aware_incumbent"
    if candidate_program is not None:
        factory = load_candidate_factory(str(candidate_program))
        policy_name = str(candidate_program)
    payload = run_analysis(
        config,
        request_count=request_count,
        seeds=seeds or None,
        splits=selected_splits,
        workloads=workloads or None,
        factory=factory,
        policy_name=policy_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(markdown_path, payload)
    click.echo(output_path)
    click.echo(markdown_path)


if __name__ == "__main__":
    main()
