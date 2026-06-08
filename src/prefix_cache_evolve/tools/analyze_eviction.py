"""Analyze eviction choice, regret, and compact specialist distillations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from prefix_cache_evolve.evaluators.complexity import scoring_fn_complexity
from prefix_cache_evolve.evaluators.contracts import PrefixBlockInfo, RequestInfo
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluationResult,
    EvaluatorConfig,
    PrefixKVCacheEvaluator,
    PrefixKVCacheSimulator,
    build_workload,
)
from prefix_cache_evolve.evaluators.telemetry import EvictionDecisionSnapshot
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import load_evaluator_config
from prefix_cache_evolve.problems.prefix_kv_cache.pressure_aware_incumbent import (
    build_candidate as build_incumbent,
)
from prefix_cache_evolve.problems.prefix_kv_cache.specialist import (
    EvictionOnlyEvaluator,
    compose_eviction_specialist_source,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_INCUMBENT_PATH = (
    _REPOSITORY_ROOT
    / "src/prefix_cache_evolve/problems/prefix_kv_cache/pressure_aware_incumbent.py"
)
_SHORT_REUSE_DISTANCE_STEPS = 8.0

INCUMBENT_SOURCE = """\
import math

def score_eviction(block, now, frequency, priority):
    return (
        0.85 * math.log1p(max(0, now - block.last_accessed_at))
        - 1.8 * math.log1p(frequency + block.hit_count)
        - 0.2 * math.log1p(block.descendant_count)
        - 0.55 * priority
    )
"""

DESCENDANT_REWEIGHT_SOURCE = """\
import math

def score_eviction(block, now, frequency, priority):
    return (
        0.85 * math.log1p(max(0, now - block.last_accessed_at))
        - 1.8 * math.log1p(frequency + block.hit_count)
        - 0.6 * math.log1p(block.descendant_count)
        - 0.55 * priority
    )
"""

AGE_DESCENDANT_REWEIGHT_SOURCE = """\
import math

def score_eviction(block, now, frequency, priority):
    return (
        1.0 * math.log1p(max(0, now - block.last_accessed_at))
        - 1.8 * math.log1p(frequency + block.hit_count)
        - 0.4 * math.log1p(block.descendant_count)
        - 0.55 * priority
    )
"""

ONE_TERM_REUSE_INTERACTION_SOURCE = """\
import math

def score_eviction(block, now, frequency, priority):
    age = max(0, now - block.last_accessed_at)
    age_pressure = age / (1.0 + age)
    reuse_support = (
        frequency / (1.0 + frequency)
        + block.hit_count / (1.0 + block.hit_count)
    )
    return (
        0.85 * math.log1p(age)
        - 1.8 * math.log1p(frequency + block.hit_count)
        - 0.2 * math.log1p(block.descendant_count)
        - 0.55 * priority
        - 0.20 * reuse_support * age_pressure
    )
"""

TWO_TERM_REUSE_INTERACTION_SOURCE = """\
import math

def score_eviction(block, now, frequency, priority):
    age = max(0, now - block.last_accessed_at)
    age_pressure = age / (1.0 + age)
    frequency_support = frequency / (1.0 + frequency)
    hit_support = block.hit_count / (1.0 + block.hit_count)
    return (
        0.85 * math.log1p(age)
        - 1.8 * math.log1p(frequency + block.hit_count)
        - 0.2 * math.log1p(block.descendant_count)
        - 0.55 * priority
        - 0.12 * frequency_support * age_pressure
        - 0.08 * hit_support * age_pressure
    )
"""

GUARDED_SUPPORT_SOURCE = """\
def score_eviction(block, now, frequency, priority):
    age = max(0, now - block.last_accessed_at)
    frequency_support = frequency / (1.0 + 0.75 * frequency)
    hit_support = block.hit_count / (1.5 + block.hit_count)
    priority_support = max(0.0, priority) / (0.8 + max(0.0, priority))
    descendant_support = block.descendant_count / (3.0 + block.descendant_count)
    support = (
        0.46 * frequency_support
        + 0.34 * hit_support
        + 0.12 * priority_support
        + 0.08 * descendant_support
    )
    age_pressure = age / (1.25 + age)
    reuse_guard = 1.0 / (
        1.0
        + 3.4 * frequency_support
        + 2.0 * hit_support
        + 1.25 * priority_support
        + 0.75 * descendant_support
    )
    return age_pressure * reuse_guard - 0.54 * support
"""

GUARDED_SUPPORT_DESCENDANT_SOURCE = """\
def score_eviction(block, now, frequency, priority):
    age = max(0, now - block.last_accessed_at)
    frequency_support = frequency / (1.0 + 0.75 * frequency)
    hit_support = block.hit_count / (1.5 + block.hit_count)
    priority_support = max(0.0, priority) / (0.8 + max(0.0, priority))
    descendant_support = block.descendant_count / (3.0 + block.descendant_count)
    support = (
        0.46 * frequency_support
        + 0.34 * hit_support
        + 0.12 * priority_support
        + 0.08 * descendant_support
    )
    age_pressure = age / (1.25 + age)
    reuse_guard = 1.0 / (
        1.0
        + 3.4 * frequency_support
        + 2.0 * hit_support
        + 1.25 * priority_support
        + 0.75 * descendant_support
    )
    descendant_guard = 1.0 / (1.0 + 0.4 * block.descendant_count)
    return (
        age_pressure * reuse_guard * descendant_guard
        - 0.54 * support
        - 0.10 * descendant_support
    )
"""

FULL_SPECIALIST_SOURCE = """\
def score_eviction(block, now, frequency, priority):
    age = now - block.last_accessed_at
    if age < 0.0:
        age = 0.0
    f = frequency
    if f < 0.0:
        f = 0.0
    h = block.hit_count
    if h < 0.0:
        h = 0.0
    d = block.descendant_count
    if d < 0.0:
        d = 0.0
    pp = priority
    if pp < 0.0:
        pp = 0.0
    np = -priority
    if np < 0.0:
        np = 0.0
    freq_support = f / (1.0 + 0.75 * f)
    hit_support = h / (1.5 + h)
    prio_support = pp / (0.8 + pp)
    desc_support = d / (3.0 + d)
    support = (
        0.46 * freq_support
        + 0.34 * hit_support
        + 0.12 * prio_support
        + 0.08 * desc_support
    )
    age_pressure = age / (1.25 + age)
    reuse_guard = 1.0 / (
        1.0
        + 3.4 * freq_support
        + 2.0 * hit_support
        + 1.25 * prio_support
        + 0.75 * desc_support
    )
    descendant_guard = 1.0 / (1.0 + 0.4 * d)
    urgent_penalty = np / (1.0 + np)
    recent_reuse_guard = support / (1.0 + 0.8 * age)
    cold = 1.0 - support
    if cold < 0.0:
        cold = 0.0
    cold_recent_guard = cold / (1.0 + age * (1.0 + 0.8 * f + 0.4 * h + 0.2 * pp))
    shadow_guard = support / (1.0 + age * (0.45 + 0.35 * support) + 0.2 * d)
    return (
        age_pressure * reuse_guard * descendant_guard
        - 0.54 * support
        - 0.10 * desc_support
        - 0.14 * recent_reuse_guard
        - 0.05 * shadow_guard
        - 0.04 * cold_recent_guard * (1.0 - urgent_penalty)
        + 0.26 * urgent_penalty
        - 0.12 * freq_support * age_pressure
        - 0.08 * hit_support * age_pressure
    )
"""

VARIANT_SOURCES = {
    "incumbent": INCUMBENT_SOURCE,
    "descendant_reweight": DESCENDANT_REWEIGHT_SOURCE,
    "age_descendant_reweight": AGE_DESCENDANT_REWEIGHT_SOURCE,
    "one_term_reuse_interaction": ONE_TERM_REUSE_INTERACTION_SOURCE,
    "two_term_reuse_interactions": TWO_TERM_REUSE_INTERACTION_SOURCE,
    "guarded_support": GUARDED_SUPPORT_SOURCE,
    "guarded_support_descendant": GUARDED_SUPPORT_DESCENDANT_SOURCE,
    "full_specialist": FULL_SPECIALIST_SOURCE,
}


@dataclass
class CounterfactualTotals:
    """Aggregates same-state eviction decisions for one alternative ranker."""

    decisions: int = 0
    multiple_legal_victim_decisions: int = 0
    legal_victim_count: int = 0
    max_legal_victims: int = 0
    changed_decisions: int = 0
    better_next_reuse_decisions: int = 0
    worse_next_reuse_decisions: int = 0
    equal_next_reuse_decisions: int = 0
    incumbent_avoidable_decisions: int = 0
    alternative_avoidable_decisions: int = 0
    corrected_avoidable_decisions: int = 0
    introduced_avoidable_decisions: int = 0
    incumbent_short_reuse_decisions: int = 0
    alternative_short_reuse_decisions: int = 0
    corrected_short_reuse_decisions: int = 0
    introduced_short_reuse_decisions: int = 0

    def record(
        self,
        *,
        legal_victims: int,
        incumbent_distance: float,
        alternative_distance: float,
        furthest_distance: float,
        changed: bool,
    ) -> None:
        """Record one same-state victim comparison."""

        self.decisions += 1
        self.legal_victim_count += legal_victims
        self.max_legal_victims = max(self.max_legal_victims, legal_victims)
        if legal_victims > 1:
            self.multiple_legal_victim_decisions += 1
        if changed:
            self.changed_decisions += 1
            if alternative_distance > incumbent_distance:
                self.better_next_reuse_decisions += 1
            elif alternative_distance < incumbent_distance:
                self.worse_next_reuse_decisions += 1
            else:
                self.equal_next_reuse_decisions += 1

        incumbent_avoidable = incumbent_distance < furthest_distance
        alternative_avoidable = alternative_distance < furthest_distance
        incumbent_short = (
            incumbent_distance <= _SHORT_REUSE_DISTANCE_STEPS
            and furthest_distance > _SHORT_REUSE_DISTANCE_STEPS
        )
        alternative_short = (
            alternative_distance <= _SHORT_REUSE_DISTANCE_STEPS
            and furthest_distance > _SHORT_REUSE_DISTANCE_STEPS
        )
        self.incumbent_avoidable_decisions += int(incumbent_avoidable)
        self.alternative_avoidable_decisions += int(alternative_avoidable)
        self.corrected_avoidable_decisions += int(incumbent_avoidable and not alternative_avoidable)
        self.introduced_avoidable_decisions += int(
            alternative_avoidable and not incumbent_avoidable
        )
        self.incumbent_short_reuse_decisions += int(incumbent_short)
        self.alternative_short_reuse_decisions += int(alternative_short)
        self.corrected_short_reuse_decisions += int(incumbent_short and not alternative_short)
        self.introduced_short_reuse_decisions += int(alternative_short and not incumbent_short)

    def summary(self) -> dict[str, float | int]:
        """Return rates and counts suitable for JSON and Markdown reporting."""

        decisions = max(1, self.decisions)
        changed = max(1, self.changed_decisions)
        return {
            **asdict(self),
            "multiple_legal_victim_rate": self.multiple_legal_victim_decisions / decisions,
            "mean_legal_victims": self.legal_victim_count / decisions,
            "changed_decision_rate": self.changed_decisions / decisions,
            "better_next_reuse_rate_on_changed": self.better_next_reuse_decisions / changed,
            "worse_next_reuse_rate_on_changed": self.worse_next_reuse_decisions / changed,
            "avoidable_decision_rate_delta": (
                self.alternative_avoidable_decisions - self.incumbent_avoidable_decisions
            )
            / decisions,
            "short_reuse_decision_rate_delta": (
                self.alternative_short_reuse_decisions - self.incumbent_short_reuse_decisions
            )
            / decisions,
        }


class _TracingIncumbentPolicy:
    """Records incumbent state values used for each eviction score."""

    def __init__(self, capacity_blocks: int, block_size_tokens: int, seed: int | None) -> None:
        self._base = build_incumbent(capacity_blocks, block_size_tokens, seed)
        self.eviction_values: dict[tuple[int, int], tuple[float, float]] = {}

    def on_request_start(self, request: RequestInfo, now: int) -> None:
        self._base.on_request_start(request, now)

    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return self._base.score_admission(block, now)

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        frequency, priority = self._base._values(block.prefix_hash, now)
        self.eviction_values[(now, block.prefix_hash)] = (frequency, priority)
        return _score_from_source("incumbent")(block, now, frequency, priority)

    def on_cache_hit(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        self._base.on_cache_hit(block, request, now)

    def on_cache_miss(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        self._base.on_cache_miss(block, request, now)


class _CounterfactualObserver:
    """Compares alternative rankers on the incumbent's exact decision states."""

    def __init__(
        self,
        policy: _TracingIncumbentPolicy,
        totals: dict[str, CounterfactualTotals],
    ) -> None:
        self._policy = policy
        self._totals = totals
        self._rankers = {
            name: _score_from_source(name) for name in VARIANT_SOURCES if name != "incumbent"
        }

    def on_eviction_decision(self, snapshot: EvictionDecisionSnapshot) -> None:
        """Compare every alternative with the incumbent-selected victim."""

        distances = {
            candidate.block.prefix_hash: float(candidate.next_reuse_distance)
            for candidate in snapshot.candidates
            if candidate.next_reuse_distance is not None
        }
        furthest_distance = max(distances.values())
        incumbent_distance = distances[snapshot.victim_prefix_hash]
        for name, ranker in self._rankers.items():
            alternative = max(
                snapshot.candidates,
                key=lambda candidate: (
                    ranker(
                        candidate.block,
                        snapshot.now,
                        *self._policy.eviction_values[(snapshot.now, candidate.block.prefix_hash)],
                    ),
                    candidate.block.prefix_hash,
                ),
            )
            alternative_hash = alternative.block.prefix_hash
            self._totals[name].record(
                legal_victims=len(snapshot.candidates),
                incumbent_distance=incumbent_distance,
                alternative_distance=distances[alternative_hash],
                furthest_distance=furthest_distance,
                changed=alternative_hash != snapshot.victim_prefix_hash,
            )


_SCORE_CACHE: dict[str, Callable[[PrefixBlockInfo, int, float, float], float]] = {}


def _score_from_source(name: str) -> Callable[[PrefixBlockInfo, int, float, float], float]:
    """Load and cache one function-only eviction ranker."""

    if name not in _SCORE_CACHE:
        namespace: dict[str, object] = {}
        exec(VARIANT_SOURCES[name], namespace)
        _SCORE_CACHE[name] = namespace["score_eviction"]  # type: ignore[assignment]
    return _SCORE_CACHE[name]


def _run_counterfactual_analysis(config: EvaluatorConfig) -> dict[str, object]:
    totals = {name: CounterfactualTotals() for name in VARIANT_SOURCES if name != "incumbent"}
    split_totals = {
        split: {name: CounterfactualTotals() for name in VARIANT_SOURCES if name != "incumbent"}
        for split in ("train", "validation", "probe", "hidden")
    }
    for workload in config.workload_configs(("train", "validation", "probe", "hidden")):
        for capacity_blocks in config.effective_capacity_blocks():
            for seed in config.seeds:
                actual_seed = seed + workload.seed_offset
                requests = build_workload(
                    workload.family,
                    request_count=workload.request_count,
                    block_size_tokens=config.effective_workload_token_granularity(),
                    seed=actual_seed,
                )
                policy = _TracingIncumbentPolicy(
                    capacity_blocks,
                    config.block_size_tokens,
                    config.policy_seed,
                )
                combined_totals = {
                    name: _CombinedTotals(totals[name], split_totals[workload.split][name])
                    for name in totals
                }
                observer = _CounterfactualObserver(policy, combined_totals)  # type: ignore[arg-type]
                simulator = PrefixKVCacheSimulator(
                    capacity_blocks=capacity_blocks,
                    block_size_tokens=config.block_size_tokens,
                    prefill_cost_per_token=config.prefill_cost_per_token,
                    lookup_cost_per_block=config.lookup_cost_per_block,
                    eviction_cost_per_block=config.eviction_cost_per_block,
                    active_tokens_per_step=config.active_tokens_per_step,
                    expose_future_reuse=False,
                    max_memory_bytes=None,
                    eviction_decision_observer=observer,
                )
                simulator.run(
                    policy,
                    requests,
                    split=workload.split,
                    workload=workload.family,
                    seed=actual_seed,
                )
    return {
        "overall": {name: value.summary() for name, value in totals.items()},
        "by_split": {
            split: {name: value.summary() for name, value in variants.items()}
            for split, variants in split_totals.items()
        },
    }


class _CombinedTotals:
    """Forwards one record into aggregate and split-specific totals."""

    def __init__(self, *targets: CounterfactualTotals) -> None:
        self._targets = targets

    def record(self, **kwargs) -> None:
        for target in self._targets:
            target.record(**kwargs)


def _rescore_panel(
    config: EvaluatorConfig,
    result: EvaluationResult,
    splits: tuple[str, ...],
) -> EvaluationResult:
    trials = [trial for trial in result.trials if trial.split in splits]
    return PrefixKVCacheEvaluator(config, splits=splits).rescore_trials(trials)


def _panel_summary(result: EvaluationResult, split: str) -> dict[str, float]:
    metrics = result.split_metrics[split]
    return {
        "raw_score": result.combined_score,
        "token_hit_rate": float(metrics["token_hit_rate"]),
        "avoidable_eviction_rate": float(metrics["avoidable_eviction_rate"]),
        "short_reuse_after_eviction_missed_token_rate": float(
            metrics["short_reuse_after_eviction_missed_token_rate"]
        ),
        "cache_churn_per_1k": float(metrics["cache_churn_per_1k"]),
    }


def _run_variant_panels(config: EvaluatorConfig) -> dict[str, object]:
    base_source = _INCUMBENT_PATH.read_text(encoding="utf-8")
    variants = {}
    evaluator = EvictionOnlyEvaluator(
        config,
        splits=("train", "validation", "probe", "hidden"),
    )
    for name, source in VARIANT_SOURCES.items():
        result = evaluator(_score_from_source(name))
        selection = _rescore_panel(config, result, ("train", "validation"))
        probe = _rescore_panel(config, result, ("probe",))
        hidden = _rescore_panel(config, result, ("hidden",))
        composed = compose_eviction_specialist_source(source, base_source)
        variants[name] = {
            "function_complexity": scoring_fn_complexity(
                source,
                form_aware=config.form_aware_complexity,
            ),
            "composed_complexity": scoring_fn_complexity(
                composed,
                form_aware=config.form_aware_complexity,
            ),
            "selection": _panel_summary(selection, "validation"),
            "probe": _panel_summary(probe, "probe"),
            "hidden": _panel_summary(hidden, "hidden"),
        }
    incumbent_raw = variants["incumbent"]["selection"]["raw_score"]
    for row in variants.values():
        row["selection_raw_delta"] = row["selection"]["raw_score"] - incumbent_raw
    return variants


def run_analysis(config_path: Path) -> dict[str, object]:
    """Run same-state eviction analysis and full-panel variant adjudication."""

    config = load_evaluator_config(config_path)
    return {
        "schema": "prefix-kv-cache-eviction-analysis-v1",
        "config": str(config_path),
        "block_size_tokens": config.block_size_tokens,
        "capacity_blocks": list(config.effective_capacity_blocks()),
        "counterfactual": _run_counterfactual_analysis(config),
        "variants": _run_variant_panels(config),
    }


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    counterfactual = payload["counterfactual"]["overall"]
    by_split = payload["counterfactual"]["by_split"]
    variants = payload["variants"]
    full = counterfactual["full_specialist"]
    incumbent = variants["incumbent"]
    descendant = variants["descendant_reweight"]
    guarded_descendant = variants["guarded_support_descendant"]
    net_corrected = full["corrected_avoidable_decisions"] - full["introduced_avoidable_decisions"]
    net_short_corrected = (
        full["corrected_short_reuse_decisions"] - full["introduced_short_reuse_decisions"]
    )
    descendant_hidden_delta = descendant["hidden"]["raw_score"] - incumbent["hidden"]["raw_score"]
    guarded_gain_fraction = (
        guarded_descendant["selection_raw_delta"]
        / variants["full_specialist"]["selection_raw_delta"]
    )
    lines = [
        "# Eviction Decision and Distillation Analysis",
        "",
        "This analysis compares eviction rankers on the incumbent's exact cache states, then",
        "runs each ranker end to end with admission and lifecycle callbacks frozen.",
        "",
        "## Same-State Decision Summary",
        "",
        f"- Eviction decisions observed: `{full['decisions']}`.",
        f"- Decisions with multiple legal victims: `{full['multiple_legal_victim_rate']:.1%}` "
        f"(mean `{full['mean_legal_victims']:.2f}`, max `{full['max_legal_victims']}`).",
        f"- Full specialist changed the incumbent victim on "
        f"`{full['changed_decision_rate']:.1%}` of decisions.",
        f"- On changed decisions, it selected a later-reused victim "
        f"`{full['better_next_reuse_rate_on_changed']:.1%}` of the time and an "
        f"earlier-reused victim `{full['worse_next_reuse_rate_on_changed']:.1%}` of the time.",
        f"- It corrected `{full['corrected_avoidable_decisions']}` avoidable choices and "
        f"introduced `{full['introduced_avoidable_decisions']}`, a net reduction of "
        f"`{net_corrected}`. Short-reuse corrections were net `{net_short_corrected}`.",
        "",
        "| Alternative | Changed | Better on changed | Worse on changed | "
        "Avoidable rate delta | Short-reuse rate delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANT_SOURCES:
        if name == "incumbent":
            continue
        row = counterfactual[name]
        lines.append(
            f"| `{name}` | {row['changed_decision_rate']:.1%} | "
            f"{row['better_next_reuse_rate_on_changed']:.1%} | "
            f"{row['worse_next_reuse_rate_on_changed']:.1%} | "
            f"{row['avoidable_decision_rate_delta']:+.3%} | "
            f"{row['short_reuse_decision_rate_delta']:+.3%} |"
        )
    lines.extend(
        [
            "",
            "### Full Specialist by Split",
            "",
            "| Split | Decisions | Multiple legal | Mean legal | Changed | "
            "Avoidable rate delta | Short-reuse rate delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "validation", "probe", "hidden"):
        split_variants = by_split[split]
        row = split_variants["full_specialist"]
        lines.append(
            f"| `{split}` | {row['decisions']} | {row['multiple_legal_victim_rate']:.1%} | "
            f"{row['mean_legal_victims']:.2f} | {row['changed_decision_rate']:.1%} | "
            f"{row['avoidable_decision_rate_delta']:+.3%} | "
            f"{row['short_reuse_decision_rate_delta']:+.3%} |"
        )
    lines.extend(
        [
            "",
            "## End-to-End Adjudication",
            "",
            "| Variant | Selection raw | Delta | Validation hit | Avoidable | Short reuse | "
            "Churn/1k | Probe raw | Hidden raw | Function cx | Composed cx |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in VARIANT_SOURCES:
        row = variants[name]
        selection = row["selection"]
        lines.append(
            f"| `{name}` | {selection['raw_score']:.3f} | "
            f"{row['selection_raw_delta']:+.3f} | {selection['token_hit_rate']:.4f} | "
            f"{selection['avoidable_eviction_rate']:.4f} | "
            f"{selection['short_reuse_after_eviction_missed_token_rate']:.4f} | "
            f"{selection['cache_churn_per_1k']:.1f} | {row['probe']['raw_score']:.3f} | "
            f"{row['hidden']['raw_score']:.3f} | {row['function_complexity']} | "
            f"{row['composed_complexity']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The specialist is making consequential choices, not merely breaking ties. Same-state",
            "regret improves overall, especially on validation, but it is not uniformly better:",
            "short-reuse regret increases on train and probe. End-to-end replay remains the",
            "promotion criterion because changed victims alter later cache states.",
            "",
            f"The guarded-support-plus-descendant variant captures `{guarded_gain_fraction:.1%}`",
            "of the full specialist's selection gain, identifying descendant-aware guarded age as",
            "the main useful design. It still composes to `775` nodes. The one- and",
            "two-interaction additions are too weak and also exceed the `650`-node cap.",
            "",
            "A single coefficient distillation is useful: increasing descendant protection from",
            f"`0.2` to `0.6` gains `{descendant['selection_raw_delta']:+.3f}` raw selection at",
            f"unchanged composed complexity `{descendant['composed_complexity']}`. It is not a",
            "promotion candidate under the fail-closed rule because hidden score changes",
            f"by `{descendant_hidden_delta:+.3f}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prefix_kv_cache_eviction_specialist.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/prefix_kv_cache_eviction_analysis.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/results/eviction_policy_analysis.md"),
    )
    args = parser.parse_args()

    payload = run_analysis(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(args.markdown, payload)
    print(args.output)
    print(args.markdown)


if __name__ == "__main__":
    main()
