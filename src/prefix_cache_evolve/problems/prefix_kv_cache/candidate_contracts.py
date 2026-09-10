"""Typed candidate module contracts for prefix KV-cache evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig


@dataclass(frozen=True)
class CandidateContract:
    """Defines the module export and signature required for one policy surface."""

    policy_surface: str
    exported_names: tuple[str, ...]
    signature: str
    load_repair_feedback: str


FULL_POLICY_CONTRACT = CandidateContract(
    policy_surface="full",
    exported_names=("candidate_factory", "build_candidate"),
    signature="build_candidate(capacity_blocks, block_size_tokens, seed=None)",
    load_repair_feedback=(
        "Define build_candidate(capacity_blocks, block_size_tokens, seed=None) and return "
        "the policy object."
    ),
)
EVICTION_ONLY_CONTRACT = CandidateContract(
    policy_surface="eviction_only",
    exported_names=("score_eviction",),
    signature="score_eviction(block, now, frequency, priority)",
    load_repair_feedback=(
        "Define score_eviction(block, now, frequency, priority) as the only candidate entry point."
    ),
)
_CONTRACTS = {
    contract.policy_surface: contract for contract in (FULL_POLICY_CONTRACT, EVICTION_ONLY_CONTRACT)
}


def candidate_contract(config: EvaluatorConfig) -> CandidateContract:
    """Return the candidate contract selected by an evaluator configuration."""
    try:
        return _CONTRACTS[config.candidate_policy_surface]
    except KeyError as exc:
        raise ValueError(
            f"unknown candidate_policy_surface {config.candidate_policy_surface!r}"
        ) from exc
