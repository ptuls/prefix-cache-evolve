"""Configurable specialist-search policy composition."""

from __future__ import annotations

import ast
import copy
from collections.abc import Sequence
from typing import Callable

from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.evaluators.contracts import PrefixBlockInfo, PrefixKVPolicy, RequestInfo
from prefix_cache_evolve.evaluators.prefix_kv_cache import PrefixKVCacheEvaluator
from prefix_cache_evolve.evaluators.results import EvaluationResult

from .incumbents import build_discovery_incumbent
from .incumbents.registry import current_incumbent

_DISCOVERY_INCUMBENT_ID = current_incumbent("discovery").incumbent_id
_FIXED_ADMISSION_POLICIES: dict[str, Callable[..., PrefixKVPolicy]] = {
    _DISCOVERY_INCUMBENT_ID: build_discovery_incumbent,
    "pressure_aware_incumbent": build_discovery_incumbent,
}
_EVICTION_ONLY_ARGUMENTS = ("block", "now", "frequency", "priority")
_FULL_POLICY_METHODS = {
    "on_request_start",
    "score_admission",
    "on_cache_hit",
    "on_cache_miss",
}
_POLICY_FACTORY_NAMES = {"build_candidate", "candidate_factory"}


def candidate_evaluator(
    config: EvaluatorConfig,
    *,
    splits: tuple[str, ...],
) -> PrefixKVCacheEvaluator | EvictionOnlyEvaluator:
    """Build an evaluator with any configured candidate-only specialist controls."""
    if config.candidate_policy_surface == "eviction_only":
        return EvictionOnlyEvaluator(config, splits=splits)
    if config.candidate_policy_surface != "full":
        raise ValueError(f"unknown candidate_policy_surface {config.candidate_policy_surface!r}")
    return PrefixKVCacheEvaluator(
        config,
        splits=splits,
        fixed_admission_factory=fixed_admission_factory(config.fixed_admission_policy),
    )


def fixed_admission_factory(
    policy_name: str | None,
) -> Callable[..., PrefixKVPolicy] | None:
    """Resolve a configured fixed-admission policy factory."""
    if policy_name is None:
        return None
    try:
        return _FIXED_ADMISSION_POLICIES[policy_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_FIXED_ADMISSION_POLICIES))
        raise ValueError(
            f"unknown fixed_admission_policy {policy_name!r}; supported values: {supported}"
        ) from exc


def candidate_exported_names(config: EvaluatorConfig) -> tuple[str, ...]:
    """Return the candidate entry points accepted by one evaluator configuration."""
    if config.candidate_policy_surface == "eviction_only":
        return ("score_eviction",)
    return ("candidate_factory", "build_candidate")


class EvictionOnlyEvaluator:
    """Composes one eviction-ranking function with a frozen complete base policy."""

    def __init__(self, config: EvaluatorConfig, *, splits: tuple[str, ...]) -> None:
        if config.fixed_admission_policy is None:
            raise ValueError("eviction_only candidates require fixed_admission_policy")
        self.config = config
        self.splits = splits
        self._base_factory = fixed_admission_factory(config.fixed_admission_policy)
        self._evaluator = PrefixKVCacheEvaluator(config, splits=splits)

    def __call__(
        self,
        score_eviction: Callable[..., float] | None = None,
        *,
        scoring_fn_complexity: int = 0,
    ) -> EvaluationResult:
        """Evaluate an eviction scorer composed with the fixed base policy."""
        return self._evaluator(
            self._composed_factory(score_eviction),
            scoring_fn_complexity=scoring_fn_complexity,
        )

    def evaluate_requests(
        self,
        score_eviction: Callable[..., float] | None,
        requests,
        *,
        workload: str = "trace_replay",
        split: str = "validation",
        seed: int = 0,
        scoring_fn_complexity: int = 0,
    ) -> EvaluationResult:
        """Evaluate one fixed request stream using frozen admission and callbacks."""
        return self._evaluator.evaluate_requests(
            self._composed_factory(score_eviction),
            requests,
            workload=workload,
            split=split,
            seed=seed,
            scoring_fn_complexity=scoring_fn_complexity,
        )

    def _composed_factory(
        self,
        score_eviction: Callable[..., float] | None,
    ) -> Callable[..., PrefixKVPolicy]:
        if score_eviction is None or not callable(score_eviction):
            raise TypeError("eviction-only candidate must expose callable score_eviction")
        base_factory = self._base_factory
        if base_factory is None:  # pragma: no cover - guarded by constructor
            raise ValueError("eviction-only evaluator has no fixed base policy")

        def build_composed_policy(
            capacity_blocks: int,
            block_size_tokens: int,
            seed: int | None = None,
        ) -> PrefixKVPolicy:
            return _EvictionOnlyPolicy(
                base_policy=base_factory(capacity_blocks, block_size_tokens, seed),
                score_eviction=score_eviction,
            )

        return build_composed_policy


class _EvictionOnlyPolicy:
    """Delegates admission and lifecycle state to the frozen incumbent."""

    def __init__(
        self,
        *,
        base_policy: PrefixKVPolicy,
        score_eviction: Callable[..., float],
    ) -> None:
        self._base_policy = base_policy
        self._score_eviction = score_eviction

    def on_request_start(self, request: RequestInfo, now: int) -> None:
        self._base_policy.on_request_start(request, now)

    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return self._base_policy.score_admission(block, now)

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        values = getattr(self._base_policy, "_values")
        frequency, priority = values(block.prefix_hash, now)
        return self._score_eviction(block, now, frequency, priority)

    def on_cache_hit(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        self._base_policy.on_cache_hit(block, request, now)

    def on_cache_miss(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        self._base_policy.on_cache_miss(block, request, now)


def eviction_only_source_violations(source: str) -> tuple[str, ...]:
    """Return violations of the function-only eviction specialist contract."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()

    violations = []
    score_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "score_eviction"
    ]
    if len(score_functions) != 1:
        violations.append(
            "eviction-only specialist must define exactly one score_eviction function"
        )
    elif isinstance(score_functions[0], ast.AsyncFunctionDef):
        violations.append("eviction-only score_eviction must be synchronous")
    else:
        function = score_functions[0]
        arguments = tuple(argument.arg for argument in function.args.args)
        if (
            arguments != _EVICTION_ONLY_ARGUMENTS
            or function.args.posonlyargs
            or function.args.kwonlyargs
            or function.args.vararg is not None
            or function.args.kwarg is not None
            or function.args.defaults
        ):
            violations.append(
                "eviction-only score_eviction signature must be "
                "score_eviction(block, now, frequency, priority)"
            )
        if function.decorator_list:
            violations.append(
                "eviction-only score_eviction must be undecorated so promotion "
                "composition preserves behavior"
            )

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            violations.append("eviction-only specialist must not define policy classes")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _FULL_POLICY_METHODS:
                violations.append(f"eviction-only specialist must not define {node.name}")
            if node.name in _POLICY_FACTORY_NAMES:
                violations.append(
                    "eviction-only specialist must expose score_eviction, not a policy factory"
                )
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append("eviction-only specialist must not use global or nonlocal state")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = _assigned_names(node)
            if names & _POLICY_FACTORY_NAMES:
                violations.append(
                    "eviction-only specialist must expose score_eviction, not a policy factory"
                )
            if any(not name.isupper() for name in names):
                violations.append(
                    "eviction-only specialist top-level assignments must be uppercase constants"
                )
            value = node.value
            if isinstance(value, (ast.Dict, ast.List, ast.Set)):
                violations.append(
                    "eviction-only specialist must not define mutable top-level state"
                )
    if any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in ast.walk(tree)):
        violations.append("eviction-only specialist must not use global or nonlocal state")
    return tuple(dict.fromkeys(violations))


def compose_eviction_specialist_source(
    specialist_source: str,
    base_source: str,
) -> str:
    """Compose one function-only eviction specialist into the complete incumbent."""
    violations = eviction_only_source_violations(specialist_source)
    if violations:
        raise ValueError("; ".join(violations))

    specialist_tree = ast.parse(specialist_source)
    base_tree = ast.parse(base_source)
    specialist_function = next(
        node
        for node in specialist_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "score_eviction"
    )
    base_imports = {
        ast.dump(node) for node in base_tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    support_nodes = [
        copy.deepcopy(node)
        for node in specialist_tree.body
        if isinstance(
            node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.FunctionDef)
        )
        and node is not specialist_function
        and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        and not (isinstance(node, (ast.Import, ast.ImportFrom)) and ast.dump(node) in base_imports)
    ]
    collisions = _top_level_names(support_nodes) & _top_level_names(base_tree.body)
    if collisions:
        raise ValueError(
            "eviction specialist support names collide with incumbent: "
            + ", ".join(sorted(collisions))
        )

    replaced = False
    for node in base_tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for method in node.body:
            if isinstance(method, ast.FunctionDef) and method.name == "score_eviction":
                method.body = [
                    ast.Assign(
                        targets=[
                            ast.Tuple(
                                elts=[
                                    ast.Name(id="frequency", ctx=ast.Store()),
                                    ast.Name(id="priority", ctx=ast.Store()),
                                ],
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_values",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.Attribute(
                                    value=ast.Name(id="block", ctx=ast.Load()),
                                    attr="prefix_hash",
                                    ctx=ast.Load(),
                                ),
                                ast.Name(id="now", ctx=ast.Load()),
                            ],
                            keywords=[],
                        ),
                    ),
                    *copy.deepcopy(specialist_function.body),
                ]
                replaced = True
                break
        if replaced:
            break
    if not replaced:
        raise ValueError("incumbent source does not define score_eviction")

    insert_at = 0
    while insert_at < len(base_tree.body) and isinstance(
        base_tree.body[insert_at],
        (ast.Expr, ast.Import, ast.ImportFrom),
    ):
        insert_at += 1
    base_tree.body[insert_at:insert_at] = support_nodes
    ast.fix_missing_locations(base_tree)
    return ast.unparse(base_tree) + "\n"


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _top_level_names(nodes: Sequence[ast.stmt]) -> set[str]:
    names = set()
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names.update(_assigned_names(node))
        elif isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names
