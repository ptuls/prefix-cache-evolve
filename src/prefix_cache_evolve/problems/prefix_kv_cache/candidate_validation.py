"""Static validation and repair guidance for candidate policy source."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.problems.prefix_kv_cache.specialist import (
    eviction_only_source_violations,
)

_UNSUPPORTED_CALLBACKS = {
    "on_request_end",
    "on_block_admitted",
    "on_block_evicted",
}
_FUTURE_REUSE_FIELDS = {
    "estimated_future_reuse",
    "estimated_next_reuse_distance",
}
_SANITIZED_REQUEST_FIELDS = {
    "prompt_tokens",
    "request_type",
}
_ALLOWED_PRIMITIVE_IMPORTS = {
    "MultiTimescaleDecay",
    "decay_vector",
    "threshold_excess",
}
_DYNAMIC_BUILTINS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "id",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
_PRIMITIVE_MODULE = "prefix_cache_evolve.problems.prefix_kv_cache.primitives"


@dataclass(frozen=True)
class CandidateSourceValidation:
    """Immutable result of validating one candidate source."""

    violations: tuple[str, ...]
    repair_feedback: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether the candidate source satisfies the static contract."""
        return not self.violations

    @property
    def violation_summary(self) -> str:
        """Return violations in their deterministic display order."""
        return "; ".join(self.violations)

    @property
    def repair_summary(self) -> str:
        """Return repair instructions in their deterministic display order."""
        return " ".join(self.repair_feedback)


def validate_candidate_source(
    source: str,
    complexity: int,
    config: EvaluatorConfig,
) -> CandidateSourceValidation:
    """Validate candidate source and derive matching repair instructions."""
    violations = candidate_source_violations(source, complexity, config)
    return CandidateSourceValidation(
        violations=violations,
        repair_feedback=static_repair_feedback(violations, complexity, config),
    )


def candidate_source_violations(
    source: str,
    complexity: int,
    config: EvaluatorConfig,
) -> tuple[str, ...]:
    """Return deterministic static violations for one candidate source."""
    violations = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        line = f" at line {exc.lineno}" if exc.lineno is not None else ""
        return (f"syntax error{line}: {exc.msg}",)

    if config.max_candidate_complexity is not None and complexity > config.max_candidate_complexity:
        violations.append(
            f"effective complexity {complexity} exceeds limit {config.max_candidate_complexity}"
        )
    if config.candidate_policy_surface == "eviction_only":
        violations.extend(eviction_only_source_violations(source))
    elif config.candidate_policy_surface != "full":
        violations.append(f"unknown candidate policy surface {config.candidate_policy_surface}")
    if not config.reject_unsupported_source_patterns:
        return tuple(violations)

    imported_names: dict[str, str] = {}
    used_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for index, node in enumerate(tree.body):
        violations.extend(_top_level_source_violations(node, index))
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                imported_names[name] = alias.name
                if alias.name != "math":
                    violations.append(f"import from unsupported module {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            for alias in node.names:
                name = alias.asname or alias.name
                imported_names[name] = alias.name
            violations.extend(_import_from_violations(node))
        elif isinstance(node, ast.ImportFrom):
            if node.module != "__future__" or tuple(alias.name for alias in node.names) != (
                "annotations",
            ):
                violations.append("only from __future__ import annotations is allowed")

    for name, imported_from in imported_names.items():
        if name == "*":
            violations.append("star imports are not allowed")
        elif name not in used_names:
            violations.append(f"unused import {imported_from}")

    for descendant in ast.walk(tree):
        if isinstance(descendant, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if descendant.name in _UNSUPPORTED_CALLBACKS:
                violations.append(f"unsupported callback {descendant.name}")
            if descendant.decorator_list:
                violations.append("decorators are not allowed in candidate code")
        elif isinstance(descendant, ast.ClassDef):
            if descendant.decorator_list:
                violations.append("decorators are not allowed in candidate code")
        elif isinstance(descendant, ast.Attribute) and descendant.attr in _FUTURE_REUSE_FIELDS:
            violations.append(f"future-knowledge field {descendant.attr} is not deployable")
        elif isinstance(descendant, ast.Attribute) and descendant.attr in _SANITIZED_REQUEST_FIELDS:
            violations.append(f"sanitized request field {descendant.attr} is not a policy signal")
        elif isinstance(descendant, ast.Attribute) and _is_dunder_name(descendant.attr):
            violations.append(f"dunder attribute {descendant.attr} is not allowed")
        elif (
            isinstance(descendant, ast.Name)
            and isinstance(descendant.ctx, ast.Load)
            and _is_dunder_name(descendant.id)
        ):
            violations.append(f"dunder name {descendant.id} is not allowed")
        elif (
            isinstance(descendant, ast.Name)
            and isinstance(descendant.ctx, ast.Load)
            and descendant.id in _DYNAMIC_BUILTINS
        ):
            violations.append(f"{descendant.id}() is not allowed in candidate code")
        elif isinstance(descendant, ast.ExceptHandler) and _is_broad_exception_handler(descendant):
            violations.append("broad exception handlers are not allowed")
        elif isinstance(descendant, ast.Call):
            called_name = _called_name(descendant.func)
            if called_name in _DYNAMIC_BUILTINS:
                violations.append(f"{called_name}() is not allowed in candidate code")
            elif called_name == "MultiTimescaleDecay":
                violations.extend(_multi_timescale_decay_violations(descendant))
            elif called_name == "threshold_excess":
                violations.extend(_threshold_excess_violations(descendant))

    return tuple(dict.fromkeys(violations))


def static_repair_feedback(
    violations: tuple[str, ...],
    complexity: int,
    config: EvaluatorConfig,
) -> tuple[str, ...]:
    """Translate static violations into concise, actionable repair instructions."""
    repairs = []
    for violation in violations:
        if violation.startswith("syntax error"):
            repairs.append(f"Fix the reported {violation} before changing policy behavior.")
        elif violation.startswith("effective complexity"):
            limit = config.max_candidate_complexity
            excess = max(1, complexity - limit) if limit is not None else 1
            repairs.append(
                f"Delete or simplify at least {excess} effective AST nodes; do not add a "
                "replacement subsystem."
            )
        elif violation.startswith("unused import "):
            repairs.append(f"Delete {violation.removeprefix('unused import ')} from the imports.")
        elif violation.startswith("unsupported callback "):
            repairs.append(f"Delete {violation.removeprefix('unsupported callback ')} entirely.")
        elif violation.startswith("eviction-only specialist"):
            repairs.append(
                "Keep only top-level imports, constants, optional helper functions, and "
                "score_eviction(block, now, frequency, priority)."
            )
        elif violation.startswith("eviction-only score_eviction must be undecorated"):
            repairs.append(
                "Remove decorators from score_eviction so exploration and promotion "
                "execute the same function body."
            )
        elif violation.startswith("eviction-only score_eviction"):
            repairs.append("Use exactly def score_eviction(block, now, frequency, priority):.")
        elif violation.startswith("future-knowledge field "):
            field = violation.removeprefix("future-knowledge field ").removesuffix(
                " is not deployable"
            )
            repairs.append(
                f"Remove {field}; use observed recurrence, subtree, gap, "
                "or pressure fields instead."
            )
        elif violation.startswith("sanitized request field "):
            field = violation.removeprefix("sanitized request field ").removesuffix(
                " is not a policy signal"
            )
            repairs.append(
                f"Remove {field}; it is deliberately scrubbed before candidate callbacks."
            )
        elif violation.endswith("() is not allowed in candidate code"):
            repairs.append("Remove the dynamic or introspective builtin call.")
        elif violation.startswith("import from unsupported module "):
            repairs.append("Remove the import; candidate code may import only math and primitives.")
        elif violation.startswith("unsupported primitive import "):
            repairs.append("Import only documented helpers from the policy primitives module.")
        elif violation == "broad exception handlers are not allowed":
            repairs.append("Remove the broad try/except and use the documented contract directly.")
        elif violation == "star imports are not allowed":
            repairs.append("Replace the star import with only the top-level names actually used.")
        elif violation.startswith("MultiTimescaleDecay"):
            repairs.append(
                "Use MultiTimescaleDecay((4.0, 20.0), max_keys=64) or delete the primitive."
            )
        elif violation.startswith("threshold_excess"):
            repairs.append("Call threshold_excess(value, threshold) with exactly two arguments.")
        else:
            repairs.append(f"Remove or repair this violation: {violation}.")
    return tuple(dict.fromkeys(repairs))


def _top_level_source_violations(node: ast.stmt, index: int) -> tuple[str, ...]:
    """Return violations of the fail-closed candidate module grammar."""
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
        return ()
    if (
        index == 0
        and isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ):
        return ()
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if len(names) != len(targets):
            return ("top-level assignments must target simple names",)
        value_node = node.value
        if value_node is None:
            return ("top-level assignments must have values",)
        if names == ["candidate_factory"]:
            if isinstance(value_node, ast.Name) and value_node.id == "build_candidate":
                return ()
            return ("candidate_factory must be a direct alias of build_candidate",)
        if names == ["__all__"]:
            try:
                value = ast.literal_eval(value_node)
            except (ValueError, TypeError):
                return ("__all__ must be a literal sequence",)
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(item, str) for item in value
            ):
                return ("__all__ must be a literal sequence of names",)
            return ()
        if not names or any(not name.isupper() for name in names):
            return ("top-level assignments must define uppercase literal constants",)
        try:
            ast.literal_eval(value_node)
        except (ValueError, TypeError):
            return ("top-level constants must use literal values",)
        return ()
    return (f"unsupported top-level statement {type(node).__name__}",)


def _import_from_violations(node: ast.ImportFrom) -> tuple[str, ...]:
    """Return violations for one non-future from-import."""
    if node.level:
        return ("relative imports are not allowed in candidate code",)
    if node.module == "math":
        return ()
    if node.module != _PRIMITIVE_MODULE:
        return (f"import from unsupported module {node.module}",)
    violations = []
    for alias in node.names:
        if alias.name not in _ALLOWED_PRIMITIVE_IMPORTS:
            violations.append(f"unsupported primitive import {alias.name}")
    return tuple(violations)


def _is_dunder_name(name: str) -> bool:
    """Return whether a loaded name accesses Python implementation internals."""
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _is_broad_exception_handler(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name):
        return handler.type.id in {"BaseException", "Exception"}
    return isinstance(handler.type, ast.Attribute) and handler.type.attr in {
        "BaseException",
        "Exception",
    }


def _called_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _multi_timescale_decay_violations(node: ast.Call) -> tuple[str, ...]:
    violations = []
    if len(node.args) > 1:
        violations.append("MultiTimescaleDecay accepts only one positional argument")
    half_lives = (
        node.args[0]
        if node.args
        else next(
            (keyword.value for keyword in node.keywords if keyword.arg == "half_lives"),
            None,
        )
    )
    if half_lives is None:
        violations.append("MultiTimescaleDecay requires a half-life sequence")
    elif isinstance(half_lives, ast.Constant):
        violations.append("MultiTimescaleDecay half-lives must be a sequence")
    elif isinstance(half_lives, (ast.List, ast.Tuple)):
        width = len(half_lives.elts)
        if not 1 <= width <= 8:
            violations.append("MultiTimescaleDecay requires one to eight half-lives")
    return tuple(violations)


def _threshold_excess_violations(node: ast.Call) -> tuple[str, ...]:
    """Validate the compact stateless threshold primitive's call shape."""
    violations = []
    if len(node.args) > 2:
        violations.append("threshold_excess accepts at most two positional arguments")
    keyword_names = {keyword.arg for keyword in node.keywords}
    if None in keyword_names or keyword_names - {"value", "threshold"}:
        violations.append("threshold_excess accepts only value and threshold arguments")
    positional_names = set(("value", "threshold")[: len(node.args)])
    if positional_names & keyword_names:
        violations.append("threshold_excess arguments must be supplied only once")
    supplied_names = positional_names | {name for name in keyword_names if name is not None}
    if supplied_names != {"value", "threshold"}:
        violations.append("threshold_excess requires value and threshold")
    return tuple(violations)
