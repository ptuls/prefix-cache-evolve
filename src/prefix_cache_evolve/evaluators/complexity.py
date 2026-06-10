"""Candidate source-complexity analysis."""

from __future__ import annotations

import ast

_FORM_AWARE_PRIMITIVE_MODULE = "prefix_cache_evolve.problems.prefix_kv_cache.primitives"
_FORM_AWARE_PRIMITIVE_CALL_CREDIT = 3
_FORM_AWARE_STATELESS_CALL_CREDIT = 1
_FORM_AWARE_MAX_DISCOUNT_FRACTION = 0.25


def scoring_fn_complexity(source: str, *, form_aware: bool = False) -> int:
    """Count effective AST nodes in the candidate policy implementation."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 10_000

    ignored_top_level_functions = {"build_candidate", "candidate_factory", "run_demo"}
    total = 0
    implementation_roots: list[ast.AST] = []
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if _is_module_docstring(node, index):
            continue
        if _is_interface_assignment(node):
            continue
        if isinstance(node, ast.ClassDef):
            total += sum(1 for _ in ast.walk(node))
            implementation_roots.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in ignored_top_level_functions:
                nested_roots = _nested_implementation_roots(node)
                total += _nested_implementation_complexity(node)
                implementation_roots.extend(nested_roots)
            else:
                total += sum(1 for _ in ast.walk(node))
                implementation_roots.append(node)
        else:
            total += sum(1 for _ in ast.walk(node))
            implementation_roots.extend(_implementation_roots(node))
    if not form_aware or total == 0:
        return total
    primitive_credit = _provided_primitive_credit(tree, implementation_roots)
    max_discount = int(total * _FORM_AWARE_MAX_DISCOUNT_FRACTION)
    discount = min(max_discount, primitive_credit)
    return max(1, total - discount)


def _nested_implementation_complexity(node: ast.AST) -> int:
    """Counts policy implementations nested inside an ignored factory wrapper."""
    return sum(sum(1 for _ in ast.walk(root)) for root in _nested_implementation_roots(node))


def _nested_implementation_roots(node: ast.AST) -> list[ast.AST]:
    """Return policy implementations nested inside an ignored factory wrapper."""
    roots: list[ast.AST] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            roots.append(child)
        else:
            roots.extend(_nested_implementation_roots(child))
    return roots


def _implementation_roots(node: ast.AST) -> list[ast.AST]:
    """Return definitions and lambdas nested in one charged module statement."""
    roots: list[ast.AST] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            roots.append(child)
        else:
            roots.extend(_implementation_roots(child))
    return roots


def _is_module_docstring(node: ast.stmt, index: int) -> bool:
    """Return whether a statement is the module docstring."""
    return (
        index == 0
        and isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_interface_assignment(node: ast.stmt) -> bool:
    """Return whether an assignment exposes metadata rather than policy logic."""
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return False
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names = [target.id for target in targets if isinstance(target, ast.Name)]
    if len(names) != len(targets):
        return False
    if names == ["__all__"]:
        if node.value is None:
            return False
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            return False
        return isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value)
    return (
        names == ["candidate_factory"]
        and isinstance(node.value, ast.Name)
        and node.value.id == "build_candidate"
    )


def _provided_primitive_credit(
    tree: ast.Module,
    implementation_roots: list[ast.AST],
) -> int:
    """Return bounded credits for canonical primitive composition call sites."""
    constructor_credits: dict[str, int] = {}
    function_credits: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != _FORM_AWARE_PRIMITIVE_MODULE:
            continue
        for alias in node.names:
            imported_name = alias.asname or alias.name
            if alias.name == "MultiTimescaleDecay":
                constructor_credits[imported_name] = _FORM_AWARE_PRIMITIVE_CALL_CREDIT
            elif alias.name == "decay_vector":
                function_credits[imported_name] = _FORM_AWARE_PRIMITIVE_CALL_CREDIT
            elif alias.name == "threshold_excess":
                function_credits[imported_name] = _FORM_AWARE_STATELESS_CALL_CREDIT

    primitive_bindings: dict[str, int] = {}
    for root in implementation_roots:
        for candidate_node in ast.walk(root):
            if not isinstance(candidate_node, (ast.Assign, ast.AnnAssign)):
                continue
            value = candidate_node.value
            if not isinstance(value, ast.Call):
                continue
            constructor_credit = constructor_credits.get(_called_name(value.func) or "")
            if constructor_credit is None:
                continue
            targets = (
                candidate_node.targets
                if isinstance(candidate_node, ast.Assign)
                else [candidate_node.target]
            )
            for target in targets:
                key = _expression_key(target)
                if key is not None:
                    primitive_bindings[key] = constructor_credit

    credit = 0
    primitive_methods = {"observe", "observe_vector", "values", "combine"}
    for root in implementation_roots:
        for candidate_node in ast.walk(root):
            if not isinstance(candidate_node, ast.Call):
                continue
            called_name = _called_name(candidate_node.func)
            if called_name in constructor_credits:
                credit += constructor_credits[called_name]
                continue
            if called_name in function_credits:
                credit += function_credits[called_name]
                continue
            if not isinstance(candidate_node.func, ast.Attribute):
                continue
            if candidate_node.func.attr not in primitive_methods:
                continue
            credit += primitive_bindings.get(
                _expression_key(candidate_node.func.value) or "",
                0,
            )
    return credit


def _called_name(node: ast.expr) -> str | None:
    """Return the direct called name when statically identifiable."""
    return node.id if isinstance(node, ast.Name) else None


def _expression_key(node: ast.expr) -> str | None:
    """Return a stable dotted key for simple assignment/call expressions."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_key(node.value)
        return f"{parent}.{node.attr}" if parent is not None else None
    return None
