"""Candidate source-complexity analysis."""

from __future__ import annotations

import ast

_FORM_AWARE_PRIMITIVE_MODULE = "prefix_cache_evolve.problems.prefix_kv_cache.primitives"
_FORM_AWARE_PRIMITIVE_CALL_CREDIT = 3
_FORM_AWARE_MAX_DISCOUNT_FRACTION = 0.25


def scoring_fn_complexity(source: str, *, form_aware: bool = False) -> int:
    """Count effective AST nodes in the candidate policy implementation."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 10_000

    ignored_top_level_functions = {"build_candidate", "candidate_factory", "run_demo"}
    total = 0
    implementation_roots = []
    for node in tree.body:
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
    if not form_aware or total == 0:
        return total
    primitive_call_count = _provided_primitive_call_count(tree, implementation_roots)
    max_discount = int(total * _FORM_AWARE_MAX_DISCOUNT_FRACTION)
    discount = min(
        max_discount,
        primitive_call_count * _FORM_AWARE_PRIMITIVE_CALL_CREDIT,
    )
    return max(1, total - discount)


def _nested_implementation_complexity(node: ast.AST) -> int:
    """Counts policy implementations nested inside an ignored factory wrapper."""

    return sum(
        sum(1 for _ in ast.walk(root)) for root in _nested_implementation_roots(node)
    )


def _nested_implementation_roots(node: ast.AST) -> list[ast.AST]:
    """Return policy implementations nested inside an ignored factory wrapper."""

    roots = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            roots.append(child)
        else:
            roots.extend(_nested_implementation_roots(child))
    return roots


def _provided_primitive_call_count(
    tree: ast.Module,
    implementation_roots: list[ast.AST],
) -> int:
    """Count canonical primitive composition call sites in candidate code."""

    constructor_names: set[str] = set()
    function_names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != _FORM_AWARE_PRIMITIVE_MODULE:
            continue
        for alias in node.names:
            imported_name = alias.asname or alias.name
            if alias.name == "MultiTimescaleDecay":
                constructor_names.add(imported_name)
            elif alias.name == "decay_vector":
                function_names.add(imported_name)

    primitive_bindings: set[str] = set()
    for root in implementation_roots:
        for node in ast.walk(root):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Call):
                continue
            if _called_name(value.func) not in constructor_names:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            primitive_bindings.update(
                key
                for target in targets
                if (key := _expression_key(target)) is not None
            )

    count = 0
    primitive_methods = {"observe", "observe_vector", "values", "combine"}
    for root in implementation_roots:
        for node in ast.walk(root):
            if not isinstance(node, ast.Call):
                continue
            if _called_name(node.func) in constructor_names | function_names:
                count += 1
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in primitive_methods:
                continue
            if _expression_key(node.func.value) in primitive_bindings:
                count += 1
    return count


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
