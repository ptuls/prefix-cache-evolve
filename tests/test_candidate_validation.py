"""Tests for static candidate source validation."""

import textwrap
from pathlib import Path

import pytest

from prefix_cache_evolve.evaluators.complexity import scoring_fn_complexity
from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.problems.prefix_kv_cache.candidate_validation import (
    candidate_source_violations,
    validate_candidate_source,
)
from prefix_cache_evolve.problems.prefix_kv_cache.incumbents.registry import current_incumbent


def test_validation_preserves_violation_and_repair_order() -> None:
    source = textwrap.dedent(
        """
        import math
        import random

        class Policy:
            @staticmethod
            def on_request_end(self):
                return self.request_type
        """
    )

    result = validate_candidate_source(
        source,
        complexity=1,
        config=EvaluatorConfig(reject_unsupported_source_patterns=True),
    )

    assert result.violations == (
        "import from unsupported module random",
        "unused import math",
        "unused import random",
        "unsupported callback on_request_end",
        "decorators are not allowed in candidate code",
        "sanitized request field request_type is not a policy signal",
    )
    assert result.repair_feedback == (
        "Remove the import; candidate code may import only math and primitives.",
        "Delete math from the imports.",
        "Delete random from the imports.",
        "Delete on_request_end entirely.",
        "Remove or repair this violation: decorators are not allowed in candidate code.",
        "Remove request_type; it is deliberately scrubbed before candidate callbacks.",
    )
    assert result.violation_summary == "; ".join(result.violations)
    assert result.repair_summary == " ".join(result.repair_feedback)


def test_syntax_violation_precedes_complexity_validation() -> None:
    result = validate_candidate_source(
        "def build_candidate(:\n    pass\n",
        complexity=100,
        config=EvaluatorConfig(max_candidate_complexity=1),
    )

    assert result.violations == ("syntax error at line 1: invalid syntax",)
    assert result.repair_feedback == (
        "Fix the reported syntax error at line 1: invalid syntax before changing policy behavior.",
    )


@pytest.mark.parametrize(
    ("invalid_source", "valid_source", "expected_violations"),
    (
        pytest.param(
            """
            from prefix_cache_evolve.problems.prefix_kv_cache.primitives import MultiTimescaleDecay

            class Policy:
                def __init__(self):
                    self._state = MultiTimescaleDecay(4, 10)
            """,
            """
            from prefix_cache_evolve.problems.prefix_kv_cache.primitives import MultiTimescaleDecay

            class Policy:
                def __init__(self):
                    self._state = MultiTimescaleDecay(half_lives=(4.0, 20.0), max_keys=64)
            """,
            (
                "MultiTimescaleDecay accepts only one positional argument",
                "MultiTimescaleDecay half-lives must be a sequence",
            ),
            id="bounded-decay-constructor",
        ),
        pytest.param(
            """
            from prefix_cache_evolve.problems.prefix_kv_cache.primitives import threshold_excess

            class Policy:
                def score_admission(self, block, now):
                    return threshold_excess(block.depth)
            """,
            """
            from prefix_cache_evolve.problems.prefix_kv_cache.primitives import threshold_excess

            class Policy:
                def score_admission(self, block, now):
                    return threshold_excess(block.depth, 2.0)
            """,
            ("threshold_excess requires value and threshold",),
            id="threshold-excess-signature",
        ),
    ),
)
def test_validation_checks_supported_primitive_signatures(
    invalid_source: str,
    valid_source: str,
    expected_violations: tuple[str, ...],
) -> None:
    config = EvaluatorConfig(reject_unsupported_source_patterns=True)
    violations = candidate_source_violations(
        textwrap.dedent(invalid_source),
        complexity=1,
        config=config,
    )

    assert all(violation in violations for violation in expected_violations)
    assert (
        candidate_source_violations(
            textwrap.dedent(valid_source),
            complexity=1,
            config=config,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("source", "expected_violations"),
    [
        pytest.param(
            f"""
            class Policy:
                def score_admission(self, block, now):
                    return {builtin_name}("0")
            """,
            (f"{builtin_name}() is not allowed in candidate code",),
            id=f"dynamic-{builtin_name}",
        )
        for builtin_name in ("exec", "eval", "compile", "vars")
    ]
    + [
        pytest.param(
            """
            class Policy:
                def score_admission(self, block, now):
                    runner = exec
                    runner("pass")
                    return 0.0
            """,
            ("exec() is not allowed in candidate code",),
            id="aliased-dynamic-builtin",
        ),
        pytest.param(
            """
            class Policy:
                def on_request_start(self, request, now):
                    self.kind = request.request_type
                    self.tokens = request.prompt_tokens
            """,
            (
                "sanitized request field request_type is not a policy signal",
                "sanitized request field prompt_tokens is not a policy signal",
            ),
            id="scrubbed-request-fields",
        ),
        pytest.param(
            """
            class Policy:
                def score_admission(self, block, now):
                    return block.__class__
            """,
            ("dunder attribute __class__ is not allowed",),
            id="dunder-attribute",
        ),
        pytest.param(
            """
            def decorate(policy):
                return policy

            @decorate
            class Policy:
                pass
            """,
            ("decorators are not allowed in candidate code",),
            id="decorated-policy",
        ),
        pytest.param(
            """
            if True:
                class Policy:
                    pass
            """,
            ("unsupported top-level statement If",),
            id="top-level-control-flow",
        ),
        pytest.param(
            "score = lambda block, now: block.depth",
            ("top-level assignments must define uppercase literal constants",),
            id="module-lambda",
        ),
        pytest.param(
            """
            from prefix_cache_evolve.evaluators.baselines import baseline_tinylfu_lru

            def build_candidate(capacity_blocks, block_size_tokens, seed=None):
                return baseline_tinylfu_lru(capacity_blocks, block_size_tokens, seed)
            """,
            ("import from unsupported module prefix_cache_evolve.evaluators.baselines",),
            id="unsupported-import",
        ),
    ],
)
def test_validation_rejects_unsafe_source_patterns(
    source: str,
    expected_violations: tuple[str, ...],
) -> None:
    normalized_source = textwrap.dedent(source)
    violations = candidate_source_violations(
        normalized_source,
        complexity=scoring_fn_complexity(normalized_source),
        config=EvaluatorConfig(reject_unsupported_source_patterns=True),
    )

    assert all(violation in violations for violation in expected_violations)


@pytest.mark.parametrize("role", ("discovery", "production"))
def test_current_incumbents_pass_static_source_contract(role: str) -> None:
    source = current_incumbent(role).source_path.read_text(encoding="utf-8")
    config = EvaluatorConfig(
        max_candidate_complexity=650,
        reject_unsupported_source_patterns=True,
    )

    assert (
        candidate_source_violations(
            source,
            complexity=scoring_fn_complexity(source, form_aware=True),
            config=config,
        )
        == ()
    )


def test_eviction_specialist_seed_passes_static_source_contract() -> None:
    source = Path(
        "src/prefix_cache_evolve/problems/prefix_kv_cache/seeds/eviction_specialist.py"
    ).read_text(encoding="utf-8")
    config = EvaluatorConfig(
        max_candidate_complexity=1000,
        reject_unsupported_source_patterns=True,
        candidate_policy_surface="eviction_only",
    )

    assert (
        candidate_source_violations(
            source,
            complexity=scoring_fn_complexity(source, form_aware=True),
            config=config,
        )
        == ()
    )


def _source_violations(source: str) -> tuple[str, ...]:
    return candidate_source_violations(
        source,
        complexity=scoring_fn_complexity(source),
        config=EvaluatorConfig(reject_unsupported_source_patterns=True),
    )


@pytest.mark.parametrize(
    "import_statement", ["import sys", "from pathlib import Path", "import math"]
)
def test_static_policy_checks_reject_imports_inside_definitions(import_statement) -> None:
    source = (
        "def build_candidate(capacity_blocks, block_size_tokens, seed=None):\n"
        f"    {import_statement}\n"
        "    return None\n"
    )
    violations = _source_violations(source)
    assert "nested imports are not allowed in candidate code" in violations


@pytest.mark.parametrize(
    "statement",
    [
        "math.log1p = lambda value: 1e12",
        "del math.log1p",
        "MultiTimescaleDecay.observe = lambda self, key, now, weight=1.0: None",
        "block.hit_count = 1000000000",
    ],
)
def test_static_policy_checks_reject_shared_attribute_mutation(statement) -> None:
    source = f"""
import math
from prefix_cache_evolve.problems.prefix_kv_cache.primitives import MultiTimescaleDecay

class Policy:
    def score_admission(self, block, now):
        {statement}
        return 1

def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return Policy()
"""
    violations = _source_violations(source)
    assert "attribute writes must target candidate-owned self state" in violations


def test_static_policy_checks_allow_candidate_owned_attribute_state() -> None:
    source = """
class Policy:
    def __init__(self):
        self.hits = 0

    def score_admission(self, block, now):
        self.hits += 1
        return self.hits

def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return Policy()
"""
    assert _source_violations(source) == ()


def test_static_policy_checks_reject_aliased_self_and_shared_runtime_types() -> None:
    source = """
def shared(self):
    self.seen = 1

class Policy:
    def poison(self):
        self.seen = 1

    def score_admission(self, block, now):
        shared(type(block))
        Policy.poison(type(block))
        return 1

def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return Policy()
"""
    violations = _source_violations(source)
    assert "attribute writes must target candidate-owned self state" in violations
    assert "type() is not allowed in candidate code" in violations
    assert "candidate classes may only be referenced as direct constructors" in violations
