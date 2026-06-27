"""Tests for static candidate source validation."""

import textwrap
from dataclasses import FrozenInstanceError

import pytest

from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.problems.prefix_kv_cache import evaluator
from prefix_cache_evolve.problems.prefix_kv_cache.candidate_validation import (
    candidate_source_violations,
    static_repair_feedback,
    validate_candidate_source,
)


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


def test_validation_result_is_immutable() -> None:
    result = validate_candidate_source(
        "VALUE = 1\n",
        complexity=1,
        config=EvaluatorConfig(),
    )

    with pytest.raises(FrozenInstanceError):
        result.violations = ("changed",)


def test_evaluator_preserves_private_validation_helpers() -> None:
    assert evaluator._candidate_source_violations is candidate_source_violations
    assert evaluator._static_repair_feedback is static_repair_feedback
