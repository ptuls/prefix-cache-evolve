"""Tests for isolated evaluator execution helpers."""

import math
import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from prefix_cache_evolve import evaluator_entry


@dataclass
class _FakeResource:
    RLIMIT_AS: int = 1
    RLIMIT_CPU: int = 2
    RLIM_INFINITY: int = -1
    inherited_limits: tuple[int, int] = (-1, -1)
    calls: list[tuple[int, tuple[int, int]]] = field(default_factory=list)

    def getrlimit(self, resource_id: int) -> tuple[int, int]:
        del resource_id
        return self.inherited_limits

    def setrlimit(self, resource_id: int, limits: tuple[int, int]) -> None:
        self.calls.append((resource_id, limits))


@pytest.fixture
def fake_resource(monkeypatch) -> _FakeResource:
    resource = _FakeResource()
    monkeypatch.setattr(evaluator_entry, "resource", resource)
    return resource


def test_apply_resource_limits_sets_cpu_and_address_space(
    monkeypatch,
    fake_resource: _FakeResource,
) -> None:
    monkeypatch.setattr(evaluator_entry, "_current_virtual_memory_bytes", lambda: 1000)

    evaluator_entry._apply_resource_limits(
        memory_limit_bytes=2000,
        cpu_limit_seconds=2.2,
    )

    assert (fake_resource.RLIMIT_CPU, (3, 4)) in fake_resource.calls
    assert (
        fake_resource.RLIMIT_AS,
        (1000 + evaluator_entry._PROCESS_MEMORY_HEADROOM_BYTES + 2000,) * 2,
    ) in fake_resource.calls


def test_apply_resource_limits_skips_address_space_without_procfs(
    monkeypatch,
    fake_resource: _FakeResource,
) -> None:
    monkeypatch.setattr(evaluator_entry, "_current_virtual_memory_bytes", lambda: None)

    evaluator_entry._apply_resource_limits(
        memory_limit_bytes=2000,
        cpu_limit_seconds=2.2,
    )

    assert fake_resource.calls == [(fake_resource.RLIMIT_CPU, (3, 4))]


def test_set_resource_limit_respects_inherited_hard_cap(
    fake_resource: _FakeResource,
) -> None:
    fake_resource.inherited_limits = (100, 200)

    evaluator_entry._set_resource_limit(fake_resource.RLIMIT_AS, 300, 400)

    assert fake_resource.calls == [(fake_resource.RLIMIT_AS, (200, 200))]


@pytest.mark.parametrize(
    ("score", "expected"),
    ((0.0, 1.0), (1.0, 0.5), (-1.0, 1.0), (math.inf, 0.0)),
)
def test_score_to_reward_is_bounded(score: float, expected: float) -> None:
    assert evaluator_entry.score_to_reward(score) == expected


def test_run_with_timeout_returns_worker_result() -> None:
    assert evaluator_entry.run_with_timeout(lambda: 17, timeout_seconds=1.0) == 17


def test_run_with_timeout_propagates_worker_errors() -> None:
    def fail() -> None:
        raise ValueError("candidate failed")

    with pytest.raises(ValueError, match="candidate failed"):
        evaluator_entry.run_with_timeout(fail, timeout_seconds=1.0)


def test_run_with_timeout_enforces_wall_clock_deadline() -> None:
    with pytest.raises(TimeoutError, match="wall-clock limit"):
        evaluator_entry.run_with_timeout(time.sleep, 0.2, timeout_seconds=0.02)


def test_run_with_timeout_refuses_daemon_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluator_entry.multiprocessing,
        "current_process",
        lambda: SimpleNamespace(daemon=True),
    )

    with pytest.raises(RuntimeError, match="supervised non-daemon worker"):
        evaluator_entry.run_with_timeout(lambda: None, timeout_seconds=1.0)


def test_run_with_timeout_enforces_resident_memory_growth() -> None:
    def allocate_memory() -> int:
        payload = bytearray(8 * 1024 * 1024)
        time.sleep(0.2)
        return len(payload)

    with pytest.raises(MemoryError, match="resident memory grew"):
        evaluator_entry.run_with_timeout(
            allocate_memory,
            timeout_seconds=1.0,
            memory_limit_bytes=1024 * 1024,
        )


def test_run_with_timeout_fails_closed_without_memory_monitoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator_entry, "_process_resident_memory_bytes", lambda _pid: None)

    with pytest.raises(RuntimeError, match="resident-memory monitoring is unavailable"):
        evaluator_entry.run_with_timeout(
            lambda: 17,
            timeout_seconds=1.0,
            memory_limit_bytes=1024,
        )


def test_run_with_timeout_fails_closed_if_memory_monitoring_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_process_id = os.getpid()
    monkeypatch.setattr(
        evaluator_entry,
        "_process_resident_memory_bytes",
        lambda _pid: 1024 if os.getpid() != parent_process_id else None,
    )

    with pytest.raises(RuntimeError, match="resident-memory monitoring became unavailable"):
        evaluator_entry.run_with_timeout(
            time.sleep,
            0.2,
            timeout_seconds=1.0,
            memory_limit_bytes=1024,
        )


def test_process_resident_memory_is_available_on_supported_platforms() -> None:
    resident_memory = evaluator_entry._process_resident_memory_bytes(os.getpid())

    assert resident_memory is not None
    assert resident_memory > 0


def test_load_candidate_factory_from_source_uses_supported_export() -> None:
    factory = evaluator_entry.load_candidate_factory_from_source(
        "def build_candidate():\n    return 23\n"
    )

    assert factory() == 23


def test_load_candidate_factory_rejects_missing_module(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        evaluator_entry.load_candidate_factory(str(tmp_path / "missing.py"))


def test_load_candidate_factory_rejects_missing_export(tmp_path) -> None:
    source = tmp_path / "candidate.py"
    source.write_text("VALUE = 3\n", encoding="utf-8")

    with pytest.raises(AttributeError, match="must expose"):
        evaluator_entry.load_candidate_factory(str(source))


def test_evaluation_entry_point_adapts_success() -> None:
    entry_point = _entry_point(lambda factory: factory())

    result = entry_point.evaluate_source("def build_candidate():\n    return 29\n")

    assert result.metrics == {"score": 29}


def test_evaluation_entry_point_adapts_load_errors() -> None:
    result = _entry_point(lambda factory: factory()).evaluate_source("VALUE = 3\n")

    assert result.metrics["error"] == "failed to load candidate factory"
    assert result.artifacts["error_type"] == "AttributeError"


def test_evaluation_entry_point_adapts_timeouts() -> None:
    result = _entry_point(lambda _factory: time.sleep(0.2), timeout_seconds=0.02).evaluate_factory(
        lambda: None
    )

    assert result.metrics["error"] == "evaluation timed out"
    assert result.artifacts["error_type"] == "TimeoutError"


def _entry_point(evaluator, *, timeout_seconds: float = 1.0):
    return evaluator_entry.EvaluationEntryPoint(
        evaluator_factory=lambda: evaluator,
        timeout_seconds=timeout_seconds,
        load_error_suggestion="repair the candidate",
        timeout_suggestion="simplify the candidate",
        success_result_builder=lambda value: evaluator_entry.EvaluatorResult(
            metrics={"score": value},
            artifacts={},
        ),
        error_result_builder=lambda message, artifacts: evaluator_entry.EvaluatorResult(
            metrics={"error": message},
            artifacts=artifacts,
        ),
    )
