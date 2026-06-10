"""Tests for isolated evaluator execution helpers."""

from dataclasses import dataclass, field

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
