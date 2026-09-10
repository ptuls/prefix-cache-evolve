"""Lifecycle checks for sleep prevention during unattended evolution."""

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from prefix_cache_evolve.problems.prefix_kv_cache import runner
from prefix_cache_evolve.workflow import power


@pytest.mark.parametrize("failure_stage", [None, "search", "report"])
def test_evolution_prevents_sleep_through_reporting(monkeypatch, tmp_path, failure_stage):
    process = Mock()
    process.poll.return_value = None
    process.__enter__ = Mock(return_value=process)
    process.__exit__ = Mock(return_value=False)
    launch = Mock(return_value=process)
    monkeypatch.setattr(power.sys, "platform", "darwin")
    monkeypatch.setattr(power.subprocess, "Popen", launch)
    stages = []

    def execute(iterations):
        launch.assert_called_once()
        process.terminate.assert_not_called()
        stages.append("search")
        if failure_stage == "search":
            raise RuntimeError("search failed")
        return SimpleNamespace()

    def save_artifacts(*args, **kwargs):
        process.terminate.assert_not_called()
        stages.append("report")
        if failure_stage == "report":
            raise RuntimeError("report failed")
        return tmp_path

    monkeypatch.setattr(
        runner, "_build_workflow", lambda *args, **kwargs: SimpleNamespace(execute=execute)
    )
    monkeypatch.setattr(runner, "save_run_artifacts", save_artifacts)

    if failure_stage:
        with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
            runner.demo_run_evolution(iterations=1, quick=True, artifact_output=tmp_path)
    else:
        runner.demo_run_evolution(iterations=1, quick=True, artifact_output=tmp_path)

    assert stages == (["search"] if failure_stage == "search" else ["search", "report"])
    assert launch.call_args.args[0] == ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())]
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=5)


def test_sleep_prevention_is_a_noop_off_macos(monkeypatch):
    monkeypatch.setattr(power.sys, "platform", "linux")
    launch = Mock(side_effect=AssertionError("non-macOS runs must not launch caffeinate"))
    monkeypatch.setattr(power.subprocess, "Popen", launch)

    with power.prevent_idle_sleep():
        pass

    launch.assert_not_called()


def test_sleep_prevention_fails_before_work_when_helper_cannot_start(monkeypatch):
    monkeypatch.setattr(power.sys, "platform", "darwin")
    monkeypatch.setattr(
        power.subprocess, "Popen", Mock(side_effect=FileNotFoundError("caffeinate"))
    )

    with pytest.raises(FileNotFoundError, match="caffeinate"):
        with power.prevent_idle_sleep():
            pytest.fail("work must not start without sleep prevention")


def test_sleep_prevention_kills_helper_if_cleanup_times_out(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("caffeinate", 5), 0]
    process.__enter__ = Mock(return_value=process)
    process.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(power.sys, "platform", "darwin")
    monkeypatch.setattr(power.subprocess, "Popen", Mock(return_value=process))

    with power.prevent_idle_sleep():
        pass

    process.kill.assert_called_once()
    assert process.wait.call_count == 2
