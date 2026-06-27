"""Shared workflow abstractions."""

from pathlib import Path
from typing import Protocol, TypeVar

from prefix_cache_evolve.workflow.config import LeviRunConfig

RunnerResultT = TypeVar("RunnerResultT", covariant=True)
ReporterResultT = TypeVar("ReporterResultT", contravariant=True)


class Runner(Protocol[RunnerResultT]):
    """Runs an evolved program against a resolved configuration."""

    def run(self, program_path: Path, config: LeviRunConfig) -> RunnerResultT:
        """Execute a program under the supplied configuration."""
        ...


class Reporter(Protocol[ReporterResultT]):
    """Publishes workflow results for humans or automation."""

    def report(
        self,
        result: ReporterResultT,
        iterations: int,
        config_label: str,
    ) -> None:
        """Publish one completed workflow result."""
        ...
