"""Lazy Click command groups for optional and heavyweight tools."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import click


@dataclass(frozen=True)
class LazyCommand:
    """Import path and help text for one deferred Click command."""

    import_path: str
    help: str


class LazyGroup(click.Group):
    """Load registered subcommands only when they are invoked."""

    def __init__(
        self,
        *args: Any,
        lazy_subcommands: Mapping[str, LazyCommand] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._lazy_subcommands = dict(lazy_subcommands or {})

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List eager and lazy commands without importing lazy modules."""
        return sorted({*super().list_commands(ctx), *self._lazy_subcommands})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Resolve an eager command or import one configured lazy command."""
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        command_spec = self._lazy_subcommands.get(cmd_name)
        if command_spec is None:
            return None
        module_name, attribute = command_spec.import_path.rsplit(":", maxsplit=1)
        loaded = getattr(importlib.import_module(module_name), attribute)
        if not isinstance(loaded, click.Command):
            raise TypeError(f"{command_spec.import_path} did not resolve to a Click command")
        return loaded

    def format_commands(
        self,
        ctx: click.Context,
        formatter: click.HelpFormatter,
    ) -> None:
        """Render lazy command help without importing command modules."""
        rows = []
        for command_name in self.list_commands(ctx):
            eager_command = super().get_command(ctx, command_name)
            if eager_command is not None:
                if eager_command.hidden:
                    continue
                help_text = eager_command.get_short_help_str()
            else:
                help_text = self._lazy_subcommands[command_name].help
            rows.append((command_name, help_text))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)
