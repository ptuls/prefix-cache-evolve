"""Operative evaluator configuration for the prefix KV-cache problem."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

import yaml

from prefix_cache_evolve.evaluators.prefix_kv_cache import EvaluatorConfig
from prefix_cache_evolve.workflow.config import WorkflowFileConfig

PREFIX_KV_CONFIG_ENV = "PREFIX_CACHE_EVOLVE_CONFIG"
PREFIX_KV_QUICK_ENV = "PREFIX_CACHE_EVOLVE_QUICK"
_REPOSITORY_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs/prefix_kv_cache.yaml"
_PACKAGED_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/prefix_kv_cache.yaml"
DEFAULT_CONFIG_PATH = (
    _REPOSITORY_CONFIG_PATH if _REPOSITORY_CONFIG_PATH.exists() else _PACKAGED_CONFIG_PATH
)


def load_evaluator_config(path: Path = DEFAULT_CONFIG_PATH) -> EvaluatorConfig:
    """Load the evaluator settings that are operative for reports and evolution."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    document = WorkflowFileConfig.model_validate(data)
    if "verifier_version" not in document.problem.settings:
        raise ValueError(f"{path} must explicitly declare problem.settings.verifier_version")
    config = evaluator_config_from_settings(document.problem.settings)
    if document.evaluator.timeout is not None:
        config = config.with_updates(timeout_s=document.evaluator.timeout)
    return config


def evaluator_config_from_settings(
    settings: Mapping[str, object],
    *,
    base: EvaluatorConfig | None = None,
) -> EvaluatorConfig:
    """Overlay YAML problem settings onto an evaluator configuration."""
    return (base or EvaluatorConfig()).with_updates(**dict(settings))


def active_evaluator_config(default: EvaluatorConfig) -> EvaluatorConfig:
    """Return an environment-selected config or the supplied test/default config."""
    path = os.environ.get(PREFIX_KV_CONFIG_ENV)
    config = load_evaluator_config(Path(path)) if path else default
    if os.environ.get(PREFIX_KV_QUICK_ENV) == "1":
        config = config.with_updates(
            request_count=36,
            seeds=(3,),
            family_request_multipliers={},
        )
    return config


@contextmanager
def prefix_kv_config_environment(path: Path, *, quick: bool = False) -> Iterator[None]:
    """Expose the operative config path to Levi evaluator worker processes."""
    previous = os.environ.get(PREFIX_KV_CONFIG_ENV)
    previous_quick = os.environ.get(PREFIX_KV_QUICK_ENV)
    os.environ[PREFIX_KV_CONFIG_ENV] = str(path.resolve())
    if quick:
        os.environ[PREFIX_KV_QUICK_ENV] = "1"
    else:
        os.environ.pop(PREFIX_KV_QUICK_ENV, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PREFIX_KV_CONFIG_ENV, None)
        else:
            os.environ[PREFIX_KV_CONFIG_ENV] = previous
        if previous_quick is None:
            os.environ.pop(PREFIX_KV_QUICK_ENV, None)
        else:
            os.environ[PREFIX_KV_QUICK_ENV] = previous_quick
