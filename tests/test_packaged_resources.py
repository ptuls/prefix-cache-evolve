"""Tests for repository and installed-package resources."""

from prefix_cache_evolve.problems.prefix_kv_cache.configuration import (
    DEFAULT_CONFIG_PATH,
    DISCOVERY_CONFIG_PATH,
    EVICTION_SPECIALIST_CONFIG_PATH,
    REDISCOVERY_CONFIG_PATH,
    TRACE_SCHEMA_PATH,
    bundled_config_path,
)


def test_all_bundled_configs_are_resolvable() -> None:
    paths = (
        DEFAULT_CONFIG_PATH,
        DISCOVERY_CONFIG_PATH,
        EVICTION_SPECIALIST_CONFIG_PATH,
        REDISCOVERY_CONFIG_PATH,
        TRACE_SCHEMA_PATH,
    )

    assert all(path.is_file() for path in paths)
    assert bundled_config_path("prefix_kv_cache_rediscovery.yaml") == REDISCOVERY_CONFIG_PATH
