"""Tests for repository and installed-package resources."""

import json
import subprocess
import sys

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
    assert bundled_config_path("prefix_kv_cache_mooncake.yaml").is_file()


def test_core_evaluation_imports_without_application_or_search_dependencies(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; "
            "import prefix_cache_evolve.evaluators.prefix_kv_cache; "
            "print(json.dumps(sorted(sys.modules)))",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = json.loads(completed.stdout)
    forbidden = (
        "prefix_cache_evolve.problems",
        "prefix_cache_evolve.workflow",
        "prefix_cache_evolve.tools",
        "levi",
    )
    assert not [
        name
        for name in loaded
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
    ]
