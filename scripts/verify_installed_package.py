"""Verify resources and entry-point dependencies in an installed wheel."""

from pathlib import Path

import prefix_cache_evolve
from prefix_cache_evolve.problems.prefix_kv_cache.configuration import (
    DEFAULT_CONFIG_PATH,
    DISCOVERY_CONFIG_PATH,
    EVICTION_SPECIALIST_CONFIG_PATH,
    REDISCOVERY_CONFIG_PATH,
    TRACE_SCHEMA_PATH,
)


def main() -> None:
    """Fail if an installed distribution is missing required resources."""
    package_root = Path(prefix_cache_evolve.__file__).resolve().parent
    paths = (
        DEFAULT_CONFIG_PATH,
        DISCOVERY_CONFIG_PATH,
        EVICTION_SPECIALIST_CONFIG_PATH,
        REDISCOVERY_CONFIG_PATH,
        TRACE_SCHEMA_PATH,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if package_root not in path.resolve().parents:
            raise RuntimeError(f"{path} did not resolve from the installed package")
    print(f"verified_installed_resources={len(paths)}")


if __name__ == "__main__":
    main()
