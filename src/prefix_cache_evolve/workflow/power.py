"""Keep unattended evolution runs awake without changing system preferences."""

import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def prevent_idle_sleep() -> Iterator[None]:
    """Hold a macOS idle-sleep assertion until the operation or process exits."""
    if sys.platform != "darwin":
        yield
        return

    # Watching our PID releases the assertion even if Python cannot run cleanup.
    with subprocess.Popen(
        ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    ) as process:
        try:
            if process.poll() is not None:
                raise RuntimeError("caffeinate exited before macOS sleep prevention started")
            yield
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
