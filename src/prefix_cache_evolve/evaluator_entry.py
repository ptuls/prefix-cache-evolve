"""Shared helpers for Levi evaluator entry points."""

from __future__ import annotations

import builtins
import ctypes
import importlib.util
import math
import multiprocessing
import os
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Generic, Sequence, TypeVar, cast

try:
    import resource
except ImportError:  # pragma: no cover - Windows does not provide resource
    resource = None  # type: ignore[assignment]

ResultT = TypeVar("ResultT")
_PROCESS_MEMORY_HEADROOM_BYTES = 256 * 1024 * 1024
_PROCESS_TERMINATE_GRACE_SECONDS = 0.05
_PROCESS_MONITOR_INTERVAL_SECONDS = 0.02
_DARWIN_PROCESS_TASK_INFO = 4
_DARWIN_PROCESS_TASK_INFO_WORDS = 12


def score_to_reward(score: float) -> float:
    """Converts a non-negative loss into a bounded reward."""
    return 0.0 if not math.isfinite(score) else 1.0 / (1.0 + max(score, 0.0))


def run_with_timeout(
    func: Callable[..., ResultT],
    *args,
    timeout_seconds: float,
    memory_limit_bytes: int | None = None,
    cpu_limit_seconds: float | None = None,
    **kwargs,
) -> ResultT:
    """Execute ``func`` in a forked worker with wall-clock and OS resource limits."""
    if multiprocessing.current_process().daemon:
        raise RuntimeError(
            "evaluation isolation cannot run inside a daemon process; "
            "use a supervised non-daemon worker"
        )

    try:
        context = multiprocessing.get_context("fork")
    except ValueError as exc:  # pragma: no cover - Python always supports fork on macOS/Linux
        raise RuntimeError("evaluation isolation requires multiprocessing fork support") from exc

    receive_conn, send_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=_run_in_subprocess,
        args=(
            send_conn,
            func,
            args,
            kwargs,
            memory_limit_bytes,
            cpu_limit_seconds or timeout_seconds,
        ),
    )
    process.start()
    send_conn.close()
    try:
        payload = _receive_worker_payload(
            receive_conn,
            process,
            timeout_seconds=timeout_seconds,
            memory_limit_bytes=memory_limit_bytes,
        )
    finally:
        receive_conn.close()
        _stop_process(process)

    status, *values = payload
    if status == "result":
        return cast(ResultT, values[0])
    if status == "error":
        raise values[0]
    error_type, error_message, full_traceback = values
    raise RuntimeError(f"evaluation worker raised {error_type}: {error_message}\n{full_traceback}")


def _receive_worker_payload(
    receive_conn: Any,
    process: Any,
    *,
    timeout_seconds: float,
    memory_limit_bytes: int | None,
) -> tuple[Any, ...]:
    """Receive a worker result while enforcing wall-clock and resident-memory limits."""
    deadline = time.monotonic() + timeout_seconds
    initial_resident_bytes: int | None = None
    while True:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            _stop_process(process, force=True)
            raise TimeoutError(f"evaluation exceeded {timeout_seconds}s wall-clock limit")

        wait_seconds = (
            min(remaining_seconds, _PROCESS_MONITOR_INTERVAL_SECONDS)
            if memory_limit_bytes is not None
            else remaining_seconds
        )
        if receive_conn.poll(wait_seconds):
            try:
                payload = receive_conn.recv()
            except EOFError as exc:
                raise RuntimeError("evaluation worker exited without returning a result") from exc
            if payload[0] != "ready":
                return cast(tuple[Any, ...], payload)
            initial_resident_bytes = payload[1]
            if memory_limit_bytes is not None and initial_resident_bytes is None:
                raise RuntimeError("candidate resident-memory monitoring is unavailable")
            continue

        if memory_limit_bytes is not None and initial_resident_bytes is not None:
            resident_bytes = _process_resident_memory_bytes(process.pid)
            if resident_bytes is None:
                _stop_process(process, force=True)
                raise RuntimeError("candidate resident-memory monitoring became unavailable")
            if resident_bytes - initial_resident_bytes > memory_limit_bytes:
                _stop_process(process, force=True)
                growth_bytes = resident_bytes - initial_resident_bytes
                raise MemoryError(
                    f"evaluation worker resident memory grew by {growth_bytes} bytes "
                    f"(> {memory_limit_bytes})"
                )


def _run_in_subprocess(
    send_conn,
    func: Callable[..., ResultT],
    args: tuple,
    kwargs: dict,
    memory_limit_bytes: int | None,
    cpu_limit_seconds: float | None,
) -> None:
    """Runs one evaluation and sends either its result or raised exception."""
    try:
        _apply_resource_limits(
            memory_limit_bytes=memory_limit_bytes,
            cpu_limit_seconds=cpu_limit_seconds,
        )
        send_conn.send(("ready", _process_resident_memory_bytes(os.getpid())))
        send_conn.send(("result", func(*args, **kwargs)))
    except BaseException as exc:  # pragma: no cover - exercised through parent process
        try:
            send_conn.send(("error", exc))
        except Exception:
            send_conn.send(
                (
                    "unserializable_error",
                    type(exc).__name__,
                    str(exc),
                    traceback.format_exc(),
                )
            )
    finally:
        send_conn.close()


def _apply_resource_limits(
    *,
    memory_limit_bytes: int | None,
    cpu_limit_seconds: float | None,
) -> None:
    """Apply best-effort POSIX limits inside an isolated evaluation worker."""
    if resource is None:
        return
    if cpu_limit_seconds is not None:
        cpu_soft = max(1, math.ceil(cpu_limit_seconds))
        _set_resource_limit(resource.RLIMIT_CPU, cpu_soft, cpu_soft + 1)
    if memory_limit_bytes is not None:
        current_virtual_bytes = _current_virtual_memory_bytes()
        if current_virtual_bytes is not None:
            address_space_limit = (
                current_virtual_bytes + _PROCESS_MEMORY_HEADROOM_BYTES + memory_limit_bytes
            )
            _set_resource_limit(
                resource.RLIMIT_AS,
                address_space_limit,
                address_space_limit,
            )


def _current_virtual_memory_bytes() -> int | None:
    """Return current Linux virtual memory size, if procfs is available."""
    try:
        statm = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(statm[0]) * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def _process_resident_memory_bytes(process_id: int) -> int | None:
    """Return resident memory for one worker on supported POSIX platforms."""
    if sys.platform == "darwin":
        try:
            process_info = _darwin_process_info()
            buffer = (ctypes.c_uint64 * _DARWIN_PROCESS_TASK_INFO_WORDS)()
            bytes_written = process_info(
                process_id,
                _DARWIN_PROCESS_TASK_INFO,
                0,
                ctypes.byref(buffer),
                ctypes.sizeof(buffer),
            )
            return int(buffer[1]) if bytes_written >= 2 * ctypes.sizeof(ctypes.c_uint64) else None
        except (AttributeError, OSError, ValueError):
            return None
    try:
        statm = Path(f"/proc/{process_id}/statm").read_text(encoding="ascii").split()
        return int(statm[1]) * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


@lru_cache(maxsize=1)
def _darwin_process_info() -> Callable[..., int]:
    """Return Darwin's process-information function without an external dependency."""
    process_info = ctypes.CDLL(None).proc_pidinfo
    process_info.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    process_info.restype = ctypes.c_int
    return cast(Callable[..., int], process_info)


def _set_resource_limit(resource_id: int, soft_limit: int, hard_limit: int) -> None:
    """Lower one resource limit without attempting to raise an inherited hard cap."""
    if resource is None:
        return
    _, inherited_hard = resource.getrlimit(resource_id)
    if inherited_hard != resource.RLIM_INFINITY:
        hard_limit = min(hard_limit, inherited_hard)
        soft_limit = min(soft_limit, hard_limit)
    resource.setrlimit(resource_id, (soft_limit, hard_limit))


def _stop_process(process, *, force: bool = False) -> None:
    """Terminates and reaps an evaluation worker if it is still running."""
    if process.is_alive():
        if force:
            process.kill()
        else:
            process.terminate()
    process.join(timeout=_PROCESS_TERMINATE_GRACE_SECONDS)
    if process.is_alive():  # pragma: no cover - terminate should normally be enough
        process.kill()
        process.join()


def load_candidate_factory(
    program_path: str,
    exported_names: Sequence[str] = ("candidate_factory", "build_candidate"),
) -> Callable[..., object]:
    """Loads a candidate factory from a Python module on disk."""
    path = Path(program_path)
    if not path.exists():
        raise FileNotFoundError(f"program path {path} does not exist")

    spec = importlib.util.spec_from_file_location("candidate_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load module from {path}")

    loader = spec.loader
    module = importlib.util.module_from_spec(spec)
    _exec_registered_module(module, lambda: loader.exec_module(module))

    factory = extract_exported_callable(module, exported_names)
    if not callable(factory):
        raise TypeError(f"{exported_names[0]} must be callable")
    return factory


def load_candidate_factory_from_source(
    source: str,
    exported_names: Sequence[str] = ("candidate_factory", "build_candidate"),
    *,
    import_overrides: Mapping[str, object] | None = None,
) -> Callable[..., object]:
    """Loads a candidate factory from Python source text."""
    module = ModuleType("candidate_module")
    if import_overrides is not None:
        candidate_builtins = vars(builtins).copy()

        def import_candidate_module(
            name: str,
            globals_: dict[str, object] | None = None,
            locals_: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            del globals_, locals_
            if level == 0 and name in import_overrides:
                return import_overrides[name]
            if level == 0 and name == "__future__" and fromlist == ("annotations",):
                return builtins.__import__(name, fromlist=fromlist)
            raise ImportError(f"candidate import is unavailable: {name}")

        candidate_builtins["__import__"] = import_candidate_module
        module.__dict__["__builtins__"] = candidate_builtins
    _exec_registered_module(
        module,
        lambda: exec(compile(source, "<candidate_source>", "exec"), module.__dict__),
    )

    factory = extract_exported_callable(module, exported_names)
    if not callable(factory):
        raise TypeError(f"{exported_names[0]} must be callable")
    return factory


def _exec_registered_module(module: ModuleType, exec_fn: Callable[[], object]) -> None:
    """Executes a candidate module while it is visible via ``sys.modules``."""
    module_name = module.__name__
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec_fn()
    except Exception:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise


def extract_exported_callable(
    module: ModuleType,
    exported_names: Sequence[str],
) -> Callable[..., object]:
    """Returns the first supported exported callable from ``module``."""
    for name in exported_names:
        if hasattr(module, name):
            return cast(Callable[..., object], getattr(module, name))
    joined_names = " or ".join(f"`{name}`" for name in exported_names)
    raise AttributeError(f"candidate module must expose {joined_names}")


@dataclass
class EvaluatorResult:
    """Levi-facing evaluator result with stable metrics/artifacts fields."""

    metrics: dict[str, Any]
    artifacts: dict[str, Any]


@dataclass(frozen=True)
class EvaluationEntryPoint(Generic[ResultT]):
    """Coordinates the common evaluator entry-point flow."""

    evaluator_factory: Callable[[], Callable[[Callable[..., object]], ResultT]]
    timeout_seconds: float
    load_error_suggestion: str
    timeout_suggestion: str
    success_result_builder: Callable[[ResultT], EvaluatorResult]
    error_result_builder: Callable[[str, dict[str, Any]], EvaluatorResult]
    unexpected_error_suggestion: str = "Unexpected evaluator failure; inspect the traceback."
    exported_names: Sequence[str] = ("candidate_factory", "build_candidate")

    def evaluate(self, program_path: str) -> EvaluatorResult:
        """Loads a candidate module, evaluates it, and adapts the result."""
        try:
            factory = load_candidate_factory(program_path, self.exported_names)
        except Exception as exc:  # pragma: no cover - defensive
            artifacts = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "full_traceback": traceback.format_exc(),
                "suggestion": self.load_error_suggestion,
            }
            return self.error_result_builder(
                "failed to load candidate factory",
                artifacts,
            )

        return self.evaluate_factory(factory)

    def evaluate_source(self, source: str) -> EvaluatorResult:
        """Loads a candidate module from source, evaluates it, and adapts the result."""
        try:
            factory = load_candidate_factory_from_source(source, self.exported_names)
        except Exception as exc:  # pragma: no cover - defensive
            artifacts = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "full_traceback": traceback.format_exc(),
                "suggestion": self.load_error_suggestion,
            }
            return self.error_result_builder(
                "failed to load candidate factory",
                artifacts,
            )

        return self.evaluate_factory(factory)

    def evaluate_factory(self, factory: Callable[..., object]) -> EvaluatorResult:
        """Evaluates an already-loaded candidate factory."""
        evaluator = self.evaluator_factory()
        try:
            result = run_with_timeout(
                evaluator,
                factory,
                timeout_seconds=self.timeout_seconds,
                cpu_limit_seconds=self.timeout_seconds,
            )
        except TimeoutError as exc:
            artifacts = {
                "error_type": "TimeoutError",
                "error_message": str(exc),
                "suggestion": self.timeout_suggestion,
            }
            return self.error_result_builder("evaluation timed out", artifacts)
        except Exception as exc:  # pragma: no cover - defensive
            artifacts = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "full_traceback": traceback.format_exc(),
                "suggestion": self.unexpected_error_suggestion,
            }
            return self.error_result_builder("evaluation failed", artifacts)

        return self.success_result_builder(result)
