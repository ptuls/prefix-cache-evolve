"""Shared helpers for Levi evaluator entry points."""

from __future__ import annotations

import importlib.util
import math
import multiprocessing
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Generic, Sequence, TypeVar

try:
    import resource
except ImportError:  # pragma: no cover - Windows does not provide resource
    resource = None  # type: ignore[assignment]

ResultT = TypeVar("ResultT")
_PROCESS_MEMORY_HEADROOM_BYTES = 256 * 1024 * 1024


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
        # Pool workers cannot create child processes. Their parent pool is
        # responsible for enforcing its evaluation timeout.
        return func(*args, **kwargs)

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
        if not receive_conn.poll(timeout_seconds):
            _stop_process(process)
            raise TimeoutError(f"evaluation exceeded {timeout_seconds}s wall-clock limit")
        try:
            payload = receive_conn.recv()
        except EOFError as exc:
            raise RuntimeError("evaluation worker exited without returning a result") from exc
    finally:
        receive_conn.close()
        _stop_process(process)

    status, *values = payload
    if status == "result":
        return values[0]  # type: ignore[no-any-return]
    if status == "error":
        raise values[0]
    error_type, error_message, full_traceback = values
    raise RuntimeError(f"evaluation worker raised {error_type}: {error_message}\n{full_traceback}")


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


def _set_resource_limit(resource_id: int, soft_limit: int, hard_limit: int) -> None:
    """Lower one resource limit without attempting to raise an inherited hard cap."""
    if resource is None:
        return
    _, inherited_hard = resource.getrlimit(resource_id)
    if inherited_hard != resource.RLIM_INFINITY:
        hard_limit = min(hard_limit, inherited_hard)
        soft_limit = min(soft_limit, hard_limit)
    resource.setrlimit(resource_id, (soft_limit, hard_limit))


def _stop_process(process) -> None:
    """Terminates and reaps an evaluation worker if it is still running."""
    if process.is_alive():
        process.terminate()
    process.join(timeout=1.0)
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

    module = importlib.util.module_from_spec(spec)
    _exec_registered_module(module, lambda: spec.loader.exec_module(module))  # type: ignore[call-arg]

    factory = extract_exported_callable(module, exported_names)
    if not callable(factory):
        raise TypeError(f"{exported_names[0]} must be callable")
    return factory


def load_candidate_factory_from_source(
    source: str,
    exported_names: Sequence[str] = ("candidate_factory", "build_candidate"),
) -> Callable[..., object]:
    """Loads a candidate factory from Python source text."""
    module = ModuleType("candidate_module")
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
            return getattr(module, name)  # type: ignore[return-value]
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
