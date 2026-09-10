"""Run source candidates in a networkless container with only selected inputs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import selectors
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import click
from pydantic import TypeAdapter

from prefix_cache_evolve.evaluator_entry import load_candidate_factory_from_source
from prefix_cache_evolve.evaluators.baselines import BASELINE_REGISTRY
from prefix_cache_evolve.evaluators.configuration import EvaluatorConfig
from prefix_cache_evolve.evaluators.policy_costs import (
    PROFILE_CONTRACT,
    baseline_source,
    profile_policy,
)
from prefix_cache_evolve.evaluators.prefix_kv_cache import (
    EvaluationResult,
    _build_policy,
    scoring_fn_complexity,
)
from prefix_cache_evolve.problems.prefix_kv_cache.primitives import (
    MultiTimescaleDecay,
    decay_vector,
    threshold_excess,
)
from prefix_cache_evolve.problems.prefix_kv_cache.reproducibility import file_sha256
from prefix_cache_evolve.problems.prefix_kv_cache.specialist import (
    candidate_evaluator,
    candidate_exported_names,
)
from prefix_cache_evolve.problems.prefix_kv_cache.trace_replay import TRACE_REPLAY_CONTRACT

_SCOPE_ENV = "PREFIX_CACHE_EVOLVE_SANDBOX_SCOPE"
_SCOPE_LABEL = "prefix-cache-evolve.scope"
_MAX_STDOUT_BYTES = 32 * 1024 * 1024
_MAX_STDERR_BYTES = 64 * 1024
_PRIMITIVE_MODULE = "prefix_cache_evolve.problems.prefix_kv_cache.primitives"


class _ReadOnlyNamespace:
    """Expose permitted imports without sharing a mutable module namespace."""

    __slots__ = ("_name", "_values")
    _name: str
    _values: MappingProxyType[str, object]

    def __init__(self, name: str, values: dict[str, object]) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_values", MappingProxyType(values))

    def __getattr__(self, name: str) -> object:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(f"{self._name} has no candidate attribute {name}") from exc

    def __setattr__(self, name: str, value: object) -> None:
        del value
        raise AttributeError(f"{self._name} is read-only: {name}")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{self._name} is read-only: {name}")


def _candidate_imports() -> MappingProxyType[str, object]:
    """Build fresh facades whose reachable mutable state is candidate-local."""

    class CandidateMultiTimescaleDecay:
        """Delegate primitive behavior without exposing its shared class object."""

        def __init__(self, *args, **kwargs) -> None:
            self._delegate = MultiTimescaleDecay(*args, **kwargs)

        @property
        def timescale_count(self) -> int:
            return self._delegate.timescale_count

        @property
        def state_size(self) -> int:
            return self._delegate.state_size

        def observe(self, *args, **kwargs):
            return self._delegate.observe(*args, **kwargs)

        def observe_vector(self, *args, **kwargs):
            return self._delegate.observe_vector(*args, **kwargs)

        def values(self, *args, **kwargs):
            return self._delegate.values(*args, **kwargs)

        def combine(self, *args, **kwargs):
            return self._delegate.combine(*args, **kwargs)

    def candidate_decay_vector(*args, **kwargs):
        return decay_vector(*args, **kwargs)

    def candidate_threshold_excess(*args, **kwargs):
        return threshold_excess(*args, **kwargs)

    return MappingProxyType(
        {
            "math": _ReadOnlyNamespace(
                "math",
                {name: value for name, value in vars(math).items() if not name.startswith("_")},
            ),
            _PRIMITIVE_MODULE: _ReadOnlyNamespace(
                _PRIMITIVE_MODULE,
                {
                    "MultiTimescaleDecay": CandidateMultiTimescaleDecay,
                    "decay_vector": candidate_decay_vector,
                    "threshold_excess": candidate_threshold_excess,
                },
            ),
        }
    )


def _run_container_process(
    command: list[str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Drain the Docker client pipes within byte and wall-clock limits.

    Container memory limits do not cover output buffered by its host client.
    Read both pipes incrementally so either stream can fail closed before the
    outer evaluator removes the named container. Docker evaluation uses POSIX
    pipe selectors, as do the supported Linux/macOS evaluation environments.
    """
    deadline = time.monotonic() + timeout
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": _MAX_STDOUT_BYTES, "stderr": _MAX_STDERR_BYTES}
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            with selectors.DefaultSelector() as selector:
                assert process.stdout is not None and process.stderr is not None
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    for key, _ in selector.select(timeout=remaining):
                        chunk = os.read(key.fd, 64 * 1024)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        stream = key.data
                        if len(buffers[stream]) + len(chunk) > limits[stream]:
                            raise RuntimeError(
                                f"sandbox {stream} exceeded {limits[stream]} output bytes"
                            )
                        buffers[stream].extend(chunk)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                process.wait(timeout=remaining)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
        return subprocess.CompletedProcess(
            command,
            process.returncode,
            buffers["stdout"].decode("utf-8", errors="replace"),
            buffers["stderr"].decode("utf-8", errors="replace"),
        )


@contextmanager
def docker_evaluation_scope(image: str | None) -> Iterator[None]:
    """Clean up this search's containers even when Levi cancels a worker."""
    if image is None:
        yield
        return
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker is required for the configured candidate sandbox")
    previous = os.environ.get(_SCOPE_ENV)
    scope = uuid.uuid4().hex
    os.environ[_SCOPE_ENV] = scope
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_SCOPE_ENV, None)
        else:
            os.environ[_SCOPE_ENV] = previous
        remaining = subprocess.run(
            [docker, "ps", "--all", "--quiet", "--filter", f"label={_SCOPE_LABEL}={scope}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.split()
        if remaining:
            subprocess.run(
                [docker, "rm", "--force", *remaining], capture_output=True, timeout=15, check=True
            )


def evaluate_in_docker(
    source: str,
    config: EvaluatorConfig,
    *,
    splits: tuple[str, ...],
    profile: bool = False,
    baseline: str | None = None,
) -> EvaluationResult:
    """Evaluate source without exposing host files, credentials, or hidden inputs.

    The image must already be built. There is no fallback to host execution.
    A named container is removed even if the client times out or fails.
    """
    if not config.sandbox_image:
        raise ValueError("sandbox_image is required for container evaluation")
    if baseline is not None and (not profile or baseline not in BASELINE_REGISTRY.factories()):
        raise ValueError("baseline profiling requires a registered deployable baseline")
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker is required for the configured candidate sandbox")
    name = f"prefix-cache-eval-{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix="prefix-cache-sandbox-") as temporary:
        inputs = Path(temporary)
        inputs.chmod(0o755)
        traces = []
        for index, trace in enumerate(config.trace_workloads):
            if trace.split in splits:
                target = inputs / f"trace-{index}.jsonl"
                shutil.copyfile(trace.path, target)
                target.chmod(0o644)
                if file_sha256(target) != trace.sha256:
                    raise ValueError(f"{trace.path}: trace SHA-256 changed before sandbox snapshot")
                path = f"/inputs/{target.name}"
            else:
                # Retain identity pins but never open or mount quarantined data.
                path = f"/unavailable/trace-{index}.jsonl"
            traces.append(trace.model_copy(update={"path": path}))
        worker_config = config.with_updates(trace_workloads=traces)
        config_payload = worker_config.model_dump(mode="json")
        for trace in config_payload["trace_workloads"]:
            if not trace["capacity_sweep_blocks"]:
                trace.pop("capacity_sweep_blocks")
        payload: dict[str, Any] = {
            "source": source,
            "config": config_payload,
            "splits": list(splits),
        }
        if profile:
            payload["profile"] = True
            payload["baseline"] = baseline
        request = inputs / "request.json"
        request.write_text(json.dumps(payload), encoding="utf-8")
        request.chmod(0o644)
        command = [
            docker,
            "run",
            "--rm",
            "--name",
            name,
            "--label",
            f"{_SCOPE_LABEL}={os.environ.get(_SCOPE_ENV, name)}",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "10001:10001",
            "--pids-limit",
            "64",
            "--memory",
            str(config.max_memory_bytes),
            "--cpus",
            "2",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--mount",
            f"type=bind,src={inputs},dst=/inputs,readonly",
            "--entrypoint",
            "timeout",
            config.sandbox_image,
            "--signal=KILL",
            str(config.timeout_s),
            "python",
            "-m",
            "prefix_cache_evolve.problems.prefix_kv_cache.sandbox",
            "/inputs/request.json",
        ]
        try:
            process = _run_container_process(command, timeout=config.timeout_s + 5)
            if process.returncode:
                raise RuntimeError(
                    f"sandbox exited {process.returncode}: {process.stderr[-4000:].strip()}"
                )
            result = TypeAdapter(EvaluationResult).validate_json(process.stdout, strict=True)
            if (
                config.trace_workloads
                and result.candidate_metadata.get("trace_replay_contract") != TRACE_REPLAY_CONTRACT
            ):
                raise ValueError(
                    "sandbox trace replay contract is stale; rebuild and pin the evaluator image"
                )
            if profile:
                measured = json.loads(
                    str(result.candidate_metadata.get("policy_cost_profile", "{}"))
                )
                if measured.get("contract") != PROFILE_CONTRACT:
                    raise ValueError(
                        "sandbox profiler is stale; rebuild and pin the evaluator image"
                    )
            return result
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"sandbox evaluation exceeded {config.timeout_s}s") from exc
        finally:
            subprocess.run(
                [docker, "rm", "--force", name],
                capture_output=True,
                timeout=15,
                check=False,
            )


def _fresh_policy(
    source: str,
    exported_names: tuple[str, ...],
    capacity_blocks: int,
    block_size_tokens: int,
    seed: int,
) -> Any:
    """Reset candidate module/class/default-argument state for every trial."""
    factory = load_candidate_factory_from_source(
        source,
        exported_names=exported_names,
        import_overrides=_candidate_imports(),
    )
    return _build_policy(cast(Any, factory), capacity_blocks, block_size_tokens, seed)


@click.command()
@click.argument("request_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
def main(request_path: Path) -> None:
    """Evaluate one mounted JSON request inside the candidate container."""
    # Import lazily to avoid the evaluator -> sandbox -> evaluator cycle.
    from prefix_cache_evolve.problems.prefix_kv_cache.evaluator import _candidate_source_violations

    payload = json.loads(request_path.read_text(encoding="utf-8"))
    config = EvaluatorConfig.model_validate(payload["config"])
    source = payload["source"]
    baseline = payload.get("baseline")
    if baseline is not None and (
        not payload.get("profile") or baseline not in BASELINE_REGISTRY.factories()
    ):
        raise click.ClickException("Invalid profiling baseline")
    complexity = scoring_fn_complexity(source, form_aware=config.form_aware_complexity)
    # Source checks and the OS sandbox form one boundary. Selecting this backend
    # always fails closed even when a trusted host workflow relaxes its checks.
    sandbox_config = config.with_updates(reject_unsupported_source_patterns=True)
    violations = (
        _candidate_source_violations(source, complexity, sandbox_config) if baseline is None else []
    )
    if violations:
        raise click.ClickException("Static policy violations: " + "; ".join(violations))
    try:
        # Candidate diagnostics must never share the trusted result channel.
        # A real sink also avoids buffering arbitrary prints inside the worker.
        with (
            open(os.devnull, "w", encoding="utf-8") as diagnostics,
            redirect_stdout(diagnostics),
            redirect_stderr(diagnostics),
        ):
            factory: Any
            if baseline is not None:
                factory = BASELINE_REGISTRY.factories()[baseline]
                complexity = 0
            else:
                exported_names = candidate_exported_names(sandbox_config)
                factory = (
                    partial(_fresh_policy, source, exported_names)
                    if sandbox_config.candidate_policy_surface == "full"
                    else load_candidate_factory_from_source(
                        source,
                        exported_names=exported_names,
                        import_overrides=_candidate_imports(),
                    )
                )
            if payload.get("profile"):
                result = profile_policy(
                    cast(Any, factory),
                    sandbox_config,
                    complexity=complexity,
                    splits=tuple(payload["splits"]),
                )
                measured_source = baseline_source(baseline) if baseline is not None else source
                result.candidate_metadata["profile_source_sha256"] = hashlib.sha256(
                    measured_source.encode()
                ).hexdigest()
            else:
                result = candidate_evaluator(sandbox_config, splits=tuple(payload["splits"]))(
                    cast(Any, factory), scoring_fn_complexity=complexity
                )
    except BaseException as exc:
        # SystemExit(0), KeyboardInterrupt, and custom BaseException subclasses
        # are failed evaluations, never successful early protocol termination.
        raise click.ClickException(
            f"Candidate evaluation failed with {type(exc).__name__}"
        ) from exc
    click.echo(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
