"""Load and validate immutable incumbent bundles."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from prefix_cache_evolve.evaluators.complexity import scoring_fn_complexity

_INCUMBENT_ROOT = Path(__file__).parent
_REGISTRY_PATH = _INCUMBENT_ROOT / "registry.json"
_REGISTRY_SCHEMA = "prefix-kv-cache-incumbent-registry-v1"
_MANIFEST_SCHEMA = "prefix-kv-cache-incumbent-manifest-v1"
_IMMUTABLE_STATUSES = frozenset({"promoted", "retained"})
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class IncumbentRecord:
    """One immutable incumbent source and its promotion evidence."""

    manifest_path: Path
    payload: Mapping[str, Any]

    @property
    def incumbent_id(self) -> str:
        """Return the stable incumbent identifier."""
        return str(self.payload["id"])

    @property
    def role(self) -> str:
        """Return the registry role served by this incumbent."""
        return str(self.payload["role"])

    @property
    def source(self) -> Mapping[str, Any]:
        """Return source identity metadata."""
        return _mapping(self.payload, "source")

    @property
    def benchmark(self) -> Mapping[str, Any]:
        """Return the pinned benchmark identity and headline metrics."""
        return _mapping(self.payload, "benchmark")

    @property
    def provenance(self) -> Mapping[str, Any]:
        """Return the originating search artifact metadata."""
        return _mapping(self.payload, "provenance")

    @property
    def source_path(self) -> Path:
        """Return the immutable candidate source path."""
        return self.manifest_path.parent / str(self.source["path"])

    @property
    def source_sha256(self) -> str:
        """Return the pinned source digest."""
        return str(self.source["sha256"])

    @property
    def effective_complexity(self) -> int:
        """Return the pinned effective source complexity."""
        return int(self.source["effective_complexity"])

    def load_factory(self) -> Callable[..., Any]:
        """Import the candidate factory named by the manifest."""
        factory = self.load_symbol(str(self.source["factory"]))
        if not callable(factory):
            raise TypeError(f"{self.incumbent_id} factory is not callable")
        return factory

    def load_symbol(self, name: str) -> Any:
        """Load one symbol from the immutable candidate source module."""
        module = importlib.import_module(str(self.source["module"]))
        return getattr(module, name)


def incumbent_records() -> tuple[IncumbentRecord, ...]:
    """Return all registered incumbents in registry order."""
    registry = _read_json(_REGISTRY_PATH)
    _require_schema(registry, _REGISTRY_SCHEMA, _REGISTRY_PATH)
    entries = registry.get("incumbents")
    if not isinstance(entries, list):
        raise ValueError("incumbent registry must contain an incumbents list")
    records = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("incumbent registry entries must be objects")
        manifest_path = _INCUMBENT_ROOT / str(entry.get("manifest", ""))
        payload = _read_json(manifest_path)
        _require_schema(payload, _MANIFEST_SCHEMA, manifest_path)
        if entry.get("id") != payload.get("id"):
            raise ValueError(f"{manifest_path} id does not match registry entry")
        records.append(IncumbentRecord(manifest_path=manifest_path, payload=payload))
    return tuple(records)


def incumbent_record(incumbent_id: str) -> IncumbentRecord:
    """Return one incumbent by stable identifier."""
    for record in incumbent_records():
        if record.incumbent_id == incumbent_id:
            return record
    raise KeyError(f"unknown incumbent {incumbent_id!r}")


def current_incumbent(role: str = "production") -> IncumbentRecord:
    """Return the incumbent currently assigned to one registry role."""
    current = current_incumbents()
    try:
        return current[role]
    except KeyError as exc:
        raise KeyError(f"unknown incumbent role {role!r}") from exc


def current_incumbents() -> Mapping[str, IncumbentRecord]:
    """Return all current role assignments."""
    registry = _read_json(_REGISTRY_PATH)
    current = _mapping(registry, "current")
    records = {}
    for role, incumbent_id in current.items():
        record = incumbent_record(str(incumbent_id))
        if record.role != role:
            raise ValueError(
                f"current {role} incumbent {record.incumbent_id!r} declares role {record.role!r}"
            )
        records[str(role)] = record
    return records


@lru_cache(maxsize=None)
def current_incumbent_factory(role: str = "production") -> Callable[..., Any]:
    """Return the cached factory for one current incumbent role."""
    return current_incumbent(role).load_factory()


def validate_incumbent_registry() -> tuple[IncumbentRecord, ...]:
    """Fail closed when any registered source or manifest identity has drifted."""
    records = incumbent_records()
    ids = [record.incumbent_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("incumbent registry contains duplicate ids")

    registered_manifests = {record.manifest_path.resolve() for record in records}
    stored_manifests = {path.resolve() for path in _INCUMBENT_ROOT.glob("*/manifest.json")}
    if registered_manifests != stored_manifests:
        raise ValueError("incumbent registry and stored manifest bundles differ")

    for record in records:
        _validate_record(record)
    _validate_lineage(records)

    current_incumbents()
    return records


def _validate_lineage(records: tuple[IncumbentRecord, ...]) -> None:
    ids = {record.incumbent_id for record in records}
    parents_by_id = {}
    for record in records:
        parent_ids = record.provenance.get("lineage_parent_ids", [])
        if not isinstance(parent_ids, list):
            raise ValueError(f"{record.incumbent_id} lineage_parent_ids must be a list")
        if any(not isinstance(parent_id, str) for parent_id in parent_ids):
            raise ValueError(f"{record.incumbent_id} lineage_parent_ids must contain strings")
        unknown_parent_ids = set(parent_ids) - ids
        if unknown_parent_ids:
            raise ValueError(
                f"{record.incumbent_id} has unknown lineage parents: {sorted(unknown_parent_ids)}"
            )
        if record.incumbent_id in parent_ids:
            raise ValueError(f"{record.incumbent_id} cannot be its own lineage parent")
        parents_by_id[record.incumbent_id] = tuple(str(parent_id) for parent_id in parent_ids)

    complete = set()
    active = set()

    def visit(incumbent_id: str) -> None:
        if incumbent_id in complete:
            return
        if incumbent_id in active:
            raise ValueError("incumbent lineage contains a cycle")
        active.add(incumbent_id)
        for parent_id in parents_by_id[incumbent_id]:
            visit(parent_id)
        active.remove(incumbent_id)
        complete.add(incumbent_id)

    for incumbent_id in ids:
        visit(incumbent_id)


def _validate_record(record: IncumbentRecord) -> None:
    status = record.payload.get("status")
    if status not in _IMMUTABLE_STATUSES:
        raise ValueError(
            f"{record.incumbent_id} status must describe immutable promotion history, "
            f"not {status!r}"
        )
    if not record.payload.get("promoted_at"):
        raise ValueError(f"{record.incumbent_id} is missing promoted_at")
    if record.manifest_path.parent.name != record.incumbent_id:
        raise ValueError(f"{record.incumbent_id} bundle directory does not match its id")
    if record.manifest_path.parent.parent.resolve() != _INCUMBENT_ROOT.resolve():
        raise ValueError(f"{record.incumbent_id} manifest must be stored in an incumbent bundle")
    if record.source_path.parent.resolve() != record.manifest_path.parent.resolve():
        raise ValueError(f"{record.incumbent_id} source must be stored beside its manifest")
    if not record.source_path.is_file():
        raise ValueError(f"{record.incumbent_id} source is missing")
    if _file_sha256(record.source_path) != record.source_sha256:
        raise ValueError(f"{record.incumbent_id} source SHA-256 does not match manifest")

    mode = record.source.get("complexity_mode")
    if mode != "form_aware":
        raise ValueError(f"{record.incumbent_id} has unsupported complexity mode {mode!r}")
    source = record.source_path.read_text(encoding="utf-8")
    complexity = scoring_fn_complexity(source, form_aware=True)
    if complexity != record.effective_complexity:
        raise ValueError(f"{record.incumbent_id} effective complexity does not match manifest")

    source_artifact_sha256 = str(record.provenance.get("source_artifact_sha256", ""))
    if source_artifact_sha256 != record.source_sha256:
        raise ValueError(f"{record.incumbent_id} does not preserve its source artifact exactly")
    for key in ("stage", "run_id", "run_artifact_root", "source_artifact"):
        if not record.provenance.get(key):
            raise ValueError(f"{record.incumbent_id} provenance is missing {key}")

    module_name = str(record.source["module"])
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ValueError(f"{record.incumbent_id} source module cannot be resolved")
    if Path(spec.origin).resolve() != record.source_path.resolve():
        raise ValueError(f"{record.incumbent_id} source path and module target differ")
    record.load_factory()

    benchmark = record.benchmark
    if not benchmark.get("verifier_version"):
        raise ValueError(f"{record.incumbent_id} benchmark is missing verifier version")
    for key in ("panel_sha256", "evaluation_context_sha256"):
        value = str(benchmark.get(key, ""))
        if len(value) != _SHA256_LENGTH or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{record.incumbent_id} benchmark has invalid {key}")


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing incumbent metadata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"incumbent metadata field {key!r} must be an object")
    return value


def _require_schema(payload: Mapping[str, Any], schema: str, path: Path) -> None:
    if payload.get("schema") != schema:
        raise ValueError(f"{path} must use schema {schema!r}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
