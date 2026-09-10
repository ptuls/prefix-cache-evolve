"""Tests for deterministic artifact filesystem helpers."""

import hashlib
import json

import pytest

from prefix_cache_evolve.artifacts import file_sha256, write_json


def test_write_json_is_stable_and_atomic_on_serialization_failure(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    write_json(path, {"value": 1})
    original = path.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o777 == 0o644
    circular = []
    circular.append(circular)

    with pytest.raises(ValueError, match="Circular reference"):
        write_json(path, circular)

    assert json.loads(original) == {"value": 1}
    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".artifact.json.*")) == []


def test_file_sha256_streams_file_content(tmp_path) -> None:
    path = tmp_path / "input.bin"
    payload = b"prefix-cache-artifact"
    path.write_bytes(payload)

    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()
