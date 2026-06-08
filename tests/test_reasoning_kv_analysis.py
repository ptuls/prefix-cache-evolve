"""Tests for the reasoning decode-KV robustness analysis."""

from pathlib import Path

from prefix_cache_evolve.problems.prefix_kv_cache.configuration import DEFAULT_CONFIG_PATH
from prefix_cache_evolve.tools.analyze_reasoning_kv import _write_markdown, run_analysis


def test_reasoning_kv_analysis_compares_capacity_modes(tmp_path: Path) -> None:
    payload = run_analysis(DEFAULT_CONFIG_PATH, request_count=4, seeds=(3,))
    markdown_path = tmp_path / "reasoning.md"
    _write_markdown(markdown_path, payload)

    assert payload["schema"] == "prefix-kv-cache-reasoning-kv-analysis-v1"
    assert set(payload["modes"]) == {"prefix_only", "shared"}
    prefix_only = payload["modes"]["prefix_only"]["incumbent"]
    shared = payload["modes"]["shared"]["incumbent"]
    assert prefix_only["decode_kv_blocks_requested"] == 0
    assert shared["decode_kv_blocks_requested"] > 0
    assert shared["decode_kv_allocation_failure_rate"] > 0
    assert "Decode allocation failure is reported but is not yet a score term" in (
        markdown_path.read_text(encoding="utf-8")
    )
