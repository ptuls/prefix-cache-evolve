"""Experimental controls must survive configuration generation and reuse."""

import copy
from pathlib import Path

import pytest
import yaml

from prefix_cache_evolve.tools.seed_comparison import prepare_campaign, validate_campaign
from prefix_cache_evolve.workflow.config import ConfigLoader


def _prepare(tmp_path: Path, monkeypatch, **overrides):
    monkeypatch.delenv("LEVI_MODEL", raising=False)
    config = tmp_path / "source.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "llm": {"primary_model": "test-mutation", "secondary_model": "test-diversity"},
                "prompt": {"system_message": "Preserve the strong incumbent."},
                "search": {"notes": "The weak seed should be replaced."},
                "problem": {
                    "description": "Biased historical prompt.",
                    "settings": {
                        "verifier_version": "1.0.0",
                        "request_count": 4,
                        "seeds": [11],
                        "train_families": ["shared_system_prompt"],
                        "validation_families": ["phase_shift_prompts"],
                        "probe_families": ["agent_trace_branching"],
                        "hidden_families": ["adversarial_unique_prompts"],
                    },
                },
            }
        )
    )
    minimal = tmp_path / "minimal.py"
    minimal.write_text("def build_candidate():\n    return 1\n")
    strong = tmp_path / "strong.py"
    strong.write_text("def build_candidate():\n    return 2\n")
    kwargs = dict(
        config_path=config,
        output=tmp_path / "campaign",
        minimal=minimal,
        strong=strong,
        search_seeds=(101, 211, 307),
        evaluations=64,
        dollars_per_run=2.0,
        seconds_per_run=14400,
    )
    kwargs.update(overrides)
    return prepare_campaign(**kwargs)


def test_pairs_differ_only_in_starting_source_and_output_location(tmp_path, monkeypatch):
    manifest = _prepare(tmp_path, monkeypatch)
    output = tmp_path / "campaign"
    common = yaml.safe_load((output / "common.yaml").read_text())
    assert len(manifest["runs"]) == 6
    assert manifest["planned_dollar_guardrails_total"] == 12
    assert manifest["status"] == "prepared_not_run"
    for run in manifest["runs"]:
        path = output / run["id"] / "evolution.yaml"
        config = yaml.safe_load(path.read_text())
        assert config.pop("output_dir") == str(output / run["id"] / "search")
        assert config.pop("search") == {
            "seed": run["search_seed"],
            "seed_program": "seed_program.py",
        }
        expected = copy.deepcopy(common)
        expected.pop("search")
        assert config == expected
        prompt = ConfigLoader().load(path).problem_description
        assert "strong incumbent" not in prompt
        assert "weak seed" not in prompt
        assert "Biased historical" not in prompt
    assert common["problem"]["settings"]["hidden_families"] == []
    assert common["problem"]["settings"]["probe_families"] == []
    assert common["cvt"]["data_driven_centroids"] is False
    assert validate_campaign(output) == manifest


@pytest.mark.parametrize("seeds", [(101,), (101, 101), (101, -1), (101, 2**32)])
def test_campaign_rejects_nonindependent_or_invalid_search_seeds(tmp_path, monkeypatch, seeds):
    with pytest.raises(ValueError, match="search seeds"):
        _prepare(tmp_path, monkeypatch, search_seeds=seeds)
    assert not (tmp_path / "campaign").exists()


def test_frozen_campaign_refuses_overwrite_and_detects_changed_treatment(tmp_path, monkeypatch):
    manifest = _prepare(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError):
        _prepare(tmp_path, monkeypatch)
    run = manifest["runs"][0]["id"]
    (tmp_path / "campaign" / run / "seed_program.py").write_text("changed")
    with pytest.raises(ValueError, match="frozen campaign input changed"):
        validate_campaign(tmp_path / "campaign")


def test_runtime_override_is_not_silently_accepted(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    monkeypatch.setenv("LEVI_MODEL", "different-model")
    with pytest.raises(ValueError, match="runtime differs"):
        validate_campaign(tmp_path / "campaign")
