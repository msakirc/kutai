"""Z1 Tier 6B — i2p_v3.json step-structure assertions for P5."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "workflows" / "i2p" / "i2p_v3.json"
)


@pytest.fixture(scope="module")
def workflow():
    with open(WORKFLOW_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _step_by_id(workflow: dict, step_id: str) -> dict:
    for s in workflow["steps"]:
        if s.get("id") == step_id:
            return s
    raise KeyError(f"step {step_id!r} not found")


def test_step_1_0_present(workflow):
    s = _step_by_id(workflow, "1.0")
    assert s["agent"] == "researcher"  # Q4 lock — reuse, no new agent
    assert "0.6" in s["depends_on"]
    assert "find_prior_art" in (s.get("tools_hint") or [])
    assert "prior_art_report" in s["output_artifacts"]
    assert any(
        "prior_art_report.json" in p for p in s.get("produces", [])
    )
    # Path uses mission_{mission_id}/ prefix, NOT workspace/
    assert any(
        p.startswith("mission_{mission_id}/.research/")
        for p in s.get("produces", [])
    )
    assert "prior_art_min_coverage" in (s.get("post_hooks") or [])
    # legacy_pre_prior_art gate was removed; step is now unconditional
    assert not s.get("skip_when") or "legacy_pre_" not in s.get("skip_when", "")


def test_step_1_14_consumes_prior_art(workflow):
    s = _step_by_id(workflow, "1.14")
    assert "prior_art_report" in s["input_artifacts"]
    # Instruction augmented to reference the report.
    assert "prior_art_report" in s["instruction"]


def test_step_2_1_consumes_prior_art(workflow):
    s = _step_by_id(workflow, "2.1")
    assert "prior_art_report" in s["input_artifacts"]
    assert "key_lessons" in s["instruction"]


def test_step_1_0_produces_path_uses_research_subdir(workflow):
    s = _step_by_id(workflow, "1.0")
    paths = s.get("produces") or []
    assert any(".research" in p for p in paths)


def test_step_1_0_difficulty_medium(workflow):
    s = _step_by_id(workflow, "1.0")
    assert s.get("difficulty") == "medium"


def test_workflow_json_still_valid(workflow):
    assert isinstance(workflow.get("steps"), list)
    assert len(workflow["steps"]) > 100
