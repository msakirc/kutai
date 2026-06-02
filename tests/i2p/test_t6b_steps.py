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


def test_step_1_0a_present(workflow):
    """1.0a: query_planner produces prior_art_queries for the mechanical fetch."""
    s = _step_by_id(workflow, "1.0a")
    assert s["agent"] == "query_planner"
    assert "0.6" in s["depends_on"]
    assert "prior_art_queries" in s["output_artifacts"]
    assert any(
        "prior_art_queries.json" in p for p in s.get("produces", [])
    )
    assert any(
        p.startswith("mission_{mission_id}/.research/")
        for p in s.get("produces", [])
    )


def test_step_1_0b_present(workflow):
    """1.0b: mechanical prior_art_fetch step."""
    s = _step_by_id(workflow, "1.0b")
    assert s["agent"] == "mechanical"
    assert "1.0a" in s["depends_on"]
    assert "prior_art_candidates" in s["output_artifacts"]


def test_step_1_0c_present(workflow):
    """1.0c: prior_art_synthesizer closes the pipeline; post_hook gates the artifact."""
    s = _step_by_id(workflow, "1.0c")
    assert s["agent"] == "prior_art_synthesizer"
    assert "1.0b" in s["depends_on"]
    assert "prior_art_report" in s["output_artifacts"]
    assert any(
        "prior_art_report.json" in p for p in s.get("produces", [])
    )
    assert any(
        p.startswith("mission_{mission_id}/.research/")
        for p in s.get("produces", [])
    )
    assert "prior_art_min_coverage" in (s.get("post_hooks") or [])


def test_step_1_14_consumes_prior_art(workflow):
    s = _step_by_id(workflow, "1.14")
    assert "prior_art_report" in s["input_artifacts"]
    # Instruction augmented to reference the report.
    assert "prior_art_report" in s["instruction"]


def test_step_2_1_consumes_prior_art(workflow):
    s = _step_by_id(workflow, "2.1")
    assert "prior_art_report" in s["input_artifacts"]
    assert "key_lessons" in s["instruction"]


def test_step_1_0c_difficulty_medium(workflow):
    """1.0c (synthesizer) is the LLM-heavy step — medium difficulty."""
    s = _step_by_id(workflow, "1.0c")
    assert s.get("difficulty") == "medium"


def test_workflow_json_still_valid(workflow):
    assert isinstance(workflow.get("steps"), list)
    assert len(workflow["steps"]) > 100
