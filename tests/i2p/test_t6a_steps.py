"""Z1 Tier 6A — i2p_v3.json step-structure assertions for A7 + P9."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "workflows" / "i2p" / "i2p_v3.json"
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


def test_step_0_1_has_find_similar_missions_post_hook(workflow):
    s = _step_by_id(workflow, "0.1")
    assert "find_similar_missions" in (s.get("post_hooks") or [])


def test_step_6_7z_index_mission_artifacts_present(workflow):
    s = _step_by_id(workflow, "6.7z")
    assert s["agent"] == "mechanical"
    assert s["executor"] == "mechanical"
    assert s["payload"]["action"] == "index_mission_artifacts"
    assert "6.6" in s["depends_on"]
    # legacy_pre_inheritance gate was removed; step is now unconditional
    assert not s.get("skip_when") or "legacy_pre_" not in s.get("skip_when", "")


def test_workflow_json_is_valid(workflow):
    assert isinstance(workflow.get("steps"), list)
    assert len(workflow["steps"]) > 100
