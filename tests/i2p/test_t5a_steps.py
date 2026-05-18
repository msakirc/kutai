"""Z1 Tier 5A — i2p_v3.json step-structure assertions for P6 + A5."""
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


def test_step_0_4a_present(workflow):
    s = _step_by_id(workflow, "0.4a")
    assert s["agent"] == "mechanical"
    assert s["executor"] == "mechanical"
    assert "0.4" in s["depends_on"]
    assert s["payload"]["action"] == "compliance_fingerprint_collection"
    assert "compliance_fingerprint" in s["output_artifacts"]
    assert any("compliance_fingerprint.json" in p for p in s.get("produces", []))
    assert "legacy_pre_compliance" in s.get("skip_when", "")


def test_step_1_11a_present(workflow):
    s = _step_by_id(workflow, "1.11a")
    # Q4 lock — no new LLM agent config; reuse `analyst`.
    assert s["agent"] == "analyst"
    assert "1.11" in s["depends_on"]
    assert "compliance_fingerprint" in s["input_artifacts"]
    assert "compliance_overlay" in s["output_artifacts"]
    assert "compliance_template_render" in (s.get("tools_hint") or [])
    # Post-hook to verify referenced templates exist on disk.
    assert "compliance_template_present" in (s.get("post_hooks") or [])
    assert "legacy_pre_compliance" in s.get("skip_when", "")


def test_step_6_6_has_compliance_blocker_post_hook(workflow):
    s = _step_by_id(workflow, "6.6")
    assert "compliance_blocker_check" in (s.get("post_hooks") or [])


def test_consumes_compliance_fingerprint_in_downstream(workflow):
    """0.6, 1.11, 3.3, 3.4 all consume compliance_fingerprint."""
    for sid in ("0.6", "1.11", "3.3", "3.4"):
        s = _step_by_id(workflow, sid)
        assert "compliance_fingerprint" in s["input_artifacts"], (
            f"step {sid} must consume compliance_fingerprint"
        )


def test_3_3_also_consumes_compliance_overlay(workflow):
    s = _step_by_id(workflow, "3.3")
    assert "compliance_overlay" in s["input_artifacts"]


def test_workflow_json_is_valid(workflow):
    """Sanity: file parses + has step list."""
    assert isinstance(workflow.get("steps"), list)
    assert len(workflow["steps"]) > 100
