"""Z6 T4A — verify the 12.1 split wiring in i2p_v3.json.

Asserts:
- 12.1 lists compliance_overlay in input_artifacts.
- 12.1 is now mechanical with payload.action=legal_document_render.
- 12.1b exists, depends on 12.1, is a writer, takes the rendered drafts
  + compliance_overlay + prd_final_summary as inputs.
- 12.2 still depends on 12.1 (carve-out per spec).
- 12.3 and 12.5 depend on 12.1b.
"""
from __future__ import annotations

import json
import os

import pytest


I2P_V3_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "src",
    "workflows",
    "i2p",
    "i2p_v3.json",
)


@pytest.fixture(scope="module")
def i2p_v3():
    with open(I2P_V3_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def steps_by_id(i2p_v3):
    steps = i2p_v3.get("steps") or i2p_v3.get("workflow_steps") or []
    return {s["id"]: s for s in steps}


def test_12_1_inputs_include_compliance_overlay(steps_by_id):
    step = steps_by_id["12.1"]
    assert "compliance_overlay" in (step.get("input_artifacts") or []), (
        "G5 fix: 12.1 must consume compliance_overlay"
    )


def test_12_1_is_mechanical_with_legal_render_action(steps_by_id):
    step = steps_by_id["12.1"]
    assert step.get("agent") == "mechanical"
    assert step.get("executor") == "mechanical"
    payload = step.get("payload") or {}
    assert payload.get("action") == "legal_document_render"


def test_12_1_outputs_three_legal_docs(steps_by_id):
    step = steps_by_id["12.1"]
    outs = set(step.get("output_artifacts") or [])
    assert {"terms_of_service", "privacy_policy", "cookie_policy"}.issubset(outs)


def test_12_1b_exists_and_depends_on_12_1(steps_by_id):
    assert "12.1b" in steps_by_id, "12.1b legal_documents_fill must exist"
    step = steps_by_id["12.1b"]
    assert "12.1" in (step.get("depends_on") or [])
    assert step.get("agent") == "writer"
    assert step.get("executor") != "mechanical"


def test_12_1b_inputs_include_drafts_overlay_and_prd(steps_by_id):
    step = steps_by_id["12.1b"]
    inputs = set(step.get("input_artifacts") or [])
    assert {
        "terms_of_service",
        "privacy_policy",
        "cookie_policy",
        "compliance_overlay",
        "prd_final_summary",
    }.issubset(inputs)


def test_12_1b_overwrites_same_three_artifacts(steps_by_id):
    step = steps_by_id["12.1b"]
    outs = set(step.get("output_artifacts") or [])
    assert outs == {"terms_of_service", "privacy_policy", "cookie_policy"}


def test_12_2_still_depends_on_12_1(steps_by_id):
    """Per T4A spec carve-out: 12.2 keeps its dep on 12.1, not 12.1b."""
    step = steps_by_id["12.2"]
    deps = step.get("depends_on") or []
    assert "12.1" in deps
    assert "12.1b" not in deps


def test_12_3_depends_on_12_1b(steps_by_id):
    step = steps_by_id["12.3"]
    deps = step.get("depends_on") or []
    assert "12.1b" in deps
    assert "12.1" not in deps, "12.3 should now depend on the filled drafts"


def test_12_5_depends_on_12_1b(steps_by_id):
    step = steps_by_id["12.5"]
    deps = step.get("depends_on") or []
    assert "12.1b" in deps
    assert "12.1" not in deps, "12.5 should now depend on the filled drafts"


def test_json_is_valid(i2p_v3):
    assert isinstance(i2p_v3, dict)
