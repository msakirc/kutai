"""Z6 polish P4 — static coherence of the 12.1 → 12.1b legal-doc chain.

Complements ``test_z6_t4a_12_1_wiring.py`` by adding:
* artifact-graph break check (every input_artifact of 12.1b / 12.2 /
  12.3 / 12.5 is produced by an upstream step in the chain),
* counsel-guard text in 12.1b instruction (do NOT fill jurisdiction
  clauses that require counsel),
* 12.1b output overlap with 12.1 (same three artifact names; 12.1b
  overwrites in place).
"""
from __future__ import annotations

import json
import os

import pytest


I2P_V3_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "src", "workflows", "i2p", "i2p_v3.json",
)


@pytest.fixture(scope="module")
def i2p_v3():
    with open(I2P_V3_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def steps_by_id(i2p_v3):
    steps = i2p_v3.get("steps") or i2p_v3.get("workflow_steps") or []
    return {s["id"]: s for s in steps}


LEGAL_DOCS = {"terms_of_service", "privacy_policy", "cookie_policy"}


def test_json_is_valid(i2p_v3):
    """Static JSON parse — the file must be loadable."""
    assert isinstance(i2p_v3, dict)


def test_12_1_and_12_1b_outputs_overlap(steps_by_id):
    """12.1b must overwrite the same three artifact names that 12.1 emits."""
    outs_12_1 = set(steps_by_id["12.1"].get("output_artifacts") or [])
    outs_12_1b = set(steps_by_id["12.1b"].get("output_artifacts") or [])
    assert LEGAL_DOCS.issubset(outs_12_1)
    assert outs_12_1b == LEGAL_DOCS, (
        "12.1b must overwrite tos+privacy+cookie in place — output set "
        "should be exactly the three legal docs"
    )


def test_12_1b_inputs_satisfied_by_12_1(steps_by_id):
    """Every legal-doc input 12.1b reads must be produced by 12.1.

    Ambient artifacts (prd_final_summary, compliance_overlay) flow from
    earlier phases via context/blackboard, not as explicit
    output_artifacts — those are workflow-wide and outside this test's
    scope. We pin only that the three docs from 12.1 are present.
    """
    step = steps_by_id["12.1b"]
    inputs = set(step.get("input_artifacts") or [])
    outs_12_1 = set(steps_by_id["12.1"].get("output_artifacts") or [])
    # Every legal doc 12.1b reads must come from 12.1.
    assert LEGAL_DOCS.issubset(inputs), (
        "12.1b must declare the three legal docs as inputs"
    )
    assert LEGAL_DOCS.issubset(outs_12_1), (
        "12.1 must produce the three legal docs that 12.1b consumes"
    )


def test_12_2_inputs_satisfied(steps_by_id):
    """12.2 cookie_consent_implementation reads cookie_policy from 12.1.

    Per the T4A carve-out, 12.2 keeps dep on 12.1 (the mechanical render
    is sufficient for the consent-banner planning — 12.1b's fill targets
    counsel/jurisdiction clauses, not the cookie catalog itself).
    """
    step = steps_by_id["12.2"]
    inputs = set(step.get("input_artifacts") or [])
    deps = set(step.get("depends_on") or [])
    assert "cookie_policy" in inputs
    # Either 12.1 or 12.1b must produce cookie_policy and be in deps.
    producers = {
        sid for sid, s in steps_by_id.items()
        if "cookie_policy" in set(s.get("output_artifacts") or [])
    }
    assert producers & deps, (
        f"12.2 reads cookie_policy but none of its deps {deps} produce it; "
        f"producers={producers}"
    )


def test_12_3_inputs_satisfied(steps_by_id):
    step = steps_by_id["12.3"]
    inputs = set(step.get("input_artifacts") or [])
    deps = set(step.get("depends_on") or [])
    # 12.3 consumes the filled docs → must depend on 12.1b.
    assert "12.1b" in deps, "12.3 must depend on 12.1b for filled drafts"
    # And cookie_consent_implementation from 12.2.
    if "cookie_consent_implementation" in inputs:
        assert "12.2" in deps


def test_12_5_inputs_satisfied(steps_by_id):
    step = steps_by_id["12.5"]
    deps = set(step.get("depends_on") or [])
    inputs = set(step.get("input_artifacts") or [])
    assert "12.1b" in deps, "12.5 must depend on 12.1b for filled drafts"
    # gdpr/ccpa from 12.3, license_* from 12.4.
    if "gdpr_compliance_result" in inputs or "ccpa_compliance_result" in inputs:
        assert "12.3" in deps
    if "license_audit_result" in inputs or "license_decision" in inputs:
        assert "12.4" in deps


def test_12_1b_instruction_has_counsel_guard(steps_by_id):
    """12.1b must explicitly warn against auto-filling counsel-required text."""
    instr = steps_by_id["12.1b"].get("instruction", "")
    # The guard wording from the v2 spec.
    assert "Do NOT" in instr, "12.1b instruction must contain a 'Do NOT' guard"
    # Counsel-required items must be called out.
    lowered = instr.lower()
    assert "counsel" in lowered, (
        "12.1b instruction must reference licensed counsel"
    )
    # Jurisdiction-specific clauses get the explicit carve-out.
    assert "jurisdiction" in lowered, (
        "12.1b instruction must mention jurisdiction-specific clauses"
    )


def test_12_1b_keeps_legal_review_markers(steps_by_id):
    """12.1b done_when must leave [LEGAL REVIEW REQUIRED] on counsel items."""
    done_when = steps_by_id["12.1b"].get("done_when", "")
    assert "[LEGAL REVIEW REQUIRED]" in done_when
