"""Coverage: review steps in i2p_v3 now declare a string `equals` gate on
their verdict/status/recommendation field. Failure values reject through
the normal grader retry → DLQ → depends_on cascade.
"""
from __future__ import annotations

import json
import pytest

from src.workflows.engine.loader import load_workflow
from src.workflows.engine.schema_dialect import validate_value


# (step_id, output_artifact_name, verdict_field, fail_value, pass_value)
#
# NOTE: steps 1.13 (research_review_result.verdict) and 1.14
# (go_no_go_decision.recommendation) are deliberately NOT hard gates — they are
# *informational* verdicts whose non-pass values (fail / needs_minor_fixes /
# Conditional / No-Go) must validate, not DLQ. No production code branches on
# their value. Their enums are covered by test_i2p_review_verdict_enum.py and
# test_i2p_go_no_go_recommendation_enum.py respectively. Keeping them here would
# re-assert the old "non-Go must DLQ" contract that DLQ'd legitimate
# Conditional/No-Go assessments (1.14 mission DLQ, 2026-06-01).
GATED_REVIEWS = [
    ("3.11", "requirements_review_result", "status", "fail", "pass"),
    ("4.16", "architecture_review_result", "status", "rejected", "approved"),
    ("5.5", "wireframe_review_result", "status", "rejected", "approved"),
    ("5.10", "design_review_result", "status", "rejected", "approved"),
    ("6.6", "project_plan_review_result", "status", "rejected", "approved"),
    ("7.16", "sprint_0_review_result", "status", "rejected", "approved"),
    ("12.5", "legal_review_result", "status", "blocked", "approved"),
    ("14.2", "checklist_review", "status", "rejected", "approved"),
]


def _step(step_id: str) -> dict:
    wf = load_workflow("i2p_v3")
    s = wf.get_step(step_id)
    assert s is not None, f"step {step_id!r} missing"
    return s


@pytest.mark.parametrize("step_id,artifact,field,fail_v,pass_v", GATED_REVIEWS)
def test_review_step_declares_equals_gate(step_id, artifact, field, fail_v, pass_v):
    step = _step(step_id)
    schema = (step.get("artifact_schema") or {}).get(artifact)
    assert schema is not None, f"{step_id} missing artifact_schema for {artifact}"
    fields = schema.get("fields") or {}
    rule = fields.get(field)
    assert rule is not None, f"{step_id} missing {field!r} field rule"
    assert rule.get("type") == "string", f"{step_id} {field} not string-typed"
    allowed = rule.get("equals")
    assert allowed is not None, f"{step_id} {field} missing equals"
    if isinstance(allowed, str):
        allowed = [allowed]
    assert pass_v in allowed, (
        f"{step_id} {field} equals={allowed} rejects expected pass value {pass_v!r}"
    )


@pytest.mark.parametrize("step_id,artifact,field,fail_v,pass_v", GATED_REVIEWS)
def test_review_step_rejects_fail_value(step_id, artifact, field, fail_v, pass_v):
    step = _step(step_id)
    schema = (step.get("artifact_schema") or {})[artifact]
    # Pass value validates clean
    other_fields = {f: "x" for f, r in (schema.get("fields") or {}).items() if f != field}
    pass_val = {field: pass_v, **other_fields}
    err = validate_value(schema, pass_val)
    assert err is None, f"{step_id} pass-value rejected unexpectedly: {err}"

    fail_val = {field: fail_v, **other_fields}
    err = validate_value(schema, fail_val)
    assert err is not None, f"{step_id} fail-value should be rejected"
    assert "VERDICT" in err or "verdict" in err.lower() or "not in allowed" in err
