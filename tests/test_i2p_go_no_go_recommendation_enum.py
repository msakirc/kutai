"""go_no_go_assessment (i2p step 1.14) must accept every recommendation its own
instruction defines — Go / Conditional / No-Go.

2026-06-01 (step 1.14 DLQ): the step instruction says "If score > 6: Go. If 4-6:
Conditional. If < 4: No-Go.", but go_no_go_decision's schema constrained
recommendation to equals=['Go', 'go']. So any assessment that wasn't an
unconditional Go (Conditional at 4-6, No-Go at <4) failed schema validation
("value 'Conditional' not in allowed set ['Go', 'go'] — this is a VERDICT field")
and DLQ'd — the analyst could never report anything but Go. This is the same bug
class fixed for the 1.13 reviewer verdict (test_i2p_review_verdict_enum.py).

recommendation is informational: step 2.1 reads go_no_go_decision but no gate
blocks on its value (Conditional drives may_need_clarification, not a DLQ). So the
schema must admit the full instructed enum.
"""
from __future__ import annotations

import json
import os

from src.workflows.engine.schema_dialect import validate_value

_I2P = os.path.join("src", "workflows", "i2p", "i2p_v3.json")


def _load():
    return json.load(open(_I2P, encoding="utf-8"))


def _step_by_name(name: str) -> dict:
    """Return the first step dict whose name matches *name*."""
    found = {}

    def rec(x):
        if found:
            return
        if isinstance(x, dict):
            if (x.get("name") or x.get("step_name")) == name and "instruction" in x:
                found["step"] = x
                return
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)

    rec(_load())
    assert "step" in found, f"step {name!r} not found"
    return found["step"]


def _decision_rule():
    return (_step_by_name("go_no_go_assessment").get("artifact_schema") or {})[
        "go_no_go_decision"
    ]


def test_recommendation_enum_admits_all_instructed_values():
    rule = _decision_rule()
    allowed = set(rule["fields"]["recommendation"]["equals"])
    assert {"Go", "Conditional", "No-Go"} <= allowed


def test_non_go_recommendation_validates():
    rule = _decision_rule()
    base = {"scores": {"market_attractiveness": 5}, "weighted_score": 5.2}
    for v in ("Go", "Conditional", "No-Go"):
        obj = {**base, "recommendation": v}
        assert validate_value(rule, obj, "go_no_go_decision") is None, v


def test_garbage_recommendation_still_rejected():
    rule = _decision_rule()
    obj = {
        "scores": {"market_attractiveness": 5},
        "weighted_score": 5.2,
        "recommendation": "totally_made_up",
    }
    assert validate_value(rule, obj, "go_no_go_decision") is not None


# ── lenient verdict matching: LLM decorates the token (mission #81) ──
# mission #81 #291858: the analyst emitted recommendation
# 'Conditional (needs_clarification)' — the right verdict, decorated with a
# parenthetical — and exact equals rejected it → DLQ. recommendation is an
# informational verdict (no downstream gate), so matching tolerates casing and
# trailing decoration: the leading token must still be one of the allowed set.


def test_decorated_recommendation_validates():
    rule = _decision_rule()
    base = {"scores": {"market_attractiveness": 5}, "weighted_score": 5.2}
    for v in (
        "Conditional (needs_clarification)",
        "Go (proceed to build)",
        "No-Go — kill the idea",
        "conditional",
        "GO",
    ):
        obj = {**base, "recommendation": v}
        assert validate_value(rule, obj, "go_no_go_decision") is None, v


def test_lenient_still_rejects_wrong_verdict():
    rule = _decision_rule()
    base = {"scores": {"market_attractiveness": 5}, "weighted_score": 5.2}
    for v in ("Maybe", "totally_made_up", "proceed"):
        obj = {**base, "recommendation": v}
        assert validate_value(rule, obj, "go_no_go_decision") is not None, v


def test_recommendation_rule_is_lenient():
    rule = _decision_rule()
    assert rule["fields"]["recommendation"].get("equals_lenient") is True


# ── schema_dialect: equals_lenient is opt-in; default stays exact ──


def _string_rule(**extra):
    return {"type": "string", "equals": ["Go", "Conditional", "No-Go"], **extra}


def test_equals_lenient_normalizes_case_and_decoration():
    rule = _string_rule(equals_lenient=True)
    assert validate_value(rule, "Conditional (needs_clarification)") is None
    assert validate_value(rule, "no-go") is None
    assert validate_value(rule, "Nope") is not None


def test_equals_default_stays_exact():
    rule = _string_rule()  # no equals_lenient
    assert validate_value(rule, "Conditional (needs_clarification)") is not None
    assert validate_value(rule, "Conditional") is None


# ── steering: go_no_go_decision.scores must drive downstream, not dead-end ──
# go_no_go used to be a near-dead artifact: produced at 1.14, weakly read by 2.1,
# ignored everywhere else. These assert the weak-dimension scores actually steer
# positioning (2.1), architecture (4.3), and the risk register (risk_assessment).


def test_scores_shape_pinned_in_instruction():
    # Downstream reads go_no_go_decision.scores.<dimension>; the producer
    # instruction must pin a flat object with the named keys so those reads land.
    instr = _step_by_name("go_no_go_assessment")["instruction"]
    for dim in (
        "market_attractiveness",
        "technical_feasibility",
        "differentiation_potential",
        "regulatory_risk",
    ):
        assert dim in instr, dim


def test_positioning_consumes_weak_dimensions():
    step = _step_by_name("product_vision_and_positioning")
    assert "go_no_go_decision" in step["input_artifacts"]
    instr = step["instruction"]
    assert "go_no_go_decision.scores" in instr
    assert "< 5" in instr or "weak" in instr.lower()


def test_architecture_consumes_go_no_go():
    step = _step_by_name("system_architecture_design")
    assert "go_no_go_decision" in step["input_artifacts"]
    assert "go_no_go_decision.scores" in step["instruction"]


def test_risk_assessment_consumes_go_no_go():
    step = _step_by_name("risk_assessment")
    assert "go_no_go_decision" in step["input_artifacts"]
    assert "go_no_go_decision" in step["instruction"]
