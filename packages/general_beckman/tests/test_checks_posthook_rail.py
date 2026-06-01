"""`checks` — the separate pot for parameterized mechanical verifiers.

`post_hooks: list[str]` stays a pure list of registry kinds whose payload is
standard/derived (grade, verify_artifacts, grounding). Parameterized shape
verifiers — converted from standalone `.verify` workflow steps — carry their
OWN payload (which file to check, min/max bounds), so they live in a SEPARATE
producer field:

    "checks": [
      {"kind": "verify_adr_shape",
       "payload": {"adr_paths": [...], "expected_schema_version": "1"}}
    ]

No union type on `post_hooks`. The two pots share the same downstream rail:
the check's kind enters _pending_posthooks, the payload-builder uses the
declared payload verbatim, and a failed verdict re-pends the PRODUCER with
feedback via _apply_simple_blocker_verdict.
"""
from __future__ import annotations

import json

import pytest

from general_beckman.posthooks import determine_posthooks
from general_beckman.apply import _posthook_agent_and_payload, _CHECK_KINDS
from general_beckman.result_router import RequestPostHook


CHK = {
    "kind": "verify_interview_script_shape",
    "payload": {
        "action": "verify_interview_script_shape",
        "script_paths": ["mission_5/.intake/interview_script.md"],
        "min_questions": 5,
        "max_questions": 7,
    },
}


# ── determine_posthooks reads the separate `checks` pot ──────────────────

def test_determine_posthooks_appends_check_kinds():
    task = {"id": 1, "agent_type": "analyst"}
    ctx = {"checks": [CHK]}
    kinds = determine_posthooks(task, ctx, {})
    assert "verify_interview_script_shape" in kinds


def test_post_hooks_stays_pure_string_list_unaffected():
    # The existing string pot still works and is independent of `checks`.
    task = {"id": 1, "agent_type": "coder"}
    ctx = {"post_hooks": ["verify_artifacts"]}
    kinds = determine_posthooks(task, ctx, {})
    assert kinds == ["grade", "verify_artifacts"]


# ── payload-builder uses the declared payload VERBATIM (no derivation) ───

def test_payload_builder_uses_declared_check_payload_verbatim():
    a = RequestPostHook(source_task_id=5, kind="verify_interview_script_shape",
                        source_ctx={})
    # produces deliberately WRONG/extra — must be ignored; the check payload wins.
    source_ctx = {
        "produces": ["mission_5/.intake/interview_script.md",
                     "mission_5/.intake/register.md"],
        "checks": [CHK],
    }
    agent, spec = _posthook_agent_and_payload(a, {"id": 5}, source_ctx)
    assert agent == "mechanical"
    assert spec["posthook_kind"] == "verify_interview_script_shape"
    # verbatim: the script path only, NOT the extra produces entry
    assert spec["payload"] == CHK["payload"]


def test_payload_builder_multiple_checks_picks_matching_kind():
    adr = {"kind": "verify_adr_shape",
           "payload": {"action": "verify_adr_shape",
                       "adr_paths": ["mission_9/.adr/tech_stack_decision.json"],
                       "expected_schema_version": "1"}}
    cost = {"kind": "verify_cost_curve_present",
            "payload": {"action": "verify_cost_curve_present",
                        "adr_paths": ["mission_9/.adr/tech_stack_decision.json"]}}
    source_ctx = {"checks": [adr, cost]}
    a = RequestPostHook(source_task_id=9, kind="verify_cost_curve_present",
                        source_ctx={})
    _, spec = _posthook_agent_and_payload(a, {"id": 9}, source_ctx)
    assert spec["payload"] == cost["payload"]


def test_check_kinds_includes_registered_shape_verbs():
    # The rail is registry-driven: a kind registered as a check is in _CHECK_KINDS.
    assert "verify_interview_script_shape" in _CHECK_KINDS
