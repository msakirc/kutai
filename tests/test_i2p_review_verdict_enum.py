"""research_quality_review (i2p step 1.13) must accept every verdict its own
instruction defines — pass / fail / needs_minor_fixes.

mission_79 #225600 (2026-05-31): the reviewer emitted verdict='needs_minor_fixes'
(exactly as instructed — checks 14/15 say to), but research_review_result's
schema constrained verdict to equals=['pass']. So ANY review that found problems
failed schema validation ("value 'needs_minor_fixes' not in allowed set
['pass']") and DLQ'd — the reviewer could never report issues. The verdict is
informational (feeds go_no_go 1.14); no gate blocks on it, so the schema must
admit the full instructed enum.
"""
from __future__ import annotations

import json
import os

from src.workflows.engine.schema_dialect import validate_value

_I2P = os.path.join("src", "workflows", "i2p", "i2p_v3.json")


def _review_rule():
    d = json.load(open(_I2P, encoding="utf-8"))
    found = {}

    def rec(x):
        if isinstance(x, dict):
            name = x.get("name") or x.get("step_name")
            if name == "research_quality_review":
                found["schema"] = x.get("artifact_schema") or {}
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)

    rec(d)
    return found["schema"]["research_review_result"]


def test_verdict_enum_admits_all_instructed_values():
    rule = _review_rule()
    allowed = set(rule["fields"]["verdict"]["equals"])
    assert {"pass", "fail", "needs_minor_fixes"} <= allowed


def test_non_pass_verdict_validates():
    rule = _review_rule()
    # Non-empty issues isolate the verdict-enum fix (an empty issues list trips
    # a separate empty-placeholder check — flagged as a latent sibling, not the
    # #225600 cause: that review carried tagged issues from checks 14/15).
    issues = [{"check": 14, "problem": "no transcripts"}]
    for v in ("pass", "fail", "needs_minor_fixes"):
        obj = {"verdict": v, "issues": issues}
        assert validate_value(rule, obj, "research_review_result") is None, v


def test_garbage_verdict_still_rejected():
    rule = _review_rule()
    obj = {"verdict": "totally_made_up", "issues": [{"check": 1, "problem": "x"}]}
    assert validate_value(rule, obj, "research_review_result") is not None
