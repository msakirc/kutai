"""direct_competitor_identification (i2p step 1.3) must require a realistic
number of competitors at the cheap mechanical schema gate — not just rely on
the expensive LLM grader.

mission_79 #225586 (2026-05-31): the researcher returned ONE direct competitor
(Habitica). artifact_schema.min_items was 1, so the mechanical schema gate
PASSED it; only the LLM grader caught the thinness (COMPLETE:NO) — after burning
a full grade cycle, then DLQ'd. Aligning the cheap gate's floor with the
grader's bar rejects a thin list immediately and retries before the grade.
"""
from __future__ import annotations

import json
import os

from src.workflows.engine.schema_dialect import validate_value

_I2P = os.path.join("src", "workflows", "i2p", "i2p_v3.json")


def _competitor_rule():
    d = json.load(open(_I2P, encoding="utf-8"))
    found = {}

    def rec(x):
        if isinstance(x, dict):
            name = x.get("name") or x.get("step_name")
            if name == "direct_competitor_identification":
                found["schema"] = x.get("artifact_schema") or {}
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)

    rec(d)
    return found["schema"]["direct_competitors_list"]


def _item():
    return {
        "name": "X",
        "website_url": "https://x.example",
        "one_line_description": "d",
        "platforms": "Web",
        "status": "live",
    }


def test_competitor_floor_is_realistic():
    rule = _competitor_rule()
    assert int(rule.get("min_items", 0)) >= 3


def test_thin_competitor_list_rejected():
    rule = _competitor_rule()
    # 1 and 2 competitors must be rejected by the mechanical gate.
    assert validate_value(rule, [_item()], "direct_competitors_list") is not None
    assert validate_value(rule, [_item()] * 2, "direct_competitors_list") is not None


def test_sufficient_competitor_list_passes():
    rule = _competitor_rule()
    assert validate_value(rule, [_item()] * 3, "direct_competitors_list") is None
