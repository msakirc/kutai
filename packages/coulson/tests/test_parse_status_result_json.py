"""A reviewer verdict `{status, issues}` must round-trip as JSON, not Python repr.

Regression for m90 task 567426: the `elif "status" in parsed` legacy branch in
`_normalize_action` stored `result = str(parsed)` (Python repr, single quotes),
which the verify_review_verdict parser (json.loads) could not parse → a real
`fail` verdict was dropped as "no parseable review verdict" → DLQ.
"""
from __future__ import annotations

import json

from coulson.parsing import parse_action


def test_status_verdict_result_is_valid_json():
    raw = (
        '{"status": "fail", "issues": [{"target_artifact": "requirements_spec.md", '
        '"severity": "blocker", "problem": "Missing falsification triples"}]}'
    )
    act = parse_action(raw)
    assert act is not None
    assert act["action"] == "final_answer"
    # The stored result must be JSON the downstream verdict parser can json.loads.
    parsed = json.loads(act["result"])
    assert parsed["status"] == "fail"
    assert parsed["issues"][0]["severity"] == "blocker"


def test_status_verdict_result_preserves_unicode():
    # Turkish content must survive (ensure_ascii=False) — no \uXXXX escaping.
    raw = '{"status": "fail", "issues": [{"problem": "eksik doğrulama"}]}'
    act = parse_action(raw)
    parsed = json.loads(act["result"])
    assert parsed["issues"][0]["problem"] == "eksik doğrulama"
