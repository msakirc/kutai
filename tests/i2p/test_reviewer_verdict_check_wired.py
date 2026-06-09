"""Wiring contract — the 7 status-verdict reviewer steps each declare the
verify_review_verdict mechanical check in their `checks` array.

This is what makes determine_posthooks emit a verify_review_verdict posthook
after each reviewer completes, so a FAIL verdict routes to the at-fault
producer (apply._apply_review_verdict) instead of dead-ending. 10.5 is
excluded (different findings/severity model).
"""
import json

import pytest

# The 7 reviewers that emit a {status, issues[]} status-verdict.
REVIEWER_STEP_IDS = ["1.13", "3.11", "4.16", "6.6", "7.16", "12.5", "14.2"]


def _steps():
    d = json.load(open("src/workflows/i2p/i2p_v3.json", encoding="utf-8"))
    out = {}

    def walk(o):
        if isinstance(o, dict):
            if o.get("id"):
                out[o["id"]] = o
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(d)
    return out


@pytest.mark.parametrize("step_id", REVIEWER_STEP_IDS)
def test_reviewer_step_declares_verify_review_verdict_check(step_id):
    steps = _steps()
    assert step_id in steps, f"step {step_id} missing from i2p_v3.json"
    step = steps[step_id]
    checks = step.get("checks") or []
    matching = [
        c for c in checks
        if isinstance(c, dict)
        and (c.get("payload") or {}).get("action") == "verify_review_verdict"
    ]
    assert matching, (
        f"step {step_id} must have a checks[] entry with "
        f"payload.action == 'verify_review_verdict'; got checks={checks!r}"
    )
    # The entry's kind must also name the check verb so _find_check_payload
    # locates it.
    assert any(c.get("kind") == "verify_review_verdict" for c in matching), (
        f"step {step_id} check entry must carry kind='verify_review_verdict'"
    )
