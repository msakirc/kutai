"""Z1 Tier 1 — verify_reverse_pitch_shape contract tests."""
from __future__ import annotations

import asyncio

from mr_roboto.verify_reverse_pitch_shape import verify_reverse_pitch_shape
from mr_roboto import run as mr_roboto_run


_GOOD_PITCH = """\
# Headline
TruthRate launches: review anything, separate facts from opinions.

## Sub-head
A universal review platform that helps consumers and business owners share
verified facts and reputation-weighted reviews.

## Customer Quote
"I finally found a place where I can check facts about a local pharmacy
before trusting it with my prescriptions." — Maya Chen, Austin TX

## Founder Quote
"We built TruthRate because the current review web is full of opinions
disguised as facts." — Founder

## FAQ
- **Is it free?** Reviewers are free; business owners pay to claim listings.
- **How are facts verified?** Community fact-check + reputation-weighted scoring.
- **Will you support other languages?** English at launch; Turkish in year two.
"""


_BAD_PITCH_MISSING_QUOTE = """\
# Headline
TruthRate launches.

## Sub-head
Review anything.

## FAQ
- **Is it free?** Yes mostly.
"""


_BAD_PITCH_PLACEHOLDER = """\
# Headline
<insert headline here>

## Sub-head
TBD

## Customer Quote
"TODO" — TODO

## Founder Quote
"FIXME" — Founder

## FAQ
- **Q1?** A1
"""


_PROTOTYPE_ACK = """\
acknowledgement: I am not building for users
"""


def test_good_pitch_passes():
    res = verify_reverse_pitch_shape(pitch_text=_GOOD_PITCH)
    assert res["ok"] is True, res
    assert set(res["found_sections"]) >= {
        "headline", "sub_head", "customer_quote", "founder_quote", "faq",
    }
    assert res["placeholders"] == []
    assert res["acknowledged_no_users"] is False


def test_pitch_missing_quote_rejected():
    res = verify_reverse_pitch_shape(pitch_text=_BAD_PITCH_MISSING_QUOTE)
    assert res["ok"] is False
    missing = set(res["missing_sections"])
    assert "customer_quote" in missing
    assert "founder_quote" in missing


def test_pitch_with_placeholders_rejected():
    res = verify_reverse_pitch_shape(pitch_text=_BAD_PITCH_PLACEHOLDER)
    assert res["ok"] is False
    assert res["placeholders"], res


def test_prototype_ack_passes_for_prototype_tier():
    res = verify_reverse_pitch_shape(
        pitch_text=_PROTOTYPE_ACK, ambition_tier="prototype"
    )
    assert res["ok"] is True
    assert res["acknowledged_no_users"] is True


def test_prototype_ack_blocked_for_private_beta():
    res = verify_reverse_pitch_shape(
        pitch_text=_PROTOTYPE_ACK, ambition_tier="private_beta"
    )
    assert res["ok"] is False
    assert res["acknowledged_no_users"] is True


def test_dispatch_via_mr_roboto_run():
    task = {
        "id": 1,
        "mission_id": 1,
        "payload": {
            "action": "verify_reverse_pitch_shape",
            "pitch_text": _GOOD_PITCH,
        },
    }
    result = asyncio.run(mr_roboto_run(task))
    assert result.status == "completed", result
    assert result.result["ok"] is True


def test_dispatch_failure_via_mr_roboto_run():
    task = {
        "id": 2,
        "mission_id": 1,
        "payload": {
            "action": "verify_reverse_pitch_shape",
            "pitch_text": _BAD_PITCH_MISSING_QUOTE,
        },
    }
    result = asyncio.run(mr_roboto_run(task))
    assert result.status == "failed"
