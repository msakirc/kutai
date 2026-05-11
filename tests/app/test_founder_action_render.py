"""Z6 T1D — founder_action card render tests."""
from __future__ import annotations

import pytest


def test_render_generic_card():
    from src.app.founder_action_render import render_action_card
    action = {
        "id": 7,
        "mission_id": 42,
        "kind": "generic",
        "title": "Send tax CSV to accountant",
        "why": "Monthly export",
        "instructions": ["Download CSV", "Email it"],
        "expected_output_kind": None,
        "blocking_step_id": "13.99",
    }
    text, kb = render_action_card(action)
    assert "#7" in text
    assert "generic" in text
    assert "13.99" in text
    assert "Download CSV" in text
    # 3 buttons total over 2 rows for non-cost_ack kinds.
    rows = kb.inline_keyboard
    assert len(rows) == 2
    btns = [b for row in rows for b in row]
    cbs = {b.callback_data for b in btns}
    assert "fa_inprogress_7" in cbs
    assert "fa_done_7" in cbs
    assert "fa_block_7" in cbs


def test_render_cost_ack_card_single_confirm():
    from src.app.founder_action_render import render_action_card
    action = {
        "id": 11,
        "mission_id": 1,
        "kind": "cost_ack",
        "title": "Confirm spend $50 for 13.1",
        "why": "irreversible deploy",
        "instructions": [],
        "expected_output_kind": "ack_only",
        "cost_estimate_usd": 50.0,
    }
    text, kb = render_action_card(action)
    assert "$50.00" in text
    assert "Confirm" in text
    rows = kb.inline_keyboard
    assert len(rows) == 1
    cbs = {b.callback_data for row in rows for b in row}
    assert "fa_done_11" in cbs
    assert "fa_block_11" in cbs
    # No in_progress for cost_ack — single decision.
    assert "fa_inprogress_11" not in cbs


def test_render_credential_paste_hint():
    from src.app.founder_action_render import render_action_card
    action = {
        "id": 3,
        "mission_id": 1,
        "kind": "credential_paste",
        "title": "Paste stripe credentials",
        "why": "13.12 needs Stripe",
        "instructions": ["Go to dashboard"],
        "expected_output_kind": "credential",
        "expected_output_schema": {
            "required_fields": ["secret_key", "webhook_secret"],
        },
    }
    text, _kb = render_action_card(action)
    assert "/credential add stripe" in text
    assert "secret_key" in text
    assert "webhook_secret" in text


def test_render_vendor_enroll_includes_followup_hint():
    from src.app.founder_action_render import render_action_card
    action = {
        "id": 4,
        "mission_id": 1,
        "kind": "vendor_enroll",
        "title": "Enroll: vercel",
        "why": "13.1 needs vercel",
        "instructions": ["Sign up at vercel.com"],
    }
    text, kb = render_action_card(action)
    assert "After enrolling" in text
    rows = kb.inline_keyboard
    # vendor_enroll uses the default 3-button layout.
    cbs = {b.callback_data for row in rows for b in row}
    assert "fa_done_4" in cbs


def test_render_kyc_uses_emoji():
    from src.app.founder_action_render import render_action_card
    action = {
        "id": 5, "mission_id": 1, "kind": "kyc",
        "title": "Apple KYC", "why": "App Store",
        "instructions": [], "expected_output_kind": None,
    }
    text, _ = render_action_card(action)
    assert "🪪" in text
