"""Z6 T5C — tests for mr_roboto.executors.stripe_payment_flow_test."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mr_roboto.executors.stripe_payment_flow_test import (
    _detect_dispute,
    run,
)


# ── unit: dispute detection ────────────────────────────────────────────────


def test_detect_dispute_true_for_dict_blob():
    assert _detect_dispute({"error": "charge.dispute.created"}) is True


def test_detect_dispute_true_for_string():
    assert _detect_dispute("Dispute opened on charge X") is True


def test_detect_dispute_false_for_unrelated():
    assert _detect_dispute({"error": "card_declined"}) is False
    assert _detect_dispute("auth failed") is False
    assert _detect_dispute(None) is False


# ── helpers ────────────────────────────────────────────────────────────────


def _provisioned():
    return {
        "items": [
            {
                "product_name": "Pro",
                "kutay_id": "deadbeef",
                "stripe_product_id": "prod_X",
                "stripe_price_id": "price_X",
            }
        ]
    }


# ── integration: full happy path ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_pipeline_success_path():
    async def _fake_load(mission_id, name):
        if name == "stripe_provisioned":
            return _provisioned()
        return None

    persisted: dict = {}

    async def _fake_persist(mission_id, name, value):
        persisted[name] = value

    counter = {"n": 0}

    async def _fake_vc(task, service, action, params):
        counter["n"] += 1
        match action:
            case "create_customer":
                return {"ok": True, "result": {"id": "cus_1"}}
            case "create_checkout_session":
                return {"ok": True, "result": {"id": "cs_1"}}
            case "confirm_test_payment":
                return {"ok": True, "result": {"id": "pi_1"}}
            case "list_subscriptions":
                return {"ok": True, "result": {"data": [{"id": "sub_1"}]}}
            case "cancel_subscription":
                return {"ok": True, "result": {"id": "sub_1"}}
            case "retrieve_balance":
                return {"ok": True, "result": {"available": []}}
            case _:
                return {"ok": False, "reason": "unknown"}

    fa_calls: list = []

    async def _fake_fa_emit(mission_id, title, why, *, kind="generic"):
        fa_calls.append((mission_id, title, why, kind))

    with patch(
        "mr_roboto.executors.stripe_payment_flow_test._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ), patch(
        "mr_roboto.executors.stripe_payment_flow_test._persist_artifact",
        new=AsyncMock(side_effect=_fake_persist),
    ), patch(
        "mr_roboto.executors.stripe_payment_flow_test._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ), patch(
        "mr_roboto.executors.stripe_payment_flow_test._emit_founder_action",
        new=AsyncMock(side_effect=_fake_fa_emit),
    ):
        res = await run({"mission_id": 7})

    assert res["ok"] is True
    assert res["pass_fail"] == "pass"
    assert set(res["flows_tested"]) >= {
        "create_customer",
        "create_checkout_session",
        "confirm_test_payment",
        "list_subscriptions",
        "cancel_subscription",
        "retrieve_balance",
    }
    assert "payment_flow_results" in persisted
    assert not fa_calls  # no founder_action on full success


# ── partial failure → generic founder_action ──────────────────────────────


@pytest.mark.asyncio
async def test_partial_failure_emits_generic_founder_action():
    async def _fake_load(mission_id, name):
        if name == "stripe_provisioned":
            return _provisioned()
        return None

    async def _fake_vc(task, service, action, params):
        if action == "create_customer":
            return {"ok": True, "result": {"id": "cus_1"}}
        if action == "list_subscriptions":
            return {
                "ok": False,
                "reason": "vendor_error",
                "error": "no_such_customer",
            }
        return {"ok": True, "result": {"id": f"{action}_id"}}

    fa_calls: list = []

    async def _fake_fa_emit(mission_id, title, why, *, kind="generic"):
        fa_calls.append((mission_id, title, why, kind))

    with patch(
        "mr_roboto.executors.stripe_payment_flow_test._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ), patch(
        "mr_roboto.executors.stripe_payment_flow_test._persist_artifact",
        new=AsyncMock(),
    ), patch(
        "mr_roboto.executors.stripe_payment_flow_test._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ), patch(
        "mr_roboto.executors.stripe_payment_flow_test._emit_founder_action",
        new=AsyncMock(side_effect=_fake_fa_emit),
    ):
        res = await run({"mission_id": 7})

    assert res["ok"] is False
    assert res["pass_fail"] == "fail"
    assert fa_calls and fa_calls[0][3] == "generic"


# ── dispute failure → legal_counsel founder_action ─────────────────────────


@pytest.mark.asyncio
async def test_dispute_failure_emits_legal_counsel():
    async def _fake_load(mission_id, name):
        if name == "stripe_provisioned":
            return _provisioned()
        return None

    async def _fake_vc(task, service, action, params):
        if action == "confirm_test_payment":
            return {
                "ok": False,
                "reason": "vendor_error",
                "error": "charge.dispute.created on charge ch_1",
            }
        if action == "create_customer":
            return {"ok": True, "result": {"id": "cus_1"}}
        return {"ok": True, "result": {"id": f"{action}_id"}}

    fa_calls: list = []

    async def _fake_fa_emit(mission_id, title, why, *, kind="generic"):
        fa_calls.append((mission_id, title, why, kind))

    with patch(
        "mr_roboto.executors.stripe_payment_flow_test._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ), patch(
        "mr_roboto.executors.stripe_payment_flow_test._persist_artifact",
        new=AsyncMock(),
    ), patch(
        "mr_roboto.executors.stripe_payment_flow_test._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ), patch(
        "mr_roboto.executors.stripe_payment_flow_test._emit_founder_action",
        new=AsyncMock(side_effect=_fake_fa_emit),
    ):
        res = await run({"mission_id": 7})

    assert res["ok"] is False
    assert fa_calls and fa_calls[0][3] == "legal_counsel"


# ── input gating ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_provisioned_artifact():
    async def _fake_load(*_a, **_kw):
        return None
    with patch(
        "mr_roboto.executors.stripe_payment_flow_test._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ):
        res = await run({"mission_id": 7})
    assert res["ok"] is False
    assert res["reason"] == "no_provisioned_products"


@pytest.mark.asyncio
async def test_first_item_missing_price_id():
    async def _fake_load(*_a, **_kw):
        return {"items": [{"product_name": "X", "stripe_price_id": None}]}
    with patch(
        "mr_roboto.executors.stripe_payment_flow_test._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ):
        res = await run({"mission_id": 7})
    assert res["ok"] is False
    assert res["reason"] == "first_provisioned_missing_price_id"


@pytest.mark.asyncio
async def test_missing_mission_id():
    res = await run({})
    assert res["ok"] is False
    assert res["reason"] == "missing_mission_id"
