"""Z6 T5D — tests for mr_roboto.executors.stripe_revenue_digest."""
from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, patch

import pytest

from mr_roboto.executors.stripe_revenue_digest import (
    _format_money_cents,
    _iso_year_week,
    _normalise_mrr,
    _render_digest,
    run,
)


# ── unit: MRR normalisation ────────────────────────────────────────────────


def _sub(unit_amount, interval, currency="usd", interval_count=1, qty=1, status="active"):
    return {
        "status": status,
        "items": {
            "data": [
                {
                    "quantity": qty,
                    "price": {
                        "unit_amount": unit_amount,
                        "currency": currency,
                        "recurring": {
                            "interval": interval,
                            "interval_count": interval_count,
                        },
                    },
                }
            ]
        },
    }


def test_normalise_mrr_monthly_active_only():
    subs = [
        _sub(1000, "month"),
        _sub(2000, "month", status="canceled"),
    ]
    n, mrr, cur = _normalise_mrr(subs)
    assert n == 1
    assert mrr == 1000
    assert cur == "USD"


def test_normalise_mrr_yearly_divided_by_12():
    subs = [_sub(12000, "year")]
    n, mrr, _ = _normalise_mrr(subs)
    assert n == 1
    assert mrr == 1000  # 12000 / 12


def test_normalise_mrr_weekly_multiplied_by_4():
    subs = [_sub(250, "week")]
    _, mrr, _ = _normalise_mrr(subs)
    assert mrr == 1000  # 250 * 4


def test_normalise_mrr_no_active():
    n, mrr, _ = _normalise_mrr([_sub(1000, "month", status="canceled")])
    assert n == 0
    assert mrr == 0


def test_normalise_mrr_quantity_aware():
    subs = [_sub(1000, "month", qty=3)]
    n, mrr, _ = _normalise_mrr(subs)
    assert n == 1
    assert mrr == 3000


# ── unit: formatting helpers ──────────────────────────────────────────────


def test_format_money_cents():
    assert _format_money_cents(1234, "usd") == "12.34 USD"
    assert _format_money_cents(100000, "EUR") == "1,000.00 EUR"


def test_iso_year_week_shape():
    yw = _iso_year_week(datetime.datetime(2026, 5, 11, 12, 0, 0))
    assert yw.startswith("2026-W")


def test_render_digest_includes_required_lines():
    body = _render_digest(
        "2026-W19", active_subs=3, mrr_cents=15000, currency="USD",
        balance_payload={
            "available": [{"amount": 50000, "currency": "usd"}],
            "pending": [{"amount": 1000, "currency": "usd"}],
        },
    )
    assert "Active subscriptions: **3**" in body
    assert "MRR (naive): **150.00 USD**" in body
    assert "ARR (12× MRR): **1,800.00 USD**" in body
    assert "Available balance:" in body
    assert "Pending balance:" in body


# ── integration: run() ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_writes_digest_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    async def _fake_vc(task, service, action, params):
        if action == "retrieve_balance":
            return {"ok": True, "result": {
                "available": [{"amount": 50000, "currency": "usd"}],
                "pending": [{"amount": 0, "currency": "usd"}],
            }}
        if action == "list_subscriptions":
            return {"ok": True, "result": {
                "data": [_sub(1500, "month")]
            }}
        return {"ok": False}

    with patch(
        "mr_roboto.executors.stripe_revenue_digest._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ):
        res = await run({"mission_id": 12})

    assert res["ok"]
    assert res["active_subscriptions"] == 1
    assert res["mrr_cents"] == 1500
    assert (tmp_path / f"mission_12" / ".stripe").is_dir()
    # The digest file is named for the current ISO year-week.
    files = list((tmp_path / "mission_12" / ".stripe").glob("digest_*.md"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "Stripe revenue digest" in content
    assert "Active subscriptions: **1**" in content


@pytest.mark.asyncio
async def test_run_balance_failure_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    async def _fake_vc(task, service, action, params):
        return {"ok": False, "reason": "vendor_error"}

    with patch(
        "mr_roboto.executors.stripe_revenue_digest._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ):
        res = await run({"mission_id": 12})
    assert res["ok"] is False
    assert res["reason"] == "retrieve_balance_failed"


@pytest.mark.asyncio
async def test_run_subscription_failure_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    async def _fake_vc(task, service, action, params):
        if action == "retrieve_balance":
            return {"ok": True, "result": {"available": [], "pending": []}}
        return {"ok": False, "reason": "vendor_error"}

    with patch(
        "mr_roboto.executors.stripe_revenue_digest._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ):
        res = await run({"mission_id": 12})
    assert res["ok"] is False
    assert res["reason"] == "list_subscriptions_failed"


@pytest.mark.asyncio
async def test_run_system_scope_no_mission_id(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    async def _fake_vc(task, service, action, params):
        if action == "retrieve_balance":
            return {"ok": True, "result": {"available": [], "pending": []}}
        return {"ok": True, "result": {"data": []}}

    with patch(
        "mr_roboto.executors.stripe_revenue_digest._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ):
        res = await run({})
    assert res["ok"]
    # System scope writes under mission_0.
    assert (tmp_path / "mission_0" / ".stripe").is_dir()
