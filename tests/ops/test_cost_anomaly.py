"""Z8 T5D — cost_anomaly detector + cost_pull executor + alerting rule."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.ops.cost_anomaly import is_anomaly
from mr_roboto.executors.cost_pull import run as cost_run


# ─────────────────────── cost_anomaly tests ──────────────────────────────


@pytest.mark.asyncio
async def test_anomaly_false_when_history_short():
    # 6 days < 7 → False regardless of magnitude.
    assert await is_anomaly("stripe", 10000.0, [1.0] * 6) is False


@pytest.mark.asyncio
async def test_anomaly_false_when_today_in_band():
    history = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 10.8]
    assert await is_anomaly("stripe", 10.5, history) is False


@pytest.mark.asyncio
async def test_anomaly_true_when_today_is_outlier():
    history = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 10.8, 9.8, 10.3, 10.1]
    # ~mean=10, ~stdev=0.6; 100.0 is wildly > 2.5σ.
    assert await is_anomaly("stripe", 100.0, history) is True


@pytest.mark.asyncio
async def test_anomaly_handles_flat_history():
    # stdev=0 path: divide-by-zero protection (effective stdev=0.01 → True).
    history = [5.0] * 8
    assert await is_anomaly("stripe", 10.0, history) is True


@pytest.mark.asyncio
async def test_anomaly_handles_empty_history():
    assert await is_anomaly("stripe", 100.0, []) is False


# ─────────────────────── cost_pull tests ─────────────────────────────────


@pytest.mark.asyncio
async def test_cost_pull_unsupported_vendor():
    res = await cost_run({"payload": {"vendor": "azure"}})
    assert res["ok"] is False
    assert "unsupported vendor" in (res["reason"] or "")


@pytest.mark.asyncio
async def test_cost_pull_stripe_aggregates_cents(monkeypatch):
    async def fake_vc(_task):
        return {"ok": True, "result": {"data": [
            {"amount": 1000},  # $10.00
            {"amount": 5050},  # $50.50
        ]}}

    monkeypatch.setattr(
        "mr_roboto.executors.vendor_call.run", fake_vc,
    )
    res = await cost_run({"payload": {"vendor": "stripe", "params": {}}})
    assert res["ok"] is True
    assert res["today_usd"] == 60.50
    assert res["skipped"] is False


@pytest.mark.asyncio
async def test_cost_pull_vercel_uses_total(monkeypatch):
    async def fake_vc(_task):
        return {"ok": True, "result": {"total": 17.5}}

    monkeypatch.setattr("mr_roboto.executors.vendor_call.run", fake_vc)
    res = await cost_run({"payload": {"vendor": "vercel"}})
    assert res["ok"] is True
    assert res["today_usd"] == 17.5


@pytest.mark.asyncio
async def test_cost_pull_aws_cost_explorer(monkeypatch):
    async def fake_vc(_task):
        return {"ok": True, "result": {"ResultsByTime": [
            {"Total": {"BlendedCost": {"Amount": "12.34"}}},
            {"Total": {"BlendedCost": {"Amount": "5.66"}}},
        ]}}

    monkeypatch.setattr("mr_roboto.executors.vendor_call.run", fake_vc)
    res = await cost_run({"payload": {"vendor": "aws"}})
    assert res["ok"] is True
    assert res["today_usd"] == 18.0


@pytest.mark.asyncio
async def test_cost_pull_marks_skipped_on_adapter_missing(monkeypatch):
    async def fake_vc(_task):
        return {"ok": False, "reason": "adapter missing for vendor"}

    monkeypatch.setattr("mr_roboto.executors.vendor_call.run", fake_vc)
    res = await cost_run({"payload": {"vendor": "aws"}})
    assert res["ok"] is False
    assert res["skipped"] is True
    assert res["today_usd"] is None


# ─────────────────────── alerting rule wired ─────────────────────────────


@pytest.mark.asyncio
async def test_alerting_fires_on_cost_slope(monkeypatch):
    """Rule 4 picks up vendor_cost_history from runtime_state and fires."""
    from src.infra import alerting, runtime_state as rs_mod

    # Stage runtime_state slot the executor would have populated.
    rs_mod.runtime_state["vendor_cost_history"] = {
        "stripe": {
            "today_usd": 200.0,
            "history_14d": [10.0, 9.8, 10.2, 10.1, 9.9, 10.0, 10.3, 9.7],
        },
    }
    rs_mod.runtime_state["telegram_available"] = False  # don't actually send

    captured = []

    async def fake_send(title, message, priority=3):
        captured.append((title, message, priority))

    monkeypatch.setattr(alerting, "_send_alert", fake_send)
    # Reset cooldown so test reliably fires.
    alerting._last_alert.clear()

    await alerting.check_alerts()

    assert any("Vendor Cost Spike" in t for t, _m, _p in captured)


@pytest.mark.asyncio
async def test_alerting_no_fire_when_within_band(monkeypatch):
    from src.infra import alerting, runtime_state as rs_mod
    rs_mod.runtime_state["vendor_cost_history"] = {
        "stripe": {
            "today_usd": 10.5,
            "history_14d": [10.0, 9.8, 10.2, 10.1, 9.9, 10.0, 10.3, 9.7],
        },
    }
    rs_mod.runtime_state["telegram_available"] = False

    captured = []

    async def fake_send(title, message, priority=3):
        captured.append((title, message, priority))

    monkeypatch.setattr(alerting, "_send_alert", fake_send)
    alerting._last_alert.clear()

    await alerting.check_alerts()

    assert not any("Vendor Cost Spike" in t for t, _m, _p in captured)
