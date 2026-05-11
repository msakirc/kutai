"""Z6 T5E — tests for mr_roboto.executors.tax_export_ledger."""
from __future__ import annotations

import csv
import datetime
import io
from unittest.mock import AsyncMock, patch

import pytest

from mr_roboto.executors.tax_export_ledger import (
    _previous_month,
    _render_csv,
    _row_for,
    run,
)


# ── unit: previous_month rollover ─────────────────────────────────────────


def test_previous_month_mid_year():
    ym, start, end = _previous_month(datetime.date(2026, 5, 11))
    assert ym == "2026-04"
    assert end > start


def test_previous_month_january_rolls_year():
    ym, _, _ = _previous_month(datetime.date(2026, 1, 15))
    assert ym == "2025-12"


def test_previous_month_december():
    ym, _, _ = _previous_month(datetime.date(2026, 12, 1))
    assert ym == "2026-11"


# ── unit: row + csv shape ──────────────────────────────────────────────────


def test_row_for_minimal_transaction():
    row = _row_for(
        {
            "id": "tx_1",
            "type": "transaction",
            "amount": 1000,
            "currency": "usd",
            "tax_amount_exclusive": 85,
            "country": "US",
            "created": 1700000000,
            "line_items": {"data": [{"id": "li_1"}, {"id": "li_2"}]},
        }
    )
    assert row["id"] == "tx_1"
    assert row["amount"] == 1000
    assert row["currency"] == "USD"
    assert row["tax_amount"] == 85
    assert row["line_item_count"] == 2


def test_row_for_line_items_as_list():
    row = _row_for({"id": "tx_2", "line_items": [{"x": 1}, {"x": 2}]})
    assert row["line_item_count"] == 2


def test_row_for_no_line_items():
    row = _row_for({"id": "tx_3"})
    assert row["line_item_count"] == 0


def test_render_csv_includes_header_only_when_empty():
    out = _render_csv([])
    reader = csv.reader(io.StringIO(out))
    rows = list(reader)
    assert len(rows) == 1  # header only
    assert rows[0][0] == "id"


def test_render_csv_writes_rows():
    out = _render_csv([
        {"id": "tx_1", "amount": 100, "currency": "usd"},
        {"id": "tx_2", "amount": 200, "currency": "eur"},
    ])
    reader = csv.DictReader(io.StringIO(out))
    rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["id"] == "tx_1"
    assert rows[1]["currency"] == "EUR"


# ── integration: run() ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_writes_csv_and_emits_ack(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    async def _fake_vc(task, service, action, params):
        return {
            "ok": True,
            "result": {
                "data": [
                    {
                        "id": "tx_1",
                        "type": "transaction",
                        "amount": 1000,
                        "currency": "usd",
                        "tax_amount_exclusive": 85,
                        "country": "US",
                        "created": 1700000000,
                        "line_items": {"data": [{"id": "li_1"}]},
                    }
                ]
            },
        }

    ack_calls: list = []

    async def _fake_ack(mission_id, ym, path, empty):
        ack_calls.append((mission_id, ym, path, empty))

    with patch(
        "mr_roboto.executors.tax_export_ledger._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ), patch(
        "mr_roboto.executors.tax_export_ledger._emit_ack",
        new=AsyncMock(side_effect=_fake_ack),
    ):
        res = await run({"mission_id": 9})

    assert res["ok"]
    assert res["transactions"] == 1
    csv_path = res["ledger_path"]
    assert (tmp_path / "mission_9" / ".tax").is_dir()
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["id"] == "tx_1"
    assert ack_calls and ack_calls[0][3] is False  # empty=False


@pytest.mark.asyncio
async def test_run_empty_month_writes_header_and_flags_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    async def _fake_vc(task, service, action, params):
        return {"ok": True, "result": {"data": []}}

    ack_calls: list = []

    async def _fake_ack(mission_id, ym, path, empty):
        ack_calls.append(empty)

    with patch(
        "mr_roboto.executors.tax_export_ledger._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ), patch(
        "mr_roboto.executors.tax_export_ledger._emit_ack",
        new=AsyncMock(side_effect=_fake_ack),
    ):
        res = await run({"mission_id": 9})

    assert res["ok"]
    assert res["transactions"] == 0
    assert ack_calls == [True]
    with open(res["ledger_path"], encoding="utf-8") as fh:
        text = fh.read()
    # Header only.
    assert text.strip().startswith("id,type,amount")
    assert len(text.strip().splitlines()) == 1


@pytest.mark.asyncio
async def test_run_vendor_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    async def _fake_vc(task, service, action, params):
        return {"ok": False, "reason": "vendor_error", "error": "401"}

    with patch(
        "mr_roboto.executors.tax_export_ledger._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ):
        res = await run({"mission_id": 9})

    assert res["ok"] is False
    assert res["reason"] == "list_tax_transactions_failed"


@pytest.mark.asyncio
async def test_run_system_scope_when_no_mission_id(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    async def _fake_vc(task, service, action, params):
        return {"ok": True, "result": {"data": []}}

    async def _fake_ack(*_a, **_kw):
        pass

    with patch(
        "mr_roboto.executors.tax_export_ledger._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ), patch(
        "mr_roboto.executors.tax_export_ledger._emit_ack",
        new=AsyncMock(side_effect=_fake_ack),
    ):
        res = await run({})
    assert res["ok"]
    assert (tmp_path / "mission_0" / ".tax").is_dir()
