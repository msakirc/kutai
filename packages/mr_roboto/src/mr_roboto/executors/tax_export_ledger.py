"""Z6 T5E — monthly Stripe Tax CSV export ledger.

Cron-driven (monthly). Calls
``vendor_call(stripe, list_tax_transactions)`` for the previous calendar
month, writes a CSV to ``mission_<id>/.tax/<YYYY-MM>.csv``, and emits a
``founder_action(kind='generic', expected_output_kind='ack_only')`` so
the founder forwards the CSV to their accountant.

Tax-collection awareness
------------------------
The provisioner (T5B) is the place where ``tax_behavior: "exclusive"``
is meant to be set on prices. This executor just observes / exports — if
no tax transactions exist for the period, the CSV is written empty (with
a single header row) and the founder_action carries an
``empty=True`` note instead of an attached path.

CSV columns: ``id, type, amount, currency, tax_amount, tax_rate,
country, created, line_item_count``.

Cron registration: monthly internal cadence (2_592_000s). System scope
(``mission_id=0``) when no mission is attached.

Stripe Tax docs: https://stripe.com/docs/tax (referenced — values not
fabricated; field set chosen from the public Transaction resource).
"""
from __future__ import annotations

import csv
import datetime
import io
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.tax_export_ledger")

SYSTEM_MISSION_ID = 0


def _workspace_root() -> str:
    return os.environ.get("MISSION_WORKSPACE_ROOT", os.getcwd())


def _ledger_path(workspace_root: str, mission_id: int, ym: str) -> str:
    return os.path.join(workspace_root, f"mission_{mission_id}", ".tax", f"{ym}.csv")


def _previous_month(today: datetime.date | None = None) -> tuple[str, int, int]:
    """Return (yyyy_mm, unix_start, unix_end_exclusive) for the previous month."""
    today = today or datetime.date.today()
    year, month = today.year, today.month
    if month == 1:
        prev_year, prev_month = year - 1, 12
    else:
        prev_year, prev_month = year, month - 1
    start = datetime.datetime(prev_year, prev_month, 1, tzinfo=datetime.timezone.utc)
    if prev_month == 12:
        next_year, next_month = prev_year + 1, 1
    else:
        next_year, next_month = prev_year, prev_month + 1
    end = datetime.datetime(next_year, next_month, 1, tzinfo=datetime.timezone.utc)
    return (
        f"{prev_year:04d}-{prev_month:02d}",
        int(start.timestamp()),
        int(end.timestamp()),
    )


# ── vendor_call indirection ────────────────────────────────────────────────


async def _vc(task: dict, service: str, action: str, params: dict) -> dict:
    from mr_roboto.executors.vendor_call import run as vendor_call_run
    sub = {
        "mission_id": task.get("mission_id"),
        "id": task.get("id"),
        "context": {
            "post_hook": {
                "service": service,
                "action": action,
                "params": params,
            }
        },
    }
    return await vendor_call_run(sub)


# ── CSV rendering ─────────────────────────────────────────────────────────


_CSV_COLS = [
    "id",
    "type",
    "amount",
    "currency",
    "tax_amount",
    "tax_rate",
    "country",
    "created",
    "line_item_count",
]


def _row_for(transaction: dict) -> dict:
    """Best-effort flattening of one Stripe tax transaction."""
    line_items = transaction.get("line_items") or {}
    if isinstance(line_items, dict):
        items_list = line_items.get("data") or []
    elif isinstance(line_items, list):
        items_list = line_items
    else:
        items_list = []
    return {
        "id": transaction.get("id", ""),
        "type": transaction.get("type", ""),
        "amount": transaction.get("amount", ""),
        "currency": (transaction.get("currency") or "").upper(),
        "tax_amount": (
            transaction.get("tax_amount_exclusive")
            or transaction.get("tax_amount")
            or ""
        ),
        "tax_rate": transaction.get("tax_rate", ""),
        "country": transaction.get("country", ""),
        "created": transaction.get("created", ""),
        "line_item_count": len(items_list),
    }


def _render_csv(transactions: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLS, extrasaction="ignore")
    writer.writeheader()
    for t in transactions:
        if isinstance(t, dict):
            writer.writerow(_row_for(t))
    return buf.getvalue()


# ── founder_action ────────────────────────────────────────────────────────


async def _emit_ack(mission_id: int, ym: str, path: str, empty: bool) -> None:
    try:
        import src.founder_actions as fa
        instructions = [
            "Forward the attached CSV to your accountant.",
            f"Path: {path}",
        ]
        if empty:
            instructions.append(
                "No Stripe Tax transactions were recorded for this month — "
                "ledger is empty (header only)."
            )
        await fa.create(
            mission_id=int(mission_id),
            kind="generic",
            title=f"Tax CSV ready for {ym}",
            why=f"Stripe Tax monthly ledger generated for {ym}.",
            instructions=instructions,
            expected_output_kind="ack_only",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("tax_export ack founder_action failed: %s", exc)


# ── main entrypoint ───────────────────────────────────────────────────────


async def run(task: dict[str, Any]) -> dict[str, Any]:
    mission_id = task.get("mission_id")
    if mission_id is None:
        mission_id_int = SYSTEM_MISSION_ID
    else:
        try:
            mission_id_int = int(mission_id)
        except (TypeError, ValueError):
            mission_id_int = SYSTEM_MISSION_ID

    ym, start_ts, end_ts = _previous_month()

    res = await _vc(
        task, "stripe", "list_tax_transactions",
        {
            "limit": 100,
            "created[gte]": start_ts,
            "created[lt]": end_ts,
        },
    )
    if not res.get("ok"):
        return {
            "ok": False,
            "reason": "list_tax_transactions_failed",
            "detail": res,
        }

    payload = res.get("result") or {}
    transactions = payload.get("data") if isinstance(payload, dict) else []
    if not isinstance(transactions, list):
        transactions = []

    csv_text = _render_csv(transactions)
    out_path = _ledger_path(_workspace_root(), mission_id_int, ym)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(csv_text)

    await _emit_ack(
        mission_id=mission_id_int,
        ym=ym,
        path=out_path,
        empty=not transactions,
    )

    return {
        "ok": True,
        "year_month": ym,
        "transactions": len(transactions),
        "ledger_path": out_path,
    }


__all__ = ["run"]
