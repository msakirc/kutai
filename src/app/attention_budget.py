"""Z7 T1D (B5) — Founder Attention Budget.

Soft-warn policy: ALL cards always surface. Over-budget p1/p2/p3 cards are
flagged ``below_fold=True`` and pushed below the fold in the UI. p0_blocking
is NEVER below_fold regardless of budget state.

Public API
----------
check_budget(product_id, day)
    Returns remaining minutes + top-priority queue summary for ``day``.
should_surface_now(card)
    Always returns True (all cards surface). Returns a dict with a
    ``below_fold`` flag that callers can use to partition the display.
get_queue(product_id)
    Returns structured queue data consumed by the A0 briefing renderer.
next_review_window(card)
    Returns the next-morning datetime for a deferred card.
record_surfaced(card_id, product_id)
    Write a founder_attention_log row when a card is surfaced.
record_acted(card_id, product_id, attention_minutes)
    Write a founder_attention_log row when the founder acts on a card.
record_deferred(card_id, product_id, deferred_to)
    Write a founder_attention_log row when the founder defers a card.
"""
from __future__ import annotations

import datetime
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("app.attention_budget")

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_DAILY_MINUTES = 60
_PRIORITY_ORDER = ["p0_blocking", "p1_today", "p2_this_week", "p3_when_idle"]
_PRIORITY_LABELS = {
    "p0_blocking": "blocking",
    "p1_today": "today",
    "p2_this_week": "this week",
    "p3_when_idle": "when idle",
}


def _daily_cap() -> int:
    """Return the daily attention cap from env or default."""
    try:
        return int(os.environ.get("FOUNDER_ATTENTION_DAILY_MINUTES", _DEFAULT_DAILY_MINUTES))
    except (ValueError, TypeError):
        return _DEFAULT_DAILY_MINUTES


def _now_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _today_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


# ── Core query helpers ─────────────────────────────────────────────────────────


async def _fetch_cards(product_id: int | None) -> list[dict]:
    """Return all pending/in_progress founder_actions with priority metadata."""
    from src.infra.db import get_db
    db = await get_db()
    where_clause = ""
    params: tuple = ()
    if product_id is not None:
        # product_id maps to mission_id in this schema
        where_clause = "AND mission_id = ?"
        params = (product_id,)
    cur = await db.execute(
        "SELECT id, mission_id, kind, title, why, priority, "
        "       defer_until, expires_at, status, created_at, urgent "
        f"FROM founder_actions "
        f"WHERE status IN ('pending', 'in_progress') {where_clause} "
        f"ORDER BY id ASC",
        params,
    )
    rows = await cur.fetchall()
    cols = [d[0] for d in cur.description]  # read before close — aiosqlite nulls description after close
    await cur.close()
    return [dict(zip(cols, r)) for r in rows]


async def _spent_today(product_id: int | None) -> int:
    """Return minutes spent today from founder_attention_log."""
    from src.infra.db import get_db
    db = await get_db()
    today = _today_str()
    where_clause = ""
    params: tuple = (today,)
    if product_id is not None:
        where_clause = "AND mission_id = ?"
        params = (today, product_id)
    cur = await db.execute(
        "SELECT COALESCE(SUM(attention_minutes), 0) "
        "FROM founder_attention_log "
        f"WHERE DATE(ts) = ? {where_clause}",
        params,
    )
    row = await cur.fetchone()
    await cur.close()
    return int(row[0]) if row and row[0] is not None else 0


# ── Public API ─────────────────────────────────────────────────────────────────


async def check_budget(
    product_id: int | None,
    day: str | None = None,
) -> dict[str, Any]:
    """Return remaining-minutes + top-priority queue for ``product_id`` on ``day``.

    ``day`` is a date string (YYYY-MM-DD). Defaults to today (UTC).

    Return shape::

        {
            "cap": int,           # daily cap in minutes
            "spent": int,         # minutes spent today
            "remaining": int,     # cap - spent (may be negative)
            "over_budget": bool,
            "top_queue": [        # first 5 cards by priority
                {"id": int, "priority": str, "title": str, "below_fold": bool}
            ]
        }
    """
    cap = _daily_cap()
    spent = await _spent_today(product_id)
    remaining = cap - spent
    cards = await _fetch_cards(product_id)

    # Classify below_fold
    surfaced = should_surface_now_batch(cards, spent, cap)

    top_queue = [
        {
            "id": c["id"],
            "priority": c.get("priority") or "p2_this_week",
            "title": c["title"],
            "below_fold": s["below_fold"],
        }
        for c, s in zip(cards, surfaced)
    ][:5]

    return {
        "cap": cap,
        "spent": spent,
        "remaining": remaining,
        "over_budget": remaining < 0,
        "top_queue": top_queue,
    }


def should_surface_now(card: dict) -> dict[str, Any]:
    """Determine surface + below_fold for a single card.

    Soft-warn policy:
    - ALL cards always surface (``surface=True``).
    - p0_blocking is NEVER below_fold.
    - p1/p2/p3 are below_fold only when the daily budget is exceeded.

    Returns::

        {"surface": True, "below_fold": bool}

    Callers that need per-card budget awareness should use
    ``should_surface_now_batch`` to get the budget context.
    """
    priority = card.get("priority") or "p2_this_week"
    # p0_blocking is always above fold
    if priority == "p0_blocking":
        return {"surface": True, "below_fold": False}
    # Default: surface but potentially below_fold — resolved by caller with budget context
    return {"surface": True, "below_fold": False}


def should_surface_now_batch(
    cards: list[dict],
    spent: int,
    cap: int,
) -> list[dict[str, Any]]:
    """Batch version of should_surface_now with budget context.

    When spent > cap, p1/p2/p3 cards are marked below_fold=True.
    p0_blocking always below_fold=False.
    """
    over_budget = spent > cap
    results = []
    for card in cards:
        priority = card.get("priority") or "p2_this_week"
        if priority == "p0_blocking":
            results.append({"surface": True, "below_fold": False})
        elif over_budget:
            results.append({"surface": True, "below_fold": True})
        else:
            results.append({"surface": True, "below_fold": False})
    return results


async def get_queue(product_id: int | None) -> dict[str, Any]:
    """Return structured queue data for the A0 briefing renderer.

    Return shape::

        {
            "cap": int,
            "spent": int,
            "remaining": int,
            "over_budget": bool,
            "today": [          # p0_blocking + p1_today (not deferred past now)
                {
                    "id": int,
                    "priority": str,
                    "title": str,
                    "why": str,
                    "kind": str,
                    "mission_id": int,
                    "below_fold": bool,
                    "urgent": bool,
                }
            ],
            "this_week": [...],  # p2_this_week (not deferred past end-of-week)
            "deferred": [...],   # cards with defer_until in the future
            "when_idle": [...],  # p3_when_idle cards
        }
    """
    cap = _daily_cap()
    spent = await _spent_today(product_id)
    remaining = cap - spent
    over_budget = remaining < 0
    now = datetime.datetime.utcnow()

    cards = await _fetch_cards(product_id)
    surfaced = should_surface_now_batch(cards, spent, cap)

    today_cards: list[dict] = []
    this_week_cards: list[dict] = []
    deferred_cards: list[dict] = []
    when_idle_cards: list[dict] = []

    for card, surf in zip(cards, surfaced):
        priority = card.get("priority") or "p2_this_week"
        defer_until = card.get("defer_until")

        # Check if card is deferred (defer_until in the future)
        is_deferred = False
        if defer_until:
            try:
                dt = datetime.datetime.strptime(defer_until, "%Y-%m-%d %H:%M:%S")
                if dt > now:
                    is_deferred = True
            except (ValueError, TypeError):
                pass

        entry = {
            "id": card["id"],
            "priority": priority,
            "title": card["title"],
            "why": card.get("why") or "",
            "kind": card.get("kind") or "",
            "mission_id": card.get("mission_id"),
            "below_fold": surf["below_fold"],
            "urgent": bool(card.get("urgent")),
            "defer_until": defer_until,
            "expires_at": card.get("expires_at"),
        }

        if is_deferred:
            deferred_cards.append(entry)
        elif priority == "p3_when_idle":
            when_idle_cards.append(entry)
        elif priority in ("p0_blocking", "p1_today"):
            today_cards.append(entry)
        else:  # p2_this_week default
            this_week_cards.append(entry)

    # Sort each bucket: p0 first, then by id (creation order)
    def _sort_key(c: dict) -> tuple:
        return (_PRIORITY_ORDER.index(c["priority"]) if c["priority"] in _PRIORITY_ORDER else 99, c["id"])

    today_cards.sort(key=_sort_key)
    this_week_cards.sort(key=_sort_key)
    when_idle_cards.sort(key=_sort_key)

    return {
        "cap": cap,
        "spent": spent,
        "remaining": remaining,
        "over_budget": over_budget,
        "today": today_cards,
        "this_week": this_week_cards,
        "deferred": deferred_cards,
        "when_idle": when_idle_cards,
    }


def next_review_window(card: dict) -> datetime.datetime:
    """Return the next-morning 09:00 UTC datetime for a deferred card.

    Used to schedule deferred cards for the next morning briefing.
    """
    now = datetime.datetime.utcnow()
    tomorrow = now.date() + datetime.timedelta(days=1)
    return datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, 9, 0, 0)


# ── Attention log helpers ──────────────────────────────────────────────────────


async def record_surfaced(card_id: int, product_id: int) -> None:
    """Record that a card was surfaced to the founder."""
    from src.infra.db import get_db
    db = await get_db()
    now = _now_str()
    await db.execute(
        "INSERT INTO founder_attention_log "
        "(mission_id, step_id, action, minutes_debited, ts, "
        " card_id, surfaced_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (product_id, "", "surfaced", 0, now, card_id, now),
    )
    await db.commit()
    logger.debug("attention_budget: surfaced card %d for product %d", card_id, product_id)


async def record_acted(card_id: int, product_id: int, attention_minutes: int) -> None:
    """Record that the founder acted on a card, consuming attention_minutes."""
    from src.infra.db import get_db
    db = await get_db()
    now = _now_str()
    await db.execute(
        "INSERT INTO founder_attention_log "
        "(mission_id, step_id, action, minutes_debited, ts, "
        " card_id, acted_at, attention_minutes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (product_id, "", "acted", attention_minutes, now, card_id, now, attention_minutes),
    )
    await db.commit()
    logger.debug(
        "attention_budget: acted on card %d for product %d (%d min)",
        card_id, product_id, attention_minutes,
    )


async def record_deferred(card_id: int, product_id: int, deferred_to: str) -> None:
    """Record that the founder deferred a card to ``deferred_to`` timestamp.

    Also updates founder_actions.defer_until to hide the card until then.
    """
    from src.infra.db import get_db
    from src.founder_actions import defer as _fa_defer
    db = await get_db()
    now = _now_str()
    await db.execute(
        "INSERT INTO founder_attention_log "
        "(mission_id, step_id, action, minutes_debited, ts, "
        " card_id, deferred_to) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (product_id, "", "deferred", 0, now, card_id, deferred_to),
    )
    await db.commit()
    await _fa_defer(card_id, deferred_to)
    logger.info(
        "attention_budget: deferred card %d (product %d) to %s",
        card_id, product_id, deferred_to,
    )


async def set_daily_budget(product_id: int, minutes: int) -> None:
    """Set the daily attention budget for a product (stored on missions row)."""
    from src.infra.db import get_db
    db = await get_db()
    await db.execute(
        "UPDATE missions SET founder_attention_budget_minutes = ? WHERE id = ?",
        (minutes, product_id),
    )
    await db.commit()
    logger.info("attention_budget: set budget for product %d = %d min", product_id, minutes)


__all__ = [
    "check_budget",
    "should_surface_now",
    "should_surface_now_batch",
    "get_queue",
    "next_review_window",
    "record_surfaced",
    "record_acted",
    "record_deferred",
    "set_daily_budget",
]
