"""Rate budget manager to prevent over-requesting.

Tracks per-domain daily request budgets in SQLite and refuses requests
once the budget is exhausted.  Call ``reset_daily_budgets`` at midnight.
"""

from __future__ import annotations

import os
import time

import aiosqlite

from src.app.config import DB_PATH
from src.infra.logging_config import get_logger

logger = get_logger("shopping.resilience.rate_budget")

# ─── Default Daily Budgets ──────────────────────────────────────────────────

DEFAULT_BUDGETS: dict[str, int] = {
    "akakce": 100,
    "trendyol": 50,
    "hepsiburada": 30,
    "amazon_tr": 40,
    "google_cse": 100,
    "forums": 30,
    "eksisozluk": 20,
    "sikayetvar": 30,
    "grocery": 50,
    "sahibinden": 20,
    "home_improvement": 40,
}

# ─── Singleton Connection ───────────────────────────────────────────────────

_budget_db: aiosqlite.Connection | None = None
BUDGET_DB_PATH = os.path.join(os.path.dirname(DB_PATH), "rate_budgets.db")


async def _get_db() -> aiosqlite.Connection:
    """Return the shared rate-budget DB connection."""
    global _budget_db
    if _budget_db is None:
        _budget_db = await aiosqlite.connect(BUDGET_DB_PATH)
        _budget_db.row_factory = aiosqlite.Row
        await _budget_db.execute("PRAGMA journal_mode=WAL")
        await _budget_db.execute("PRAGMA synchronous=NORMAL")
        await _budget_db.execute("PRAGMA busy_timeout=5000")
    return _budget_db


# ─── Schema ─────────────────────────────────────────────────────────────────

async def init_rate_budget_db() -> None:
    """Create the ``rate_budgets`` table and seed default budgets."""
    db = await _get_db()

    await db.execute("""
        CREATE TABLE IF NOT EXISTS rate_budgets (
            domain      TEXT PRIMARY KEY,
            daily_limit INTEGER NOT NULL,
            used_today  INTEGER NOT NULL DEFAULT 0,
            reset_at    REAL    NOT NULL
        )
    """)

    now = time.time()
    for domain, limit in DEFAULT_BUDGETS.items():
        await db.execute(
            """
            INSERT OR IGNORE INTO rate_budgets (domain, daily_limit, used_today, reset_at)
            VALUES (?, ?, 0, ?)
            """,
            (domain, limit, now),
        )

    await db.commit()
    logger.info("Rate budget database initialised with %d domains", len(DEFAULT_BUDGETS))


# ─── Public API ─────────────────────────────────────────────────────────────

async def get_remaining_budget(domain: str) -> int:
    """Return how many requests are left today for *domain*.

    Returns 0 if the domain is unknown or fully consumed.
    """
    db = await _get_db()
    cursor = await db.execute(
        "SELECT daily_limit, used_today FROM rate_budgets WHERE domain = ?",
        (domain,),
    )
    row = await cursor.fetchone()
    if row is None:
        return 0
    return max(0, row["daily_limit"] - row["used_today"])


async def consume_budget(domain: str, count: int = 1) -> None:
    """Deduct *count* requests from *domain*'s daily budget.

    Logs a warning if the budget is fully consumed after the deduction.
    """
    db = await _get_db()
    await db.execute(
        "UPDATE rate_budgets SET used_today = used_today + ? WHERE domain = ?",
        (count, domain),
    )
    await db.commit()

    remaining = await get_remaining_budget(domain)
    if remaining <= 0:
        logger.warning("Daily budget exhausted for %s", domain)
    elif remaining <= 5:
        logger.info("Budget low for %s: %d remaining", domain, remaining)


async def get_budget_summary() -> dict:
    """Return all domains with their remaining and total budgets.

    Returns
    -------
    Dict mapping domain to ``{"remaining": int, "total": int, "used": int}``.
    """
    db = await _get_db()
    cursor = await db.execute("SELECT domain, daily_limit, used_today FROM rate_budgets")
    rows = await cursor.fetchall()
    return {
        row["domain"]: {
            "remaining": max(0, row["daily_limit"] - row["used_today"]),
            "total": row["daily_limit"],
            "used": row["used_today"],
        }
        for row in rows
    }


async def reset_daily_budgets() -> None:
    """Reset all ``used_today`` counters to zero.  Call at midnight."""
    db = await _get_db()
    now = time.time()
    await db.execute("UPDATE rate_budgets SET used_today = 0, reset_at = ?", (now,))
    await db.commit()
    logger.info("Daily rate budgets reset at %.0f", now)
