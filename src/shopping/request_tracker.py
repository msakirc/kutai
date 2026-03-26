"""Request tracking for anti-detection monitoring and rate budget management."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from src.shopping.cache import get_cache_db
from src.shopping.config import get_rate_limit
from src.infra.logging_config import get_logger

logger = get_logger("shopping.request_tracker")


# ─── Schema ──────────────────────────────────────────────────────────────────

async def init_request_db() -> None:
    """Create request_log and domain_health tables in the shopping cache DB."""
    db = await get_cache_db()

    await db.execute("""
        CREATE TABLE IF NOT EXISTS request_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            domain        TEXT    NOT NULL,
            url           TEXT    NOT NULL,
            status_code   INTEGER,
            response_time_ms INTEGER,
            cache_hit     INTEGER NOT NULL DEFAULT 0,
            scraper_used  TEXT,
            session_id    TEXT,
            created_at    REAL    NOT NULL
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_request_log_domain_created
        ON request_log (domain, created_at)
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS domain_health (
            domain          TEXT PRIMARY KEY,
            success_count   INTEGER NOT NULL DEFAULT 0,
            failure_count   INTEGER NOT NULL DEFAULT 0,
            last_success    REAL,
            last_failure    REAL,
            current_status  TEXT NOT NULL DEFAULT 'unknown'
        )
    """)

    await db.commit()
    logger.info("Request tracking tables initialised")


# ─── Request Logging ─────────────────────────────────────────────────────────

async def log_request(
    domain: str,
    url: str,
    status_code: int | None,
    response_time_ms: int | None,
    cache_hit: bool,
    scraper_used: str | None = None,
    session_id: str | None = None,
) -> None:
    """Record a single request to the log and update domain health."""
    db = await get_cache_db()
    now = time.time()

    await db.execute(
        """
        INSERT INTO request_log
            (domain, url, status_code, response_time_ms, cache_hit, scraper_used, session_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (domain, url, status_code, response_time_ms, int(cache_hit), scraper_used, session_id, now),
    )
    await db.commit()

    # Update domain health based on status code
    if status_code is not None:
        success = 200 <= status_code < 400
        await update_domain_health(domain, success)


# ─── Domain Health ───────────────────────────────────────────────────────────

async def get_domain_health(domain: str) -> dict:
    """Return health stats for a domain over the last 24 hours.

    Returns dict with: success_count_24h, failure_count_24h,
    last_success, last_failure, current_status.
    """
    db = await get_cache_db()
    cutoff = time.time() - 86400  # 24 hours ago

    # Count successes (2xx/3xx) in last 24h
    cursor = await db.execute(
        """
        SELECT COUNT(*) as cnt FROM request_log
        WHERE domain = ? AND created_at >= ? AND status_code IS NOT NULL
              AND status_code >= 200 AND status_code < 400
        """,
        (domain, cutoff),
    )
    row = await cursor.fetchone()
    success_count_24h = row[0] if row else 0

    # Count failures in last 24h
    cursor = await db.execute(
        """
        SELECT COUNT(*) as cnt FROM request_log
        WHERE domain = ? AND created_at >= ? AND (
            status_code IS NULL OR status_code < 200 OR status_code >= 400
        )
        """,
        (domain, cutoff),
    )
    row = await cursor.fetchone()
    failure_count_24h = row[0] if row else 0

    # Get stored health record
    cursor = await db.execute(
        "SELECT last_success, last_failure, current_status FROM domain_health WHERE domain = ?",
        (domain,),
    )
    health_row = await cursor.fetchone()

    if health_row:
        last_success = health_row[0]
        last_failure = health_row[1]
        current_status = health_row[2]
    else:
        last_success = None
        last_failure = None
        current_status = "unknown"

    return {
        "success_count_24h": success_count_24h,
        "failure_count_24h": failure_count_24h,
        "last_success": last_success,
        "last_failure": last_failure,
        "current_status": current_status,
    }


async def update_domain_health(domain: str, success: bool) -> None:
    """Update the domain_health table after a request."""
    db = await get_cache_db()
    now = time.time()

    if success:
        await db.execute(
            """
            INSERT INTO domain_health (domain, success_count, last_success, current_status)
            VALUES (?, 1, ?, 'healthy')
            ON CONFLICT(domain) DO UPDATE SET
                success_count = success_count + 1,
                last_success = excluded.last_success,
                current_status = 'healthy'
            """,
            (domain, now),
        )
    else:
        await db.execute(
            """
            INSERT INTO domain_health (domain, failure_count, last_failure, current_status)
            VALUES (?, 1, ?, 'degraded')
            ON CONFLICT(domain) DO UPDATE SET
                failure_count = failure_count + 1,
                last_failure = excluded.last_failure,
                current_status = CASE
                    WHEN domain_health.failure_count + 1 >= 5 THEN 'down'
                    ELSE 'degraded'
                END
            """,
            (domain, now),
        )

    await db.commit()


# ─── Rate Budget ─────────────────────────────────────────────────────────────

async def get_daily_request_count(domain: str) -> int:
    """Return the number of non-cache-hit requests to a domain today (last 24h)."""
    db = await get_cache_db()
    cutoff = time.time() - 86400

    cursor = await db.execute(
        """
        SELECT COUNT(*) FROM request_log
        WHERE domain = ? AND created_at >= ? AND cache_hit = 0
        """,
        (domain, cutoff),
    )
    row = await cursor.fetchone()
    return row[0] if row else 0


async def get_rate_limit_info(domain: str) -> dict:
    """Return rate budget info: used, limit, remaining."""
    used = await get_daily_request_count(domain)
    limit_cfg = get_rate_limit(domain)
    daily_limit = limit_cfg["daily_budget"]

    return {
        "used": used,
        "limit": daily_limit,
        "remaining": max(0, daily_limit - used),
    }
