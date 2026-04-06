"""Unified time utilities for KutAI.

ALL datetime operations in the project should use these helpers instead of
calling ``datetime.now()``, ``datetime.utcnow()``, or inline timezone
conversions.

Rules
-----
* Internal timestamps → ``utc_now()``  (timezone-aware UTC)
* DB storage         → ``db_now()`` or ``to_db(dt)``  (naive UTC string ``%Y-%m-%d %H:%M:%S``)
* Parsing DB values  → ``from_db(s)``  (returns aware UTC datetime)
* User display (TR)  → ``to_turkey(dt)``
* ISO strings        → ``utc_now().isoformat()``
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone  # noqa: F401 — re-exported

try:
    from zoneinfo import ZoneInfo

    TZ_TR = ZoneInfo("Europe/Istanbul")
except Exception:  # pragma: no cover — fallback for minimal envs
    TZ_TR = timezone(timedelta(hours=3))  # Turkey is UTC+3, no DST

DB_FMT = "%Y-%m-%d %H:%M:%S"
"""SQLite-compatible datetime format.  Must match what ``datetime('now')``
returns so that ``<=`` comparisons work in SQL."""

_TR_UTC_OFFSET = 3  # Turkey is always UTC+3, no DST


# ── Core helpers ─────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def db_now() -> str:
    """Return current UTC time as a SQLite-compatible string."""
    return utc_now().strftime(DB_FMT)


def to_db(dt: datetime) -> str:
    """Convert *any* datetime to a SQLite-compatible UTC string.

    Handles both naive (assumed UTC) and aware datetimes.
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime(DB_FMT)


def from_db(value: str) -> datetime:
    """Parse a DB datetime string into a timezone-aware UTC datetime.

    Handles both ``2026-04-06 12:00:00`` and ``2026-04-06T12:00:00+00:00``.
    """
    if value is None:
        return None
    # Strip timezone suffix if present (e.g. +00:00) and normalize separator
    clean = value[:19].replace("T", " ")
    dt = datetime.strptime(clean, DB_FMT)
    return dt.replace(tzinfo=timezone.utc)


# ── Turkey time ──────────────────────────────────────────────────────────────

def turkey_now() -> datetime:
    """Return current time in Turkey (Europe/Istanbul) timezone."""
    return utc_now().astimezone(TZ_TR)


def to_turkey(dt: datetime) -> datetime:
    """Convert any datetime to Turkey local time.

    Naive datetimes are assumed UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TZ_TR)


def turkey_to_utc_naive(dt: datetime) -> str:
    """Convert a Turkey-local datetime to a naive UTC DB string.

    Used by Telegram input parsing: user gives times in Turkey local,
    we store as UTC in the database.
    """
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).strftime(DB_FMT)
    # Naive — assume Turkey local
    dt = dt.replace(tzinfo=TZ_TR)
    return dt.astimezone(timezone.utc).strftime(DB_FMT)


def tr_hour_to_utc(hour: int) -> int:
    """Convert a Turkey local hour (0-23) to UTC hour (0-23)."""
    return (hour - _TR_UTC_OFFSET) % 24
