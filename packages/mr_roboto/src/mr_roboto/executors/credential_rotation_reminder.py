"""Z6 T7A — weekly credential rotation reminder.

Queries ``credentials`` for rows that either:

* Have an ``expires_at`` within 14 days, OR
* Have ``rotated_at IS NULL`` AND ``created_at < now - 90 days``.

For each match emits one ``founder_action(kind='credential_paste')`` per
service. Idempotent: skips a service if a pending/in_progress
``credential_paste`` already exists with the same deterministic title.

Cron registration: weekly via ``general_beckman.cron_seed`` as the
``credential_rotation_reminder`` internal cadence.

Mission scoping: rotation is a system-wide concern. We use
``mission_id=0`` (sentinel) — same convention as
:mod:`mr_roboto.executors.compliance_template_staleness` (T4D).
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.credential_rotation_reminder")

# Sentinel for system-wide actions (no specific mission).
SYSTEM_MISSION_ID = 0

# Lookahead window for upcoming expiry.
EXPIRY_LOOKAHEAD_DAYS = 14
# Hard rotation cadence used when expires_at is absent.
DEFAULT_ROTATION_DAYS = 90

# Filesystem root for per-vendor credential schemas — used to derive the
# vendor's docs_url for the founder_action instructions.
_SCHEMA_ROOT = "credential_schemas"


def _docs_url_for(service: str) -> str | None:
    """Best-effort: read the vendor schema and return its ``docs_url``.

    Returns ``None`` if no schema file exists or the field is absent —
    the founder_action falls back to a generic "Generate new credential"
    instruction in that case.
    """
    candidates = [
        os.path.join(_SCHEMA_ROOT, f"{service}.json"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "credential_rotation_reminder: schema read failed %s: %s",
                path, e,
            )
            continue
        url = data.get("docs_url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


def _build_title(service: str) -> str:
    return f"Rotate {service} credential"


def _why_text(
    service: str,
    expires_at: str | None,
    rotated_at: str | None,
    created_at: str | None,
    days_to_expiry: int | None,
    days_since_seen: int | None,
) -> str:
    parts: list[str] = []
    if days_to_expiry is not None and days_to_expiry <= EXPIRY_LOOKAHEAD_DAYS:
        if days_to_expiry < 0:
            parts.append(
                f"Token EXPIRED {-days_to_expiry}d ago "
                f"(expires_at={expires_at})."
            )
        else:
            parts.append(
                f"Token expires in {days_to_expiry}d "
                f"(expires_at={expires_at})."
            )
    if (rotated_at is None
            and days_since_seen is not None
            and days_since_seen > DEFAULT_ROTATION_DAYS):
        parts.append(
            f"Credential has not been rotated in {days_since_seen}d "
            f"(created_at={created_at})."
        )
    if not parts:
        parts.append(
            f"{service} credential is due for rotation."
        )
    return " ".join(parts)


async def _existing_titles_for_credential_paste() -> set[str]:
    """Return titles of pending/in_progress credential_paste actions
    so we can de-duplicate per service."""
    try:
        from src.infra.db import get_db
    except ImportError:
        return set()
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT title FROM founder_actions "
            "WHERE kind = 'credential_paste' "
            "AND status IN ('pending', 'in_progress')"
        )
        rows = await cur.fetchall()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "credential_rotation_reminder: existing-titles query failed: %s",
            e,
        )
        return set()
    return {str(r[0]) for r in rows if r and r[0]}


async def credential_rotation_reminder(
    *,
    expiry_lookahead_days: int = EXPIRY_LOOKAHEAD_DAYS,
    rotation_days: int = DEFAULT_ROTATION_DAYS,
) -> dict[str, Any]:
    """Emit a founder_action per credential due for rotation.

    Returns ``{ok, scanned, due, emitted, skipped_duplicate}``.
    """
    try:
        from src.infra.db import get_db
    except ImportError:
        return {
            "ok": False,
            "scanned": 0,
            "due": 0,
            "emitted": [],
            "skipped_duplicate": 0,
            "error": "db unavailable",
        }
    db = await get_db()
    # Pull every row — the predicate set is small (single-digit count of
    # service rows in practice) and we want to compute days-since locally
    # so missing/empty values don't trip SQLite datetime arithmetic.
    try:
        cur = await db.execute(
            "SELECT service_name, expires_at, rotated_at, created_at "
            "FROM credentials"
        )
        rows = await cur.fetchall()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "credential_rotation_reminder: credentials query failed: %s", e,
        )
        return {
            "ok": False,
            "scanned": 0,
            "due": 0,
            "emitted": [],
            "skipped_duplicate": 0,
            "error": str(e),
        }
    if not rows:
        return {
            "ok": True,
            "scanned": 0,
            "due": 0,
            "emitted": [],
            "skipped_duplicate": 0,
        }

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    def _parse_dt(raw: str | None) -> datetime | None:
        if not raw:
            return None
        s = str(raw).strip()
        if not s:
            return None
        # Tolerate both ISO 8601 and "YYYY-MM-DD HH:MM:SS" sqlite format.
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
                    "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None

    due_entries: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, tuple):
            service, expires_at, rotated_at, created_at = row
        else:
            service = row["service_name"]
            expires_at = row["expires_at"]
            rotated_at = row["rotated_at"]
            created_at = row["created_at"]
        if not service:
            continue

        exp_dt = _parse_dt(expires_at)
        days_to_expiry: int | None = None
        if exp_dt is not None:
            days_to_expiry = int((exp_dt - now).total_seconds() // 86400)

        created_dt = _parse_dt(created_at)
        days_since_created: int | None = None
        if created_dt is not None:
            days_since_created = int((now - created_dt).total_seconds() // 86400)

        # Predicates: expiring soon OR never-rotated-and-old.
        expiring_soon = (
            days_to_expiry is not None
            and days_to_expiry <= expiry_lookahead_days
        )
        never_rotated_and_old = (
            not rotated_at
            and days_since_created is not None
            and days_since_created > rotation_days
        )
        if not (expiring_soon or never_rotated_and_old):
            continue

        due_entries.append({
            "service": service,
            "expires_at": expires_at,
            "rotated_at": rotated_at,
            "created_at": created_at,
            "days_to_expiry": days_to_expiry,
            "days_since_created": days_since_created,
        })

    if not due_entries:
        return {
            "ok": True,
            "scanned": len(rows),
            "due": 0,
            "emitted": [],
            "skipped_duplicate": 0,
        }

    existing_titles = await _existing_titles_for_credential_paste()

    try:
        from src.founder_actions import create as create_founder_action
    except ImportError:
        logger.warning(
            "credential_rotation_reminder: founder_actions module missing"
        )
        return {
            "ok": False,
            "scanned": len(rows),
            "due": len(due_entries),
            "emitted": [],
            "skipped_duplicate": 0,
            "error": "founder_actions module unavailable",
        }

    emitted: list[int] = []
    skipped = 0
    for entry in due_entries:
        service = entry["service"]
        title = _build_title(service)
        if title in existing_titles:
            skipped += 1
            continue
        docs_url = _docs_url_for(service)
        why = _why_text(
            service=service,
            expires_at=entry.get("expires_at"),
            rotated_at=entry.get("rotated_at"),
            created_at=entry.get("created_at"),
            days_to_expiry=entry.get("days_to_expiry"),
            days_since_seen=entry.get("days_since_created"),
        )
        gen_step = (
            f"Generate a new credential at {docs_url}"
            if docs_url else
            f"Generate a new credential in the {service} dashboard."
        )
        instructions = [
            gen_step,
            (
                f"Send `/credential add {service} "
                f"{{\"token\": \"<new value>\"}}` to the bot."
            ),
            "Mark this action done after the new credential is stored.",
        ]
        try:
            action = await create_founder_action(
                mission_id=SYSTEM_MISSION_ID,
                kind="credential_paste",
                title=title,
                why=why,
                instructions=instructions,
                expected_output_kind="credential",
                notify_telegram=False,
            )
            emitted.append(action.id)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "credential_rotation_reminder: founder_action create failed "
                "for %s: %s", service, e,
            )

    return {
        "ok": True,
        "scanned": len(rows),
        "due": len(due_entries),
        "emitted": emitted,
        "skipped_duplicate": skipped,
    }
