"""Z6 T4D — weekly staleness check for compliance templates.

Walks ``compliance_templates/**/*.meta.json``. For every template whose
``last_reviewed`` date is more than ``STALE_DAYS`` (180) in the past,
emits a ``legal_counsel`` founder_action so the founder can refresh the
template with counsel review.

Idempotent: before creating a new founder_action this executor scans the
existing pending/in-progress ``legal_counsel`` rows and skips any whose
``title`` matches the deterministic title we'd create for the same
template.

Cron registration: weekly via ``general_beckman.cron_seed`` as the
``compliance_template_staleness`` internal cadence.

Mission scoping
---------------
Staleness is a system-wide concern (templates live in the repo, not a
mission). When invoked from cron we use ``mission_id=0`` as a sentinel.
Founder_actions table requires a mission_id (NOT NULL); 0 reads as
"system / unscoped" in the Telegram surface and the founder_actions
listing query.
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.compliance_template_staleness")

TEMPLATE_ROOT = "compliance_templates"
STALE_DAYS = 180
SYSTEM_MISSION_ID = 0


def _walk_meta_files(root: str) -> list[str]:
    out: list[str] = []
    if not os.path.isdir(root):
        return out
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".meta.json"):
                out.append(os.path.join(dirpath, fn))
    return out


def _parse_meta(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning("compliance_template_staleness: meta read fail %s: %s", path, e)
        return None


def _build_title(doc_type: str, jurisdiction: str, lang: str) -> str:
    return (
        f"Review compliance template {doc_type} ({jurisdiction}/{lang})"
    )


def _is_stale(last_reviewed: str | None, today: datetime.date | None = None) -> bool:
    if not last_reviewed:
        # Missing review date — treat as stale (forces a counsel review).
        return True
    try:
        lr = datetime.date.fromisoformat(str(last_reviewed))
    except ValueError:
        return False
    ref = today or datetime.date.today()
    return (ref - lr).days > STALE_DAYS


async def _existing_titles_for_legal_counsel() -> set[str]:
    """Return titles of pending/in_progress legal_counsel founder_actions
    so the executor can skip duplicates."""
    try:
        from dabidabi import get_db
    except ImportError:
        return set()
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT title FROM founder_actions "
            "WHERE kind = 'legal_counsel' "
            "AND status IN ('pending', 'in_progress')"
        )
        rows = await cur.fetchall()
    except Exception as e:
        logger.warning(
            "compliance_template_staleness: existing-titles query failed: %s", e,
        )
        return set()
    return {str(r[0]) for r in rows if r and r[0]}


async def compliance_template_staleness(
    template_root: str | None = None,
    today: datetime.date | None = None,
) -> dict[str, Any]:
    """Emit a founder_action per stale compliance template.

    Returns ``{ok, scanned, stale, emitted, skipped_duplicate}``.
    """
    root = template_root or TEMPLATE_ROOT
    metas = _walk_meta_files(root)
    if not metas:
        return {
            "ok": True,
            "scanned": 0,
            "stale": 0,
            "emitted": [],
            "skipped_duplicate": 0,
        }

    stale_entries: list[dict[str, Any]] = []
    for path in metas:
        meta = _parse_meta(path)
        if not meta:
            continue
        last_reviewed = meta.get("last_reviewed")
        if not _is_stale(last_reviewed, today=today):
            continue
        # Best-effort field defaults — derive from path when the meta is sparse.
        rel = os.path.relpath(path, root).replace("\\", "/")
        parts = rel.split("/")
        # Expected layout: <jurisdiction>/<lang>/<doc_type>.meta.json
        path_juris = parts[0] if len(parts) >= 3 else "default"
        path_lang = parts[1] if len(parts) >= 3 else "en"
        doc_type = (
            meta.get("doc_type")
            or (
                os.path.basename(path).replace(".meta.json", "")
                if path.endswith(".meta.json") else "unknown"
            )
        )
        jurisdiction = meta.get("jurisdiction") or path_juris
        lang = meta.get("lang") or path_lang
        stale_entries.append({
            "path": path,
            "doc_type": doc_type,
            "jurisdiction": jurisdiction,
            "lang": lang,
            "last_reviewed": last_reviewed,
        })

    if not stale_entries:
        return {
            "ok": True,
            "scanned": len(metas),
            "stale": 0,
            "emitted": [],
            "skipped_duplicate": 0,
        }

    existing_titles = await _existing_titles_for_legal_counsel()

    emitted: list[int] = []
    skipped = 0
    try:
        from src.founder_actions import create as create_founder_action
    except ImportError:
        logger.warning("compliance_template_staleness: founder_actions module missing")
        return {
            "ok": False,
            "scanned": len(metas),
            "stale": len(stale_entries),
            "emitted": [],
            "skipped_duplicate": 0,
            "error": "founder_actions module unavailable",
        }

    for entry in stale_entries:
        title = _build_title(
            entry["doc_type"], entry["jurisdiction"], entry["lang"],
        )
        if title in existing_titles:
            skipped += 1
            continue
        why = (
            f"last reviewed {entry['last_reviewed'] or 'never'}; "
            f"templates older than {STALE_DAYS} days may be stale and "
            "should be refreshed with counsel review."
        )
        instructions = [
            f"Open the template at {entry['path']}.",
            "Have counsel review the wording for current legal accuracy.",
            "Update the body if needed and bump .meta.json `last_reviewed` "
            "to today's date when finished.",
            "Mark this action done.",
        ]
        try:
            action = await create_founder_action(
                mission_id=SYSTEM_MISSION_ID,
                kind="legal_counsel",
                title=title,
                why=why,
                instructions=instructions,
                expected_output_kind="ack_only",
                notify_telegram=False,
            )
            emitted.append(action.id)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "compliance_template_staleness: founder_action create failed: %s", e,
            )

    return {
        "ok": True,
        "scanned": len(metas),
        "stale": len(stale_entries),
        "emitted": emitted,
        "skipped_duplicate": skipped,
    }
