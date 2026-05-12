"""Z1 Tier 5A (P6) — phase-boundary compliance blocker check.

Post-hook on phase-6 terminal step (``6.6 project_plan_review``). Reads
``mission_<id>/compliance_overlay.json`` and fails the step when any
``required_documents[i]`` declares ``blocker_for_phase <= current_phase``
AND either:

  * the rendered template at ``generated_template_path`` is missing, OR
  * the doc has ``founder_review_required=true`` and no row exists in
    ``founder_signoffs`` for ``(mission_id, doc_type)``.

The signoff log is read via ``src.infra.db.get_founder_signoffs``; an
empty set is returned when the table is missing or unreadable, so the
"rendered file exists" path stays usable as a fallback.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.compliance_blocker_check")


def _resolve_workspace(mission_id: int, workspace_path: str | None) -> str:
    if workspace_path:
        return workspace_path
    from src.tools.workspace import get_mission_workspace
    return get_mission_workspace(int(mission_id))


def _load_overlay(workspace_path: str) -> dict[str, Any] | None:
    candidates = [
        os.path.join(workspace_path, "compliance_overlay.json"),
        os.path.join(workspace_path, ".compliance", "overlay.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as e:
                logger.warning("compliance_blocker_check: read failed %s: %s", path, e)
    return None


def _load_signoffs_sync(mission_id: int) -> set[str]:
    """Synchronous wrapper around get_founder_signoffs.

    compliance_blocker_check is sync (called from a sync dispatch path),
    so we spin a small event loop call. Empty set on any error so the
    legacy file-exists path stays the safety net.
    """
    try:
        from src.infra.db import get_founder_signoffs
    except Exception:
        return set()
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Called from inside a running loop — schedule + wait via
                # asyncio.run on a thread would deadlock; defer to a future.
                # Cheapest safe path: return empty so the file-exists rule
                # still applies. Async callers should hit get_founder_signoffs
                # directly.
                return set()
        except RuntimeError:
            pass
        return asyncio.run(get_founder_signoffs(int(mission_id)))
    except Exception:
        return set()


def compliance_blocker_check(
    mission_id: int,
    current_phase: int = 6,
    workspace_path: str | None = None,
    signoffs: set[str] | None = None,
) -> dict[str, Any]:
    """Check compliance overlay against current phase boundary.

    Returns ``{"ok", "pending": [...], "checked_phase", "overlay_present"}``.
    ``ok=False`` when at least one required doc has not been rendered yet
    AND its ``blocker_for_phase <= current_phase``, OR when a doc with
    ``founder_review_required=true`` has no row in ``founder_signoffs``.

    ``signoffs`` can be passed in by async callers to skip the sync-bridge
    lookup. When omitted, ``_load_signoffs_sync`` reads the table itself.
    """
    ws = _resolve_workspace(mission_id, workspace_path)
    overlay = _load_overlay(ws)
    if overlay is None:
        # Per spec: missing overlay is NOT an automatic fail at phase 6 —
        # the prototype fast-path can have an empty overlay. The reviewer
        # at 6.6 still inspects 'overlay_present=False' and warns.
        return {
            "ok": True,
            "pending": [],
            "checked_phase": current_phase,
            "overlay_present": False,
        }

    if signoffs is None:
        signoffs = _load_signoffs_sync(int(mission_id))

    pending: list[dict[str, Any]] = []
    for doc in overlay.get("required_documents") or []:
        bfp = doc.get("blocker_for_phase")
        if bfp is None:
            continue
        try:
            bfp_int = int(bfp)
        except (ValueError, TypeError):
            continue
        if bfp_int > current_phase:
            continue
        doc_type = doc.get("doc_type")
        gen_path = doc.get("generated_template_path") or ""
        # Treat absolute paths as-is; relative paths are resolved against
        # the repo root (workspace would be wrong for a workspace-relative
        # spec path). Fall back to absent → still pending.
        file_present = False
        if gen_path:
            abs_path = gen_path if os.path.isabs(gen_path) else os.path.abspath(gen_path)
            file_present = os.path.isfile(abs_path)

        needs_signoff = bool(doc.get("founder_review_required"))
        signoff_present = bool(doc_type and doc_type in signoffs)

        if not file_present:
            pending.append({
                "doc_type": doc_type,
                "blocker_for_phase": bfp_int,
                "generated_template_path": gen_path,
                "reason": "template_not_rendered",
            })
            continue
        if needs_signoff and not signoff_present:
            pending.append({
                "doc_type": doc_type,
                "blocker_for_phase": bfp_int,
                "generated_template_path": gen_path,
                "reason": "founder_signoff_missing",
            })
            continue

    return {
        "ok": not pending,
        "pending": pending,
        "checked_phase": current_phase,
        "overlay_present": True,
    }
