"""Z1 Tier 5A (P6) — phase-boundary compliance blocker check.

Post-hook on phase-6 terminal step (``6.6 project_plan_review``). Reads
``mission_<id>/compliance_overlay.json`` and fails the step when any
``required_documents[i]`` declares ``blocker_for_phase <= current_phase``
AND the rendered template file at ``generated_template_path`` is not yet
present.

Per spec (plan-v3 lines 1587-1608) the v3 form additionally checks
founder-signoffs; that signoff log is z1-Tier-5-not-yet-shipped, so for
T5A we use the simpler "rendered file exists" rule. Founder-signoff
gating is a deferred Tier-5C/D follow-up.
"""
from __future__ import annotations

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


def compliance_blocker_check(
    mission_id: int,
    current_phase: int = 6,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Check compliance overlay against current phase boundary.

    Returns ``{"ok", "pending": [...], "checked_phase", "overlay_present"}``.
    ``ok=False`` when at least one required doc has not been rendered yet
    AND its ``blocker_for_phase <= current_phase``.
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
        gen_path = doc.get("generated_template_path") or ""
        # Treat absolute paths as-is; relative paths are resolved against
        # the repo root (workspace would be wrong for a workspace-relative
        # spec path). Fall back to absent → still pending.
        if gen_path:
            abs_path = gen_path if os.path.isabs(gen_path) else os.path.abspath(gen_path)
            if os.path.isfile(abs_path):
                continue
        pending.append({
            "doc_type": doc.get("doc_type"),
            "blocker_for_phase": bfp_int,
            "generated_template_path": gen_path,
        })

    return {
        "ok": not pending,
        "pending": pending,
        "checked_phase": current_phase,
        "overlay_present": True,
    }
