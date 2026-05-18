"""Mechanical helper: canonical ``tag_signature`` from taste_emphasis.

Used downstream (Z1 Tier 3 C HTML prototype generation) to name files
in the paraflow style (``mobile_fact_primary_light.style-guide.md``
→ ``tag_signature: "fact_primary"``).

Pure function — no LLM, no I/O. Accepts either a parsed dict or a
filesystem path under the mission workspace.
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.tools.workspace import get_mission_workspace
from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.derive_token_tag_signature")

# Canonical content-type slugs accepted by verify_taste_emphasis_shape.
_CANONICAL_CONTENT_TYPES = {
    "fact_primary",
    "community_primary",
    "discovery_primary",
    "transactional_primary",
    "informational_primary",
    "other",
}


def _resolve_under(workspace_root: str, rel_path: str) -> str | None:
    if not isinstance(rel_path, str) or not rel_path:
        return None
    if os.path.isabs(rel_path):
        return None
    joined = os.path.normpath(os.path.join(workspace_root, rel_path))
    root_real = os.path.realpath(workspace_root)
    joined_real = os.path.realpath(joined)
    if not (joined_real == root_real or joined_real.startswith(root_real + os.sep)):
        return None
    return joined_real


def derive_from_payload(payload: dict[str, Any]) -> str:
    """Return canonical ``tag_signature`` for a parsed taste_emphasis dict.

    Falls back to ``"other"`` when the primary content type is missing
    or unrecognised — matches the verifier's contract that ``other``
    requires populated secondary_emphasis (caller responsibility).
    """
    if not isinstance(payload, dict):
        return "other"
    pct = payload.get("primary_content_type")
    if isinstance(pct, str) and pct in _CANONICAL_CONTENT_TYPES:
        return pct
    return "other"


async def derive_token_tag_signature(
    mission_id: int | None,
    path: str = ".style/taste_emphasis.json",
    workspace_path: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive ``tag_signature`` for a mission.

    If ``payload`` is supplied, it is used directly. Otherwise the
    file at ``path`` (relative to the mission workspace) is read.
    """
    if payload is not None:
        return {"ok": True, "tag_signature": derive_from_payload(payload)}

    if workspace_path is None:
        if mission_id is None:
            return {"ok": False, "error": "no mission_id and no workspace_path"}
        workspace_path = get_mission_workspace(mission_id)

    abs_path = _resolve_under(workspace_path, path)
    if abs_path is None:
        return {"ok": False, "path": path, "error": "path rejected (absolute or traversal)"}
    if not os.path.isfile(abs_path):
        return {"ok": False, "path": path, "error": "file not found"}

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        return {"ok": False, "path": path, "error": f"invalid JSON: {e}"}
    except OSError as e:
        return {"ok": False, "path": path, "error": f"read failed: {e}"}

    return {
        "ok": True,
        "path": path,
        "tag_signature": derive_from_payload(payload),
    }
