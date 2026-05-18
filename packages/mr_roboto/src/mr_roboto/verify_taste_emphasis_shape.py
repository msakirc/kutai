"""Mechanical shape-verifier for Z1 Tier 3 ``taste_emphasis.json``.

Pure parse + structural assertions; no LLM. Wired as the sibling
``verify`` step of i2p_v3 ``5.0 taste_extraction_from_charter`` so the
generated artifact is rejected immediately if any required field is
missing or empty.

Schema (``_schema_version: "1"``)::

    {
      "_schema_version": "1",
      "mission_id": <int>,
      "primary_content_type":
          "fact_primary | community_primary | discovery_primary
           | transactional_primary | informational_primary | other",
      "primary_content_type_rationale": "...",
      "secondary_emphasis": [...],
      "tone_keywords": [...],
      "tone_rationale": "...",
      "key_visual_priorities": [...],
      "anti_emphasis": [...]
    }
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.tools.workspace import get_mission_workspace
from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.verify_taste_emphasis_shape")

SCHEMA_VERSION = "1"

ALLOWED_PRIMARY_CONTENT_TYPES = {
    "fact_primary",
    "community_primary",
    "discovery_primary",
    "transactional_primary",
    "informational_primary",
    "other",
}

REQUIRED_FIELDS = (
    "_schema_version",
    "mission_id",
    "primary_content_type",
    "primary_content_type_rationale",
    "secondary_emphasis",
    "tone_keywords",
    "tone_rationale",
    "key_visual_priorities",
    "anti_emphasis",
)


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


def verify_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a parsed taste_emphasis dict. Returns ``{"ok": bool, ...}``.

    Pure function — does not touch the filesystem. Useful for unit tests
    and for callers that already hold the parsed object.
    """
    if not isinstance(payload, dict):
        return {"ok": False, "error": f"payload must be object, got {type(payload).__name__}"}

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return {"ok": False, "error": f"missing required fields: {missing}"}

    if str(payload.get("_schema_version")) != SCHEMA_VERSION:
        return {
            "ok": False,
            "error": (
                f"_schema_version mismatch: expected {SCHEMA_VERSION!r}, "
                f"got {payload.get('_schema_version')!r}"
            ),
        }

    pct = payload.get("primary_content_type")
    if pct not in ALLOWED_PRIMARY_CONTENT_TYPES:
        return {
            "ok": False,
            "error": (
                f"primary_content_type {pct!r} not in "
                f"{sorted(ALLOWED_PRIMARY_CONTENT_TYPES)}"
            ),
        }

    rationale = payload.get("primary_content_type_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        return {"ok": False, "error": "primary_content_type_rationale must be non-empty string"}

    tone_rationale = payload.get("tone_rationale")
    if not isinstance(tone_rationale, str) or not tone_rationale.strip():
        return {"ok": False, "error": "tone_rationale must be non-empty string"}

    for list_field in ("secondary_emphasis", "tone_keywords", "key_visual_priorities", "anti_emphasis"):
        v = payload.get(list_field)
        if not isinstance(v, list):
            return {"ok": False, "error": f"{list_field} must be a list"}

    # tone_keywords and key_visual_priorities must have at least one entry
    if not payload["tone_keywords"]:
        return {"ok": False, "error": "tone_keywords must not be empty"}
    if not payload["key_visual_priorities"]:
        return {"ok": False, "error": "key_visual_priorities must not be empty"}

    # Reviewer rule: primary_content_type == 'other' demands secondary_emphasis populated.
    if pct == "other" and not payload["secondary_emphasis"]:
        return {
            "ok": False,
            "error": "primary_content_type 'other' requires non-empty secondary_emphasis",
        }

    return {"ok": True, "primary_content_type": pct}


async def verify_taste_emphasis_shape(
    mission_id: int | None,
    path: str = ".style/taste_emphasis.json",
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Resolve, parse, and validate a taste_emphasis.json artifact.

    Returns ``{"ok": bool, "path": str, "error": str | None, ...}``.
    """
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

    res = verify_payload(payload)
    res["path"] = path
    return res
