"""Verify mission `.charter/surfaces.json` shape (C12 multi-surface).

Mechanical post-step verifier for step ``5.0b.verify``. No LLM. Asserts:

- File exists and parses as JSON.
- ``_schema_version`` == ``"1"``.
- ``surfaces`` is a non-empty list of valid surface tokens.
- ``primary_surface`` is present and is one of the declared surfaces.
- ``founder_confirmed_at`` is a non-empty string (ISO timestamp).
"""

from __future__ import annotations

import json
import os
from typing import Any

VALID_SURFACES = {"mobile", "web", "desktop", "admin"}


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


async def verify_surfaces_shape(
    *,
    mission_id: int | None,
    path: str | None = None,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Validate surfaces.json shape.

    Returns ``{"ok": bool, "errors": [str, ...], "surfaces": [...], "primary": str}``.
    """
    errors: list[str] = []

    if not workspace_path:
        # Lazy import to avoid hard dependency in unit tests.
        try:
            from src.tools.workspace import get_mission_workspace
            workspace_path = str(get_mission_workspace(mission_id))
        except Exception as e:
            return {"ok": False, "errors": [f"workspace resolve failed: {e}"]}

    rel = path or ".charter/surfaces.json"
    abs_path = _resolve_under(workspace_path, rel)
    if abs_path is None:
        return {"ok": False, "errors": [f"path rejected: {rel}"]}
    if not os.path.isfile(abs_path):
        return {"ok": False, "errors": [f"missing file: {rel}"]}

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"ok": False, "errors": [f"invalid JSON: {e}"]}

    if not isinstance(data, dict):
        return {"ok": False, "errors": ["surfaces.json must be a JSON object"]}

    if str(data.get("_schema_version") or "") != "1":
        errors.append("_schema_version must be \"1\"")

    surfaces = data.get("surfaces")
    if not isinstance(surfaces, list) or len(surfaces) == 0:
        errors.append("surfaces must be a non-empty list")
        surfaces = []
    else:
        for s in surfaces:
            if s not in VALID_SURFACES:
                errors.append(f"invalid surface token: {s!r}")

    primary = data.get("primary_surface")
    if not primary or not isinstance(primary, str):
        errors.append("primary_surface missing or not a string")
    elif surfaces and primary not in surfaces:
        errors.append(f"primary_surface {primary!r} not in surfaces {surfaces!r}")

    confirmed = data.get("founder_confirmed_at")
    if not confirmed or not isinstance(confirmed, str):
        errors.append("founder_confirmed_at missing")

    return {
        "ok": not errors,
        "errors": errors,
        "surfaces": surfaces if isinstance(surfaces, list) else [],
        "primary": primary if isinstance(primary, str) else "",
    }
