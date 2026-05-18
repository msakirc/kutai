"""Verify mission `.flow/user_flow.md` shape (C4+A12 user-flow with Mermaid).

Mechanical post-step verifier for step ``5.0c.verify``. No LLM. Asserts:

- File exists.
- YAML frontmatter present (between two ``---`` markers).
- ``surfaces`` declared in frontmatter (read from surfaces.json if absent).
- At least one ```` ```mermaid ```` block per declared surface.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any


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


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_MERMAID_BLOCK_RE = re.compile(r"```mermaid\b", re.IGNORECASE)


def _parse_frontmatter_surfaces(text: str) -> list[str] | None:
    """Extract ``surfaces`` list from YAML frontmatter without yaml dep."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None
    fm = m.group(1)
    # Look for `surfaces: ["a", "b"]` or `surfaces: [a, b]`.
    sm = re.search(r"^\s*surfaces:\s*\[(.*?)\]", fm, re.MULTILINE)
    if not sm:
        return None
    raw = sm.group(1)
    items = [x.strip().strip("\"'") for x in raw.split(",") if x.strip()]
    return items


async def verify_user_flow_shape(
    *,
    mission_id: int | None,
    path: str | None = None,
    surfaces: list[str] | None = None,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Validate user_flow.md shape.

    Returns ``{"ok": bool, "errors": [...], "surfaces": [...], "mermaid_count": int}``.
    """
    errors: list[str] = []

    if not workspace_path:
        try:
            from src.tools.workspace import get_mission_workspace
            workspace_path = str(get_mission_workspace(mission_id))
        except Exception as e:
            return {"ok": False, "errors": [f"workspace resolve failed: {e}"]}

    rel = path or ".flow/user_flow.md"
    abs_path = _resolve_under(workspace_path, rel)
    if abs_path is None:
        return {"ok": False, "errors": [f"path rejected: {rel}"]}
    if not os.path.isfile(abs_path):
        return {"ok": False, "errors": [f"missing file: {rel}"]}

    with open(abs_path, "r", encoding="utf-8") as f:
        text = f.read()

    fm_surfaces = _parse_frontmatter_surfaces(text)
    if fm_surfaces is None:
        errors.append("missing or unparseable YAML frontmatter (need surfaces: [...])")

    declared = surfaces if surfaces is not None else (fm_surfaces or [])

    # Try surfaces.json fallback when caller didn't supply.
    if not declared:
        sjson = _resolve_under(workspace_path, ".charter/surfaces.json")
        if sjson and os.path.isfile(sjson):
            try:
                with open(sjson, "r", encoding="utf-8") as f:
                    sdata = json.load(f)
                if isinstance(sdata.get("surfaces"), list):
                    declared = list(sdata["surfaces"])
            except Exception:
                pass

    mermaid_count = len(_MERMAID_BLOCK_RE.findall(text))
    if mermaid_count == 0:
        errors.append("no ```mermaid blocks found")

    if declared and mermaid_count < len(declared):
        errors.append(
            f"mermaid block count ({mermaid_count}) < declared surfaces ({len(declared)})"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "surfaces": declared,
        "mermaid_count": mermaid_count,
    }
