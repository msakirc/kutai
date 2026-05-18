"""Verify mission `.flow/shared_shell.md` shape (C18 shared-shell invariants).

Mechanical post-step verifier for step ``5.0d.verify``. No LLM. Asserts:

- File exists.
- YAML frontmatter present.
- Required shell components declared (header / empty_state / error_state /
  loading_state at minimum) — each as a ``## <Name>`` markdown heading.
"""

from __future__ import annotations

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
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

REQUIRED_SHELLS = ("header", "empty_state", "error_state", "loading_state")


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


async def verify_shared_shell_shape(
    *,
    mission_id: int | None,
    path: str | None = None,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Validate shared_shell.md shape.

    Returns ``{"ok": bool, "errors": [...], "shells": [...]}``.
    """
    errors: list[str] = []

    if not workspace_path:
        try:
            from src.tools.workspace import get_mission_workspace
            workspace_path = str(get_mission_workspace(mission_id))
        except Exception as e:
            return {"ok": False, "errors": [f"workspace resolve failed: {e}"]}

    rel = path or ".flow/shared_shell.md"
    abs_path = _resolve_under(workspace_path, rel)
    if abs_path is None:
        return {"ok": False, "errors": [f"path rejected: {rel}"]}
    if not os.path.isfile(abs_path):
        return {"ok": False, "errors": [f"missing file: {rel}"]}

    with open(abs_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not _FRONTMATTER_RE.match(text):
        errors.append("missing or unparseable YAML frontmatter")

    headings = [_normalize(h) for h in _HEADING_RE.findall(text)]
    # Aliases — accept singular and EmptyState/Empty State variants alike.
    alias_map = {
        "empty": "empty_state",
        "error": "error_state",
        "loading": "loading_state",
        "emptystate": "empty_state",
        "errorstate": "error_state",
        "loadingstate": "loading_state",
    }
    norm_headings = {alias_map.get(h, h) for h in headings}

    missing = [s for s in REQUIRED_SHELLS if s not in norm_headings]
    if missing:
        errors.append(f"missing required shell sections: {missing}")

    return {
        "ok": not errors,
        "errors": errors,
        "shells": sorted(norm_headings),
    }
