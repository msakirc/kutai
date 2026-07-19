"""Verify mission `.flow/screen_inventory.md` shape (C18 shared-shell).

Mechanical post-step verifier for step ``5.0d.verify``. No LLM. Asserts:

- File exists.
- YAML frontmatter present with ``total_screens``, ``chunk_size``, ``chunks``.
- ``chunks`` is a list of lists; each chunk size <= ``chunk_size``.
- ``total_screens`` equals the sum of chunk lengths.
- Every screen line in the body has a route (``(`/path`)``) annotation.
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
_SCREEN_LINE_RE = re.compile(r"^\s*-\s+(.+?)\s*$")
_ROUTE_RE = re.compile(r"\(`/[^`]*`\)")


def _parse_frontmatter(text: str) -> dict[str, Any] | None:
    """Parse the artifact's YAML frontmatter with a real YAML parser.

    Accepts BOTH the flow-style ``chunks: [[...], [...]]`` inline list AND a
    block-style YAML sequence::

        chunks:
          - ["Landing", "Auth", ...]
          - ["Detail", "Settings", ...]

    The prior hand-rolled regex only matched flow style, so a model that
    emitted equally-valid block-style YAML got a false ``frontmatter missing
    chunks`` reject → degenerate DLQ of a correct artifact (m90 567453).
    ``yaml.safe_load`` is the codebase's frontmatter arbiter (verify_artifacts,
    visual_review already use it). Returns ``None`` when no ``---`` block is
    present or the block is not parseable YAML mapping.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None
    import yaml  # lazy; pyyaml is already a dependency

    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None

    out: dict[str, Any] = {}
    for key in ("total_screens", "chunk_size"):
        if isinstance(data.get(key), int):
            out[key] = data[key]
    chunks = data.get("chunks")
    if isinstance(chunks, list):
        out["chunks"] = [
            [str(x) for x in c] if isinstance(c, list) else c for c in chunks
        ]
    return out


async def verify_screen_inventory_shape(
    *,
    mission_id: int | None,
    path: str | None = None,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Validate screen_inventory.md shape.

    Returns ``{"ok": bool, "errors": [...], "total": int, "chunk_size": int}``.
    """
    errors: list[str] = []

    if not workspace_path:
        try:
            from src.tools.workspace import get_mission_workspace
            workspace_path = str(get_mission_workspace(mission_id))
        except Exception as e:
            return {"ok": False, "errors": [f"workspace resolve failed: {e}"]}

    rel = path or ".flow/screen_inventory.md"
    abs_path = _resolve_under(workspace_path, rel)
    if abs_path is None:
        return {"ok": False, "errors": [f"path rejected: {rel}"]}
    if not os.path.isfile(abs_path):
        return {"ok": False, "errors": [f"missing file: {rel}"]}

    with open(abs_path, "r", encoding="utf-8") as f:
        text = f.read()

    fm = _parse_frontmatter(text)
    if fm is None:
        return {"ok": False, "errors": ["missing or unparseable YAML frontmatter"]}

    total = fm.get("total_screens")
    chunk_size = fm.get("chunk_size")
    chunks = fm.get("chunks")

    if total is None:
        errors.append("frontmatter missing total_screens")
    if chunk_size is None:
        errors.append("frontmatter missing chunk_size")
    if not isinstance(chunks, list) or not chunks:
        errors.append("frontmatter missing chunks (non-empty list of lists)")
        chunks = []

    if isinstance(chunk_size, int) and isinstance(chunks, list):
        for i, ch in enumerate(chunks):
            if not isinstance(ch, list):
                errors.append(f"chunks[{i}] is not a list")
                continue
            if len(ch) > chunk_size:
                errors.append(
                    f"chunks[{i}] has {len(ch)} items > chunk_size={chunk_size}"
                )

    if isinstance(total, int) and isinstance(chunks, list):
        actual_sum = sum(len(c) for c in chunks if isinstance(c, list))
        if actual_sum != total:
            errors.append(
                f"total_screens={total} != sum-of-chunks={actual_sum}"
            )

    # Every screen bullet under a header section must include a route.
    body = text[text.find("---", 3) + 3 :] if text.startswith("---") else text
    missing_route_lines: list[str] = []
    for raw in body.splitlines():
        line = raw.rstrip()
        m = _SCREEN_LINE_RE.match(line)
        if not m:
            continue
        item = m.group(1)
        if not _ROUTE_RE.search(item):
            missing_route_lines.append(item)
    if missing_route_lines:
        errors.append(
            f"{len(missing_route_lines)} screen line(s) missing route: "
            f"{missing_route_lines[:3]}"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "total": total if isinstance(total, int) else 0,
        "chunk_size": chunk_size if isinstance(chunk_size, int) else 0,
    }
