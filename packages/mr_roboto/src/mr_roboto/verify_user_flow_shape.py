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
_BARE_MERMAID_START_RE = re.compile(r"^\s*(?:graph|flowchart)\s+\w+", re.IGNORECASE)


def _load_surfaces_json(workspace_path: str) -> list[str]:
    """Read the declared surfaces from ``.charter/surfaces.json`` (or [])."""
    sjson = _resolve_under(workspace_path, ".charter/surfaces.json")
    if sjson and os.path.isfile(sjson):
        try:
            with open(sjson, "r", encoding="utf-8") as f:
                sdata = json.load(f)
            if isinstance(sdata.get("surfaces"), list):
                return [str(s) for s in sdata["surfaces"]]
        except Exception:  # noqa: BLE001
            pass
    return []


def _inject_surfaces_frontmatter(text: str, surfaces: list[str]) -> tuple[str, bool]:
    """Ensure the YAML frontmatter declares ``surfaces:``.

    Adds a ``surfaces: [...]`` line to an existing frontmatter that lacks one,
    or prepends a minimal frontmatter when none is present. No-op when surfaces
    is empty (never inject an empty list) or a surfaces line already exists.
    """
    if not surfaces:
        return text, False
    surf_line = "surfaces: [" + ", ".join(surfaces) + "]"
    m = _FRONTMATTER_RE.match(text)
    if m:
        fm = m.group(1)
        if re.search(r"^\s*surfaces\s*:", fm, re.MULTILINE):
            return text, False
        new_fm = fm.rstrip("\n") + "\n" + surf_line
        return text[: m.start(1)] + new_fm + text[m.end(1) :], True
    return "---\n" + surf_line + "\n---\n\n" + text, True


def _fence_bare_mermaid(text: str) -> tuple[str, bool]:
    """Wrap any unfenced ``graph``/``flowchart`` block in a ```mermaid fence.

    A block is the diagram-start line plus the following blank or indented
    lines (the node/edge body). Trailing blank lines are kept outside the fence.
    Blocks already inside a ``` fence are left untouched.
    """
    lines = text.split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    in_fence = False
    changed = False
    while i < n:
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if not in_fence and _BARE_MERMAID_START_RE.match(line):
            j = i + 1
            while j < n and (lines[j].strip() == "" or lines[j][:1].isspace()):
                j += 1
            block = lines[i:j]
            trailing: list[str] = []
            while block and block[-1].strip() == "":
                trailing.insert(0, block.pop())
            out.append("```mermaid")
            out.extend(block)
            out.append("```")
            out.extend(trailing)
            changed = True
            i = j
            continue
        out.append(line)
        i += 1
    return ("\n".join(out), True) if changed else (text, False)


def normalize_user_flow(
    text: str, surfaces: list[str] | None
) -> tuple[str, bool]:
    """Deterministically repair a user_flow.md to the shape gate's contract.

    Injects a missing ``surfaces:`` frontmatter line (from *surfaces*) and
    fences bare mermaid blocks. Returns ``(repaired_text, changed)``. Never
    fabricates diagram content — a genuinely under-produced multi-surface flow
    still fails verification.
    """
    changed = False
    text, c1 = _inject_surfaces_frontmatter(text, surfaces or [])
    text, c2 = _fence_bare_mermaid(text)
    return text, (c1 or c2)


def _parse_frontmatter_surfaces(text: str) -> list[str] | None:
    """Extract ``surfaces`` from YAML frontmatter (flow OR block style).

    Both ``surfaces: ["web", "mobile"]`` (flow) and a block-style sequence::

        surfaces:
          - web
          - mobile

    are valid YAML; the prior flow-only regex false-rejected block style
    (same class as m90 567453 screen_inventory chunks). ``yaml.safe_load`` is
    the codebase's frontmatter arbiter. Returns ``None`` when no ``---`` block
    is present, the block is unparseable, or ``surfaces`` is absent.
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
    surfaces = data.get("surfaces")
    if isinstance(surfaces, list):
        return [str(s) for s in surfaces]
    if isinstance(surfaces, str) and surfaces.strip():
        return [surfaces.strip()]
    return None


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

    # ── Deterministic shape repair (before validation) ──
    # The analyst's flow graph is the deliverable; the frontmatter `surfaces:`
    # key and the ```mermaid fence are mechanical shell the engine guarantees
    # regardless of model compliance (m90 567452). Repair, write back, validate
    # the repaired form. On write failure, validate the on-disk (unrepaired)
    # text so the gate reflects disk truth.
    repair_surfaces = surfaces if surfaces is not None else _load_surfaces_json(workspace_path)
    repaired, changed = normalize_user_flow(text, repair_surfaces)
    if changed:
        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(repaired)
            text = repaired
        except OSError:
            pass

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
