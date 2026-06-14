"""Regen primitives — Z1 Tier 4A (C11+A15 + C19).

Two mechanical actions live here:

- :func:`regen_artifact` — re-emit one existing artifact against a founder
  change description, preserving the previous version as
  ``{stem}.v{N}{ext}`` siblings, recording a row in ``regen_log``.

- :func:`regen_bundle` — directional change ("darker / less warm / more
  clinical") fans out to the slice of artifacts that the axis touches
  (style guide + design tokens + per-screen plans + HTML prototypes),
  re-emitting each in dependency order with a shared axis prompt fragment.

Versioning scheme picked: **versioned siblings.** For an artifact at
``mission_42/charter.md``, the previous body is preserved at
``mission_42/charter.v{N}.md`` and the original path always holds the
latest body. ``regen_log.prev_version`` / ``new_version`` carry the
versioned-sibling absolute paths. This keeps the produces/consumes graph
intact (referencers always resolve to the canonical name) and bounds the
on-disk explosion to one extra file per regen.

Both primitives are LLM-aware but the LLM call is delegated to a private
``_invoke_emitter`` shim so unit tests can ``patch`` it directly without
needing a live model loaded.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Axis registry — C19 directional axes and their artifact-slice resolvers.
# Add an axis here, both regen_bundle and the Telegram surface pick it up.
# ---------------------------------------------------------------------------

_KNOWN_AXES: dict[str, dict[str, Any]] = {
    "tone": {
        # Style + every HTML + every screen plan referencing color/tone tokens.
        "globs": (
            ".style/style_guide.md",
            ".style/design_tokens.json",
            ".prototype/*.html",
            "screen_plans/*.md",
        ),
        "fragment": "Apply a {direction} tonal shift across this artifact. Preserve structure, content, and componentry; only the tonal axis changes.",
    },
    "density": {
        "globs": (
            ".style/style_guide.md",
            ".style/design_tokens.json",
            ".prototype/*.html",
            "screen_plans/*.md",
        ),
        "fragment": "Apply a {direction} density shift (whitespace, type sizing, hit-areas) while keeping the visual identity intact.",
    },
    "scope": {
        # Charter-side: scope changes ripple across charter + PRD + non_goals.
        "globs": (
            "charter.md",
            "PRD.md",
            "non_goals.md",
        ),
        "fragment": "Apply a {direction} scope shift to this artifact, keeping the falsification triple and ADR references intact.",
    },
}


def known_axes() -> tuple[str, ...]:
    """Public helper — Telegram surface lists axes here."""
    return tuple(_KNOWN_AXES.keys())


# ---------------------------------------------------------------------------
# Versioning helpers
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"\.v(\d+)$")


def _split_versioned(path: str) -> tuple[str, str]:
    """Split ``foo.md`` → ``("foo", ".md")``; ``foo`` → ``("foo", "")``."""
    base, ext = os.path.splitext(path)
    return base, ext


def _next_version(artifact_abs: str) -> int:
    """Scan sibling versioned files; return next free version number (>=2).

    First regen produces v2 — version 1 is the body that lives at the
    canonical path before any regen has happened.
    """
    base, ext = _split_versioned(artifact_abs)
    parent = os.path.dirname(artifact_abs)
    stem = os.path.basename(base)
    if not os.path.isdir(parent):
        return 2
    seen: list[int] = []
    prefix = stem + ".v"
    for name in os.listdir(parent):
        if not name.startswith(prefix):
            continue
        s = name[len(prefix):]
        # name = "<stem>.v<N>.<ext>"  → strip "<ext>" off s
        if ext and s.endswith(ext):
            s = s[: -len(ext)]
        if s.isdigit():
            seen.append(int(s))
    if not seen:
        return 2
    return max(seen) + 1


def _versioned_path(artifact_abs: str, version: int) -> str:
    base, ext = _split_versioned(artifact_abs)
    return f"{base}.v{version}{ext}"


# ---------------------------------------------------------------------------
# Emitter shim — patched in unit tests; production path delegates to coulson.
# ---------------------------------------------------------------------------

async def _invoke_emitter(
    *,
    artifact_path: str,
    current_body: str,
    change_description: str,
    axis_fragment: str | None = None,
    mission_id: int | None = None,
) -> dict[str, Any]:
    """Re-emit an artifact against a change description.

    Production path: ask the LLM via coulson with a small wrapper task
    carrying the current body + change description + optional axis fragment.

    Returns an envelope ``{"ok": bool, "text": str | None, "error": str | None}``.
    Failure modes (coulson unavailable / coulson raised / empty text) return
    ``ok=False`` with an error string. Callers MUST NOT write the original
    body back to disk on failure — that would silently pretend the regen
    succeeded.
    """
    try:  # pragma: no cover — exercised only when coulson is wired live.
        from coulson import execute as _coulson_execute  # type: ignore
    except Exception:
        _coulson_execute = None  # type: ignore[assignment]

    if _coulson_execute is None:
        return {
            "ok": False,
            "text": None,
            "error": "coulson unavailable; regen requires an LLM emitter",
        }

    prompt = [
        f"Re-emit the artifact at `{artifact_path}` against the change below.",
        "Preserve format and schema. Output ONLY the new body, no commentary.",
        "",
        f"Change: {change_description}",
    ]
    if axis_fragment:
        prompt += ["", f"Axis: {axis_fragment}"]
    prompt += ["", "Current body:", current_body]
    task = {
        "id": 0,
        "mission_id": mission_id,
        "title": f"regen:{os.path.basename(artifact_path)}",
        "context": json.dumps({"prompt": "\n".join(prompt)}),
    }
    try:
        res = await _coulson_execute("overhead", task)  # type: ignore[misc]
    except Exception as exc:
        logger.warning("regen_artifact: coulson failed (%s)", exc)
        return {"ok": False, "text": None, "error": f"coulson failed: {exc}"}
    text = (res or {}).get("text") if isinstance(res, dict) else None
    if not isinstance(text, str) or not text.strip():
        return {
            "ok": False,
            "text": None,
            "error": "emitter returned empty text",
        }
    return {"ok": True, "text": text, "error": None}


# ---------------------------------------------------------------------------
# DB log — written through an injected helper so tests can capture rows
# without touching the real DB.
# ---------------------------------------------------------------------------

async def _record_regen_log(
    *,
    mission_id: int,
    artifact_path: str,
    change_description: str,
    prev_version: str,
    new_version: str,
    scope: str,
) -> int | None:
    """Insert a row into ``regen_log``. Returns the row id or None on error."""
    try:
        from dabidabi import get_db  # type: ignore
    except Exception:
        logger.debug("regen_log: db module unavailable; skipping persistence")
        return None
    try:
        db = await get_db()  # type: ignore[misc]
        cur = await db.execute(
                """
                INSERT INTO regen_log
                  (mission_id, artifact_path, change_description,
                   prev_version, new_version, scope)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
            (
                mission_id,
                artifact_path,
                change_description,
                prev_version,
                new_version,
                scope,
            ),
        )
        await db.commit()
        return cur.lastrowid
    except Exception as exc:
        logger.warning("regen_log: insert failed (%s)", exc)
        return None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_workspace(workspace_path: str | None) -> str:
    if workspace_path:
        return workspace_path
    try:
        from src.tools.workspace import WORKSPACE_DIR  # type: ignore
        return WORKSPACE_DIR
    except Exception:
        return os.getcwd()


def _resolve_artifact_abs(workspace: str, rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(workspace, rel_or_abs))


# ---------------------------------------------------------------------------
# Public — single-artifact regen
# ---------------------------------------------------------------------------

async def regen_artifact(
    *,
    mission_id: int,
    artifact_path: str,
    change_description: str,
    workspace_path: str | None = None,
    scope: str = "artifact",
    axis_fragment: str | None = None,
) -> dict[str, Any]:
    """Re-emit a single artifact; preserve the prior body as `.v{N}` sibling.

    The original path always holds the latest version. The previous body
    is copied to ``{base}.v{N}{ext}`` BEFORE the new body is written so a
    crash mid-write doesn't lose the prior text.
    """
    workspace = _resolve_workspace(workspace_path)
    abs_path = _resolve_artifact_abs(workspace, artifact_path)

    if not os.path.isfile(abs_path):
        return {
            "ok": False,
            "error": f"regen_artifact: artifact not found at {abs_path}",
        }

    try:
        with open(abs_path, encoding="utf-8") as fh:
            current_body = fh.read()
    except OSError as exc:
        return {"ok": False, "error": f"regen_artifact: read failed ({exc})"}

    # Snapshot prev body to .v{N-1} (or v1 on first regen).
    prev_version_n = _next_version(abs_path) - 1
    prev_path = _versioned_path(abs_path, prev_version_n if prev_version_n >= 1 else 1)
    if not os.path.exists(prev_path):
        try:
            with open(prev_path, "w", encoding="utf-8") as fh:
                fh.write(current_body)
        except OSError as exc:
            return {"ok": False, "error": f"regen_artifact: snapshot prev failed ({exc})"}

    new_version_n = prev_version_n + 1
    new_path = _versioned_path(abs_path, new_version_n)

    emit = await _invoke_emitter(
        artifact_path=artifact_path,
        current_body=current_body,
        change_description=change_description,
        axis_fragment=axis_fragment,
        mission_id=mission_id,
    )
    # Back-compat: accept legacy `{"text": ...}` shape (treat as success when
    # text is a non-empty string). New shape: `{"ok": bool, "text", "error"}`.
    if "ok" in emit:
        if not emit.get("ok"):
            return {
                "ok": False,
                "artifact_path": artifact_path,
                "error": f"regen_artifact: emitter failed ({emit.get('error') or 'unknown error'})",
            }
        new_body = emit.get("text")
    else:
        new_body = emit.get("text")
    if not isinstance(new_body, str) or not new_body.strip():
        return {
            "ok": False,
            "artifact_path": artifact_path,
            "error": "regen_artifact: emitter produced empty body",
        }

    try:
        with open(new_path, "w", encoding="utf-8") as fh:
            fh.write(new_body)
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write(new_body)
    except OSError as exc:
        return {"ok": False, "error": f"regen_artifact: write failed ({exc})"}

    log_id = await _record_regen_log(
        mission_id=mission_id,
        artifact_path=artifact_path,
        change_description=change_description,
        prev_version=prev_path,
        new_version=new_path,
        scope=scope,
    )

    return {
        "ok": True,
        "artifact_path": artifact_path,
        "prev_version": prev_path,
        "new_version": new_path,
        "version_n": new_version_n,
        "log_id": log_id,
        "scope": scope,
    }


# ---------------------------------------------------------------------------
# Public — bundle regen
# ---------------------------------------------------------------------------

# Dependency-order priority: lower runs first. Tokens/style ripple into HTML
# + screen plans, so they regen first; HTML is last (consumes everything).
def _dep_priority(rel_path: str) -> int:
    p = rel_path.replace("\\", "/")
    if "/.style/" in p or p.startswith(".style/") or p.endswith("design_tokens.json"):
        return 0
    if p.endswith("style_guide.md"):
        return 0
    if "/charter.md" in p or p.endswith("charter.md"):
        return 1
    if p.endswith("PRD.md") or p.endswith("non_goals.md"):
        return 1
    if "screen_plans" in p:
        return 2
    if p.endswith(".html") or "/.prototype/" in p:
        return 3
    return 4


def _expand_globs(mission_root: str, globs: Iterable[str]) -> list[str]:
    """Expand each axis glob under the mission root; return absolute paths."""
    import glob as _glob
    out: list[str] = []
    for g in globs:
        full = os.path.normpath(os.path.join(mission_root, g))
        # If it has a wildcard, glob it; otherwise existence-check.
        if any(ch in g for ch in "*?["):
            out.extend(_glob.glob(full))
        elif os.path.isfile(full):
            out.append(full)
    # Dedupe + skip versioned siblings (don't regen .v2.md of an artifact).
    seen: set[str] = set()
    deduped: list[str] = []
    for p in out:
        if p in seen:
            continue
        if _VERSION_RE.search(os.path.splitext(p)[0]):
            # path like foo.v2.md → splitext drops .md → base ends with .v2
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


async def regen_bundle(
    *,
    mission_id: int,
    axis: str,
    direction: str,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Re-emit the slice of artifacts that ``axis`` touches.

    Returns ``{"ok": True, "affected": [<per-artifact result>, ...]}``.
    Order of ``affected`` reflects dependency-order regen.
    """
    if axis not in _KNOWN_AXES:
        return {
            "ok": False,
            "error": f"regen_bundle: unknown axis {axis!r} (known: {sorted(_KNOWN_AXES)})",
        }

    spec = _KNOWN_AXES[axis]
    fragment = spec["fragment"].format(direction=direction)
    workspace = _resolve_workspace(workspace_path)
    mission_root = os.path.join(workspace, f"mission_{mission_id}")

    candidates = _expand_globs(mission_root, spec["globs"])
    candidates.sort(key=lambda p: (_dep_priority(os.path.relpath(p, mission_root)), p))

    affected: list[dict[str, Any]] = []
    for abs_path in candidates:
        rel = os.path.relpath(abs_path, workspace).replace("\\", "/")
        res = await regen_artifact(
            mission_id=mission_id,
            artifact_path=rel,
            change_description=direction,
            workspace_path=workspace,
            scope="bundle",
            axis_fragment=fragment,
        )
        affected.append(res)

    return {
        "ok": True,
        "axis": axis,
        "direction": direction,
        "fragment": fragment,
        "affected": affected,
    }
