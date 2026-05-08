"""Verify declared build-step artifacts exist and are non-trivial.

Mechanical post-step verifier. Caller declares paths the previous step was
supposed to produce; mr_roboto resolves each under the mission workspace and
checks: file exists, size >= min_bytes, optional syntax/parse check.

No LLM, no narration. The whole point is to ground "the coder said it wrote
this" against the filesystem.
"""

from __future__ import annotations

import asyncio
import glob as _glob_mod
import hashlib
import json
import os
import py_compile
from typing import Any

from src.tools.workspace import get_mission_workspace
from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.verify_artifacts")


_COMPILE_CHECKERS: dict[str, str] = {
    ".py": "py_compile",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def _resolve_under(workspace_root: str, rel_path: str) -> str | None:
    """Return absolute path if rel_path stays inside workspace_root.

    Refuses absolute paths and traversal. Returns None on rejection.
    """
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


def _is_glob(p: str) -> bool:
    return any(c in p for c in "*?[")


def _glob_under(workspace_root: str, pattern: str) -> list[str]:
    """Expand glob pattern under workspace_root. Returns absolute file paths.

    Stack-variant scaffolds (7.4 db, 7.6 test infra) declare patterns like
    ``migrations/**/*`` rather than canonical paths, so the same step can
    verify alembic / prisma / drizzle output without per-stack branches.
    Refuses absolute patterns and excludes directories + traversal escapes.
    """
    if not isinstance(pattern, str) or not pattern or os.path.isabs(pattern):
        return []
    full_pattern = os.path.join(workspace_root, pattern)
    root_real = os.path.realpath(workspace_root)
    out: list[str] = []
    for hit in _glob_mod.glob(full_pattern, recursive=True):
        if not os.path.isfile(hit):
            continue
        hit_real = os.path.realpath(hit)
        if not (hit_real == root_real or hit_real.startswith(root_real + os.sep)):
            continue
        out.append(hit_real)
    return out


async def _compile_check_one(abs_path: str) -> str | None:
    """Return None on pass, error string on fail. Skips unsupported extensions."""
    ext = os.path.splitext(abs_path)[1].lower()
    checker = _COMPILE_CHECKERS.get(ext)
    if not checker:
        return None
    loop = asyncio.get_event_loop()
    try:
        if checker == "py_compile":
            await loop.run_in_executor(
                None, lambda: py_compile.compile(abs_path, doraise=True)
            )
        elif checker == "json":
            def _parse_json():
                with open(abs_path, "r", encoding="utf-8") as f:
                    json.load(f)
            await loop.run_in_executor(None, _parse_json)
        elif checker == "yaml":
            try:
                import yaml  # type: ignore
            except ImportError:
                return None  # yaml not installed → don't block
            def _parse_yaml():
                with open(abs_path, "r", encoding="utf-8") as f:
                    yaml.safe_load(f)
            await loop.run_in_executor(None, _parse_yaml)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


async def verify_artifacts(
    mission_id: int | None,
    paths: list[str],
    min_bytes: int = 1,
    compile_check: bool = False,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Verify each path exists under the mission workspace.

    Parameters
    ----------
    mission_id:
        Used to locate the mission workspace if ``workspace_path`` is None.
    paths:
        Relative paths the previous step claimed to produce.
    min_bytes:
        Minimum file size. Files smaller than this count as failed.
    compile_check:
        If True, run a syntax check appropriate to the file extension
        (``py_compile`` for .py, ``json.load`` for .json, ``yaml.safe_load``
        for .yaml/.yml). Other extensions are skipped.
    workspace_path:
        Override workspace root (used by tests). Defaults to
        ``get_mission_workspace(mission_id)``.

    Returns
    -------
    dict
        ``{"verified": [...], "missing": [...], "failed": [...], "all_ok": bool}``.
        Each entry in ``verified`` is ``{"path", "bytes", "sha256"}``.
        Each entry in ``failed`` is ``{"path", "reason"}``.
    """
    if not isinstance(paths, list) or not paths:
        return {
            "verified": [],
            "missing": [],
            "failed": [],
            "all_ok": False,
            "error": "no paths supplied",
        }

    if workspace_path is None:
        if mission_id is None:
            return {
                "verified": [],
                "missing": list(paths),
                "failed": [],
                "all_ok": False,
                "error": "no mission_id and no workspace_path",
            }
        workspace_path = get_mission_workspace(mission_id)

    verified: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[dict[str, str]] = []

    async def _check_file(display: str, abs_path: str) -> bool:
        """Verify one resolved file. Appends to verified/failed. Returns ok."""
        try:
            size = os.path.getsize(abs_path)
        except OSError as e:
            failed.append({"path": display, "reason": f"stat failed: {e}"})
            return False
        if size < min_bytes:
            failed.append({"path": display, "reason": f"size {size} < min_bytes {min_bytes}"})
            return False
        if compile_check:
            err = await _compile_check_one(abs_path)
            if err is not None:
                failed.append({"path": display, "reason": f"compile check: {err}"})
                return False
        try:
            sha = _sha256_of(abs_path)
        except OSError as e:
            failed.append({"path": display, "reason": f"hash failed: {e}"})
            return False
        verified.append({"path": display, "bytes": size, "sha256": sha})
        return True

    async def _verify_string(rel: str) -> None:
        """Single string entry: glob expansion if pattern, else literal."""
        if _is_glob(rel):
            matches = _glob_under(workspace_path, rel)
            if not matches:
                missing.append(rel)
                return
            for abs_path in matches:
                display = f"{rel} -> {os.path.relpath(abs_path, workspace_path)}"
                await _check_file(display, abs_path)
            return
        abs_path = _resolve_under(workspace_path, rel)
        if abs_path is None:
            failed.append({"path": rel, "reason": "path rejected (absolute or traversal)"})
            return
        if not os.path.isfile(abs_path):
            missing.append(rel)
            return
        await _check_file(rel, abs_path)

    for entry in paths:
        # any_of semantic: nested list satisfies the slot if ANY alternative
        # has at least one file. Used by stack-variant scaffolds where the
        # exact path depends on framework choice (pytest conftest vs jest
        # config vs vitest config). First alternative that matches wins.
        if isinstance(entry, list):
            satisfied = False
            tried: list[str] = []
            for alt in entry:
                if not isinstance(alt, str) or not alt:
                    continue
                tried.append(alt)
                if _is_glob(alt):
                    matches = _glob_under(workspace_path, alt)
                    if not matches:
                        continue
                    for abs_path in matches:
                        display = f"any_of[{alt}] -> {os.path.relpath(abs_path, workspace_path)}"
                        await _check_file(display, abs_path)
                    satisfied = True
                    break
                abs_path = _resolve_under(workspace_path, alt)
                if abs_path is None:
                    continue  # rejected silently — try next alternative
                if not os.path.isfile(abs_path):
                    continue
                await _check_file(f"any_of[{alt}]", abs_path)
                satisfied = True
                break
            if not satisfied:
                missing.append(f"any_of[{', '.join(tried)}]")
            continue
        if isinstance(entry, str):
            await _verify_string(entry)
            continue
        failed.append({"path": str(entry), "reason": f"unsupported path entry type {type(entry).__name__}"})

    all_ok = not missing and not failed
    return {
        "verified": verified,
        "missing": missing,
        "failed": failed,
        "all_ok": all_ok,
    }
