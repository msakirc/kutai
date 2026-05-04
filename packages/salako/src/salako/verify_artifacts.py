"""Verify declared build-step artifacts exist and are non-trivial.

Mechanical post-step verifier. Caller declares paths the previous step was
supposed to produce; salako resolves each under the mission workspace and
checks: file exists, size >= min_bytes, optional syntax/parse check.

No LLM, no narration. The whole point is to ground "the coder said it wrote
this" against the filesystem.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import py_compile
from typing import Any

from src.tools.workspace import get_mission_workspace
from src.infra.logging_config import get_logger

logger = get_logger("salako.verify_artifacts")


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

    for rel in paths:
        abs_path = _resolve_under(workspace_path, rel)
        if abs_path is None:
            failed.append({"path": rel, "reason": "path rejected (absolute or traversal)"})
            continue
        if not os.path.isfile(abs_path):
            missing.append(rel)
            continue
        try:
            size = os.path.getsize(abs_path)
        except OSError as e:
            failed.append({"path": rel, "reason": f"stat failed: {e}"})
            continue
        if size < min_bytes:
            failed.append({"path": rel, "reason": f"size {size} < min_bytes {min_bytes}"})
            continue
        if compile_check:
            err = await _compile_check_one(abs_path)
            if err is not None:
                failed.append({"path": rel, "reason": f"compile check: {err}"})
                continue
        try:
            sha = _sha256_of(abs_path)
        except OSError as e:
            failed.append({"path": rel, "reason": f"hash failed: {e}"})
            continue
        verified.append({"path": rel, "bytes": size, "sha256": sha})

    all_ok = not missing and not failed
    return {
        "verified": verified,
        "missing": missing,
        "failed": failed,
        "all_ok": all_ok,
    }
