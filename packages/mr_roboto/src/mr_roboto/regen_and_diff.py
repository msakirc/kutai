"""Regenerate an artifact to stdout and diff it against the committed target.

Mechanical executor. No LLM. Shells out to ``generator_cmd`` (argv list),
captures stdout, and computes a unified diff against ``target_path``.

Return shape
------------
``ok``
    True when the generator ran (or was soft-skipped) without an internal
    error.  Drift (diff_present=True) still yields ok=True — the caller
    (verdict fn) decides whether to blocker-retry.
``diff_present``
    True when stdout != target file contents.
``diff_excerpt``
    First 50 lines of unified diff, or ``""`` when diff_present=False.
``target_path``
    Echoed for the verdict fn.
``generator_cmd``
    Echoed for feedback messages.
``exit``
    Raw process exit code.  -1 on spawn failure / soft-skip.
``error``
    Human-readable error string on failure / skipped reason.
``duration_s``
    Wall time of the subprocess.
``skipped``
    True when the generator command was not found on PATH (FileNotFoundError
    or exit 127).  v1 ramp: callers treat this as a soft pass.
``reason``
    Human-readable skip reason when ``skipped=True``.

Soft-skip
---------
If ``generator_cmd[0]`` is not found (FileNotFoundError) or the subprocess
exits 127 (command-not-found shell error), the verb returns
``ok=True, skipped=True, reason="generator not installed"``.  This prevents
the post-hook from blocking missions that haven't yet integrated FastAPI /
openapi-typescript.

Slow-regen warning
------------------
If ``duration_s > 30`` the result carries ``warning="slow_regen: {n:.1f}s > 30s"``.
Surfaced in ctx by the verdict fn; never blocks.

File-missing case
-----------------
If ``target_path`` does not exist in the workspace, ``diff_present=True``
and ``diff_excerpt`` contains ``"file missing: {target_path}"``.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger
from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.regen_and_diff")

_SLOW_REGEN_THRESHOLD_S = 30.0
_DIFF_MAX_LINES = 50


async def regen_and_diff(
    mission_id: int | None,
    generator_cmd: list[str],
    target_path: str,
    workspace_path: str,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Regenerate content to stdout; diff against committed ``target_path``.

    Parameters
    ----------
    mission_id:
        Forwarded to ``run_cmd``.  May be None when ``workspace_path`` is set.
    generator_cmd:
        Argv list for the generator.  stdout = the regenerated content.
    target_path:
        Path (relative to workspace_path or absolute) to the committed file
        to compare against.
    workspace_path:
        Root of the mission workspace.  Used by ``run_cmd`` for cwd
        resolution AND for resolving a relative ``target_path``.
    timeout_s:
        Hard cap for the generator subprocess.  Default 60s.

    Returns
    -------
    dict — see module docstring.
    """
    _base: dict[str, Any] = {
        "ok": False,
        "diff_present": False,
        "diff_excerpt": "",
        "target_path": target_path,
        "generator_cmd": generator_cmd,
        "exit": -1,
        "error": None,
        "duration_s": 0.0,
        "skipped": False,
        "reason": None,
    }

    if not generator_cmd or not all(isinstance(a, str) for a in generator_cmd):
        return {**_base, "ok": False, "error": "generator_cmd must be a non-empty list of strings"}

    if not workspace_path:
        return {**_base, "ok": False, "error": "workspace_path is required"}

    # Run generator — captures stdout as the "regenerated" content.
    raw = await run_cmd(
        mission_id=mission_id,
        cmd=generator_cmd,
        cwd=None,  # workspace root
        timeout_s=timeout_s,
        require_exit_zero=False,
        workspace_path=workspace_path,
    )

    duration = float(raw.get("duration_s") or 0.0)
    exit_code = int(raw.get("exit", -1))
    spawn_error = raw.get("error")

    # Soft-skip: generator not installed.
    if spawn_error and "executable not found" in str(spawn_error):
        logger.warning(
            "regen_and_diff: generator not installed — soft-skip",
            cmd=generator_cmd[0],
        )
        return {
            **_base,
            "ok": True,
            "skipped": True,
            "reason": "generator not installed",
            "exit": exit_code,
            "duration_s": duration,
        }

    if exit_code == 127:
        logger.warning(
            "regen_and_diff: exit 127 (command not found) — soft-skip",
            cmd=generator_cmd[0],
        )
        return {
            **_base,
            "ok": True,
            "skipped": True,
            "reason": "generator not installed",
            "exit": 127,
            "duration_s": duration,
        }

    # Other spawn errors (OSError, etc.) → hard failure.
    if spawn_error and "executable not found" not in str(spawn_error):
        return {
            **_base,
            "ok": False,
            "error": f"generator spawn failed: {spawn_error}",
            "exit": exit_code,
            "duration_s": duration,
        }

    if raw.get("timed_out"):
        return {
            **_base,
            "ok": False,
            "error": f"generator timed out after {timeout_s}s",
            "exit": exit_code,
            "duration_s": duration,
        }

    generated_content: str = raw.get("stdout_tail") or ""

    # Resolve target path.
    abs_target: str
    if os.path.isabs(target_path):
        abs_target = target_path
    else:
        abs_target = str(Path(workspace_path) / target_path)

    # Read committed content.
    if not os.path.exists(abs_target):
        diff_excerpt = f"file missing: {target_path}"
        result = {
            **_base,
            "ok": True,
            "diff_present": True,
            "diff_excerpt": diff_excerpt,
            "exit": exit_code,
            "duration_s": duration,
        }
        _attach_slow_warning(result, duration)
        return result

    try:
        committed_content = Path(abs_target).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {
            **_base,
            "ok": False,
            "error": f"cannot read target: {exc}",
            "exit": exit_code,
            "duration_s": duration,
        }

    # Compute diff.
    diff_present = generated_content != committed_content
    diff_excerpt = ""
    if diff_present:
        gen_lines = generated_content.splitlines(keepends=True)
        cmt_lines = committed_content.splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(
            cmt_lines,
            gen_lines,
            fromfile=f"committed/{target_path}",
            tofile=f"regenerated/{target_path}",
            lineterm="",
        ))
        diff_excerpt = "".join(diff_lines[:_DIFF_MAX_LINES])
        if len(diff_lines) > _DIFF_MAX_LINES:
            diff_excerpt += f"\n... ({len(diff_lines) - _DIFF_MAX_LINES} more lines truncated)"

    result = {
        **_base,
        "ok": True,
        "diff_present": diff_present,
        "diff_excerpt": diff_excerpt,
        "exit": exit_code,
        "duration_s": duration,
    }
    _attach_slow_warning(result, duration)
    return result


def _attach_slow_warning(result: dict, duration: float) -> None:
    if duration > _SLOW_REGEN_THRESHOLD_S:
        result["warning"] = f"slow_regen: {duration:.1f}s > {_SLOW_REGEN_THRESHOLD_S:.0f}s"


__all__ = ["regen_and_diff"]
