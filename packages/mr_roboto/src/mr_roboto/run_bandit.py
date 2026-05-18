"""Run bandit over Python files and return normalised findings.

Mechanical executor — no LLM.  Shells out to ``bandit -f json`` and converts
the JSON output to the shared findings schema used by ``security_review``.

Severity mapping
----------------
bandit HIGH/MEDIUM/LOW confidence × HIGH/MEDIUM/LOW severity maps to:
  severity HIGH   → blocker
  severity MEDIUM → warning
  severity LOW    → info

Missing bandit
--------------
If bandit is not installed (FileNotFoundError or exit 127) the function returns
a soft-skipped result — ``skipped=True``, ``findings=[]`` — so the caller is
not penalised.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger
from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.run_bandit")

_BANDIT_SEVERITY_MAP = {
    "HIGH": "blocker",
    "MEDIUM": "warning",
    "LOW": "info",
}

_SKIPPED_RESULT: dict[str, Any] = {
    "ok": True,
    "skipped": True,
    "findings": [],
    "blocker_count": 0,
    "warning_count": 0,
    "exit": -1,
    "error": None,
    "duration_s": 0.0,
}


def _locate_bandit() -> str:
    import shutil
    path = shutil.which("bandit")
    if path is None:
        raise FileNotFoundError("bandit not found on PATH")
    return path


async def run_bandit(
    target_files: list[str] | None = None,
    workspace_path: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run bandit on Python files and return normalised findings.

    Parameters
    ----------
    target_files:
        Relative or absolute paths to Python files to scan.  ``None`` or ``[]``
        scans the workspace root (``"."``).
    workspace_path:
        Explicit workspace root forwarded to ``run_cmd``.
    timeout_s:
        Hard cap for the subprocess.

    Returns
    -------
    dict with keys:
        ok           – True on success or soft-skip
        skipped      – True when bandit not installed
        findings     – list of {severity, file, line, why, source: "bandit"}
        blocker_count
        warning_count
        exit
        error
        duration_s
    """
    # Filter to .py files only.
    py_files: list[str]
    if not target_files:
        py_files = ["."]
    else:
        py_files = [f for f in target_files if f.endswith(".py")]
        if not py_files:
            # No Python files — nothing to scan, soft-skip.
            logger.debug("run_bandit: no Python files in target_files — skipping")
            return {**_SKIPPED_RESULT, "skipped": True, "error": None}

    try:
        bandit_exe = _locate_bandit()
    except FileNotFoundError:
        logger.warning("bandit not installed — security_review bandit pass skipped")
        return {**_SKIPPED_RESULT}

    cmd = [bandit_exe, "-f", "json", "-q", *py_files]

    raw = await run_cmd(
        mission_id=None,
        cmd=cmd,
        cwd=None,
        timeout_s=timeout_s,
        require_exit_zero=False,
        workspace_path=workspace_path,
    )

    exit_code = int(raw.get("exit", -1))
    timed_out = bool(raw.get("timed_out"))
    spawn_error = raw.get("error")
    stdout = raw.get("stdout_tail") or ""
    stderr = raw.get("stderr_tail") or ""
    duration_s = float(raw.get("duration_s") or 0.0)

    if exit_code == 127 or spawn_error:
        logger.warning(
            "bandit not available or spawn error — skipped",
            exit_code=exit_code, error=spawn_error,
        )
        return {**_SKIPPED_RESULT, "duration_s": duration_s}

    if timed_out:
        return {
            "ok": False,
            "skipped": False,
            "findings": [],
            "blocker_count": 0,
            "warning_count": 0,
            "exit": exit_code,
            "error": "bandit timed out",
            "duration_s": duration_s,
        }

    # bandit exits 0 (no issues), 1 (issues found), 2 (usage error).
    if exit_code not in (0, 1):
        logger.warning("bandit exited with unexpected code", exit_code=exit_code)
        return {
            "ok": False,
            "skipped": False,
            "findings": [],
            "blocker_count": 0,
            "warning_count": 0,
            "exit": exit_code,
            "error": f"bandit exit {exit_code}: {stderr[:200]}",
            "duration_s": duration_s,
        }

    findings: list[dict] = []
    try:
        data = json.loads(stdout)
        for result in (data.get("results") or []):
            sev = str(result.get("issue_severity") or "LOW").upper()
            norm_sev = _BANDIT_SEVERITY_MAP.get(sev, "info")
            findings.append({
                "severity": norm_sev,
                "file": result.get("filename") or "",
                "line": int(result.get("line_number") or 0),
                "why": result.get("issue_text") or "",
                "source": "bandit",
                "rule_id": result.get("test_id") or "",
            })
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning(
            "bandit JSON parse failed — treating as empty findings",
            exc=str(exc), stdout_head=stdout[:200],
        )

    blocker_count = sum(1 for f in findings if f["severity"] == "blocker")
    warning_count = sum(1 for f in findings if f["severity"] == "warning")

    return {
        "ok": True,
        "skipped": False,
        "findings": findings,
        "blocker_count": blocker_count,
        "warning_count": warning_count,
        "exit": exit_code,
        "error": None,
        "duration_s": duration_s,
    }
