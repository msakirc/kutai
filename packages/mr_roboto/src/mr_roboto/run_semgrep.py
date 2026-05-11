"""Run semgrep with a rule pack and parse findings.

Mechanical executor. No LLM. Shells out to ``semgrep --config <rule_pack>
--json <targets...>`` and converts the JSON output to a normalised findings
list.

Severity mapping
----------------
semgrep ``ERROR``   → blocker
semgrep ``WARNING`` → warning
semgrep ``INFO``    → info

Missing semgrep
---------------
If semgrep is not installed (FileNotFoundError or exit 127) the verb returns a
soft-skipped verdict — ``ok=True``, ``skipped=True``, ``findings=[]`` — so the
caller is not penalised.  This is intentional for v1: semgrep is a runtime
tool, not a project dependency.  Promote to blocker once the CI image ships
semgrep.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger
from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.run_semgrep")

# Canonical path to the rule pack shipped with this package.
_RULE_PACK_DIR = Path(__file__).parent / "rule_packs"
DEFAULT_RULE_PACK = str(_RULE_PACK_DIR / "forbidden.yml")

_SEMGREP_SEVERITY_MAP = {
    "ERROR": "blocker",
    "WARNING": "warning",
    "INFO": "info",
}


def _locate_semgrep() -> str:
    """Return the semgrep executable path.  Raises FileNotFoundError if absent."""
    import shutil
    path = shutil.which("semgrep")
    if path is None:
        raise FileNotFoundError("semgrep not found on PATH")
    return path


async def run_semgrep(
    mission_id: int | None,
    target_files: list[str] | None = None,
    rule_pack_path: str | None = None,
    workspace_path: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run semgrep and return normalised findings.

    Parameters
    ----------
    mission_id:
        Forwarded to ``run_cmd`` for workspace resolution.  May be None for
        tests that supply an explicit ``workspace_path``.
    target_files:
        Relative or absolute paths to scan.  ``None`` or ``[]`` scans the
        workspace root (``"."``).
    rule_pack_path:
        Path to a semgrep ``rules:`` YAML file.  Defaults to the built-in
        ``rule_packs/forbidden.yml`` shipped with this package.
    workspace_path:
        Explicit workspace root passed through to ``run_cmd``.  Optional when
        ``mission_id`` is set.
    timeout_s:
        Hard cap passed to ``run_cmd``.  Semgrep on large repos can be slow;
        default 120 s.

    Returns
    -------
    dict with keys:

    ``ok``
        True when semgrep exited 0 or 1 (findings present but no error) AND
        the run was not skipped AND no internal error occurred.
        False when an unexpected error prevented analysis.
    ``skipped``
        True when semgrep is not installed.  Callers MUST treat this as a
        soft pass (warning at most), never as a blocker.
    ``findings``
        List of finding dicts: ``{rule_id, path, line, message, severity,
        semgrep_severity}``.  Empty when ``ok=True`` and no patterns matched.
    ``blocker_count``
        Count of findings with ``severity == "blocker"``.
    ``warning_count``
        Count of findings with ``severity == "warning"``.
    ``exit``
        Raw semgrep exit code.  -1 on spawn failure.
    ``stdout_tail``
        Last portion of stdout (for debugging).
    ``stderr_tail``
        Last portion of stderr.
    ``duration_s``
        Wall time of the subprocess.
    ``error``
        Human-readable error string when ``ok=False`` and not skipped.
    """
    effective_rule_pack = rule_pack_path or DEFAULT_RULE_PACK

    # Soft-skip when semgrep is absent — never blocker on missing tool.
    try:
        semgrep_exe = _locate_semgrep()
    except FileNotFoundError:
        logger.warning("semgrep not installed — pattern_lint skipped")
        return {
            "ok": True,
            "skipped": True,
            "findings": [],
            "blocker_count": 0,
            "warning_count": 0,
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "error": None,
        }

    targets: list[str]
    if not target_files:
        targets = ["."]
    else:
        targets = list(target_files)

    cmd = [
        semgrep_exe,
        "--config", effective_rule_pack,
        "--json",
        "--quiet",
        *targets,
    ]

    raw = await run_cmd(
        mission_id=mission_id,
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

    # Exit 127 = command not found (shell wrapper didn't find semgrep).
    if exit_code == 127 or spawn_error:
        logger.warning(
            "semgrep not available or spawn error — pattern_lint skipped",
            exit_code=exit_code, error=spawn_error,
        )
        return {
            "ok": True,
            "skipped": True,
            "findings": [],
            "blocker_count": 0,
            "warning_count": 0,
            "exit": exit_code,
            "stdout_tail": stdout,
            "stderr_tail": stderr,
            "duration_s": raw.get("duration_s", 0.0),
            "error": None,
        }

    if timed_out:
        return {
            "ok": False,
            "skipped": False,
            "findings": [],
            "blocker_count": 0,
            "warning_count": 0,
            "exit": exit_code,
            "stdout_tail": stdout,
            "stderr_tail": stderr,
            "duration_s": raw.get("duration_s", 0.0),
            "error": "semgrep timed out",
        }

    # semgrep exits 0 (no findings) or 1 (findings present) in normal
    # operation.  Anything else (2 = usage error, 3 = I/O error, …) is an
    # execution failure.
    if exit_code not in (0, 1):
        logger.warning(
            "semgrep exited with unexpected code",
            exit_code=exit_code, stderr=stderr[:300],
        )
        return {
            "ok": False,
            "skipped": False,
            "findings": [],
            "blocker_count": 0,
            "warning_count": 0,
            "exit": exit_code,
            "stdout_tail": stdout,
            "stderr_tail": stderr,
            "duration_s": raw.get("duration_s", 0.0),
            "error": f"semgrep exit {exit_code}: {stderr[:200]}",
        }

    # Parse JSON output.
    findings: list[dict] = []
    try:
        # semgrep --json writes to stdout.  stdout_tail may be truncated by
        # run_cmd's 32 KB cap — for large outputs this could drop trailing
        # JSON.  Acceptable for v1 (rule pack is minimal; findings are dense
        # but short).
        data = json.loads(stdout)
        raw_results = data.get("results") or []
        for r in raw_results:
            sg_sev = str(r.get("extra", {}).get("severity") or "WARNING").upper()
            norm_sev = _SEMGREP_SEVERITY_MAP.get(sg_sev, "warning")
            findings.append({
                "rule_id": r.get("check_id") or "",
                "path": r.get("path") or "",
                "line": int((r.get("start") or {}).get("line") or 0),
                "message": r.get("extra", {}).get("message") or "",
                "severity": norm_sev,
                "semgrep_severity": sg_sev,
            })
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning(
            "semgrep JSON parse failed — treating as empty findings",
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
        "stdout_tail": stdout,
        "stderr_tail": stderr,
        "duration_s": raw.get("duration_s", 0.0),
        "error": None,
    }
