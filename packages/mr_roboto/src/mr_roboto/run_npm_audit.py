"""Run ``npm audit`` over a workspace and return normalised findings.

Mechanical executor — no LLM.  Shells out to ``npm audit --json`` and converts
the JSON output to the shared findings schema used by ``security_review``.

Severity mapping (npm → beckman)
---------------------------------
critical  → blocker
high      → blocker
moderate  → warning
low       → info

Soft-skip conditions
--------------------
- ``npm`` not on PATH
- No ``package.json`` in the workspace root
- ``npm audit`` exits with code 127 (npm absent in shell)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger
from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.run_npm_audit")

_NPM_SEVERITY_MAP = {
    "critical": "blocker",
    "high": "blocker",
    "moderate": "warning",
    "low": "info",
    "info": "info",
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


def _locate_npm() -> str:
    import shutil
    path = shutil.which("npm")
    if path is None:
        raise FileNotFoundError("npm not found on PATH")
    return path


def _has_package_json(workspace_path: str | None) -> bool:
    if not workspace_path:
        return Path("package.json").exists()
    return (Path(workspace_path) / "package.json").exists()


async def run_npm_audit(
    workspace_path: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run ``npm audit --json`` and return normalised findings.

    Parameters
    ----------
    workspace_path:
        Directory containing package.json. Defaults to cwd.
    timeout_s:
        Hard cap for the subprocess.

    Returns
    -------
    dict with keys:
        ok           – True on success or soft-skip
        skipped      – True when npm absent or no package.json
        findings     – list of {severity, file, line, why, source: "npm_audit"}
        blocker_count
        warning_count
        exit
        error
        duration_s
    """
    # Soft-skip when no package.json.
    if not _has_package_json(workspace_path):
        logger.debug("run_npm_audit: no package.json found — skipping")
        return {**_SKIPPED_RESULT}

    try:
        npm_exe = _locate_npm()
    except FileNotFoundError:
        logger.warning("npm not installed — security_review npm audit pass skipped")
        return {**_SKIPPED_RESULT}

    cmd = [npm_exe, "audit", "--json"]

    raw = await run_cmd(
        mission_id=None,
        cmd=cmd,
        cwd=workspace_path or None,
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
            "npm not available or spawn error — skipped",
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
            "error": "npm audit timed out",
            "duration_s": duration_s,
        }

    # npm audit exits 0 (no vulns), 1 (vulns found). Other codes are errors.
    if exit_code not in (0, 1):
        logger.warning("npm audit exited with unexpected code", exit_code=exit_code)
        return {
            "ok": False,
            "skipped": False,
            "findings": [],
            "blocker_count": 0,
            "warning_count": 0,
            "exit": exit_code,
            "error": f"npm audit exit {exit_code}: {stderr[:200]}",
            "duration_s": duration_s,
        }

    findings: list[dict] = []
    try:
        data = json.loads(stdout)
        # npm audit JSON v2 (npm >= 7): top-level "vulnerabilities" dict.
        # npm audit JSON v1 (npm < 7): "advisories" dict.
        vulnerabilities = data.get("vulnerabilities") or {}
        advisories = data.get("advisories") or {}

        # v2 format
        for pkg_name, vuln in vulnerabilities.items():
            sev = str(vuln.get("severity") or "low").lower()
            norm_sev = _NPM_SEVERITY_MAP.get(sev, "info")
            # Collect via array
            via = vuln.get("via") or []
            advisory_titles = []
            for v in via:
                if isinstance(v, dict):
                    advisory_titles.append(v.get("title") or v.get("name") or "")
                elif isinstance(v, str):
                    advisory_titles.append(v)
            why = "; ".join(t for t in advisory_titles if t) or f"severity={sev}"
            findings.append({
                "severity": norm_sev,
                "file": "package.json",
                "line": 0,
                "why": f"{pkg_name}: {why}",
                "source": "npm_audit",
                "rule_id": f"npm/{pkg_name}",
            })

        # v1 format (fallback)
        for adv_id, advisory in advisories.items():
            sev = str(advisory.get("severity") or "low").lower()
            norm_sev = _NPM_SEVERITY_MAP.get(sev, "info")
            title = advisory.get("title") or advisory.get("module_name") or str(adv_id)
            findings.append({
                "severity": norm_sev,
                "file": "package.json",
                "line": 0,
                "why": title,
                "source": "npm_audit",
                "rule_id": f"npm/{advisory.get('module_name') or adv_id}",
            })

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning(
            "npm audit JSON parse failed — treating as empty findings",
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
