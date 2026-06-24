"""Composite security-review verb (Z3 T3A).

Aggregates findings from three tools:
  1. semgrep with security.yml rule pack (OWASP Top10 + secret patterns)
  2. bandit   (Python-specific SAST)
  3. npm audit (JavaScript/Node.js dependency vulnerabilities)

Each tool runs independently.  Missing tools are soft-skipped (never block).
Verdict is ``"fail"`` when ANY finding has severity ``"blocker"``; otherwise
``"pass"``.

Severity normalisation across all sources
------------------------------------------
semgrep:   ERROR → blocker,  WARNING → warning,  INFO → info
bandit:    HIGH  → blocker,  MEDIUM  → warning,   LOW  → info
npm audit: critical/high → blocker, moderate → warning, low → info
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.security_review")

# Path to the security rule pack shipped with this package.
_SECURITY_RULE_PACK = str(
    Path(__file__).parent / "rule_packs" / "security.yml"
)


async def security_review(
    target_files: list[str] | None = None,
    workspace_path: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run composite security review and return aggregated findings.

    Parameters
    ----------
    target_files:
        Paths to scan.  ``None`` / ``[]`` scans the workspace root.
    workspace_path:
        Explicit workspace root.  Used for npm audit and as semgrep/bandit cwd.
    timeout_s:
        Per-tool timeout cap (each tool gets this budget independently).

    Returns
    -------
    dict:
        verdict      – ``"pass"`` or ``"fail"`` (fail when any blocker found)
        findings     – combined list of {severity, file, line, why, source, rule_id}
        tools_used   – list of tool names actually invoked (not skipped)
        blocker_count
        warning_count
        semgrep_skipped
        bandit_skipped
        npm_audit_skipped
    """
    from mr_roboto.run_semgrep import run_semgrep as _run_semgrep
    from mr_roboto.run_bandit import run_bandit as _run_bandit
    from mr_roboto.run_npm_audit import run_npm_audit as _run_npm_audit

    all_findings: list[dict] = []
    tools_used: list[str] = []

    # ------------------------------------------------------------------ semgrep
    try:
        sg_result = await _run_semgrep(
            mission_id=None,
            target_files=target_files,
            rule_pack_path=_SECURITY_RULE_PACK,
            workspace_path=workspace_path,
            timeout_s=timeout_s,
        )
        semgrep_skipped = bool(sg_result.get("skipped"))
        if not semgrep_skipped:
            tools_used.append("semgrep")
        # Normalise: semgrep findings already have severity in blocker/warning/info.
        # Add source field for each finding.
        for f in (sg_result.get("findings") or []):
            all_findings.append({
                "severity": f.get("severity", "info"),
                "file": f.get("path", ""),
                "line": f.get("line", 0),
                "why": f.get("message", ""),
                "source": "semgrep",
                "rule_id": f.get("rule_id", ""),
            })
    except Exception as exc:
        logger.warning("semgrep sub-run failed — skipped", exc=str(exc))
        semgrep_skipped = True

    # ------------------------------------------------------------------ bandit
    # Only run on Python files.
    py_files: list[str] | None
    if target_files:
        py_files = [f for f in target_files if f.endswith(".py")] or None
    else:
        py_files = None  # will scan workspace root

    try:
        bandit_result = await _run_bandit(
            target_files=py_files,
            workspace_path=workspace_path,
            timeout_s=timeout_s,
        )
        bandit_skipped = bool(bandit_result.get("skipped"))
        if not bandit_skipped:
            tools_used.append("bandit")
        all_findings.extend(bandit_result.get("findings") or [])
    except Exception as exc:
        logger.warning("bandit sub-run failed — skipped", exc=str(exc))
        bandit_skipped = True

    # ------------------------------------------------------------------ npm audit
    try:
        npm_result = await _run_npm_audit(
            workspace_path=workspace_path,
            timeout_s=timeout_s,
        )
        npm_audit_skipped = bool(npm_result.get("skipped"))
        if not npm_audit_skipped:
            tools_used.append("npm_audit")
        all_findings.extend(npm_result.get("findings") or [])
    except Exception as exc:
        logger.warning("npm_audit sub-run failed — skipped", exc=str(exc))
        npm_audit_skipped = True

    # ------------------------------------------------------------------ aggregate
    blocker_count = sum(1 for f in all_findings if f.get("severity") == "blocker")
    warning_count = sum(1 for f in all_findings if f.get("severity") == "warning")

    verdict = "fail" if blocker_count > 0 else "pass"

    return {
        "verdict": verdict,
        "findings": all_findings,
        "tools_used": tools_used,
        "blocker_count": blocker_count,
        "warning_count": warning_count,
        "semgrep_skipped": semgrep_skipped,
        "bandit_skipped": bandit_skipped,
        "npm_audit_skipped": npm_audit_skipped,
    }
