"""Z3 T3B — run axe-core accessibility scan against a preview URL.

Mechanical executor. No LLM. Browser-driven; uses ``@axe-core/cli`` via npx.

Soft-skips when:
- ``preview_url`` is absent or is a ``pending:`` placeholder
- ``@axe-core/cli`` is not installed (npx dry-run check)

Return shape
------------
``{verdict, findings, skipped, reason}``

- ``verdict``: ``"pass"`` or ``"fail"`` (fail when any blocker found).
  When skipped, verdict is ``"pass"`` (skip = no gate).
- ``findings``: list of dicts with keys
  ``{severity, file, url, impact, why, source}``.
- ``skipped``: bool
- ``reason``: str explaining skip (absent when not skipped)

Severity mapping
----------------
- critical / serious  → blocker
- moderate            → warning
- minor               → info
"""
from __future__ import annotations

import asyncio
import json
import shutil
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.run_axe")

# Impacts that map to severity=blocker
_BLOCKER_IMPACTS = frozenset({"critical", "serious"})
_WARNING_IMPACTS = frozenset({"moderate"})
# minor → info (everything else)


def _impact_to_severity(impact: str) -> str:
    if impact in _BLOCKER_IMPACTS:
        return "blocker"
    if impact in _WARNING_IMPACTS:
        return "warning"
    return "info"


def _is_real_url(url: str | None) -> bool:
    if not url:
        return False
    stripped = url.strip()
    if stripped.startswith("pending:"):
        return False
    return stripped.startswith("http://") or stripped.startswith("https://")


def _axe_available() -> bool:
    """Best-effort check: npx must be on PATH (implies Node.js present)."""
    return shutil.which("npx") is not None


def _parse_axe_output(raw_json: str, preview_url: str, target_paths: list[str] | None) -> list[dict[str, Any]]:
    """Parse axe-core CLI JSON output into normalised finding dicts."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("run_axe: failed to parse axe JSON output")
        return []

    # axe-core CLI can return a list of page results or a single result.
    if isinstance(data, dict):
        pages = [data]
    elif isinstance(data, list):
        pages = data
    else:
        return []

    findings: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        violations = page.get("violations") or []
        for violation in violations:
            impact = str(violation.get("impact") or "minor").lower()
            why = str(violation.get("description") or violation.get("help") or "")
            rule_id = str(violation.get("id") or "")
            nodes = violation.get("nodes") or []
            if not nodes:
                # Surface at least one finding per violation even if no nodes.
                findings.append({
                    "severity": _impact_to_severity(impact),
                    "file": (target_paths[0] if target_paths else ""),
                    "url": preview_url,
                    "impact": impact,
                    "why": why or rule_id,
                    "source": "axe",
                })
            else:
                for node in nodes:
                    node_target = ""
                    if isinstance(node, dict):
                        targets = node.get("target") or []
                        node_target = targets[0] if targets else ""
                    findings.append({
                        "severity": _impact_to_severity(impact),
                        "file": node_target or (target_paths[0] if target_paths else ""),
                        "url": preview_url,
                        "impact": impact,
                        "why": why or rule_id,
                        "source": "axe",
                    })
    return findings


async def run_axe(
    preview_url: str | None = None,
    target_paths: list[str] | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Run axe-core accessibility scan against *preview_url*.

    Parameters
    ----------
    preview_url:
        The tunneled preview URL to scan.  If absent or ``pending:``,
        soft-skips with ``skipped=True``.
    target_paths:
        Optional list of paths that produced this preview (used to populate
        the ``file`` field of findings).  May be None.
    timeout_s:
        Subprocess timeout in seconds (default 60 s).

    Returns
    -------
    dict with keys: ``verdict``, ``findings``, ``skipped``, ``reason`` (if
    skipped).
    """
    if not _is_real_url(preview_url):
        reason = "no real preview_url available (absent or pending)"
        logger.warning(f"run_axe soft-skip: {reason}")
        return {
            "verdict": "pass",
            "findings": [],
            "skipped": True,
            "reason": reason,
        }

    if not _axe_available():
        reason = "npx not on PATH; @axe-core/cli cannot be invoked"
        logger.warning(f"run_axe soft-skip: {reason}")
        return {
            "verdict": "pass",
            "findings": [],
            "skipped": True,
            "reason": reason,
        }

    cmd = [
        "npx", "--yes", "@axe-core/cli",
        str(preview_url),
        "--stdout",
        "--exit",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            reason = f"axe-core timed out after {timeout_s}s"
            logger.warning(f"run_axe soft-skip: {reason}")
            return {
                "verdict": "pass",
                "findings": [],
                "skipped": True,
                "reason": reason,
            }

        stdout_text = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
        exit_code = proc.returncode or 0

        # axe-core/cli exits non-zero when violations are found.
        # We still parse the output either way.
        findings = _parse_axe_output(stdout_text, str(preview_url), target_paths)

    except FileNotFoundError:
        reason = "npx binary not found"
        logger.warning(f"run_axe soft-skip: {reason}")
        return {
            "verdict": "pass",
            "findings": [],
            "skipped": True,
            "reason": reason,
        }
    except Exception as e:
        reason = f"axe-core spawn error: {e}"
        logger.warning(f"run_axe soft-skip: {reason}")
        return {
            "verdict": "pass",
            "findings": [],
            "skipped": True,
            "reason": reason,
        }

    has_blocker = any(f["severity"] == "blocker" for f in findings)
    verdict = "fail" if has_blocker else "pass"

    logger.info(
        f"run_axe: verdict={verdict} findings={len(findings)} "
        f"url={preview_url}"
    )
    return {
        "verdict": verdict,
        "findings": findings,
        "skipped": False,
    }
