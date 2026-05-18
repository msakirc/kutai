"""Run Vitest under the mission workspace and parse its result honestly.

Mechanical executor. No LLM. Thin wrapper around the existing ``run_cmd``
primitive. Shells out via ``npx vitest run --reporter=json`` so the output
is machine-readable.

Return contract mirrors ``run_pytest`` exactly:
    ok, passed, failed, errors, skipped, total, exit, timed_out,
    stdout_tail, stderr_tail, duration_s, report_path
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.infra.logging_config import get_logger

from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.run_vitest")

_COUNT_PATTERNS = {
    "passed": re.compile(r"(\d+)\s+passed", re.IGNORECASE),
    "failed": re.compile(r"(\d+)\s+failed", re.IGNORECASE),
    "skipped": re.compile(r"(\d+)\s+skipped", re.IGNORECASE),
}


def _parse_vitest_json(stdout: str) -> dict[str, int] | None:
    """Best-effort parse of Vitest --reporter=json output.

    Vitest JSON reporter writes a JSON object with a ``testResults`` array.
    Top-level aggregate keys: ``numPassedTests``, ``numFailedTests``,
    ``numPendingTests``, ``numTotalTests`` — same shape as Jest.
    Returns None if stdout doesn't look like Vitest JSON.
    """
    text = stdout.strip()
    if not text.startswith("{"):
        # Vitest may have ANSI preamble; scan for the first '{'
        idx = text.find("{")
        if idx == -1:
            return None
        text = text[idx:]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try last balanced block
        try:
            start = text.rfind("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                data = json.loads(text[start : end + 1])
            else:
                return None
        except json.JSONDecodeError:
            return None

    # Vitest v1+ uses numTotalTests; v0.x uses testResults array only
    if "numTotalTests" in data:
        return {
            "passed": int(data.get("numPassedTests", 0)),
            "failed": int(data.get("numFailedTests", 0)),
            "errors": 0,  # Vitest collapses errors into failed
            "skipped": int(data.get("numPendingTests", 0)),
            "total": int(data.get("numTotalTests", 0)),
        }

    # Fallback: sum across testResults array
    if "testResults" in data:
        passed = failed = skipped = 0
        for suite in data.get("testResults") or []:
            for t in suite.get("assertionResults") or []:
                status = t.get("status", "")
                if status == "passed":
                    passed += 1
                elif status == "failed":
                    failed += 1
                elif status in ("pending", "skipped", "todo"):
                    skipped += 1
        total = passed + failed + skipped
        if total > 0:
            return {
                "passed": passed,
                "failed": failed,
                "errors": 0,
                "skipped": skipped,
                "total": total,
            }

    return None


def _parse_stdout_summary(stdout: str) -> dict[str, int]:
    """Last-ditch fallback: scrape Vitest text output for counts."""
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "total": 0}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if not any(k in line.lower() for k in ("passed", "failed", "skipped")):
            continue
        for key, pat in _COUNT_PATTERNS.items():
            m = pat.search(line)
            if m:
                counts[key] = int(m.group(1))
        counts["total"] = counts["passed"] + counts["failed"] + counts["skipped"]
        if counts["total"] > 0:
            break
    return counts


async def run_vitest(
    mission_id: int | None,
    target: str | list[str] | None = None,
    cwd: str | None = None,
    timeout_s: float = 600.0,
    extra_args: list[str] | None = None,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Run Vitest in the mission workspace.

    Parameters
    ----------
    target:
        Test file(s) to pass to Vitest.  ``None`` runs discovery from
        ``cwd``.
    cwd:
        Workspace-relative cwd. Resolved + jailed by ``run_cmd``.
    timeout_s:
        Hard cap; default 10 min.
    extra_args:
        Additional Vitest flags (argv list; no shell expansion).

    Returns
    -------
    dict with keys: ``ok``, ``passed``, ``failed``, ``errors``, ``skipped``,
    ``total``, ``exit``, ``timed_out``, ``stdout_tail``, ``stderr_tail``,
    ``duration_s``, ``report_path`` (always None for Vitest — JSON inline).
    """
    targets: list[str]
    if target is None:
        targets = []
    elif isinstance(target, str):
        targets = [target]
    else:
        targets = list(target)

    # ``vitest run`` is the non-watch invocation; --reporter=json gives
    # machine-readable output on stdout.
    cmd = ["npx", "vitest", "run", "--reporter=json"]
    if extra_args:
        cmd.extend(str(a) for a in extra_args)
    cmd.extend(targets)

    raw = await run_cmd(
        mission_id=mission_id,
        cmd=cmd,
        cwd=cwd,
        timeout_s=timeout_s,
        require_exit_zero=False,
        workspace_path=workspace_path,
    )

    stdout = raw.get("stdout_tail", "") or ""
    counts = _parse_vitest_json(stdout) or _parse_stdout_summary(stdout)

    timed_out = bool(raw.get("timed_out"))
    exit_code = int(raw.get("exit", -1))
    spawn_error = raw.get("error")

    if spawn_error or timed_out:
        ok = False
    else:
        ok = (
            exit_code == 0
            and counts["total"] > 0
            and counts["failed"] == 0
            and counts["errors"] == 0
        )

    return {
        "ok": ok,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "errors": counts["errors"],
        "skipped": counts["skipped"],
        "total": counts["total"],
        "exit": exit_code,
        "timed_out": timed_out,
        "stdout_tail": stdout,
        "stderr_tail": raw.get("stderr_tail", ""),
        "duration_s": raw.get("duration_s", 0.0),
        "report_path": None,
        "error": spawn_error,
    }
