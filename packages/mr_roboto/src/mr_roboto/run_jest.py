"""Run Jest under the mission workspace and parse its result honestly.

Mechanical executor. No LLM. Thin wrapper around the existing ``run_cmd``
primitive. Shells out via ``npx jest --json`` so the output is machine-
readable without the plugin-availability dance required by pytest-json-report.

Return contract mirrors ``run_pytest`` exactly:
    ok, passed, failed, errors, skipped, total, exit, timed_out,
    stdout_tail, stderr_tail, duration_s, report_path
"""

from __future__ import annotations

import json
import re
from typing import Any

from yazbunu import get_logger

from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.run_jest")

_COUNT_PATTERNS = {
    "passed": re.compile(r"(\d+)\s+passed", re.IGNORECASE),
    "failed": re.compile(r"(\d+)\s+failed", re.IGNORECASE),
    "skipped": re.compile(r"(\d+)\s+(?:skipped|pending)", re.IGNORECASE),
}


def _parse_jest_json(stdout: str) -> dict[str, int] | None:
    """Best-effort parse of Jest --json output.

    Jest writes a single JSON object to stdout when invoked with ``--json``.
    The top-level keys we care about are ``numPassedTests``,
    ``numFailedTests``, ``numPendingTests``, ``numTotalTests``.
    Returns None if stdout doesn't look like Jest JSON.
    """
    text = stdout.strip()
    # Jest JSON starts with '{' and has the discriminator key
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # stdout may contain extra lines before the JSON blob; try to
        # extract the last {...} block (Jest sometimes logs setup noise first).
        try:
            start = text.rfind("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                data = json.loads(text[start : end + 1])
            else:
                return None
        except json.JSONDecodeError:
            return None

    if "numTotalTests" not in data:
        return None

    return {
        "passed": int(data.get("numPassedTests", 0)),
        "failed": int(data.get("numFailedTests", 0)),
        "errors": int(data.get("numRuntimeErrorTestSuites", 0)),
        "skipped": int(data.get("numPendingTests", 0)),
        "total": int(data.get("numTotalTests", 0)),
    }


def _parse_stdout_summary(stdout: str) -> dict[str, int]:
    """Last-ditch fallback: scrape Jest text output for counts."""
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


async def run_jest(
    mission_id: int | None,
    target: str | list[str] | None = None,
    cwd: str | None = None,
    timeout_s: float = 600.0,
    extra_args: list[str] | None = None,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Run Jest in the mission workspace.

    Parameters
    ----------
    target:
        Test file(s) or glob pattern(s) to pass to Jest.  ``None`` runs
        discovery from ``cwd``.
    cwd:
        Workspace-relative cwd. Resolved + jailed by ``run_cmd``.
    timeout_s:
        Hard cap; default 10 min.
    extra_args:
        Additional Jest flags (argv list; no shell expansion).

    Returns
    -------
    dict with keys: ``ok``, ``passed``, ``failed``, ``errors``, ``skipped``,
    ``total``, ``exit``, ``timed_out``, ``stdout_tail``, ``stderr_tail``,
    ``duration_s``, ``report_path`` (always None for Jest — JSON inline).
    """
    targets: list[str]
    if target is None:
        targets = []
    elif isinstance(target, str):
        targets = [target]
    else:
        targets = list(target)

    # --json emits machine-readable output on stdout; --no-coverage avoids
    # slow instrumentation in the post-hook runner.
    cmd = ["npx", "jest", "--json", "--no-coverage"]
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
    counts = _parse_jest_json(stdout) or _parse_stdout_summary(stdout)

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
