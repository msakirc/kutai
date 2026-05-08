"""Run pytest under the mission workspace and parse its result honestly.

Mechanical executor. No LLM. Wraps the existing ``run_cmd`` primitive with
two extra concerns specific to test execution:

- pytest is invoked with ``--json-report --json-report-file=<tmp>`` when the
  ``pytest-json-report`` plugin is available (best-effort: if the plugin is
  missing, we fall back to parsing the stdout summary line).
- The verb's ``ok`` is True iff pytest exited 0 AND we can confirm at least
  one test ran AND the failed/errors counts are zero. A green exit code on
  zero collected tests counts as failure (silent skip is a regression class
  in itself).

This is the runner end of the build/run-then-verify pattern: an LLM step
authors test files, a mechanical sibling step runs them, the post-hook
machinery cascades the verdict back. The runner refuses to fabricate.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any

from src.infra.logging_config import get_logger

from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.run_pytest")


_COUNT_PATTERNS = {
    "passed": re.compile(r"(\d+)\s+passed", re.IGNORECASE),
    "failed": re.compile(r"(\d+)\s+failed", re.IGNORECASE),
    "errors": re.compile(r"(\d+)\s+errors?", re.IGNORECASE),
    "skipped": re.compile(r"(\d+)\s+skipped", re.IGNORECASE),
}


def _parse_json_report(path: str) -> dict[str, int] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    summary = data.get("summary") or {}
    return {
        "passed": int(summary.get("passed", 0)),
        "failed": int(summary.get("failed", 0)),
        "errors": int(summary.get("errors", 0)),
        "skipped": int(summary.get("skipped", 0)),
        "total": int(summary.get("total", 0) or summary.get("collected", 0)),
    }


def _parse_stdout_summary(stdout: str) -> dict[str, int]:
    """Last-ditch parser for the pytest summary line.

    pytest emits the summary on the final non-empty line. With ``-q`` the
    line has no ``===`` prefix; with the default it does. We scan from the
    end for the first line containing one of the count keywords.
    """
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "total": 0}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if not any(k in line for k in (
            "passed", "failed", "error", "skipped",
        )):
            continue
        for key, pat in _COUNT_PATTERNS.items():
            m = pat.search(line)
            if m:
                counts[key] = int(m.group(1))
        counts["total"] = (
            counts["passed"] + counts["failed"]
            + counts["errors"] + counts["skipped"]
        )
        if counts["total"] > 0:
            break
    return counts


async def run_pytest(
    mission_id: int | None,
    target: str | list[str] | None = None,
    cwd: str | None = None,
    timeout_s: float = 600.0,
    extra_args: list[str] | None = None,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Run pytest in the mission workspace.

    Parameters
    ----------
    target:
        File(s) or dir(s) to pass to pytest. ``None`` means run pytest with
        no targets (collects from cwd).
    cwd:
        Workspace-relative cwd. Resolved + jailed by ``run_cmd``.
    timeout_s:
        Hard cap; default 10 min. Tests that exceed this are killed.
    extra_args:
        Additional pytest flags (no shell expansion; argv list).

    Returns
    -------
    dict with keys: ``ok``, ``passed``, ``failed``, ``errors``, ``skipped``,
    ``total``, ``exit``, ``timed_out``, ``stdout_tail``, ``stderr_tail``,
    ``duration_s``, ``report_path`` (None if no JSON report).
    """
    targets: list[str]
    if target is None:
        targets = []
    elif isinstance(target, str):
        targets = [target]
    else:
        targets = list(target)

    # Best-effort JSON report — only enabled when pytest-json-report is
    # importable. With the plugin missing pytest exits 4 ("usage error") on
    # the unknown flag, so we must NOT pass it unconditionally.
    try:
        import pytest_jsonreport  # type: ignore  # noqa: F401
        _have_json_report = True
    except ImportError:
        try:
            import pytest_json_report  # type: ignore  # noqa: F401
            _have_json_report = True
        except ImportError:
            _have_json_report = False

    report_path: str | None = None
    if _have_json_report:
        report_fd, report_path = tempfile.mkstemp(suffix=".json", prefix="pytest_")
        os.close(report_fd)

    cmd = ["python", "-m", "pytest", "-q"]
    if _have_json_report and report_path:
        cmd.append("--json-report")
        cmd.append(f"--json-report-file={report_path}")
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

    counts: dict[str, int] | None = None
    report_used: str | None = None
    if report_path:
        counts = _parse_json_report(report_path)
        if counts is not None:
            report_used = report_path
        try:
            os.unlink(report_path)
        except OSError:
            pass
    if counts is None:
        counts = _parse_stdout_summary(raw.get("stdout_tail", "") or "")

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
        "stdout_tail": raw.get("stdout_tail", ""),
        "stderr_tail": raw.get("stderr_tail", ""),
        "duration_s": raw.get("duration_s", 0.0),
        "report_path": report_used,
        "error": spawn_error,
    }
