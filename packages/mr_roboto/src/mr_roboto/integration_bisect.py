"""Integration bisect — binary-search commit list for regression introduction.

Given a list of commits and a failing test suite, narrows down the
smallest (commit_a, commit_b) pair where checking out commit_b causes the
suite to go red.

Mechanical executor. No LLM. Best-effort: returns ``{"breaking_pair": None}``
when the commit list is too short or bisect fails to isolate.
"""
from __future__ import annotations

import asyncio
import glob as _glob
import sys
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.integration_bisect")

_PYTEST_TIMEOUT = 120  # per-run budget during bisect


async def _git_checkout(sha: str, cwd: str) -> bool:
    """Checkout sha in detached-HEAD mode. Returns True on success."""
    proc = await asyncio.create_subprocess_exec(
        "git", "checkout", sha, "--detach",
        cwd=cwd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate()
    return (proc.returncode or 0) == 0


async def _run_suite(suite_glob: str, cwd: str) -> tuple[bool, str]:
    """Run suite_glob under cwd. Returns (passed, stdout_tail)."""
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", suite_glob]
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            ),
            timeout=_PYTEST_TIMEOUT,
        )
        out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=_PYTEST_TIMEOUT)
    except asyncio.TimeoutError:
        return False, "timed_out"
    except Exception as e:
        return False, str(e)

    out = out_b.decode("utf-8", errors="replace") if out_b else ""
    passed = (proc.returncode or 0) == 0
    return passed, out[-2000:]


async def integration_bisect(
    commits: list[str],
    suite_glob: str,
    workspace_path: str,
) -> dict[str, Any]:
    """Binary-search the commit list to find the breaking pair.

    Parameters
    ----------
    commits:
        Ordered list of SHAs (most-recent first, i.e. HEAD at index 0).
    suite_glob:
        Test glob to run at each candidate commit.
    workspace_path:
        Absolute path to git repo root.

    Returns
    -------
    ``{"breaking_pair": [commit_a, commit_b], "failing_test": str,
       "diagnostic": str}``
    or ``{"breaking_pair": None}`` when unable to isolate.
    """
    cwd = workspace_path

    if len(commits) < 2:
        return {"breaking_pair": None}

    import os
    if not os.path.isdir(os.path.join(cwd, ".git")):
        return {"breaking_pair": None}

    if not _glob.glob(os.path.join(cwd, suite_glob), recursive=True):
        return {"breaking_pair": None}

    # Verify that the suite is actually failing at head (first commit).
    await _git_checkout(commits[0], cwd=cwd)
    head_passed, head_diag = await _run_suite(suite_glob, cwd=cwd)
    if head_passed:
        # Suite passes at HEAD — nothing to bisect.
        return {"breaking_pair": None}

    # Check if the suite passes at the oldest commit in the list.
    await _git_checkout(commits[-1], cwd=cwd)
    oldest_passed, _ = await _run_suite(suite_glob, cwd=cwd)
    if not oldest_passed:
        # Already failing at oldest — can't isolate with this range.
        # Restore head.
        await _git_checkout(commits[0], cwd=cwd)
        return {"breaking_pair": None}

    # Binary search: find smallest index i such that suite passes at
    # commits[i] but fails at commits[i-1].
    lo, hi = 1, len(commits) - 1
    breaking_idx = lo

    while lo <= hi:
        mid = (lo + hi) // 2
        ok = await _git_checkout(commits[mid], cwd=cwd)
        if not ok:
            lo = mid + 1
            continue
        passed, _ = await _run_suite(suite_glob, cwd=cwd)
        if passed:
            # Suite is green here; the break is earlier (lower index = newer)
            breaking_idx = mid
            hi = mid - 1
        else:
            # Suite is red here too; break is older
            lo = mid + 1

    # Restore HEAD.
    await _git_checkout(commits[0], cwd=cwd)

    commit_a = commits[breaking_idx]      # last good
    commit_b = commits[breaking_idx - 1]  # first bad (more recent)

    # Collect diagnostic from running at the bad commit.
    await _git_checkout(commit_b, cwd=cwd)
    _, diag = await _run_suite(suite_glob, cwd=cwd)
    await _git_checkout(commits[0], cwd=cwd)

    # Extract first failing test name.
    failing_test = ""
    for line in (diag or "").splitlines():
        if "FAILED" in line:
            failing_test = line.strip()[:200]
            break

    return {
        "breaking_pair": [commit_a, commit_b],
        "failing_test": failing_test,
        "diagnostic": diag[:1000] if diag else "",
    }
