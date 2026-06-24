"""Integration replay — re-run test suite against current + N prior commits.

Mechanical executor. No LLM. Three modes:

- quick:    spot-check only tests touched by the current HEAD's diff.
- standard: full suite_glob against current commit only (one run).
- strict:   full suite_glob against current + N prior commits in random
            shuffle order (shuffle_seed for determinism).

Returns a verdict dict with findings per commit pair.

Soft-skips when:
- workspace_path has no ``.git/`` directory.
- no test files match suite_glob.
"""
from __future__ import annotations

import asyncio
import glob
import os
import random
import sys
import tempfile
import time
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.integration_replay")

_DEFAULT_N_PRIOR = 3
_PYTEST_TIMEOUT = 60  # per-test timeout flag passed to pytest


async def _git_cmd(args: list[str], cwd: str) -> tuple[int, str, str]:
    """Run a git command, return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout_b.decode("utf-8", errors="replace").strip(),
        stderr_b.decode("utf-8", errors="replace").strip(),
    )


async def _run_pytest(suite_glob: str, cwd: str, timeout_s: float) -> dict[str, Any]:
    """Run pytest on suite_glob in cwd. Returns ok + counts + stdout_tail."""
    import glob as _glob

    # Collect test files matching the glob (relative to cwd).
    abs_patterns = os.path.join(cwd, suite_glob)
    matched = _glob.glob(abs_patterns, recursive=True)
    if not matched:
        return {
            "ok": True,
            "skipped": True,
            "reason": f"no test files match {suite_glob!r}",
            "passed": 0,
            "failed": 0,
            "stdout_tail": "",
        }

    cmd = [
        sys.executable, "-m", "pytest",
        "-q",
        f"--timeout={_PYTEST_TIMEOUT}",
        "--tb=short",
        suite_glob,
    ]
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            ),
            timeout=timeout_s,
        )
        stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return {
            "ok": False,
            "timed_out": True,
            "passed": 0,
            "failed": 0,
            "stdout_tail": "",
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "passed": 0, "failed": 0, "stdout_tail": ""}

    stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
    tail = stdout[-4096:] if len(stdout) > 4096 else stdout

    # Parse summary line.
    passed = failed = 0
    for line in reversed(stdout.splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            import re
            m_pass = re.search(r"(\d+)\s+passed", line)
            m_fail = re.search(r"(\d+)\s+(failed|error)", line)
            if m_pass:
                passed = int(m_pass.group(1))
            if m_fail:
                failed = int(m_fail.group(1))
            if passed or failed:
                break

    exit_code = proc.returncode or 0
    ok = (exit_code == 0 and failed == 0)
    return {
        "ok": ok,
        "exit": exit_code,
        "passed": passed,
        "failed": failed,
        "stdout_tail": tail,
    }


async def _get_prior_commits(cwd: str, n: int) -> list[str]:
    """Return up to n prior commit SHAs before HEAD."""
    rc, out, _ = await _git_cmd(
        ["log", "--format=%H", f"-{n + 1}"],
        cwd=cwd,
    )
    if rc != 0:
        return []
    shas = [s.strip() for s in out.splitlines() if s.strip()]
    # shas[0] is HEAD; return next n
    return shas[1 : n + 1]


async def _get_head_diff_files(cwd: str) -> list[str]:
    """Return files touched by HEAD diff vs parent."""
    rc, out, _ = await _git_cmd(
        ["diff", "--name-only", "HEAD~1", "HEAD"],
        cwd=cwd,
    )
    if rc != 0:
        # Might be first commit; diff against empty tree.
        rc2, out2, _ = await _git_cmd(
            ["diff", "--name-only", "4b825dc642cb6eb9a060e54bf8d69288fbee4904", "HEAD"],
            cwd=cwd,
        )
        if rc2 != 0:
            return []
        out = out2
    return [f.strip() for f in out.splitlines() if f.strip()]


async def integration_replay(
    commits: list[str],
    suite_glob: str,
    shuffle_seed: int,
    mode: str,
    workspace_path: str | None = None,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Run test suite against commits in the given mode.

    Parameters
    ----------
    commits:
        For strict mode: list of SHAs to replay (in addition to HEAD).
        Empty = use last _DEFAULT_N_PRIOR commits.
    suite_glob:
        Glob pattern for test files (e.g. ``"tests/integration/**"``).
    shuffle_seed:
        RNG seed for strict-mode shuffle (use mission_id for determinism).
    mode:
        One of ``"quick"``, ``"standard"``, ``"strict"``.
    workspace_path:
        Path to git repo root. Defaults to cwd when None.
    timeout_s:
        Total wall-clock budget across all runs.

    Returns
    -------
    dict with verdict, findings, mode, shuffle_seed, commits_replayed,
    skipped, reason.
    """
    if mode not in ("quick", "standard", "strict"):
        mode = "standard"

    cwd = workspace_path or os.getcwd()

    # Soft-skip when no .git present.
    if not os.path.isdir(os.path.join(cwd, ".git")):
        return {
            "verdict": "pass",
            "skipped": True,
            "reason": "no .git directory — not a git repo",
            "findings": [],
            "mode": mode,
            "shuffle_seed": shuffle_seed,
            "commits_replayed": [],
        }

    # Pre-check: any test files?
    import glob as _glob
    if not _glob.glob(os.path.join(cwd, suite_glob), recursive=True):
        return {
            "verdict": "pass",
            "skipped": True,
            "reason": f"no test files match {suite_glob!r}",
            "findings": [],
            "mode": mode,
            "shuffle_seed": shuffle_seed,
            "commits_replayed": [],
        }

    # Resolve HEAD sha.
    rc_head, head_sha, _ = await _git_cmd(["rev-parse", "HEAD"], cwd=cwd)
    if rc_head != 0:
        return {
            "verdict": "pass",
            "skipped": True,
            "reason": "could not resolve HEAD",
            "findings": [],
            "mode": mode,
            "shuffle_seed": shuffle_seed,
            "commits_replayed": [],
        }
    head_sha = head_sha.strip()

    findings: list[dict] = []
    commits_replayed: list[str] = []

    if mode == "quick":
        # Only test files touched by HEAD diff.
        diff_files = await _get_head_diff_files(cwd=cwd)
        test_files = [
            f for f in diff_files
            if _glob.fnmatch.fnmatch(f, suite_glob) or "test" in f.lower()
        ]
        if not test_files:
            # Nothing to spot-check.
            return {
                "verdict": "pass",
                "skipped": False,
                "reason": "quick: no test files in HEAD diff",
                "findings": [],
                "mode": mode,
                "shuffle_seed": shuffle_seed,
                "commits_replayed": [head_sha],
            }
        quick_glob = " ".join(test_files)
        result = await _run_pytest(test_files[0] if len(test_files) == 1 else suite_glob, cwd=cwd, timeout_s=min(timeout_s, 120.0))
        commits_replayed = [head_sha]
        if not result.get("ok") and not result.get("skipped"):
            findings.append({
                "severity": "blocker",
                "commit_pair": [head_sha, head_sha],
                "test_id": "quick_spot_check",
                "why": result.get("stdout_tail", "")[-500:],
            })

    elif mode == "standard":
        result = await _run_pytest(suite_glob, cwd=cwd, timeout_s=timeout_s)
        commits_replayed = [head_sha]
        if not result.get("ok") and not result.get("skipped"):
            findings.append({
                "severity": "blocker",
                "commit_pair": [head_sha, head_sha],
                "test_id": "standard_full_suite",
                "why": result.get("stdout_tail", "")[-500:],
            })

    else:  # strict
        # Build commit list: HEAD + provided commits or last N prior.
        if commits:
            replay_shas = [head_sha] + list(commits)
        else:
            prior = await _get_prior_commits(cwd=cwd, n=_DEFAULT_N_PRIOR)
            replay_shas = [head_sha] + prior

        # Shuffle deterministically.
        rng = random.Random(shuffle_seed)
        shuffled = list(replay_shas)
        rng.shuffle(shuffled)

        # Budget time per commit.
        per_commit_budget = timeout_s / max(len(shuffled), 1)

        # Save current HEAD so we can restore.
        _, stash_out, _ = await _git_cmd(["stash", "list"], cwd=cwd)
        original_branch_rc, original_ref, _ = await _git_cmd(
            ["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd
        )

        for sha in shuffled:
            commits_replayed.append(sha)
            # Checkout this commit in detached HEAD state.
            co_rc, _, co_err = await _git_cmd(["checkout", sha, "--detach"], cwd=cwd)
            if co_rc != 0:
                findings.append({
                    "severity": "warning",
                    "commit_pair": [sha, sha],
                    "test_id": "checkout_failed",
                    "why": co_err[:300],
                })
                continue
            try:
                result = await _run_pytest(suite_glob, cwd=cwd, timeout_s=per_commit_budget)
            except Exception as e:
                result = {"ok": False, "stdout_tail": str(e)}

            if not result.get("ok") and not result.get("skipped"):
                findings.append({
                    "severity": "blocker",
                    "commit_pair": [head_sha, sha],
                    "test_id": f"suite_at_{sha[:8]}",
                    "why": result.get("stdout_tail", "")[-500:],
                })

        # Restore HEAD.
        await _git_cmd(["checkout", head_sha, "--detach"], cwd=cwd)
        if original_ref and original_ref not in ("HEAD", ""):
            await _git_cmd(["checkout", original_ref], cwd=cwd)

    verdict = "fail" if any(f["severity"] == "blocker" for f in findings) else "pass"
    return {
        "verdict": verdict,
        "findings": findings,
        "mode": mode,
        "shuffle_seed": shuffle_seed,
        "commits_replayed": commits_replayed,
        "skipped": False,
        "reason": None,
    }
