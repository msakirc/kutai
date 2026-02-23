# tools/git_ops.py
"""
Git integration for the workspace.
Init, commit, branch, log, diff, rollback — with subdirectory support.
"""

import asyncio
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORKSPACE_DIR: str = os.environ.get(
    "WORKSPACE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace")),
)

DEFAULT_GITIGNORE = (
    "__pycache__/\n*.pyc\n*.pyo\nnode_modules/\n.env\nvenv/\n"
    ".venv/\ndist/\nbuild/\n*.egg-info/\n.DS_Store\nThumbs.db\n"
)

MAX_DIFF_CHARS = 5000
MAX_COMMIT_MSG_LEN = 200


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------
def _resolve_repo(path: str = "") -> Optional[str]:
    """
    Resolve *path* to an absolute directory under WORKSPACE_DIR.
    Returns None if the result would escape the workspace.
    """
    joined = os.path.normpath(os.path.join(WORKSPACE_DIR, path)) if path else WORKSPACE_DIR
    real = os.path.realpath(joined)
    ws_real = os.path.realpath(WORKSPACE_DIR)
    if real == ws_real or real.startswith(ws_real + os.sep):
        return real
    return None


# ---------------------------------------------------------------------------
# Internal runner
# ---------------------------------------------------------------------------
async def _run_git(
    args: list[str],
    cwd: Optional[str] = None,
) -> tuple[int, str, str]:
    """
    Run a git command and return (returncode, stdout, stderr).

    All three values are always populated; stdout and stderr are
    decoded strings (never None).
    """
    cwd = cwd or WORKSPACE_DIR
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    raw_out, raw_err = await proc.communicate()
    return (
        proc.returncode,
        raw_out.decode(errors="replace").strip(),
        raw_err.decode(errors="replace").strip(),
    )


# ---------------------------------------------------------------------------
# Repo initialisation
# ---------------------------------------------------------------------------
async def git_init(path: str = "") -> str:
    """
    Initialize a git repo in the workspace (or a subdirectory).
    Creates a sensible .gitignore and an initial commit.
    Idempotent — safe to call on an existing repo.

    Args:
        path: Subdirectory relative to workspace root (default: root).

    Returns:
        Status message.
    """
    target = _resolve_repo(path)
    if target is None:
        return "❌ Access denied: path is outside workspace."

    os.makedirs(target, exist_ok=True)

    # Already a repo?
    if os.path.isdir(os.path.join(target, ".git")):
        return f"ℹ️ Git repo already exists at {path or 'workspace root'}."

    code, out, err = await _run_git(["init"], cwd=target)
    if code != 0:
        return f"❌ git init failed: {err}"

    # Configure user identity if not set (required for commits)
    await _run_git(
        ["config", "user.email", "orchestrator@local"], cwd=target,
    )
    await _run_git(
        ["config", "user.name", "Orchestrator"], cwd=target,
    )

    # Create .gitignore
    gitignore_path = os.path.join(target, ".gitignore")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w") as f:
            f.write(DEFAULT_GITIGNORE)

    # Initial commit (--allow-empty so it succeeds even in empty dirs)
    await _run_git(["add", "-A"], cwd=target)
    code, out, err = await _run_git(
        ["commit", "-m", "Initial commit by orchestrator", "--allow-empty"],
        cwd=target,
    )
    if code != 0:
        return f"❌ Initial commit failed: {err}"

    return f"✅ Git repo initialized at {path or 'workspace root'}."


async def _ensure_repo(path: str = "") -> Optional[str]:
    """
    Resolve *path* and verify it contains a git repo.
    Returns the absolute path, or None (after logging) if invalid.
    """
    target = _resolve_repo(path)
    if target is None:
        return None
    if not os.path.isdir(os.path.join(target, ".git")):
        return None
    return target
async def ensure_git_repo() -> str:
    """Initialize git repo in workspace if not already."""
    git_dir = os.path.join(WORKSPACE_ROOT, ".git")
    if os.path.isdir(git_dir):
        return "Git repo already initialized."

    code, output = await _run_git(["init"])
    if code != 0:
        return f"❌ git init failed: {output}"

    # Create .gitignore
    gitignore_path = os.path.join(WORKSPACE_ROOT, ".gitignore")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w") as f:
            f.write("__pycache__/\n*.pyc\nnode_modules/\n.env\n.venv/\nvenv/\n")

    await _run_git(["add", "-A"])
    await _run_git(["commit", "-m", "Initial commit by orchestrator"])
    return "✅ Git repo initialized with initial commit."

# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------
async def git_commit(
    message: str,
    path: str = "",
    add_all: bool = True,
) -> str:
    """
    Stage and commit changes.

    Args:
        message:  Commit message (truncated to 200 chars).
        path:     Subdirectory containing the repo.
        add_all:  If True, run `git add -A` first.

    Returns:
        Status message.
    """
    target = _resolve_repo(path)
    if target is None:
        return "❌ Access denied: path is outside workspace."

    # Auto-init if not a repo yet
    if not os.path.isdir(os.path.join(target, ".git")):
        init_result = await git_init(path)
        if init_result.startswith("❌"):
            return init_result

    safe_msg = message[:MAX_COMMIT_MSG_LEN]

    if add_all:
        await _run_git(["add", "-A"], cwd=target)

    # Anything staged?
    code, status_out, _ = await _run_git(["status", "--porcelain"], cwd=target)
    if not status_out:
        return "ℹ️ Nothing to commit (workspace clean)."

    code, out, err = await _run_git(
        ["commit", "-m", safe_msg], cwd=target,
    )
    if code != 0:
        return f"❌ Commit failed: {err}"

    return f"✅ Committed: {safe_msg}\n{out}"


# ---------------------------------------------------------------------------
# Branch
# ---------------------------------------------------------------------------
async def git_branch(
    branch_name: str,
    path: str = "",
) -> str:
    """
    Create and switch to a new branch, or switch to an existing one.

    Args:
        branch_name: Name of the branch.
        path:        Subdirectory containing the repo.

    Returns:
        Status message.
    """
    target = await _ensure_repo(path)
    if target is None:
        return f"❌ No git repo at {path or 'workspace root'}. Run git_init first."

    # Try to create new branch
    code, out, err = await _run_git(
        ["checkout", "-b", branch_name], cwd=target,
    )
    if code == 0:
        return f"✅ Created and switched to branch '{branch_name}'."

    # Branch may already exist — try plain checkout
    code, out, err = await _run_git(
        ["checkout", branch_name], cwd=target,
    )
    if code == 0:
        return f"✅ Switched to existing branch '{branch_name}'."

    return f"❌ Branch operation failed: {err}"


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------
async def git_log(
    path: str = "",
    count: int = 10,
) -> str:
    """
    Show recent commit log (one-line format).

    Args:
        path:  Subdirectory containing the repo.
        count: Number of commits to show.

    Returns:
        Commit log or status message.
    """
    target = await _ensure_repo(path)
    if target is None:
        return f"❌ No git repo at {path or 'workspace root'}."

    count = max(1, min(count, 100))  # clamp to sane range

    code, out, err = await _run_git(
        ["log", "--oneline", "--no-color", f"-{count}"],
        cwd=target,
    )
    if code != 0:
        return f"❌ git log failed: {err}"

    return out or "(no commits yet)"


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------
async def git_diff(
    path: str = "",
    stat_only: bool = False,
) -> str:
    """
    Show current uncommitted changes.

    Args:
        path:      Subdirectory containing the repo.
        stat_only: If True, show diffstat summary instead of full diff.

    Returns:
        Diff output (truncated if very large).
    """
    target = await _ensure_repo(path)
    if target is None:
        return f"❌ No git repo at {path or 'workspace root'}."

    stat_flag = ["--stat"] if stat_only else []

    # Try diff against HEAD (staged + unstaged)
    code, out, err = await _run_git(
        ["diff", "HEAD"] + stat_flag, cwd=target,
    )
    if code != 0:
        # HEAD may not exist yet (no commits); fall back to plain diff
        code, out, err = await _run_git(
            ["diff"] + stat_flag, cwd=target,
        )

    if not out:
        # Maybe only staged changes (no HEAD yet)
        _, out, _ = await _run_git(
            ["diff", "--staged"] + stat_flag, cwd=target,
        )

    if not out:
        return "(no changes)"

    if len(out) > MAX_DIFF_CHARS:
        out = out[:MAX_DIFF_CHARS] + f"\n\n… (diff truncated — {len(out)} chars total)"

    return out


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------
async def git_rollback(
    steps: int = 1,
    path: str = "",
) -> str:
    """
    Soft-reset the last N commits. Files are preserved; commits are undone.
    The changes become staged so they can be re-committed.

    Args:
        steps: Number of commits to undo (default 1).
        path:  Subdirectory containing the repo.

    Returns:
        Status message.
    """
    target = await _ensure_repo(path)
    if target is None:
        return f"❌ No git repo at {path or 'workspace root'}."

    steps = max(1, min(steps, 50))  # clamp to sane range

    # Verify there are enough commits to roll back
    code, log_out, _ = await _run_git(
        ["rev-list", "--count", "HEAD"], cwd=target,
    )
    if code != 0:
        return "❌ Cannot rollback: no commits exist yet."

    try:
        total_commits = int(log_out)
    except ValueError:
        return f"❌ Cannot determine commit count: {log_out}"

    if steps >= total_commits:
        return (
            f"❌ Cannot roll back {steps} commit(s) — only {total_commits} exist. "
            f"Maximum rollback is {total_commits - 1}."
        )

    code, out, err = await _run_git(
        ["reset", "--soft", f"HEAD~{steps}"], cwd=target,
    )
    if code != 0:
        return f"❌ Rollback failed: {err}"

    return f"✅ Rolled back {steps} commit(s). Changes are staged for re-commit."


# ---------------------------------------------------------------------------
# Status (bonus — useful for agents)
# ---------------------------------------------------------------------------
async def git_status(path: str = "") -> str:
    """
    Show the current branch and working-tree status.

    Args:
        path: Subdirectory containing the repo.

    Returns:
        Status output.
    """
    target = await _ensure_repo(path)
    if target is None:
        return f"❌ No git repo at {path or 'workspace root'}."

    code, out, err = await _run_git(
        ["status", "--short", "--branch"], cwd=target,
    )
    if code != 0:
        return f"❌ git status failed: {err}"

    return out or "(clean)"
