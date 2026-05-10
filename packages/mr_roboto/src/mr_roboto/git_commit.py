"""Git auto-commit executor.

Ported from src/core/mechanical/git_commit.py during Phase 2a. Invoked via
``mr_roboto.run({"executor": "mechanical", "payload": {"action": "git_commit", ...}})``.

Z10 T1A — atomic commit↔push contract.
====================================
The verb now treats ``commit`` and ``push`` as a single transaction:

  * Default (``allow_orphan=False``): a successful local commit followed by
    a failed ``git push`` is rolled back via ``git reset --soft HEAD~1``.
    The result reports ``status='failed'``, ``pushed=False``, and the
    ``commit_sha`` of the (now reverted) commit.
  * Opt-in (``allow_orphan=True``): the local commit is kept; result is
    ``status='partial'`` with ``pushed=False`` and ``commit_sha`` of the
    surviving local commit. Useful for offline development sessions.
  * Push disabled (``push=False`` in payload): legacy best-effort behaviour
    — commit-only, ``status='completed'``, ``pushed=False``,
    ``commit_sha`` always present on a real commit.

Result dict shape (always populated):

    {
        "committed": bool,
        "empty":     bool,
        "message":   str,
        "pushed":    bool,
        "commit_sha": str | None,
        "status":    "completed" | "failed" | "partial" | "skipped",
        "error":     str | None,    # populated on failure paths
    }
"""

from src.tools.git_ops import (
    git_commit,
    ensure_git_repo,
    git_push,
    git_reset_soft_one,
    git_head_sha,
)
from src.tools.workspace import get_mission_workspace_relative
from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.git_commit")


async def auto_commit(task: dict, result: dict) -> dict:
    """Auto-commit workspace changes, optionally pairing with an atomic push.

    Args
    ----
    task   : task dict (must have ``id``; ``mission_id`` and ``title``
             optional).
    result : prior agent result; reserved for future provenance hooks.

    Push contract (read from ``task["payload"]`` when present, falling back
    to safe defaults):

      * ``push``         (bool, default False)        — opt in to push
      * ``allow_orphan`` (bool, default False)        — keep local commit
                                                        on push failure
      * ``remote``       (str,  default "origin")
      * ``branch``       (str | None, default None)   — push HEAD when None

    Existing callers that don't set ``push`` keep getting commit-only
    behaviour (back-compat).
    """
    payload = task.get("payload") or {}
    push_enabled = bool(payload.get("push", False))
    allow_orphan = bool(payload.get("allow_orphan", False))
    remote = payload.get("remote", "origin")
    branch = payload.get("branch")

    try:
        mission_id = task.get("mission_id")
        repo_path = (
            get_mission_workspace_relative(mission_id) if mission_id else ""
        )
        await ensure_git_repo(repo_path)
        commit_msg = f"Task #{task['id']}: {task.get('title', 'untitled')[:60]}"
        commit_result = await git_commit(commit_msg, path=repo_path)
        empty = "Nothing to commit" in commit_result
        committed = not empty

        commit_sha = await git_head_sha(repo_path) if committed else None

        out = {
            "committed":  committed,
            "empty":      empty,
            "message":    commit_msg,
            "pushed":     False,
            "commit_sha": commit_sha,
            "status":     "completed",
            "error":      None,
        }

        if not committed:
            if not empty:
                out["status"] = "failed"
                out["error"] = commit_result
            return out

        logger.info(f"[Task #{task['id']}] Auto-committed: {commit_msg}")

        if not push_enabled:
            return out

        # Atomic push attempt.
        code, stdout, stderr = await git_push(
            path=repo_path, remote=remote, branch=branch
        )
        if code == 0:
            out["pushed"] = True
            return out

        # Push failed — decide between rollback and orphan-kept partial.
        push_err = (stderr or stdout or f"git push exited {code}").strip()
        if allow_orphan:
            out["status"] = "partial"
            out["error"] = f"push failed (orphan kept): {push_err}"
            logger.warning(
                f"[Task #{task['id']}] push failed; keeping local commit "
                f"(allow_orphan=True): {push_err}"
            )
            return out

        # Default: rollback the local commit so the caller's state matches
        # the remote. Staged changes survive (--soft).
        rcode, _rout, rerr = await git_reset_soft_one(repo_path)
        if rcode != 0:
            out["status"] = "failed"
            out["error"] = (
                f"push failed and rollback failed: push_err={push_err}; "
                f"reset_err={rerr}"
            )
            logger.error(out["error"])
            return out

        out["committed"] = False
        out["status"] = "failed"
        out["error"] = f"push failed; local commit reverted: {push_err}"
        logger.warning(
            f"[Task #{task['id']}] push failed; local commit reverted: "
            f"{push_err}"
        )
        return out
    except Exception as e:
        logger.debug(f"Auto-commit skipped: {e}")
        return {
            "committed":  False,
            "empty":      False,
            "message":    "",
            "pushed":     False,
            "commit_sha": None,
            "status":     "failed",
            "error":      str(e),
        }
