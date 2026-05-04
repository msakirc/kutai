"""Git auto-commit executor.

Ported from src/core/mechanical/git_commit.py during Phase 2a. Invoked via
`salako.run({"executor": "mechanical", "payload": {"action": "git_commit", ...}})`.
"""

from src.tools.git_ops import git_commit, ensure_git_repo
from src.tools.workspace import get_mission_workspace_relative
from src.infra.logging_config import get_logger

logger = get_logger("salako.git_commit")


async def auto_commit(task: dict, result: dict) -> dict:
    """Auto-commit workspace changes after a successful coder task.

    Ported from orchestrator._auto_commit() → core.mechanical.git_commit.

    Returns a dict describing the outcome:

    - ``{"committed": True,  "message": "...", "empty": False}`` on a real commit
    - ``{"committed": False, "message": "...", "empty": True}`` when nothing changed
    - ``{"committed": False, "error": "..."}`` on exception (still swallowed for the
      legacy fire-and-forget callers; salako.run() inspects this dict)

    Exceptions are caught and logged at debug to preserve "best-effort" semantics
    for existing callers that ignore the return value.
    """
    try:
        # Use mission-specific workspace path if available
        mission_id = task.get("mission_id")
        repo_path = (
            get_mission_workspace_relative(mission_id) if mission_id else ""
        )
        await ensure_git_repo(repo_path)
        commit_msg = f"Task #{task['id']}: {task.get('title', 'untitled')[:60]}"
        commit_result = await git_commit(commit_msg, path=repo_path)
        empty = "Nothing to commit" in commit_result
        if not empty:
            logger.info(f"[Task #{task['id']}] Auto-committed: {commit_msg}")
        return {"committed": not empty, "message": commit_msg, "empty": empty}
    except Exception as e:
        logger.debug(f"Auto-commit skipped: {e}")
        return {"committed": False, "error": str(e)}
