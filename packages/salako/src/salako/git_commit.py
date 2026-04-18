"""Git auto-commit executor.

Ported from src/core/mechanical/git_commit.py during Phase 2a. Invoked via
`salako.run({"executor": "mechanical", "payload": {"action": "git_commit", ...}})`.
"""

from src.tools.git_ops import git_commit, ensure_git_repo
from src.tools.workspace import get_mission_workspace_relative
from src.infra.logging_config import get_logger

logger = get_logger("salako.git_commit")


async def auto_commit(task: dict, result: dict):
    """Auto-commit workspace changes after a successful coder task.

    Ported verbatim from orchestrator._auto_commit() → core.mechanical.git_commit.
    Exceptions are swallowed and logged at debug level; callers continue.
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
        if "Nothing to commit" not in commit_result:
            logger.info(f"[Task #{task['id']}] Auto-committed: {commit_msg}")
    except Exception as e:
        logger.debug(f"Auto-commit skipped: {e}")
