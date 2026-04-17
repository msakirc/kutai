"""Git auto-commit executor. Dormant in Phase 1 — i2p refactor will re-wire.

Moved from src/core/orchestrator.py _auto_commit(). Behavior unchanged;
only the call site was disconnected from the orchestrator main loop.

Phase 2a: invoked via Dispatch(executor='mechanical', payload={'action': 'git_commit', ...}).
"""

from src.tools.git_ops import git_commit, ensure_git_repo
from src.tools.workspace import get_mission_workspace_relative
from src.infra.logging_config import get_logger

logger = get_logger("core.mechanical.git_commit")


async def auto_commit(task: dict, result: dict):
    """Auto-commit workspace changes after a successful coder task.

    Absorbed from orchestrator._auto_commit(). Call site disconnected in
    Phase 1; this function is dormant until i2p workflow refactor re-wires it.
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
