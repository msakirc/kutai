"""Workspace snapshot executor: compute file hashes + git SHA + branch, save to DB.

Ported from src/core/mechanical/workspace_snapshot.py during Phase 2a.
"""

from src.infra.db import save_workspace_snapshot
from src.tools.git_ops import get_commit_sha, get_current_branch
from src.tools.workspace import compute_workspace_hashes
from src.infra.logging_config import get_logger

logger = get_logger("salako.workspace_snapshot")


async def snapshot_workspace(
    mission_id: int,
    task_id: int,
    workspace_path: str,
    repo_path: str | None = None,
) -> dict | None:
    """Snapshot workspace state. Returns dict with hashes/branch/commit_sha, or None on failure.

    Failure is non-fatal (logged at debug). Caller should proceed.
    """
    try:
        hashes = compute_workspace_hashes(workspace_path)
        sha = await get_commit_sha(path=repo_path or workspace_path)
        branch = await get_current_branch(path=repo_path or workspace_path)
        await save_workspace_snapshot(
            mission_id=mission_id,
            file_hashes=hashes,
            task_id=task_id,
            branch_name=branch,
            commit_sha=sha,
        )
        return {"hashes": hashes, "commit_sha": sha, "branch": branch}
    except Exception as e:
        logger.debug(f"snapshot skipped task={task_id}: {e}")
        return None
