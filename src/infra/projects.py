# infra/projects.py
"""
Project context helpers — operates on missions table columns.

The separate `projects` table has been retired. Project metadata
(repo_path, language, framework) now lives directly on missions.
These helpers provide a convenient API for project-related queries.
"""
from __future__ import annotations
from typing import Optional

from .logging_config import get_logger
from .db import get_db, update_mission

logger = get_logger("infra.projects")


async def set_mission_project(
    mission_id: int,
    repo_path: str = "",
    language: str = "",
    framework: str = "",
) -> None:
    """Attach project/codebase context to a mission."""
    kwargs = {}
    if repo_path:
        kwargs["repo_path"] = repo_path
    if language:
        kwargs["language"] = language
    if framework:
        kwargs["framework"] = framework
    if kwargs:
        await update_mission(mission_id, **kwargs)


async def get_missions_by_repo(repo_path: str) -> list[dict]:
    """Find missions linked to a specific codebase."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM missions WHERE repo_path = ? AND repo_path != '' ORDER BY created_at DESC",
        (repo_path,),
    )
    return [dict(row) for row in await cursor.fetchall()]


async def detect_project_info(path: str) -> dict:
    """Detect language/framework from a filesystem path (best-effort)."""
    import os
    info = {"repo_path": path, "language": "", "framework": ""}
    if os.path.exists(os.path.join(path, "package.json")):
        info["language"] = "javascript"
    elif os.path.exists(os.path.join(path, "pyproject.toml")) or os.path.exists(os.path.join(path, "setup.py")):
        info["language"] = "python"
    elif os.path.exists(os.path.join(path, "go.mod")):
        info["language"] = "go"
    elif os.path.exists(os.path.join(path, "Cargo.toml")):
        info["language"] = "rust"
    return info


# ── Backward compatibility ──────────────────────────────────────────────────
# These stubs prevent ImportError in modules that still import old functions.
# They log deprecation warnings and delegate to the new API where possible.

async def create_project(**kwargs) -> dict:
    """DEPRECATED: Projects are now mission metadata."""
    logger.warning("create_project() is deprecated — use add_mission() with repo_path/language/framework")
    return {"id": 0, "name": kwargs.get("name", ""), "status": "deprecated"}

async def get_project(project_id: int) -> Optional[dict]:
    """DEPRECATED."""
    return None

async def get_project_by_name(name: str) -> Optional[dict]:
    """DEPRECATED."""
    return None

async def list_projects(status: Optional[str] = None) -> list[dict]:
    """DEPRECATED: Query missions with repo_path instead."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, title as name, repo_path, language, framework, status FROM missions WHERE repo_path != '' ORDER BY created_at DESC"
    )
    return [dict(row) for row in await cursor.fetchall()]

async def update_project(project_id: int, **kwargs) -> None:
    """DEPRECATED."""
    pass

async def link_goal_to_project(goal_id: int, project_id: int) -> None:
    """DEPRECATED: Use set_mission_project() instead."""
    pass

async def get_project_goals(project_id: int) -> list[dict]:
    """DEPRECATED."""
    return []

def format_project_status_badge(status: str) -> str:
    badges = {"active": "🟢", "completed": "✅", "archived": "📦"}
    return badges.get(status, "⚪")
