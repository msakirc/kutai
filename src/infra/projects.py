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


async def list_missions_with_repo() -> list[dict]:
    """Return missions that have a repo_path set."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, title, repo_path, language, framework, status FROM missions WHERE repo_path != '' ORDER BY created_at DESC"
    )
    return [dict(row) for row in await cursor.fetchall()]


def format_project_status_badge(status: str) -> str:
    badges = {"active": "🟢", "completed": "✅", "archived": "📦"}
    return badges.get(status, "⚪")
