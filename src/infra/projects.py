# infra/projects.py
"""
Phase 7.1 — Project Registry

DB-backed project registry. Every goal can link to a project.
Table: projects {id, name, description, language, framework, repo_path, workspace_path, status, created_at, updated_at}
"""
from __future__ import annotations
import json
from typing import Optional

from .logging_config import get_logger
from .db import get_db

logger = get_logger("infra.projects")

STATUS_ACTIVE = "active"
STATUS_COMPLETED = "completed"
STATUS_ARCHIVED = "archived"


async def _ensure_table(db) -> None:
    await db.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT '',
            language TEXT DEFAULT '',
            framework TEXT DEFAULT '',
            repo_path TEXT DEFAULT '',
            workspace_path TEXT DEFAULT '',
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Add project_id column to goals if it doesn't exist
    try:
        await db.execute("ALTER TABLE goals ADD COLUMN project_id INTEGER REFERENCES projects(id)")
        await db.commit()
    except Exception:
        pass  # Column already exists


async def create_project(
    name: str,
    description: str = "",
    language: str = "",
    framework: str = "",
    repo_path: str = "",
    workspace_path: str = "",
) -> dict:
    """Create a new project. Returns the created project dict."""
    db = await get_db()
    await _ensure_table(db)
    cursor = await db.execute(
        """INSERT INTO projects (name, description, language, framework, repo_path, workspace_path)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (name, description, language, framework, repo_path, workspace_path),
    )
    await db.commit()
    project_id = cursor.lastrowid
    logger.info("Project created", project_id=project_id, name=name)
    return await get_project(project_id)


async def get_project(project_id: int) -> Optional[dict]:
    """Get a project by ID."""
    db = await get_db()
    await _ensure_table(db)
    cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    row = await cursor.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


async def get_project_by_name(name: str) -> Optional[dict]:
    """Get a project by name."""
    db = await get_db()
    await _ensure_table(db)
    cursor = await db.execute("SELECT * FROM projects WHERE name = ?", (name,))
    row = await cursor.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


async def list_projects(status: Optional[str] = None) -> list[dict]:
    """List all projects, optionally filtered by status."""
    db = await get_db()
    await _ensure_table(db)
    if status:
        cursor = await db.execute("SELECT * FROM projects WHERE status = ? ORDER BY updated_at DESC", (status,))
    else:
        cursor = await db.execute("SELECT * FROM projects ORDER BY updated_at DESC")
    rows = await cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in rows]


async def update_project(project_id: int, **kwargs) -> None:
    """Update project fields."""
    if not kwargs:
        return
    db = await get_db()
    await _ensure_table(db)
    kwargs["updated_at"] = "CURRENT_TIMESTAMP"
    set_clause = ", ".join(f"{k} = ?" for k in kwargs if k != "updated_at")
    set_clause += ", updated_at = CURRENT_TIMESTAMP"
    values = [v for k, v in kwargs.items() if k != "updated_at"] + [project_id]
    await db.execute(f"UPDATE projects SET {set_clause} WHERE id = ?", values)
    await db.commit()


async def link_goal_to_project(goal_id: int, project_id: int) -> None:
    """Associate a goal with a project."""
    db = await get_db()
    await _ensure_table(db)
    await db.execute("UPDATE goals SET project_id = ? WHERE id = ?", (project_id, goal_id))
    await db.commit()


async def get_project_goals(project_id: int) -> list[dict]:
    """Get all goals linked to a project."""
    db = await get_db()
    await _ensure_table(db)
    cursor = await db.execute(
        "SELECT * FROM goals WHERE project_id = ? ORDER BY created_at DESC", (project_id,)
    )
    rows = await cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in rows]


def format_project_status_badge(status: str) -> str:
    badges = {"active": "🟢", "completed": "✅", "archived": "📦"}
    return badges.get(status, "⚪")
