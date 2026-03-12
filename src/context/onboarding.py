# context/onboarding.py
"""
Phase 12.6 — Project Onboarding.

When a project is registered via ``/project add <path>``, this module:
1. Detects language & framework
2. Runs structural index (tree-sitter)
3. Builds code embeddings
4. Generates repo map
5. Detects coding conventions
6. Stores a *project profile* dict for later injection into agent prompts
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


async def onboard_project(project_path: str, project_name: str = "") -> dict:
    """Run the full onboarding pipeline for a project.

    Returns a *project profile* dict suitable for storage and prompt
    injection.
    """
    root = os.path.normpath(project_path)
    if not os.path.isdir(root):
        return {"error": f"Path not found: {root}"}

    if not project_name:
        project_name = os.path.basename(root) or "unnamed"

    profile: dict = {
        "name": project_name,
        "path": root,
        "language": "unknown",
        "framework": None,
        "conventions": {},
        "repo_map_summary": "",
        "files_indexed": 0,
        "symbols_embedded": 0,
    }

    # ── Step 1: Detect language & framework ──
    try:
        from ..tools.workspace import detect_project as _detect_project
        detection = await _detect_project(root)
        if isinstance(detection, str):
            try:
                detection = json.loads(detection)
            except (json.JSONDecodeError, TypeError):
                detection = {}
        if isinstance(detection, dict):
            profile["language"] = detection.get("primary_language", "unknown")
            profile["framework"] = detection.get("framework")
    except Exception as exc:
        logger.debug(f"Language detection failed: {exc}")

    # ── Step 2: Structural index (tree-sitter) ──
    try:
        from ..tools.codebase_index import build_index, detect_conventions
        index = build_index(root)
        profile["files_indexed"] = len(index)

        # ── Step 5: Detect conventions ──
        conventions = detect_conventions(index)
        profile["conventions"] = conventions
    except Exception as exc:
        logger.debug(f"Structural indexing failed: {exc}")

    # ── Step 3: Code embeddings ──
    try:
        from ..parsing.code_embeddings import index_codebase
        embed_result = await index_codebase(root)
        profile["symbols_embedded"] = embed_result.get("symbols_embedded", 0)
    except Exception as exc:
        logger.debug(f"Code embedding failed (non-critical): {exc}")

    # ── Step 4: Repository map ──
    try:
        from ..context.repo_map import generate_repo_map, format_repo_map, save_repo_map
        repo_map = generate_repo_map(root)
        profile["repo_map_summary"] = format_repo_map(repo_map, max_lines=40)

        # Save full repo map to disk alongside the project
        map_path = os.path.join(root, ".repo_map.json")
        save_repo_map(repo_map, map_path)
    except Exception as exc:
        logger.debug(f"Repo map generation failed (non-critical): {exc}")

    return profile


async def store_project_profile(profile: dict) -> None:
    """Persist project profile to the database."""
    try:
        from ..db import get_db
        db = await get_db()

        # Create table if it doesn't exist
        await db.execute("""
            CREATE TABLE IF NOT EXISTS project_profiles (
                name TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                profile JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await db.execute(
            """INSERT OR REPLACE INTO project_profiles
               (name, path, profile, updated_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
            (profile["name"], profile["path"], json.dumps(profile)),
        )
        await db.commit()
        logger.info(f"Stored project profile: {profile['name']}")
    except Exception as exc:
        logger.warning(f"Failed to store project profile: {exc}")


async def load_project_profile(project_name: str) -> Optional[dict]:
    """Load a stored project profile from the database."""
    try:
        from ..db import get_db
        db = await get_db()
        cursor = await db.execute(
            "SELECT profile FROM project_profiles WHERE name = ?",
            (project_name,),
        )
        row = await cursor.fetchone()
        if row:
            return json.loads(row[0])
    except Exception as exc:
        logger.debug(f"Failed to load project profile: {exc}")
    return None


async def load_all_project_profiles() -> list[dict]:
    """Load all stored project profiles."""
    try:
        from ..db import get_db
        db = await get_db()
        cursor = await db.execute(
            "SELECT profile FROM project_profiles ORDER BY name"
        )
        rows = await cursor.fetchall()
        return [json.loads(row[0]) for row in rows]
    except Exception as exc:
        logger.debug(f"Failed to load project profiles: {exc}")
    return []


def format_project_profile(profile: dict) -> str:
    """Format a project profile for injection into agent system prompts."""
    if not profile:
        return ""

    parts = [f"## Project Profile: {profile.get('name', '?')}"]
    parts.append(f"- Language: {profile.get('language', 'unknown')}")

    fw = profile.get("framework")
    if fw:
        parts.append(f"- Framework: {fw}")

    parts.append(f"- Files indexed: {profile.get('files_indexed', 0)}")

    conv = profile.get("conventions", {})
    if conv and not conv.get("error"):
        parts.append(f"- Naming: {conv.get('naming_style', '?')}")
        if conv.get("has_docstrings"):
            parts.append(f"- Docstrings: yes ({conv.get('docstring_ratio', 0):.0%})")
        if conv.get("async_style"):
            parts.append("- Uses async/await")
        common = conv.get("common_imports", [])
        if common:
            parts.append(f"- Common imports: {', '.join(common[:5])}")
        avg_len = conv.get("avg_function_length", 0)
        if avg_len:
            parts.append(f"- Avg function length: {avg_len:.0f} lines")

    summary = profile.get("repo_map_summary", "")
    if summary:
        # Keep it compact — first 10 lines
        summary_lines = summary.strip().split("\n")[:10]
        parts.append(f"- Structure:\n  " + "\n  ".join(summary_lines))

    return "\n".join(parts)


async def get_project_profile_for_task(task: dict) -> Optional[dict]:
    """Detect which project a task belongs to and return its profile.

    Uses task context (workspace path) to match against stored profiles.
    """
    task_context = task.get("context")
    if isinstance(task_context, str):
        try:
            task_context = json.loads(task_context)
        except (json.JSONDecodeError, TypeError):
            task_context = {}
    if not isinstance(task_context, dict):
        task_context = {}

    # Check for explicit project name
    project_name = task_context.get("project")
    if project_name:
        return await load_project_profile(project_name)

    # Check for workspace path matching
    workspace = task_context.get("workspace") or task_context.get("project_path")
    if workspace:
        profiles = await load_all_project_profiles()
        for p in profiles:
            if os.path.normpath(p.get("path", "")) == os.path.normpath(workspace):
                return p

    return None
