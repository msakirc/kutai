# collaboration/blackboard.py
"""
Phase 13.1 — Shared Blackboard.

Per-goal structured state store with typed entries.  Backed by a
``blackboards`` DB table (goal_id → data JSON).  Agents read/write
structured data instead of parsing prior_steps text blobs.

Schema per goal::

    {
      "architecture": {plan_json},
      "files": {"path": {"status": "implemented|planned|failed", "interface_hash": "..."}},
      "decisions": [{"what": "...", "why": "...", "by": "architect"}],
      "open_issues": [...],
      "constraints": [...]
    }
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import Any, Optional

from src.infra.logging_config import get_logger
from ..infra.db import get_db

logger = get_logger("collaboration.blackboard")

# ── Default empty blackboard ─────────────────────────────────────────────────

DEFAULT_BLACKBOARD: dict = {
    "architecture": {},
    "files": {},
    "decisions": [],
    "open_issues": [],
    "constraints": [],
}

# In-memory cache to reduce DB round-trips within a single run cycle.
_BLACKBOARD_CACHE: dict[int, dict] = {}

# Per-goal locks to prevent concurrent coroutines from corrupting cache/DB.
_BLACKBOARD_LOCKS: dict[int, asyncio.Lock] = {}


def _get_lock(goal_id: int) -> asyncio.Lock:
    """Return (or create) the asyncio.Lock for a specific goal_id."""
    if goal_id not in _BLACKBOARD_LOCKS:
        _BLACKBOARD_LOCKS[goal_id] = asyncio.Lock()
    return _BLACKBOARD_LOCKS[goal_id]


# ── Core API ─────────────────────────────────────────────────────────────────

async def get_or_create_blackboard(goal_id: int) -> dict:
    """Load the blackboard for a goal, creating a fresh one if needed."""
    async with _get_lock(goal_id):
        if goal_id in _BLACKBOARD_CACHE:
            return _BLACKBOARD_CACHE[goal_id]

        try:
            db = await get_db()

            await _ensure_table(db)

            cursor = await db.execute(
                "SELECT data FROM blackboards WHERE goal_id = ?", (goal_id,)
            )
            row = await cursor.fetchone()
            if row:
                board = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            else:
                board = json.loads(json.dumps(DEFAULT_BLACKBOARD))  # deep copy
                await db.execute(
                    "INSERT INTO blackboards (goal_id, data) VALUES (?, ?)",
                    (goal_id, json.dumps(board)),
                )
                await db.commit()
        except Exception as exc:
            logger.debug(f"Blackboard DB access failed, using defaults: {exc}")
            board = json.loads(json.dumps(DEFAULT_BLACKBOARD))

        _BLACKBOARD_CACHE[goal_id] = board
        return board


async def read_blackboard(goal_id: int, key: Optional[str] = None) -> Any:
    """Read the entire blackboard or a specific key."""
    board = await get_or_create_blackboard(goal_id)
    if key is None:
        return board
    return board.get(key)


async def write_blackboard(goal_id: int, key: str, value: Any) -> None:
    """Overwrite a top-level key in the blackboard."""
    async with _get_lock(goal_id):
        board = await get_or_create_blackboard(goal_id)
        board[key] = value
        _BLACKBOARD_CACHE[goal_id] = board
    await _persist(goal_id, board)


async def update_blackboard_entry(
    goal_id: int, key: str, sub_key: str, value: Any
) -> None:
    """Update a nested entry (e.g., files["app.py"] = {...})."""
    async with _get_lock(goal_id):
        board = await get_or_create_blackboard(goal_id)
        section = board.get(key)
        if isinstance(section, dict):
            section[sub_key] = value
        elif isinstance(section, list):
            section.append({sub_key: value})
        else:
            board[key] = {sub_key: value}
        _BLACKBOARD_CACHE[goal_id] = board
    await _persist(goal_id, board)


async def append_blackboard(goal_id: int, key: str, item: Any) -> None:
    """Append an item to a list-typed key (decisions, open_issues, etc.)."""
    async with _get_lock(goal_id):
        board = await get_or_create_blackboard(goal_id)
        section = board.get(key, [])
        if not isinstance(section, list):
            section = [section]
        section.append(item)
        board[key] = section
        _BLACKBOARD_CACHE[goal_id] = board
    await _persist(goal_id, board)


def clear_cache(goal_id: int | None = None) -> None:
    """Clear in-memory cache (useful for tests)."""
    if goal_id is not None:
        _BLACKBOARD_CACHE.pop(goal_id, None)
        _BLACKBOARD_LOCKS.pop(goal_id, None)
    else:
        _BLACKBOARD_CACHE.clear()
        _BLACKBOARD_LOCKS.clear()


# ── Prompt formatting ────────────────────────────────────────────────────────

def format_blackboard_for_prompt(board: dict, max_chars: int = 3000) -> str:
    """Format a blackboard for injection into an agent system prompt."""
    if not board:
        return ""

    # Skip if everything is empty
    has_content = any([
        board.get("architecture"),
        board.get("files"),
        board.get("decisions"),
        board.get("open_issues"),
        board.get("constraints"),
    ])
    if not has_content:
        return ""

    parts = ["## Shared Blackboard (Project State)"]

    # Architecture
    arch = board.get("architecture", {})
    if arch:
        arch_json = json.dumps(arch, indent=2)
        if len(arch_json) > 500:
            arch_json = arch_json[:500] + "\n..."
        parts.append(f"### Architecture\n```json\n{arch_json}\n```")

    # Files
    files = board.get("files", {})
    if files:
        file_lines = []
        for path, info in list(files.items())[:20]:
            status = info.get("status", "?") if isinstance(info, dict) else str(info)
            file_lines.append(f"  - `{path}`: {status}")
        parts.append("### File Status\n" + "\n".join(file_lines))

    # Decisions
    decisions = board.get("decisions", [])
    if decisions:
        dec_lines = [
            f"  - **{d.get('what', '?')}** — {d.get('why', '')} (by {d.get('by', '?')})"
            for d in decisions[-5:]  # last 5
            if isinstance(d, dict)
        ]
        if dec_lines:
            parts.append("### Key Decisions\n" + "\n".join(dec_lines))

    # Open Issues
    issues = board.get("open_issues", [])
    if issues:
        issue_lines = [f"  - {i}" for i in issues[-5:] if isinstance(i, str)]
        if issue_lines:
            parts.append("### Open Issues\n" + "\n".join(issue_lines))

    # Constraints
    constraints = board.get("constraints", [])
    if constraints:
        constraint_lines = [f"  - {c}" for c in constraints if isinstance(c, str)]
        if constraint_lines:
            parts.append("### Constraints\n" + "\n".join(constraint_lines))

    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [blackboard truncated]"
    return text


# ── Internal helpers ─────────────────────────────────────────────────────────

async def _ensure_table(db) -> None:
    """Create blackboards table if not exists."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS blackboards (
            goal_id INTEGER PRIMARY KEY,
            data JSON NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


async def _persist(goal_id: int, board: dict) -> None:
    """Write board state back to DB with 1 retry on SQLite busy errors."""
    last_exc: Exception | None = None
    for attempt in range(2):  # initial + 1 retry
        try:
            db = await get_db()
            await _ensure_table(db)
            await db.execute(
                """INSERT OR REPLACE INTO blackboards (goal_id, data, updated_at)
                   VALUES (?, ?, CURRENT_TIMESTAMP)""",
                (goal_id, json.dumps(board)),
            )
            await db.commit()
            return  # success
        except sqlite3.OperationalError as exc:
            last_exc = exc
            if attempt == 0 and "database is locked" in str(exc).lower():
                logger.debug("Blackboard persist: SQLite busy, retrying in 100ms")
                await asyncio.sleep(0.1)
                continue
            raise  # non-busy OperationalError — propagate
        except Exception:
            raise  # non-SQLite errors always propagate
    # Exhausted retries — raise the last busy error
    raise last_exc  # type: ignore[misc]
