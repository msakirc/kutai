# collaboration/blackboard.py
"""
Phase 13.1 — Shared Blackboard.

Per-mission structured state store with typed entries.  Backed by a
``blackboards`` DB table (mission_id → data JSON).  Agents read/write
structured data instead of parsing prior_steps text blobs.

Schema per mission::

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
    "dependency_map": {},
    "version": 0,
}

# In-memory cache to reduce DB round-trips within a single run cycle.
_BLACKBOARD_CACHE: dict[int, dict] = {}

# Per-mission locks to prevent concurrent coroutines from corrupting cache/DB.
_BLACKBOARD_LOCKS: dict[int, asyncio.Lock] = {}


def _get_lock(mission_id: int) -> asyncio.Lock:
    """Return (or create) the asyncio.Lock for a specific mission_id."""
    if mission_id not in _BLACKBOARD_LOCKS:
        _BLACKBOARD_LOCKS[mission_id] = asyncio.Lock()
    return _BLACKBOARD_LOCKS[mission_id]


# ── Core API ─────────────────────────────────────────────────────────────────

async def _load_board(mission_id: int) -> dict:
    """Load board from cache or DB. Caller MUST hold _get_lock(mission_id)."""
    if mission_id in _BLACKBOARD_CACHE:
        return _BLACKBOARD_CACHE[mission_id]

    try:
        db = await get_db()
        await _ensure_table(db)
        cursor = await db.execute(
            "SELECT data FROM blackboards WHERE mission_id = ?", (mission_id,)
        )
        row = await cursor.fetchone()
        if row:
            board = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        else:
            board = json.loads(json.dumps(DEFAULT_BLACKBOARD))  # deep copy
            await db.execute(
                "INSERT INTO blackboards (mission_id, data) VALUES (?, ?)",
                (mission_id, json.dumps(board)),
            )
            await db.commit()
    except Exception as exc:
        logger.debug(f"Blackboard DB access failed, using defaults: {exc}")
        board = json.loads(json.dumps(DEFAULT_BLACKBOARD))

    _BLACKBOARD_CACHE[mission_id] = board
    return board


async def get_or_create_blackboard(mission_id: int) -> dict:
    """Load the blackboard for a mission, creating a fresh one if needed."""
    async with _get_lock(mission_id):
        return await _load_board(mission_id)


async def read_blackboard(mission_id: int, key: Optional[str] = None) -> Any:
    """Read the entire blackboard or a specific key."""
    board = await get_or_create_blackboard(mission_id)
    if key is None:
        return board
    return board.get(key)


async def write_blackboard(mission_id: int, key: str, value: Any) -> None:
    """Overwrite a top-level key in the blackboard."""
    async with _get_lock(mission_id):
        board = await _load_board(mission_id)
        board[key] = value
        if key != "version":
            board["version"] = board.get("version", 0) + 1
        _BLACKBOARD_CACHE[mission_id] = board
    await _persist(mission_id, board)


async def update_blackboard_entry(
    mission_id: int, key: str, sub_key: str, value: Any
) -> None:
    """Update a nested entry (e.g., files["app.py"] = {...})."""
    async with _get_lock(mission_id):
        board = await _load_board(mission_id)
        section = board.get(key)
        if isinstance(section, dict):
            section[sub_key] = value
        elif isinstance(section, list):
            section.append({sub_key: value})
        else:
            board[key] = {sub_key: value}
        _BLACKBOARD_CACHE[mission_id] = board
    await _persist(mission_id, board)


async def append_blackboard(mission_id: int, key: str, item: Any) -> None:
    """Append an item to a list-typed key (decisions, open_issues, etc.)."""
    async with _get_lock(mission_id):
        board = await _load_board(mission_id)
        section = board.get(key, [])
        if not isinstance(section, list):
            section = [section]
        section.append(item)
        board[key] = section
        _BLACKBOARD_CACHE[mission_id] = board
    await _persist(mission_id, board)


async def write_blackboard_versioned(
    mission_id: int, key: str, value: Any, expected_version: int
) -> None:
    """Write to blackboard with optimistic locking (version checking).

    Raises ValueError if the blackboard version doesn't match expected_version.
    """
    async with _get_lock(mission_id):
        board = await _load_board(mission_id)
        current_version = board.get("version", 0)
        if current_version != expected_version:
            raise ValueError(
                f"Blackboard conflict: expected version {expected_version}, "
                f"got {current_version}"
            )
        board[key] = value
        board["version"] = current_version + 1
        _BLACKBOARD_CACHE[mission_id] = board
    await _persist(mission_id, board)


def clear_cache(mission_id: int | None = None) -> None:
    """Clear in-memory cache (useful for tests)."""
    if mission_id is not None:
        _BLACKBOARD_CACHE.pop(mission_id, None)
        _BLACKBOARD_LOCKS.pop(mission_id, None)
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

    # Dependency Map
    dependency_map = board.get("dependency_map", {})
    if dependency_map:
        num_deps = len(dependency_map)
        parts.append(f"### Task Dependencies\n  {num_deps} task dependencies tracked")

    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [blackboard truncated]"
    return text


# ── Internal helpers ─────────────────────────────────────────────────────────

async def _ensure_table(db) -> None:
    """Create blackboards table if not exists."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS blackboards (
            mission_id INTEGER PRIMARY KEY,
            data JSON NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


async def _persist(mission_id: int, board: dict) -> None:
    """Write board state back to DB with 1 retry on SQLite busy errors."""
    last_exc: Exception | None = None
    for attempt in range(2):  # initial + 1 retry
        try:
            db = await get_db()
            await _ensure_table(db)
            await db.execute(
                """INSERT OR REPLACE INTO blackboards (mission_id, data, updated_at)
                   VALUES (?, ?, CURRENT_TIMESTAMP)""",
                (mission_id, json.dumps(board)),
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
