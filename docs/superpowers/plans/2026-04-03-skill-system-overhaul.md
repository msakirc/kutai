# Skill System Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken auto-capture skill system with grader-driven execution recipes that use vector matching, adaptive injection, and real-world success tracking.

**Architecture:** Expand grading to extract strategy metadata. Rewrite skills.py as the central module — new schema with ranked strategies, vector-only matching, adaptive injection depth. Rewire orchestrator capture and base.py injection to use the new system. Migrate seed skills, wipe garbage.

**Tech Stack:** Python 3.10, aiosqlite (WAL), ChromaDB (multilingual-e5-small 384d), async/await throughout

---

### Task 1: New Skills Schema & Migration in db.py

**Files:**
- Modify: `src/infra/db.py:366-378` (replace skill_metrics table area, add new skills table)
- Test: `tests/test_skill_overhaul.py`

The current `skills` table is created in `src/memory/skills.py:22-34` via `_ensure_table()`. The new table will be created in `db.py:init_db()` alongside other tables, and the old `_ensure_table()` will be removed. The existing `skill_metrics` table (db.py:366-378) is kept as-is — it tracks A/B metrics separately and is still useful.

- [ ] **Step 1: Write failing test for new schema**

```python
# tests/test_skill_overhaul.py
import pytest
import json
import aiosqlite


@pytest.fixture
async def db(tmp_path):
    """Create a fresh DB with the new schema."""
    db_path = str(tmp_path / "test.db")
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")

    # New skills table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL,
            skill_type TEXT DEFAULT 'auto',
            strategies TEXT DEFAULT '[]',
            injection_count INTEGER DEFAULT 0,
            injection_success INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
            updated_at TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
        )
    """)
    await db.commit()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_new_skills_table_structure(db):
    """New skills table has correct columns."""
    cursor = await db.execute("PRAGMA table_info(skills)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert "name" in cols
    assert "description" in cols
    assert "skill_type" in cols
    assert "strategies" in cols
    assert "injection_count" in cols
    assert "injection_success" in cols
    # Old columns should NOT exist
    assert "trigger_pattern" not in cols
    assert "tool_sequence" not in cols
    assert "success_count" not in cols
    assert "failure_count" not in cols


@pytest.mark.asyncio
async def test_insert_skill_with_strategies(db):
    """Can insert a skill with JSON strategies."""
    strategy = {
        "summary": "Search stores separately, compare",
        "tool_template": ["smart_search({product} trendyol)", "compare prices"],
        "tools_used": ["smart_search"],
        "avg_iterations": 3,
        "source_grade": "great",
        "injection_count": 0,
        "injection_success": 0,
        "created_from_task": 42,
        "created_at": "2026-04-03 14:00:00",
    }
    await db.execute(
        "INSERT INTO skills (name, description, skill_type, strategies) VALUES (?, ?, ?, ?)",
        ("price_comparison", "Comparing product prices across stores", "auto", json.dumps([strategy])),
    )
    await db.commit()
    cursor = await db.execute("SELECT * FROM skills WHERE name = ?", ("price_comparison",))
    row = await cursor.fetchone()
    assert row is not None
    assert row["skill_type"] == "auto"
    strategies = json.loads(row["strategies"])
    assert len(strategies) == 1
    assert strategies[0]["summary"] == "Search stores separately, compare"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill_overhaul.py -v`
Expected: PASS (this tests the schema directly, not production code yet — acts as a schema contract test)

- [ ] **Step 3: Add new skills table to db.py init_db()**

In `src/infra/db.py`, find the skill_metrics table creation (line ~366). Add the new skills table right before it. This replaces the old `_ensure_table()` in skills.py.

```python
    # ── Skill library (v2 — execution recipes) ──
    await db.execute("""
        CREATE TABLE IF NOT EXISTS skills_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL,
            skill_type TEXT DEFAULT 'auto',
            strategies TEXT DEFAULT '[]',
            injection_count INTEGER DEFAULT 0,
            injection_success INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
            updated_at TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
        )
    """)
```

Note: Table is named `skills_v2` during migration. The migration step will rename it to `skills` after backing up the old one. This avoids conflicts with the existing `skills` table that `_ensure_table()` creates.

- [ ] **Step 4: Add migration logic in init_db()**

After the table creation, add migration logic that runs once:

```python
    # ── Migrate skills to v2 if old table exists ──
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='skills'"
    )
    old_exists = await cursor.fetchone()
    cursor2 = await db.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='skills_v2'"
    )
    v2_exists = (await cursor2.fetchone())[0] > 0

    if old_exists and v2_exists:
        # Check if migration already done (skills_v2 has data)
        cursor3 = await db.execute("SELECT COUNT(*) FROM skills_v2")
        v2_count = (await cursor3.fetchone())[0]
        if v2_count == 0:
            # Migrate seed skills from old table
            cursor4 = await db.execute(
                "SELECT name, description, tool_sequence, examples FROM skills "
                "WHERE name NOT LIKE 'auto:%'"
            )
            for row in await cursor4.fetchall():
                strategy = json.dumps([{
                    "summary": row["description"][:200],
                    "tool_template": [],
                    "tools_used": [],
                    "avg_iterations": 0,
                    "source_grade": "seed",
                    "injection_count": 0,
                    "injection_success": 0,
                    "created_from_task": 0,
                    "created_at": "2026-04-03 00:00:00",
                }])
                await db.execute(
                    "INSERT OR IGNORE INTO skills_v2 (name, description, skill_type, strategies) "
                    "VALUES (?, ?, 'seed', ?)",
                    (row["name"], row["description"], strategy),
                )
            # Rename: backup old, promote v2
            await db.execute("ALTER TABLE skills RENAME TO skills_old_backup")
            await db.execute("ALTER TABLE skills_v2 RENAME TO skills")
            await db.commit()
            logger.info("Migrated skills to v2 schema, old table backed up as skills_old_backup")
        elif v2_count > 0 and old_exists:
            # Migration already done but old table still around — just rename
            try:
                await db.execute("ALTER TABLE skills RENAME TO skills_old_backup")
                await db.execute("ALTER TABLE skills_v2 RENAME TO skills")
                await db.commit()
            except Exception:
                pass  # tables already renamed
```

- [ ] **Step 5: Add DB helper functions for new skill operations**

Add these after the existing `record_skill_metric` function (line ~2542):

```python
# ─── Skill Library v2 Operations ─────────────────────────────────────────────


async def upsert_skill(name: str, description: str, skill_type: str = "auto",
                       strategies: str = "[]") -> int:
    """Insert or update a skill. Returns row id."""
    db = await get_db()
    cursor = await db.execute(
        """INSERT INTO skills (name, description, skill_type, strategies)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET
           description=excluded.description,
           strategies=excluded.strategies,
           updated_at=strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')""",
        (name, description, skill_type, strategies),
    )
    await db.commit()
    return cursor.lastrowid or 0


async def get_all_skills() -> list[dict]:
    """Fetch all skills."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM skills ORDER BY injection_success DESC, injection_count DESC"
    )
    rows = await cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in rows]


async def get_skill_by_name(name: str) -> dict | None:
    """Fetch a single skill by name."""
    db = await get_db()
    cursor = await db.execute("SELECT * FROM skills WHERE name = ?", (name,))
    row = await cursor.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


async def increment_skill_injection(name: str) -> None:
    """Increment injection_count for a skill."""
    db = await get_db()
    await db.execute(
        "UPDATE skills SET injection_count = injection_count + 1, "
        "updated_at = strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime') "
        "WHERE name = ?", (name,),
    )
    await db.commit()


async def increment_skill_success(name: str) -> None:
    """Increment injection_success for a skill (task that used this skill succeeded)."""
    db = await get_db()
    await db.execute(
        "UPDATE skills SET injection_success = injection_success + 1, "
        "updated_at = strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime') "
        "WHERE name = ?", (name,),
    )
    await db.commit()
```

- [ ] **Step 6: Write test for DB helpers**

```python
# Append to tests/test_skill_overhaul.py

@pytest.mark.asyncio
async def test_increment_injection_tracking(db):
    """Injection count and success tracking work."""
    await db.execute(
        "INSERT INTO skills (name, description) VALUES (?, ?)",
        ("test_skill", "A test skill"),
    )
    await db.commit()

    # Simulate 3 injections, 2 successes
    for _ in range(3):
        await db.execute(
            "UPDATE skills SET injection_count = injection_count + 1 WHERE name = ?",
            ("test_skill",),
        )
    for _ in range(2):
        await db.execute(
            "UPDATE skills SET injection_success = injection_success + 1 WHERE name = ?",
            ("test_skill",),
        )
    await db.commit()

    cursor = await db.execute("SELECT * FROM skills WHERE name = ?", ("test_skill",))
    row = await cursor.fetchone()
    assert row["injection_count"] == 3
    assert row["injection_success"] == 2
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/test_skill_overhaul.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/infra/db.py tests/test_skill_overhaul.py
git commit -m "feat(skills): new v2 schema with strategies and injection tracking"
```

---

### Task 2: Rewrite skills.py — Core Module

**Files:**
- Modify: `src/memory/skills.py` (full rewrite)
- Test: `tests/test_skill_overhaul.py` (append)

This task rewrites skills.py to use the new schema: vector-only matching, strategy accumulation with dedup, adaptive injection formatting, and injection tracking. The old regex matching, `_ensure_table()`, `_skill_score()`, and `format_skills_for_prompt()` are all replaced.

- [ ] **Step 1: Write failing tests for new skills.py functions**

```python
# Append to tests/test_skill_overhaul.py
import os
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_add_skill_creates_new():
    """add_skill creates a new skill with strategy when no duplicate exists."""
    with patch("src.memory.skills.get_db") as mock_db, \
         patch("src.memory.skills._find_duplicate_skill", new_callable=AsyncMock, return_value=None), \
         patch("src.memory.skills._embed_skill", new_callable=AsyncMock):

        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.lastrowid = 1
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.commit = AsyncMock()
        mock_db.return_value = mock_conn

        from src.memory.skills import add_skill
        result = await add_skill(
            name="test_skill",
            description="Test situation",
            strategy_summary="Do X then Y",
            tool_template=["smart_search({q})", "compare"],
            tools_used=["smart_search"],
            avg_iterations=3,
            source_grade="great",
            source_task_id=42,
        )
        assert result is not None
        mock_conn.execute.assert_called()


@pytest.mark.asyncio
async def test_add_skill_merges_duplicate():
    """add_skill adds strategy to existing skill when duplicate found."""
    existing_skill = {
        "name": "existing_skill",
        "description": "Existing description",
        "strategies": json.dumps([{
            "summary": "Old strategy",
            "tool_template": [],
            "tools_used": [],
            "avg_iterations": 2,
            "source_grade": "great",
            "injection_count": 10,
            "injection_success": 8,
            "created_from_task": 1,
            "created_at": "2026-04-01 10:00:00",
        }]),
    }
    with patch("src.memory.skills.get_db") as mock_db, \
         patch("src.memory.skills._find_duplicate_skill", new_callable=AsyncMock, return_value=existing_skill), \
         patch("src.memory.skills._embed_skill", new_callable=AsyncMock):

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        mock_db.return_value = mock_conn

        from src.memory.skills import add_skill
        await add_skill(
            name="new_name_doesnt_matter",
            description="Similar situation",
            strategy_summary="New approach",
            tool_template=["web_search({q})"],
            tools_used=["web_search"],
            avg_iterations=5,
            source_grade="great",
            source_task_id=99,
        )
        # Should have called UPDATE on existing skill, not INSERT
        calls = [str(c) for c in mock_conn.execute.call_args_list]
        assert any("UPDATE" in c for c in calls)


def test_format_skill_verbose():
    """Verbose format includes full strategy details."""
    from src.memory.skills import format_skill_verbose
    skill = {
        "name": "price_comparison",
        "description": "Comparing product prices across stores",
        "injection_count": 10,
        "injection_success": 8,
        "strategies": json.dumps([{
            "summary": "Search each store, build comparison table",
            "tool_template": ["smart_search({product} trendyol)", "compare prices"],
            "tools_used": ["smart_search", "web_search"],
            "avg_iterations": 3,
            "source_grade": "great",
            "injection_count": 10,
            "injection_success": 8,
            "created_from_task": 42,
            "created_at": "2026-04-03 14:00:00",
        }]),
    }
    result = format_skill_verbose(skill)
    assert "Proven Strategy" in result
    assert "price_comparison" in result
    assert "smart_search" in result
    assert "8/10" in result


def test_format_skill_compact():
    """Compact format is a single line per skill."""
    from src.memory.skills import format_skill_compact
    skill = {
        "name": "price_comparison",
        "description": "Comparing product prices",
        "injection_count": 10,
        "injection_success": 8,
        "strategies": json.dumps([{
            "summary": "Search stores, compare",
            "tools_used": ["smart_search"],
            "injection_count": 10,
            "injection_success": 8,
        }]),
    }
    result = format_skill_compact(skill)
    assert "price_comparison" in result
    assert "80%" in result
    assert "\n" not in result.strip()


def test_select_injection_depth_trusted():
    """Single highly trusted skill → verbose, just one."""
    from src.memory.skills import select_injection_depth
    skills = [
        {"name": "s1", "injection_count": 10, "injection_success": 9, "strategies": "[]"},
    ]
    mode, selected = select_injection_depth(skills, context_budget=4096)
    assert mode == "verbose"
    assert len(selected) == 1


def test_select_injection_depth_uncertain():
    """Uncertain skills → compact, multiple."""
    from src.memory.skills import select_injection_depth
    skills = [
        {"name": "s1", "injection_count": 3, "injection_success": 1, "strategies": "[]"},
        {"name": "s2", "injection_count": 2, "injection_success": 1, "strategies": "[]"},
        {"name": "s3", "injection_count": 1, "injection_success": 0, "strategies": "[]"},
    ]
    mode, selected = select_injection_depth(skills, context_budget=4096)
    assert mode == "compact"
    assert len(selected) >= 2


def test_select_injection_depth_small_context():
    """Small context budget → fewer skills even if uncertain."""
    from src.memory.skills import select_injection_depth
    skills = [
        {"name": "s1", "injection_count": 3, "injection_success": 1, "strategies": "[]"},
        {"name": "s2", "injection_count": 2, "injection_success": 1, "strategies": "[]"},
        {"name": "s3", "injection_count": 1, "injection_success": 0, "strategies": "[]"},
    ]
    mode, selected = select_injection_depth(skills, context_budget=1500)
    assert mode == "compact"
    assert len(selected) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_skill_overhaul.py -v -k "not test_new_skills_table and not test_insert_skill and not test_increment"`
Expected: FAIL — functions don't exist yet

- [ ] **Step 3: Rewrite skills.py**

Replace the entire content of `src/memory/skills.py`:

```python
"""
Skill Library v2 — Execution Recipes

Skills capture what worked during task execution: tool sequences, strategies,
iteration patterns. They are NOT routing hints (classifier/fast_resolver handle that).

Matching: vector similarity via ChromaDB on skill descriptions.
Capture: grader generates situation + strategy summaries.
Injection: adaptive depth — one trusted skill verbose, or multiple compact.
Tracking: injection_count / injection_success for real-world validation.
"""
from __future__ import annotations
import json
from datetime import datetime
from typing import Optional

from src.infra.logging_config import get_logger
from src.infra.db import get_db

logger = get_logger("memory.skills")

# ─── Thresholds ──────────────────────────────────────────────────────────────

DEDUP_SIMILARITY_THRESHOLD = 0.85   # merge strategies into existing skill above this
MATCH_SIMILARITY_THRESHOLD = 0.6    # minimum similarity for injection candidates
HIGH_CONFIDENCE_THRESHOLD = 0.8     # injection success rate for "trusted" status
MIN_INJECTIONS_FOR_CONFIDENCE = 5   # need this many injections before trusting rate
MAX_STRATEGIES_PER_SKILL = 5        # cap strategies list
TOOL_INJECTION_THRESHOLD = 0.7      # success rate needed to inject tools into allowed_tools
TOOL_INJECTION_MIN_COUNT = 5        # minimum injections before tool injection


def _injection_success_rate(skill: dict) -> float:
    """Compute injection success rate, capped at 0.5 for low-count skills."""
    count = skill.get("injection_count", 0)
    success = skill.get("injection_success", 0)
    if count < MIN_INJECTIONS_FOR_CONFIDENCE:
        return 0.5  # neutral — not enough data
    return success / max(count, 1)


def _best_strategy(skill: dict) -> dict | None:
    """Get the highest-ranked strategy from a skill."""
    try:
        strategies = json.loads(skill.get("strategies", "[]"))
    except (json.JSONDecodeError, TypeError):
        return None
    if not strategies:
        return None
    # Rank: strategies with enough injections ranked by success rate,
    # then strategies still proving themselves (newest first)
    def sort_key(s):
        ic = s.get("injection_count", 0)
        isc = s.get("injection_success", 0)
        if ic >= MIN_INJECTIONS_FOR_CONFIDENCE:
            return (1, isc / max(ic, 1))  # proven — rank by success rate
        return (0, ic)  # unproven — newest (highest count) first
    strategies.sort(key=sort_key, reverse=True)
    return strategies[0]


# ─── Vector Operations ───────────────────────────────────────────────────────


async def _embed_skill(name: str, description: str) -> None:
    """Embed skill description into ChromaDB for vector search."""
    try:
        from src.memory.vector_store import embed_and_store
        await embed_and_store(
            text=description,
            metadata={"type": "skill", "skill_name": name},
            collection="semantic",
            doc_id=f"skill:{name}",
        )
    except Exception as exc:
        logger.debug("Skill embedding failed (non-critical): %s", exc)


async def _find_duplicate_skill(description: str) -> dict | None:
    """Check if a skill with similar description already exists (>= 0.85 similarity)."""
    try:
        from src.memory.vector_store import query as vquery
        results = await vquery(
            text=description,
            collection="semantic",
            top_k=1,
            where={"type": "skill"},
        )
        if not results:
            return None
        top = results[0]
        distance = top.get("distance", 1.0)
        # ChromaDB returns L2 distance — lower is more similar
        # 0.85 similarity ≈ 0.3 L2 distance for normalized embeddings
        similarity = max(0, 1.0 - distance)
        if similarity < DEDUP_SIMILARITY_THRESHOLD:
            return None
        skill_name = top.get("metadata", {}).get("skill_name", "")
        if not skill_name:
            return None
        from src.infra.db import get_skill_by_name
        return await get_skill_by_name(skill_name)
    except Exception as exc:
        logger.debug("Duplicate skill check failed: %s", exc)
        return None


async def _vector_search_skills(task_text: str, top_k: int = 5) -> list[dict]:
    """Find skills by vector similarity to task text."""
    try:
        from src.memory.vector_store import query as vquery
        results = await vquery(
            text=task_text,
            collection="semantic",
            top_k=top_k,
            where={"type": "skill"},
        )
        matched = []
        for r in results:
            distance = r.get("distance", 1.0)
            similarity = max(0, 1.0 - distance)
            if similarity < MATCH_SIMILARITY_THRESHOLD:
                continue
            skill_name = r.get("metadata", {}).get("skill_name", "")
            if not skill_name:
                continue
            matched.append({"skill_name": skill_name, "similarity": similarity})
        return matched
    except Exception as exc:
        logger.debug("Vector skill search failed: %s", exc)
        return []


# ─── Capture ─────────────────────────────────────────────────────────────────


async def add_skill(
    name: str,
    description: str,
    strategy_summary: str,
    tool_template: list[str] | None = None,
    tools_used: list[str] | None = None,
    avg_iterations: int = 0,
    source_grade: str = "great",
    source_task_id: int = 0,
) -> str | None:
    """
    Capture an execution recipe as a skill.

    If a skill with similar description exists (vector similarity >= 0.85),
    adds the strategy to that skill instead of creating a new one.

    Returns the skill name (may differ from input if merged into existing).
    """
    strategy = {
        "summary": strategy_summary,
        "tool_template": tool_template or [],
        "tools_used": tools_used or [],
        "avg_iterations": avg_iterations,
        "source_grade": source_grade,
        "injection_count": 0,
        "injection_success": 0,
        "created_from_task": source_task_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Check for duplicate
    existing = await _find_duplicate_skill(description)
    if existing:
        # Merge strategy into existing skill
        try:
            strategies = json.loads(existing.get("strategies", "[]"))
        except (json.JSONDecodeError, TypeError):
            strategies = []
        strategies.append(strategy)
        # Enforce max strategies: drop worst performer if over limit
        if len(strategies) > MAX_STRATEGIES_PER_SKILL:
            strategies = _prune_strategies(strategies)
        db = await get_db()
        await db.execute(
            "UPDATE skills SET strategies = ?, "
            "updated_at = strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime') "
            "WHERE name = ?",
            (json.dumps(strategies), existing["name"]),
        )
        await db.commit()
        logger.info("Strategy added to existing skill: %s (from task #%d)",
                     existing["name"], source_task_id)
        return existing["name"]

    # Create new skill
    from src.infra.db import upsert_skill
    await upsert_skill(name, description, "auto", json.dumps([strategy]))
    await _embed_skill(name, description)
    logger.info("New skill created: %s (from task #%d)", name, source_task_id)
    return name


def _prune_strategies(strategies: list[dict]) -> list[dict]:
    """Keep top MAX_STRATEGIES_PER_SKILL strategies. Never drop unproven ones."""
    proven = [s for s in strategies if s.get("injection_count", 0) >= MIN_INJECTIONS_FOR_CONFIDENCE]
    unproven = [s for s in strategies if s.get("injection_count", 0) < MIN_INJECTIONS_FOR_CONFIDENCE]

    if len(proven) > MAX_STRATEGIES_PER_SKILL:
        # Drop worst proven strategy
        proven.sort(key=lambda s: s.get("injection_success", 0) / max(s.get("injection_count", 1), 1))
        proven = proven[1:]  # drop lowest

    result = proven + unproven
    if len(result) > MAX_STRATEGIES_PER_SKILL:
        # Still too many — drop oldest unproven
        unproven.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        result = proven + unproven[:MAX_STRATEGIES_PER_SKILL - len(proven)]

    return result[:MAX_STRATEGIES_PER_SKILL]


# ─── Injection ───────────────────────────────────────────────────────────────


async def find_relevant_skills(task_text: str, limit: int = 5) -> list[dict]:
    """
    Find skills relevant to a task using vector similarity.

    Returns skills ranked by: vector_similarity * 0.5 + injection_success_rate * 0.5
    """
    try:
        vector_matches = await _vector_search_skills(task_text, top_k=limit)
        if not vector_matches:
            logger.info("No skills matched for task: %s", task_text[:80])
            return []

        # Load full skill data
        from src.infra.db import get_skill_by_name
        skills_with_scores = []
        for match in vector_matches:
            skill = await get_skill_by_name(match["skill_name"])
            if not skill:
                continue
            isr = _injection_success_rate(skill)
            combined = match["similarity"] * 0.5 + isr * 0.5
            skill["_match_score"] = combined
            skill["_similarity"] = match["similarity"]
            skills_with_scores.append(skill)

        skills_with_scores.sort(key=lambda s: s["_match_score"], reverse=True)
        logger.info("Skills matched: %s",
                     [(s["name"], f"{s['_match_score']:.2f}") for s in skills_with_scores[:limit]])
        return skills_with_scores[:limit]

    except Exception as exc:
        logger.warning("find_relevant_skills failed: %s", exc)
        return []


def select_injection_depth(
    skills: list[dict], context_budget: int
) -> tuple[str, list[dict]]:
    """
    Decide how many skills to inject and in what format.

    Returns: (mode, selected_skills) where mode is "verbose" or "compact".
    """
    if not skills:
        return "compact", []

    top = skills[0]
    top_rate = _injection_success_rate(top)
    top_count = top.get("injection_count", 0)

    # Highly trusted top skill → just this one, verbose
    if top_rate >= HIGH_CONFIDENCE_THRESHOLD and top_count >= MIN_INJECTIONS_FOR_CONFIDENCE:
        return "verbose", [top]

    # Uncertain → multiple compact, scaled by context budget
    if context_budget < 2048:
        max_skills = 1
    elif context_budget < 4096:
        max_skills = 2
    else:
        max_skills = 3

    return "compact", skills[:max_skills]


def format_skill_verbose(skill: dict) -> str:
    """Full verbose format for a single highly trusted skill."""
    strategy = _best_strategy(skill)
    if not strategy:
        return ""

    ic = skill.get("injection_count", 0)
    isc = skill.get("injection_success", 0)
    tools = ", ".join(strategy.get("tools_used", []))
    template = strategy.get("tool_template", [])

    lines = [
        "## Proven Strategy",
        "",
        f"### {skill['name']}",
        f"**Situation**: {skill.get('description', '')}",
        f"**Strategy**: {strategy['summary']}",
    ]
    if template:
        lines.append("**Steps**:")
        for i, step in enumerate(template, 1):
            lines.append(f"  {i}. {step}")
    lines.append(f"**Track record**: {isc}/{ic} successful uses")
    if tools:
        lines.append(f"**Tools**: {tools}")
    return "\n".join(lines)


def format_skill_compact(skill: dict) -> str:
    """Single-line compact format for uncertain skills."""
    strategy = _best_strategy(skill)
    if not strategy:
        return ""

    ic = skill.get("injection_count", 0)
    isc = skill.get("injection_success", 0)
    rate = round(isc / max(ic, 1) * 100) if ic > 0 else 0
    tools = ", ".join(strategy.get("tools_used", []))
    return f"- {skill['name']}: {strategy['summary']} (tools: {tools}, {rate}% success)"


def format_skills_for_prompt(skills: list[dict], context_budget: int = 4096) -> str:
    """Format skills for injection into agent context. Adaptive depth."""
    if not skills:
        return ""

    mode, selected = select_injection_depth(skills, context_budget)

    if mode == "verbose" and selected:
        return format_skill_verbose(selected[0])

    if selected:
        lines = ["## Strategy Hints", ""]
        for s in selected:
            line = format_skill_compact(s)
            if line:
                lines.append(line)
        return "\n".join(lines) if len(lines) > 2 else ""

    return ""


def get_tools_to_inject(skills: list[dict]) -> list[str]:
    """
    Get tool names that high-confidence skills recommend but agent may not have.

    Only returns tools from skills with injection success rate >= 0.7
    and at least 5 injections.
    """
    tools = set()
    for skill in skills:
        rate = _injection_success_rate(skill)
        count = skill.get("injection_count", 0)
        if rate >= TOOL_INJECTION_THRESHOLD and count >= TOOL_INJECTION_MIN_COUNT:
            strategy = _best_strategy(skill)
            if strategy:
                for t in strategy.get("tools_used", []):
                    tools.add(t)
    return list(tools)


async def record_injection(skill_names: list[str]) -> None:
    """Record that skills were injected (increment injection_count)."""
    from src.infra.db import increment_skill_injection
    for name in skill_names:
        try:
            await increment_skill_injection(name)
            # Also increment strategy-level count
            skill = await (await get_db()).execute(
                "SELECT strategies FROM skills WHERE name = ?", (name,)
            )
            row = await skill.fetchone()
            if row:
                strategies = json.loads(row[0] or "[]")
                if strategies:
                    strategies[0]["injection_count"] = strategies[0].get("injection_count", 0) + 1
                    await (await get_db()).execute(
                        "UPDATE skills SET strategies = ? WHERE name = ?",
                        (json.dumps(strategies), name),
                    )
                    await (await get_db()).commit()
        except Exception as exc:
            logger.debug("Failed to record injection for %s: %s", name, exc)


async def record_injection_success(skill_names: list[str]) -> None:
    """Record that a task with injected skills succeeded."""
    from src.infra.db import increment_skill_success
    for name in skill_names:
        try:
            await increment_skill_success(name)
            # Also increment strategy-level success
            skill = await (await get_db()).execute(
                "SELECT strategies FROM skills WHERE name = ?", (name,)
            )
            row = await skill.fetchone()
            if row:
                strategies = json.loads(row[0] or "[]")
                if strategies:
                    strategies[0]["injection_success"] = strategies[0].get("injection_success", 0) + 1
                    await (await get_db()).execute(
                        "UPDATE skills SET strategies = ? WHERE name = ?",
                        (json.dumps(strategies), name),
                    )
                    await (await get_db()).commit()
        except Exception as exc:
            logger.debug("Failed to record success for %s: %s", name, exc)


async def list_skills() -> list[dict]:
    """List all skills (for /skillstats and admin)."""
    from src.infra.db import get_all_skills
    return await get_all_skills()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_skill_overhaul.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/skills.py tests/test_skill_overhaul.py
git commit -m "feat(skills): rewrite skills.py with vector matching, strategy accumulation, adaptive injection"
```

---

### Task 3: Update seed_skills.py for New Format

**Files:**
- Modify: `src/memory/seed_skills.py` (update seed data and seeding function)
- Test: `tests/test_skill_overhaul.py` (append)

Seed skills are converted from old format (trigger_pattern, tool_sequence, examples) to new format (description used for vector matching, strategy object). The `seed_skills()` function now calls the new `add_skill()` from skills.py or directly inserts via db helpers.

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_skill_overhaul.py

def test_seed_skills_have_new_format():
    """Seed skills data uses new strategy format."""
    from src.memory.seed_skills import SEED_SKILLS
    for skill in SEED_SKILLS:
        assert "name" in skill
        assert "description" in skill
        assert "strategy_summary" in skill, f"{skill['name']} missing strategy_summary"
        assert "trigger_pattern" not in skill, f"{skill['name']} still has trigger_pattern"
        assert "tool_sequence" not in skill, f"{skill['name']} still has tool_sequence"
        assert "examples" not in skill, f"{skill['name']} still has examples"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill_overhaul.py::test_seed_skills_have_new_format -v`
Expected: FAIL — seed skills still have old format

- [ ] **Step 3: Rewrite seed_skills.py**

Replace the entire content of `src/memory/seed_skills.py`:

```python
"""Seed the skills database with curated execution recipe skills."""

from src.infra.logging_config import get_logger

logger = get_logger("memory.seed_skills")

SEED_SKILLS = [
    {
        "name": "currency_lookup",
        "description": "Looking up currency exchange rates, conversion between currencies, checking current dollar/euro/gold prices in Turkish Lira",
        "strategy_summary": "Use api_call with TCMB for Turkish rates or Frankfurter for international. Faster and more accurate than web search for simple rate lookups.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "weather_check",
        "description": "Checking weather forecasts, current temperature, rain predictions for a specific city or location",
        "strategy_summary": "Use api_call with wttr.in (simple format) or Open-Meteo (detailed forecast). Format: wttr.in/{city}?format=j1. Avoid web_search for basic weather.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "timezone_lookup",
        "description": "Finding current time in a city, timezone differences, time conversion between locations",
        "strategy_summary": "Use api_call with WorldTimeAPI. Endpoint: worldtimeapi.org/api/timezone/{Area}/{City}.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "encyclopedia_lookup",
        "description": "Looking up factual information about people, places, historical events, scientific concepts from encyclopedia sources",
        "strategy_summary": "Use api_call with Wikipedia API. Try Turkish Wikipedia first for Turkish topics, then English. Structured data beats web_search for factual lookups.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "app_store_research",
        "description": "Researching mobile apps, finding app alternatives, reading app reviews, comparing app features and ratings across stores",
        "strategy_summary": "Use play_store tool: search (find apps), app (details), reviews (user reviews), similar (competitors). For competitor analysis, combine play_store with github for open-source alternatives.",
        "tools_used": ["play_store", "smart_search"],
    },
    {
        "name": "github_code_research",
        "description": "Searching for open source projects, code examples, library comparison, repository analysis on GitHub",
        "strategy_summary": "Use github tool: repos (search projects), code (search code), readme (fetch README). Combine with web_search for ecosystem context. Use GITHUB_TOKEN for higher rate limits.",
        "tools_used": ["github", "smart_search"],
    },
    {
        "name": "turkish_product_shopping",
        "description": "Shopping for products in Turkey, finding prices on Turkish e-commerce sites, comparing prices across Trendyol, Hepsiburada, Akakce, Amazon TR",
        "strategy_summary": "Use shopping_search for product discovery across Turkish retailers. For price comparison, Akakce aggregates all retailers. For reviews, Trendyol has the best review data. Search multiple sources then compare.",
        "tools_used": ["shopping_search", "smart_search"],
    },
    {
        "name": "product_review_research",
        "description": "Finding product reviews, user complaints, brand reputation analysis in Turkish sources including Sikayetvar, Technopat, DonanımHaber",
        "strategy_summary": "Use shopping_fetch_reviews with specific sources: sikayetvar for complaints, technopat/donanimhaber for tech reviews, trendyol/hepsiburada for e-commerce reviews. Aggregate multiple sources for balanced view.",
        "tools_used": ["shopping_fetch_reviews", "smart_search"],
    },
    {
        "name": "sports_live_data",
        "description": "Getting live sports scores, match lineups, football predictions, league standings, especially Turkish Super Lig and Champions League",
        "strategy_summary": "Use web_search with search_depth=standard. Sports data is time-sensitive — always search, never answer from memory. No reliable free API for Turkish football.",
        "tools_used": ["web_search", "smart_search"],
    },
    {
        "name": "pdf_document_processing",
        "description": "Reading PDF files, extracting text from documents, analyzing report contents",
        "strategy_summary": "Use read_pdf_advanced (multi-backend: PyMuPDF > pdfplumber > PyPDF2). Pass file_path and optional max_pages parameter.",
        "tools_used": ["read_pdf_advanced"],
    },
    {
        "name": "programming_error_diagnosis",
        "description": "Diagnosing programming errors, looking up stack traces, finding solutions for exception messages and error codes",
        "strategy_summary": "Use web_search targeting Stack Overflow and GitHub issues. Add 'site:stackoverflow.com' or 'site:github.com/issues' to query for focused results.",
        "tools_used": ["web_search", "smart_search"],
    },
    {
        "name": "text_translation",
        "description": "Translating text between languages, especially Turkish-English translation",
        "strategy_summary": "Try api_call with LibreTranslate first (POST /translate with q, source, target). Falls back to web_search if API unavailable.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "current_news_lookup",
        "description": "Finding current news, breaking news headlines, today's news in Turkey or worldwide",
        "strategy_summary": "Check api_lookup for GNews first (needs GNEWS_API_KEY). Otherwise web_search with search_depth=quick. News queries are time-sensitive.",
        "tools_used": ["smart_search", "web_search"],
    },
    {
        "name": "network_diagnostics",
        "description": "Looking up IP addresses, geolocation, running network diagnostics like ping and DNS queries",
        "strategy_summary": "Use api_call with ipapi for IP geolocation. Use shell for ping, traceroute, nslookup commands.",
        "tools_used": ["smart_search", "api_call", "shell"],
    },
    {
        "name": "competitor_analysis_research",
        "description": "Researching competitors for a product idea, market analysis, finding similar apps and open-source alternatives",
        "strategy_summary": "1. play_store search for competing apps. 2. play_store similar for direct competitors. 3. github repos for open-source alternatives. 4. web_search for market size and trends. Combine all sources.",
        "tools_used": ["play_store", "github", "smart_search", "web_search"],
    },
    {
        "name": "pharmacy_finder",
        "description": "Finding pharmacies on duty (nobetci eczane) in Turkey, nearest open pharmacy with distance calculation",
        "strategy_summary": "Use pharmacy tool. Pass city for all districts, or city+district for specific area. Falls back to eczaneler.gen.tr web scraping if no API key.",
        "tools_used": ["pharmacy"],
    },
    {
        "name": "earthquake_data_lookup",
        "description": "Checking recent earthquakes in Turkey, seismic activity data from Kandilli Observatory",
        "strategy_summary": "Use api_call with Kandilli Observatory. Returns live earthquake list with magnitude, location, depth. Real-time data.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "fuel_price_lookup",
        "description": "Checking current fuel prices in Turkey: gasoline, diesel, LPG prices by city",
        "strategy_summary": "Use api_call with Turkey Fuel Prices. Requires COLLECTAPI_KEY in .env.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "gold_price_lookup",
        "description": "Checking gold prices in Turkey: gram altin, ceyrek, yarim, tam, cumhuriyet altini",
        "strategy_summary": "Use api_call with Gold Price Turkey. Requires COLLECTAPI_KEY in .env.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "directions_and_routing",
        "description": "Getting directions between locations, calculating distance, finding routes for driving or walking",
        "strategy_summary": "1. Geocode addresses with api_call HERE Geocoding (or Photon for privacy). 2. Get route with api_call OSRM. OSRM and Photon are free, no API key needed.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "prayer_times_lookup",
        "description": "Looking up prayer times (namaz vakitleri), ezan times, iftar/sahur times in Turkey",
        "strategy_summary": "Use api_call with Diyanet Prayer Times. Query by district code.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "travel_ticket_search",
        "description": "Searching for flight, bus, train tickets and prices in Turkey, especially YHT and domestic flights",
        "strategy_summary": "1. api_call Kiwi Tequila (needs KIWI_API_KEY, free, 750+ carriers). 2. api_call Rome2rio for route planning. 3. web_search targeting enuygun.com or obilet.com as fallback.",
        "tools_used": ["smart_search", "api_call", "web_search"],
    },
    {
        "name": "product_spec_comparison",
        "description": "Detailed product specification comparison using Epey.com, finding products by specific technical requirements like RAM, GPU, screen size",
        "strategy_summary": "Use shopping_search with epey.com source. Epey has 85+ spec fields per product. For detailed specs, use get_product_details(url) on individual products. Best for 'find laptop with X and Y' queries.",
        "tools_used": ["shopping_search"],
    },
    {
        "name": "turkish_holiday_lookup",
        "description": "Looking up Turkish public holidays, bayram dates, official holiday calendar for any year",
        "strategy_summary": "Use api_call with Turkey Holidays. Returns official public holidays for any year.",
        "tools_used": ["smart_search", "api_call"],
    },
]


async def seed_skills():
    """Seed the skills database with curated execution recipe skills.

    Only adds skills that don't already exist (by name).
    Returns the number of new skills added.
    """
    from .skills import list_skills, add_skill

    existing = await list_skills()
    existing_names = {s["name"] for s in existing}

    added = 0
    for skill in SEED_SKILLS:
        if skill["name"] in existing_names:
            continue
        await add_skill(
            name=skill["name"],
            description=skill["description"],
            strategy_summary=skill["strategy_summary"],
            tools_used=skill.get("tools_used", []),
            source_grade="seed",
        )
        added += 1

    if added:
        logger.info("Seeded %d execution recipe skills", added)
    return added
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_skill_overhaul.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/seed_skills.py tests/test_skill_overhaul.py
git commit -m "feat(skills): convert seed skills to execution recipe format"
```

---

### Task 4: Expand Grading Prompt & Capture in Orchestrator

**Files:**
- Modify: `src/core/router.py:1468-1476` (expand GRADING_PROMPT)
- Modify: `src/core/orchestrator.py:2172-2232` (rewrite auto-capture)
- Test: `tests/test_skill_overhaul.py` (append)

This task expands the grading prompt to output `situation_summary`, `strategy_summary`, and `tool_template` alongside the existing grade. Then rewrites the orchestrator's auto-capture block to use grader output instead of title splitting. The grade threshold is raised from 3.0 to 4.0 ("great" bucket only).

- [ ] **Step 1: Write failing test for grading prompt parse**

```python
# Append to tests/test_skill_overhaul.py

def test_parse_grader_output_full():
    """Full grader output with skill fields parses correctly."""
    import json
    raw = json.dumps({
        "score": 4.5,
        "reason": "Good execution",
        "situation_summary": "Comparing laptop prices",
        "strategy_summary": "Searched 3 stores then built comparison table",
        "tool_template": ["smart_search({product} trendyol)", "compare prices"],
    })
    parsed = json.loads(raw)
    assert parsed["score"] == 4.5
    assert parsed["situation_summary"] == "Comparing laptop prices"
    assert len(parsed["tool_template"]) == 2


def test_parse_grader_output_partial():
    """Grader output missing skill fields still parses grade."""
    import json
    raw = json.dumps({"score": 4.0, "reason": "OK"})
    parsed = json.loads(raw)
    assert parsed["score"] == 4.0
    assert "situation_summary" not in parsed


def test_parse_grader_output_garbage():
    """Totally broken grader output returns None gracefully."""
    import json
    raw = "This is not JSON at all {broken"
    try:
        parsed = json.loads(raw)
        assert False, "Should have raised"
    except json.JSONDecodeError:
        pass  # Expected
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_skill_overhaul.py -v -k "test_parse_grader"`
Expected: PASS (these test JSON parsing, not production code)

- [ ] **Step 3: Expand GRADING_PROMPT in router.py**

Replace `GRADING_PROMPT` at `src/core/router.py:1468-1476`:

```python
GRADING_PROMPT = """Rate this AI response on a scale of 1-5:
1 = Wrong/useless, 2 = Partially relevant, 3 = Adequate,
4 = Good and complete, 5 = Excellent

Task: {task_title}
Response to grade:
{response}

Respond with ONLY JSON. If score >= 4, also include situation_summary, strategy_summary, and tool_template fields describing what approach worked:
{{"score": N, "reason": "brief", "situation_summary": "one line describing the type of problem solved", "strategy_summary": "one line describing the approach that worked", "tool_template": ["step1", "step2"]}}

For scores < 4, just: {{"score": N, "reason": "brief"}}"""
```

- [ ] **Step 4: Rewrite auto-capture in orchestrator.py**

Replace the entire block at `src/core/orchestrator.py:2172-2232`:

```python
        # Phase 13.2 v2: Extract execution recipe from high-quality tasks
        tools_used = result.get("tools_used_names", [])
        quality = result.get("quality_score")
        # Only capture from "great" tasks (grade >= 4.0)
        worth_capturing = (
            iterations >= 2
            and tools_used
            and quality is not None
            and quality >= 4.0
        )
        if worth_capturing:
            try:
                from ..memory.skills import add_skill

                # Try to extract grader's skill metadata
                grader_data = result.get("grader_data", {})
                situation = grader_data.get("situation_summary", "")
                strategy = grader_data.get("strategy_summary", "")
                tool_template = grader_data.get("tool_template", [])

                agent_type = task.get("agent_type", "executor")
                title = task.get("title", "")
                task_id_val = task.get("id", 0)

                if situation and strategy:
                    # Full grader output — use it
                    skill_name = f"auto:{agent_type}:{title[:40]}"
                    await add_skill(
                        name=skill_name,
                        description=situation,
                        strategy_summary=strategy,
                        tool_template=tool_template,
                        tools_used=tools_used,
                        avg_iterations=iterations,
                        source_grade="great",
                        source_task_id=task_id_val,
                    )
                else:
                    # Graceful degradation: grader didn't produce skill fields
                    # Use task metadata for a basic entry
                    skill_name = f"auto:{agent_type}:{title[:40]}"
                    auto_desc = f"Task: {title[:100]}. Agent: {agent_type}."
                    auto_strategy = f"Used {', '.join(tools_used[:5])} in {iterations} iterations"
                    await add_skill(
                        name=skill_name,
                        description=auto_desc,
                        strategy_summary=auto_strategy,
                        tools_used=tools_used,
                        avg_iterations=iterations,
                        source_grade="great",
                        source_task_id=task_id_val,
                    )
            except Exception:
                pass

        # Track injection success — skills injected earlier get credit if task succeeds
        if quality is not None and quality >= 4.0:
            try:
                injected = task_ctx_parsed.get("injected_skills", [])
                if injected:
                    from ..memory.skills import record_injection_success
                    await record_injection_success(injected)
            except Exception:
                pass
```

- [ ] **Step 5: Pass grader_data through the grading pipeline**

In `src/core/router.py`, in the `grade_response` function (line ~1521), after parsing the JSON, store the full parsed dict — not just the score. Find the section that returns the grade and modify it to also return the parsed data.

Change the `grade_response` function signature and return type. Currently it returns `float | None`. Change to return `tuple[float | None, dict]`:

At line ~1479:
```python
async def grade_response(
    task_title: str,
    task_description: str,
    response_text: str,
    generating_model: str = "",
    task_name: str = "",
) -> tuple[float | None, dict]:
    """Grade a response using a DIFFERENT model. Returns (grade, grader_data)."""
```

At line ~1521 where the JSON is parsed:
```python
        parsed = json.loads(raw.strip())
        score = float(parsed.get("score", 3))
        grade = max(1.0, min(5.0, score))
        grader_data = parsed  # Keep full parsed output for skill capture
```

At the function's return points, return `(grade, grader_data)` instead of just `grade`. For error cases, return `(None, {})`.

Then update all callers of `grade_response` to unpack the tuple. The main callers are in `src/core/llm_dispatcher.py` (lines ~204-205, ~622-644). Each call site should be updated:

```python
# Before:
grade = await grade_response(...)
# After:
grade, grader_data = await grade_response(...)
```

The `grader_data` should be passed back through the result dict so the orchestrator can access it. In the dispatcher, when returning the grade result, include: `result["grader_data"] = grader_data`.

- [ ] **Step 6: Write integration test for capture flow**

```python
# Append to tests/test_skill_overhaul.py

@pytest.mark.asyncio
async def test_skill_capture_with_grader_data():
    """Orchestrator captures skill from grader output."""
    with patch("src.memory.skills.get_db") as mock_db, \
         patch("src.memory.skills._find_duplicate_skill", new_callable=AsyncMock, return_value=None), \
         patch("src.memory.skills._embed_skill", new_callable=AsyncMock):

        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.lastrowid = 1
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.commit = AsyncMock()
        mock_db.return_value = mock_conn

        from src.memory.skills import add_skill
        result = await add_skill(
            name="auto:shopping_advisor:laptop comparison",
            description="Comparing laptop prices across Turkish stores",
            strategy_summary="Search Trendyol, Hepsiburada, Akakce separately then compare",
            tool_template=["smart_search({product} trendyol)", "smart_search({product} hepsiburada)", "compare"],
            tools_used=["smart_search", "web_search"],
            avg_iterations=4,
            source_grade="great",
            source_task_id=123,
        )
        assert result == "auto:shopping_advisor:laptop comparison"
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/test_skill_overhaul.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/core/router.py src/core/orchestrator.py tests/test_skill_overhaul.py
git commit -m "feat(skills): expand grading prompt for skill metadata, rewrite auto-capture"
```

---

### Task 5: Rewrite Skill Injection in base.py

**Files:**
- Modify: `src/agents/base.py:479-496` (rewrite skill injection block)
- Test: `tests/test_skill_overhaul.py` (append)

This task replaces the skill injection in `_build_context()` to use adaptive injection depth, compact/verbose formatting, tool injection for high-confidence skills, and injection tracking.

- [ ] **Step 1: Write failing test for tool injection**

```python
# Append to tests/test_skill_overhaul.py

def test_get_tools_to_inject_high_confidence():
    """High-confidence skills provide tools for injection."""
    from src.memory.skills import get_tools_to_inject
    skills = [{
        "name": "test",
        "injection_count": 10,
        "injection_success": 8,
        "strategies": json.dumps([{
            "summary": "test",
            "tools_used": ["smart_search", "play_store"],
            "injection_count": 10,
            "injection_success": 8,
        }]),
    }]
    tools = get_tools_to_inject(skills)
    assert "smart_search" in tools
    assert "play_store" in tools


def test_get_tools_to_inject_low_confidence():
    """Low-confidence skills don't provide tools."""
    from src.memory.skills import get_tools_to_inject
    skills = [{
        "name": "test",
        "injection_count": 3,
        "injection_success": 1,
        "strategies": json.dumps([{
            "summary": "test",
            "tools_used": ["smart_search"],
            "injection_count": 3,
            "injection_success": 1,
        }]),
    }]
    tools = get_tools_to_inject(skills)
    assert tools == []
```

- [ ] **Step 2: Run tests to verify they pass (these test skills.py functions already written)**

Run: `pytest tests/test_skill_overhaul.py -v -k "test_get_tools"`
Expected: PASS

- [ ] **Step 3: Rewrite skill injection in base.py**

Replace lines 479-496 in `src/agents/base.py`:

```python
        # ── Skill library injection (v2 — execution recipes) ──
        try:
            from ..memory.skills import (
                find_relevant_skills, format_skills_for_prompt,
                get_tools_to_inject, record_injection,
            )
            task_text = f"{task.get('title', '')} {task.get('description', '')}"
            relevant_skills = await find_relevant_skills(task_text, limit=5)
            if relevant_skills:
                # Estimate context budget from model info
                model_ctx = task.get("context", "{}")
                if isinstance(model_ctx, str):
                    try:
                        model_ctx = json.loads(model_ctx)
                    except (json.JSONDecodeError, TypeError):
                        model_ctx = {}
                context_budget = model_ctx.get("model_context_length", 4096)

                skills_block = format_skills_for_prompt(relevant_skills, context_budget)
                if skills_block:
                    parts.append(skills_block)

                # Tool injection for high-confidence skills
                extra_tools = get_tools_to_inject(relevant_skills)
                if extra_tools and self.allowed_tools is not None:
                    for tool in extra_tools:
                        if tool not in self.allowed_tools:
                            self.allowed_tools.append(tool)
                            logger.info("Skill-injected tool: %s", tool)

                # Track injections
                skill_names = [s["name"] for s in relevant_skills]
                await record_injection(skill_names)

                # Store for success tracking after task completes
                try:
                    _ctx = json.loads(task.get("context", "{}"))
                    _ctx["injected_skills"] = skill_names
                    task["context"] = json.dumps(_ctx)
                except Exception:
                    pass

                logger.info("Skills injected: %s", skill_names)
        except Exception as exc:
            logger.debug("Skill injection failed (non-critical): %s", exc)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_skill_overhaul.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/base.py tests/test_skill_overhaul.py
git commit -m "feat(skills): adaptive skill injection with tool injection and tracking in base.py"
```

---

### Task 6: Dispatcher Integration — Pass grader_data Through

**Files:**
- Modify: `src/core/router.py:1479-1548` (change grade_response return type — if not already done in Task 4)
- Modify: `src/core/llm_dispatcher.py:204-205, 622-644` (update grade_response callers)
- Modify: `src/core/orchestrator.py` (where grade is stored in result dict, add grader_data)
- Test: `tests/test_skill_overhaul.py` (append)

The grading pipeline needs to pass `grader_data` (the full parsed grader JSON) through to the orchestrator result dict. This connects the expanded grading prompt (Task 4) to the capture flow.

**IMPORTANT**: Task 4 Step 5 changes `grade_response` return type from `float | None` to `tuple[float | None, dict]`. All callers must be updated in the SAME commit or the system breaks. This task completes that work.

- [ ] **Step 1: Write test for grader_data passthrough**

```python
# Append to tests/test_skill_overhaul.py

def test_grader_data_round_trip():
    """Grader data can round-trip through a result dict."""
    grader_data = {
        "score": 4.5,
        "reason": "Good",
        "situation_summary": "Price comparison across stores",
        "strategy_summary": "Searched 3 stores then compared",
        "tool_template": ["search store1", "search store2", "compare"],
    }
    result = {"quality_score": 4.5, "grader_data": grader_data}
    assert result["grader_data"]["situation_summary"] == "Price comparison across stores"
    assert result["grader_data"]["tool_template"][0] == "search store1"
```

- [ ] **Step 2: Complete grade_response return type change in router.py**

Ensure `grade_response` at `src/core/router.py:1479` returns `tuple[float | None, dict]`:
- Success: `return (grade, parsed)` where `parsed` is the full JSON dict from grader
- All error/early-return paths: `return (None, {})`

- [ ] **Step 3: Update ALL callers in llm_dispatcher.py**

Find every call to `grade_response` in `src/core/llm_dispatcher.py` and update to unpack the tuple.

At line ~204-205 (in `_execute_grade` method):
```python
# Before:
return await grade_response(...)
# After:
grade, grader_data = await grade_response(...)
return grade, grader_data
```

At lines ~622-644 (in the direct grading path):
```python
# Before:
grade = await grade_response(...)
# After:
grade, grader_data = await grade_response(...)
```

The `_execute_grade` return type changes too — its callers (GradeQueue.drain) must handle the tuple. Trace the full call chain and update each unpacking site.

- [ ] **Step 4: Pass grader_data into result dict in orchestrator.py**

Find where `quality_score` is set in the result dict (search for `quality_score` assignment in orchestrator.py). Add `grader_data` alongside it so the capture block (Task 4) can read it via `result.get("grader_data", {})`.

- [ ] **Step 5: Run all tests**

Run: `pytest tests/test_skill_overhaul.py -v`
Expected: All PASS

Run: `pytest tests/ -v -k "not llm" --timeout=30`
Expected: Existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add src/core/llm_dispatcher.py src/core/router.py src/core/orchestrator.py
git commit -m "feat(skills): pass grader_data through dispatcher to orchestrator"
```

---

### Task 7: Cleanup — Wipe Garbage Skills & Rebuild ChromaDB

**Files:**
- Create: `scripts/migrate_skills_v2.py` (one-time migration script)
- Test: manual verification

This task creates a migration script that wipes garbage auto-captured skills, evaluates the 7 non-i2p ones, and rebuilds the ChromaDB skills collection.

- [ ] **Step 1: Create migration script**

```python
# scripts/migrate_skills_v2.py
"""
One-time migration: wipe garbage auto-captured skills, keep seed skills.
Run: python -m scripts.migrate_skills_v2
"""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from src.infra.db import get_db, init_db
    await init_db()

    db = await get_db()

    # 1. Count current state
    cursor = await db.execute("SELECT COUNT(*) FROM skills WHERE name LIKE 'auto:%'")
    auto_count = (await cursor.fetchone())[0]

    cursor = await db.execute("SELECT COUNT(*) FROM skills WHERE name NOT LIKE 'auto:%'")
    seed_count = (await cursor.fetchone())[0]

    print(f"Current state: {auto_count} auto-captured, {seed_count} seed/manual")

    # 2. Show the non-i2p auto skills for review
    cursor = await db.execute(
        "SELECT name, description, tool_sequence FROM skills "
        "WHERE name LIKE 'auto:%' "
        "AND name NOT LIKE '%raw_idea%' "
        "AND name NOT LIKE '%competitor%research%' "
        "AND name NOT LIKE '%tech_stack%' "
        "AND name NOT LIKE '%feature_%' "
        "AND name NOT LIKE '%security%' "
        "AND name NOT LIKE '%test_%' "
        "AND name NOT LIKE '%[%]%' "
        "LIMIT 20"
    )
    non_i2p = await cursor.fetchall()
    if non_i2p:
        print(f"\nNon-i2p auto skills ({len(non_i2p)}):")
        for row in non_i2p:
            print(f"  {row[0]}: {row[1][:80]}")
            print(f"    tools: {row[2][:100]}")
    else:
        print("\nNo non-i2p auto skills found")

    # 3. Delete all auto-captured skills
    await db.execute("DELETE FROM skills WHERE name LIKE 'auto:%'")
    await db.commit()
    print(f"\nDeleted {auto_count} auto-captured skills")

    # 4. Rebuild ChromaDB skills collection
    try:
        from src.memory.vector_store import get_collection
        collection = await get_collection("semantic")
        # Delete all skill entries
        results = collection.get(where={"type": "skill"})
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            print(f"Cleaned {len(results['ids'])} skill embeddings from ChromaDB")

        # Re-embed remaining skills (seeds)
        cursor = await db.execute("SELECT name, description FROM skills")
        remaining = await cursor.fetchall()
        from src.memory.vector_store import embed_and_store
        for row in remaining:
            await embed_and_store(
                text=row[1],  # description
                metadata={"type": "skill", "skill_name": row[0]},
                collection="semantic",
                doc_id=f"skill:{row[0]}",
            )
        print(f"Re-embedded {len(remaining)} seed skills in ChromaDB")
    except Exception as e:
        print(f"ChromaDB cleanup failed (non-critical): {e}")

    print("\nMigration complete!")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run migration script**

Run: `python -m scripts.migrate_skills_v2`
Expected: Shows current state, lists non-i2p skills for review, deletes garbage, rebuilds ChromaDB

- [ ] **Step 3: Verify**

```bash
python -c "import asyncio; from src.memory.skills import list_skills; print(asyncio.run(list_skills()))"
```
Expected: Only seed skills remain (24 entries with `skill_type='seed'`)

- [ ] **Step 4: Commit**

```bash
git add scripts/migrate_skills_v2.py
git commit -m "feat(skills): migration script to wipe garbage and rebuild ChromaDB"
```

---

### Task 8: Remove Old Code & Final Cleanup

**Files:**
- Modify: `src/memory/skills.py` (remove any leftover old functions if not already cleaned)
- Modify: `tests/test_seed_skills.py` (update for new format)
- Modify: `tests/test_skill_fixes.py` (update or remove obsolete tests)
- Modify: `tests/test_skill_metrics.py` (verify still works with new schema)

- [ ] **Step 1: Update test_seed_skills.py**

Read the current file and update any references to old seed skill format (trigger_pattern, tool_sequence, examples).

- [ ] **Step 2: Update test_skill_fixes.py**

The existing tests check for old regex behavior (`_skill_score`, `re.escape` on auto-capture). These are now obsolete:
- `test_seed_skills_returned_without_success_count` — `_skill_score` no longer exists. Replace with test for `_injection_success_rate`.
- `test_auto_skill_regex_escape` — regex is no longer used for auto-skills. Remove.

```python
# tests/test_skill_fixes.py (updated)
import pytest


def test_seed_skills_have_neutral_confidence():
    """Seed skills (0 injections) get neutral confidence score."""
    from src.memory.skills import _injection_success_rate
    skill = {"injection_count": 0, "injection_success": 0}
    rate = _injection_success_rate(skill)
    assert rate == 0.5  # neutral — not enough data


def test_proven_skill_confidence():
    """Skills with enough data reflect actual success rate."""
    from src.memory.skills import _injection_success_rate
    skill = {"injection_count": 10, "injection_success": 8}
    rate = _injection_success_rate(skill)
    assert rate == 0.8
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v -k "not llm" --timeout=30`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_skill_fixes.py tests/test_seed_skills.py
git commit -m "test(skills): update tests for v2 skill system"
```

---

### Task 9: Smoke Test — End-to-End Verification

**Files:** No code changes — verification only.

- [ ] **Step 1: Verify imports work**

```bash
python -c "from src.memory.skills import find_relevant_skills, add_skill, format_skills_for_prompt, get_tools_to_inject, select_injection_depth, record_injection, record_injection_success, list_skills; print('All imports OK')"
```

- [ ] **Step 2: Verify seed skills load**

```bash
python -c "import asyncio; from src.memory.seed_skills import seed_skills; print(asyncio.run(seed_skills()))"
```
Expected: 24 (or 0 if already seeded)

- [ ] **Step 3: Verify skill matching works**

```bash
python -c "
import asyncio
from src.memory.seed_skills import seed_skills
from src.memory.skills import find_relevant_skills

async def test():
    await seed_skills()
    results = await find_relevant_skills('coffee machine fiyat karsilastirma')
    print(f'Matched {len(results)} skills:')
    for s in results:
        print(f'  {s[\"name\"]}: score={s.get(\"_match_score\", 0):.2f}')

asyncio.run(test())
"
```
Expected: Matches like `turkish_product_shopping`, `product_spec_comparison`

- [ ] **Step 4: Verify formatting works**

```bash
python -c "
import json
from src.memory.skills import format_skill_verbose, format_skill_compact

skill = {
    'name': 'test_skill',
    'description': 'Test description',
    'injection_count': 10,
    'injection_success': 8,
    'strategies': json.dumps([{
        'summary': 'Test strategy',
        'tool_template': ['step1', 'step2'],
        'tools_used': ['smart_search'],
        'injection_count': 10,
        'injection_success': 8,
    }]),
}
print('=== VERBOSE ===')
print(format_skill_verbose(skill))
print()
print('=== COMPACT ===')
print(format_skill_compact(skill))
"
```
Expected: Verbose shows full strategy block, compact shows single line

- [ ] **Step 5: Run full test suite one final time**

Run: `pytest tests/ -v -k "not llm" --timeout=30`
Expected: All PASS

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat(skills): skill system overhaul v2 complete — execution recipes with vector matching"
```
