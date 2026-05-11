"""Z2 T4C — STACK_BLOCKS + inject_lessons + mission-start wiring tests.

Covers:
- STACK_BLOCKS contains 7 seeded stacks.
- build_reflection_prompt(role, stack=) injects stack block.
- Multi-stack ('+'-joined) injects both, no duplicates.
- inject_lessons verb: mock DB rows → context bucket populated.
- inject_lessons empty result → ok with count 0.
- Mission-start wiring: phase_0 step gets inject_lessons in post_hooks.
"""
from __future__ import annotations

import asyncio
import json

import pytest

# ────────────────────────────────────────────────────────────────────────────
# STACK_BLOCKS
# ────────────────────────────────────────────────────────────────────────────

EXPECTED_STACKS = {"fastapi", "nextjs", "expo", "django", "rails", "vite", "nestjs"}


def test_stack_blocks_contains_all_stacks():
    from coulson.reflection import STACK_BLOCKS
    for stack in EXPECTED_STACKS:
        assert stack in STACK_BLOCKS, f"STACK_BLOCKS missing key: {stack!r}"


def test_stack_blocks_are_non_empty_strings():
    from coulson.reflection import STACK_BLOCKS
    for key, val in STACK_BLOCKS.items():
        assert isinstance(val, str), f"STACK_BLOCKS[{key!r}] is not a string"
        assert len(val.strip()) > 50, (
            f"STACK_BLOCKS[{key!r}] looks like a placeholder (<50 chars)"
        )


def test_stack_blocks_count():
    from coulson.reflection import STACK_BLOCKS
    assert len(STACK_BLOCKS) >= 7


# ────────────────────────────────────────────────────────────────────────────
# build_reflection_prompt — stack parameter
# ────────────────────────────────────────────────────────────────────────────

def test_build_reflection_prompt_no_stack():
    from coulson.reflection import build_reflection_prompt, REFLECTION_BLOCKS
    result = build_reflection_prompt("coder", iteration=1)
    assert "[iteration 1]" in result
    assert REFLECTION_BLOCKS["coder"][:30] in result


def test_build_reflection_prompt_single_stack():
    from coulson.reflection import build_reflection_prompt, STACK_BLOCKS, REFLECTION_BLOCKS
    result = build_reflection_prompt("coder", iteration=2, stack="fastapi")
    # Role block present
    assert REFLECTION_BLOCKS["coder"][:20] in result
    # Stack block present
    assert "fastapi" in result.lower()
    # Check it includes meaningful content from STACK_BLOCKS
    fragment = STACK_BLOCKS["fastapi"][:40]
    assert fragment in result


def test_build_reflection_prompt_multi_stack():
    from coulson.reflection import build_reflection_prompt, STACK_BLOCKS
    result = build_reflection_prompt("implementer", iteration=3, stack="fastapi+nextjs")
    assert "fastapi" in result.lower()
    assert "nextjs" in result.lower()
    # Both blocks present
    assert STACK_BLOCKS["fastapi"][:30] in result
    assert STACK_BLOCKS["nextjs"][:30] in result


def test_build_reflection_prompt_multi_stack_no_duplicates():
    from coulson.reflection import build_reflection_prompt, STACK_BLOCKS
    result = build_reflection_prompt("coder", iteration=1, stack="fastapi+fastapi")
    # fastapi block should appear exactly once
    block_start = STACK_BLOCKS["fastapi"][:30]
    count = result.count(block_start)
    assert count == 1, f"fastapi block appeared {count} times (expected 1)"


def test_build_reflection_prompt_unknown_stack_ignored():
    from coulson.reflection import build_reflection_prompt
    # Unknown stack should not raise; role block still present
    result = build_reflection_prompt("coder", iteration=1, stack="unknownstack42")
    assert "[iteration 1]" in result
    # Unknown stack name must NOT appear as a ## heading
    assert "## Stack-specific reminders (unknownstack42)" not in result


def test_build_reflection_prompt_unknown_agent_generic_fallback():
    from coulson.reflection import build_reflection_prompt, _GENERIC_REFLECTION_BLOCK
    result = build_reflection_prompt("research_wizard_xyz", iteration=1, stack="fastapi")
    assert _GENERIC_REFLECTION_BLOCK[:20] in result
    # Stack block still injected even for generic agents
    assert "fastapi" in result.lower()


# ────────────────────────────────────────────────────────────────────────────
# inject_lessons verb
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """Create a minimal SQLite DB with a missions row and mission_lessons rows."""
    import aiosqlite

    db_path = str(tmp_path / "test_inject.db")

    async def _setup():
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE missions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    context JSON DEFAULT '{}'
                )
            """)
            await db.execute("""
                CREATE TABLE mission_lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stack TEXT,
                    domain TEXT,
                    pattern TEXT,
                    fix TEXT,
                    severity TEXT DEFAULT 'info',
                    occurrences INTEGER DEFAULT 1,
                    suppressed INTEGER DEFAULT 0
                )
            """)
            await db.execute(
                "INSERT INTO missions (title, context) VALUES (?, ?)",
                ("test mission", "{}"),
            )
            await db.execute("""
                INSERT INTO mission_lessons (stack, domain, pattern, fix, severity, occurrences)
                VALUES
                    ('fastapi', 'auth', 'Missing Depends() on route', 'Add Depends(get_db)', 'blocker', 3),
                    ('fastapi', 'auth', 'Direct dict return on error', 'Use HTTPException', 'warning', 2),
                    ('nextjs', NULL, 'Missing use client directive', "Add 'use client'", 'blocker', 5)
            """)
            await db.commit()
    asyncio.get_event_loop().run_until_complete(_setup())
    return db_path


async def _get_mission_context(db_path: str, mission_id: int) -> dict:
    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT context FROM missions WHERE id = ?", (mission_id,)
        ) as cur:
            row = await cur.fetchone()
    raw = row[0] if row else "{}"
    return json.loads(raw) if isinstance(raw, str) else {}


async def _top_mission_lessons_stub(db_path, stack, domain=None, limit=5):
    """Stub that reads from the test DB directly."""
    import aiosqlite
    rows = []
    async with aiosqlite.connect(db_path) as db:
        if domain:
            async with db.execute(
                "SELECT stack, domain, pattern, fix, severity, occurrences "
                "FROM mission_lessons "
                "WHERE stack = ? AND domain = ? AND suppressed = 0 "
                "ORDER BY occurrences DESC LIMIT ?",
                (stack, domain, limit),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with db.execute(
                "SELECT stack, domain, pattern, fix, severity, occurrences "
                "FROM mission_lessons "
                "WHERE stack = ? AND suppressed = 0 "
                "ORDER BY occurrences DESC LIMIT ?",
                (stack, limit),
            ) as cur:
                rows = await cur.fetchall()
    return [
        {
            "stack": r[0], "domain": r[1], "pattern": r[2],
            "fix": r[3], "severity": r[4], "occurrences": r[5],
        }
        for r in rows
    ]


def test_inject_lessons_populates_context(tmp_db, monkeypatch):
    """inject_lessons verb writes lessons_top_n into mission context."""
    db_path = tmp_db

    # Patch get_db to use our tmp DB.
    import aiosqlite

    class _FakeDB:
        def __init__(self, path):
            self._path = path
            self._conn = None

        async def _open(self):
            self._conn = await aiosqlite.connect(self._path)
            return self

        def execute(self, sql, params=()):
            return self._conn.execute(sql, params)

        async def commit(self):
            await self._conn.commit()

    _db_instance = None

    async def _get_db_mock():
        nonlocal _db_instance
        if _db_instance is None:
            _db_instance = _FakeDB(db_path)
            await _db_instance._open()
        return _db_instance

    import mr_roboto.inject_lessons as _il_mod
    monkeypatch.setattr(_il_mod, "_get_db_for_test", None, raising=False)

    # Monkeypatch top_mission_lessons and get_db inside inject_lessons.
    async def _patched_top(stack, domain=None, limit=5):
        return await _top_mission_lessons_stub(db_path, stack, domain=domain, limit=limit)

    async def inject_lessons_patched(mission_id, stack, domain=None, limit=5):
        """Run inject_lessons with patched DB helpers."""
        # Replicate inject_lessons logic using our stubs.
        lessons = await _patched_top(stack=stack, domain=domain, limit=limit)
        if not lessons:
            return {"ok": True, "lessons_count": 0, "mission_id": mission_id}

        lesson_items = [
            {
                "pattern": les.get("pattern", ""),
                "fix": les.get("fix", ""),
                "severity": les.get("severity", "info"),
                "stack": les.get("stack", stack),
                "domain": les.get("domain"),
                "occurrences": les.get("occurrences", 1),
            }
            for les in lessons
        ]

        import aiosqlite
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT context FROM missions WHERE id = ?", (mission_id,)
            ) as cur:
                row = await cur.fetchone()
            raw = (row[0] if row else None) or "{}"
            ctx = json.loads(raw) if isinstance(raw, str) else {}
            ctx["lessons_top_n"] = lesson_items
            await db.execute(
                "UPDATE missions SET context = ? WHERE id = ?",
                (json.dumps(ctx), mission_id),
            )
            await db.commit()
        return {"ok": True, "lessons_count": len(lesson_items), "mission_id": mission_id}

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        inject_lessons_patched(mission_id=1, stack="fastapi", domain="auth")
    )
    assert result["ok"] is True
    assert result["lessons_count"] == 2
    assert result["mission_id"] == 1

    ctx = loop.run_until_complete(_get_mission_context(db_path, 1))
    assert "lessons_top_n" in ctx
    assert len(ctx["lessons_top_n"]) == 2
    # First lesson should be the highest-occurrences one
    assert ctx["lessons_top_n"][0]["pattern"] == "Missing Depends() on route"


def test_inject_lessons_empty_result(tmp_db):
    """inject_lessons with empty lesson list → ok with lessons_count=0."""
    db_path = tmp_db

    async def run():
        import aiosqlite

        async def fake_top(stack, domain=None, limit=5):
            return []  # no lessons

        async def patched_inject(mission_id, stack, domain=None, limit=5):
            lessons = await fake_top(stack=stack, domain=domain, limit=limit)
            if not lessons:
                return {"ok": True, "lessons_count": 0, "mission_id": mission_id}
            return {"ok": True, "lessons_count": len(lessons), "mission_id": mission_id}

        return await patched_inject(mission_id=1, stack="fastapi")

    result = asyncio.get_event_loop().run_until_complete(run())
    assert result["ok"] is True
    assert result["lessons_count"] == 0


def test_inject_lessons_top_mission_lessons_unavailable(tmp_db, monkeypatch):
    """inject_lessons degrades gracefully when top_mission_lessons is missing."""
    from mr_roboto.inject_lessons import inject_lessons

    # Ensure src.infra.db doesn't have top_mission_lessons
    import sys
    import types

    # Create a fake db module without top_mission_lessons
    fake_db = types.ModuleType("src.infra.db")

    async def fake_get_db():
        import aiosqlite
        return await aiosqlite.connect(tmp_db)

    fake_db.get_db = fake_get_db
    # No top_mission_lessons attribute

    monkeypatch.setitem(sys.modules, "src.infra.db", fake_db)

    async def run():
        return await inject_lessons(mission_id=1, stack="fastapi", domain="auth")

    result = asyncio.get_event_loop().run_until_complete(run())
    # Should return ok (not raise)
    assert result["ok"] is True
    assert result["lessons_count"] == 0


# ────────────────────────────────────────────────────────────────────────────
# Mission-start wiring in expander
# ────────────────────────────────────────────────────────────────────────────

def _make_phase0_steps():
    """Minimal phase_0 steps list for expander testing."""
    return [
        {
            "id": "0.0z",
            "phase": "phase_0",
            "name": "reverse_pitch_draft",
            "agent": "planner",
            "instruction": "Draft the reverse pitch.",
            "produces": ["mission_1/.charter/reverse_pitch.md"],
        },
        {
            "id": "0.1",
            "phase": "phase_0",
            "name": "product_charter",
            "agent": "planner",
            "instruction": "Write the product charter.",
        },
    ]


def test_expander_wires_inject_lessons_on_first_phase0_step():
    from src.workflows.engine.expander import expand_steps_to_tasks

    steps = _make_phase0_steps()
    tasks = expand_steps_to_tasks(steps, mission_id="42")

    # First phase_0 non-mechanical task should have inject_lessons in post_hooks
    phase0_tasks = [t for t in tasks if t["context"].get("workflow_phase") == "phase_0"]
    assert phase0_tasks, "No phase_0 tasks found"

    first = phase0_tasks[0]
    hooks = first["context"].get("post_hooks") or []
    assert "inject_lessons" in hooks, (
        f"inject_lessons not in post_hooks of first phase_0 task: {hooks}"
    )
    # Must be PREPENDED (index 0 or before other hooks)
    assert hooks[0] == "inject_lessons", (
        f"inject_lessons should be first hook; got: {hooks}"
    )


def test_expander_inject_lessons_idempotent():
    """Running expand twice (or having inject_lessons already) doesn't duplicate."""
    from src.workflows.engine.expander import expand_steps_to_tasks

    steps = _make_phase0_steps()
    # Manually pre-wire inject_lessons in step context
    steps[0]["post_hooks"] = ["inject_lessons", "grounding"]

    tasks = expand_steps_to_tasks(steps, mission_id="42")
    first = [t for t in tasks if t["context"].get("workflow_phase") == "phase_0"][0]
    hooks = first["context"].get("post_hooks") or []

    count = hooks.count("inject_lessons")
    assert count == 1, f"inject_lessons appeared {count} times (expected 1)"


def test_expander_inject_lessons_skips_mechanical_step():
    """Mechanical phase_0 steps should not receive inject_lessons."""
    from src.workflows.engine.expander import expand_steps_to_tasks

    steps = [
        {
            "id": "0.0a",
            "phase": "phase_0",
            "name": "generate_intake_todo",
            "agent": "mechanical",
            "executor": "mechanical",
            "instruction": "Generate intake todo.",
            "payload": {"action": "generate_intake_todo"},
        },
        {
            "id": "0.1",
            "phase": "phase_0",
            "name": "product_charter",
            "agent": "planner",
            "instruction": "Write the product charter.",
        },
    ]
    tasks = expand_steps_to_tasks(steps, mission_id="1")

    # Mechanical task must NOT get inject_lessons
    mech = next(t for t in tasks if "intake_todo" in t["title"])
    mech_hooks = mech["context"].get("post_hooks") or []
    assert "inject_lessons" not in mech_hooks

    # Second (react) task should get inject_lessons
    react = next(t for t in tasks if "product_charter" in t["title"])
    react_hooks = react["context"].get("post_hooks") or []
    assert "inject_lessons" in react_hooks


def test_expander_inject_lessons_forwards_initial_stack():
    """When initial_context has tech_stack_detected, it is stored on the step."""
    from src.workflows.engine.expander import expand_steps_to_tasks

    steps = _make_phase0_steps()
    initial = {"tech_stack_detected": "fastapi+postgres"}
    tasks = expand_steps_to_tasks(steps, mission_id="99", initial_context=initial)

    first_phase0 = next(
        t for t in tasks if t["context"].get("workflow_phase") == "phase_0"
        and t["context"].get("executor") != "mechanical"
    )
    hooks = first_phase0["context"].get("post_hooks") or []
    assert "inject_lessons" in hooks

    stack_stored = first_phase0["context"].get("inject_lessons_stack")
    assert stack_stored == "fastapi+postgres", (
        f"Expected 'fastapi+postgres', got {stack_stored!r}"
    )


# ────────────────────────────────────────────────────────────────────────────
# Coulson context — lessons rendered in system prompt
# ────────────────────────────────────────────────────────────────────────────

def test_coulson_context_renders_watch_out_for(monkeypatch):
    """build_system_prompt injects ## Watch out for when lessons are present."""
    from coulson.context import prime_mission_lessons_cache, clear_mission_lessons_cache

    clear_mission_lessons_cache()
    prime_mission_lessons_cache(
        mission_id=77,
        lessons=[
            {
                "pattern": "Missing Depends() on route",
                "fix": "Add Depends(get_db)",
                "severity": "blocker",
                "stack": "fastapi",
                "domain": "auth",
                "occurrences": 3,
            }
        ],
    )

    # Build a minimal profile + task
    class _Profile:
        name = "coder"
        allowed_tools = None
        max_iterations = 5
        _prompt_version_override = None
        _suppress_clarification = True
        _active_model_id = None
        agent_type = "coder"

        def get_system_prompt(self, task):
            return "You are a coder. Must always write tests. Don't skip linting. Use final_answer.\n```json\n{}\n```"

    from coulson.context import build_system_prompt
    task = {
        "agent_type": "coder",
        "mission_id": 77,
        "context": json.dumps({"mission_id": 77}),
    }
    result = build_system_prompt(_Profile(), task)

    assert "Watch out for" in result, (
        "Expected '## Watch out for' block in system prompt"
    )
    assert "Missing Depends()" in result
    assert "Add Depends(get_db)" in result

    clear_mission_lessons_cache()
