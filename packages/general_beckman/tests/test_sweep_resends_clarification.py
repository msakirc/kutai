"""waiting_human escalation reminders must re-send the ORIGINAL interactive
clarification, not just a bare "Task #N needs your input" pointer.

Founder complaint (2026-06-24): the 48h reminder for task #525000
([0.6a] non_goals_confirm) carried only the title — to act, the founder had
to scroll the entire chat history for the original question + OK/Regenerate/
Edit buttons. Each escalation tier (4h nudge / 24h / 48h) now also enqueues a
mechanical ``resend_clarification`` task targeting the waiting row, so the
reminder is self-contained.
"""
from __future__ import annotations

import json

import pytest
import aiosqlite


def _reset_db(db_module, db_path: str):
    db_module._db_connection = None
    db_module._db_connection_path = None
    db_module.DB_PATH = db_path


_CLARIFY_CTX = {
    "executor": "mechanical",
    "payload": {
        "action": "clarify",
        "kind": "non_goals_confirm",
        "question": "Mission-wide non-goals draft below.",
        "attach_file_paths": ["mission_89/.charter/non_goals.md"],
    },
}


async def _insert_waiting(db_path, *, hours_ago, escalation_count):
    ctx = dict(_CLARIFY_CTX, escalation_count=escalation_count)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO tasks (title, status, agent_type, runner, context, "
            "started_at) VALUES ('[0.6a] non_goals_confirm', 'waiting_human', "
            "'mechanical', 'mechanical', ?, datetime('now', ?))",
            (json.dumps(ctx), f"-{hours_ago} hours"),
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        tid = (await cur.fetchone())[0]
        await db.commit()
    return tid


async def _resend_tasks_for(db_path, source_tid):
    """Pending mechanical resend_clarification tasks targeting source_tid."""
    out = []
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT context FROM tasks WHERE agent_type='mechanical' "
            "AND status='pending'"
        )
        for row in await cur.fetchall():
            ctx = json.loads(row["context"] or "{}")
            payload = ctx.get("payload") or {}
            if (payload.get("action") == "resend_clarification"
                    and payload.get("source_task_id") == source_tid):
                out.append(payload)
    return out


async def _run_sweep(monkeypatch, db_module, db_path):
    _reset_db(db_module, db_path)
    import general_beckman.sweep as sweep_mod

    async def _noop(*a, **k):
        return None

    # Silence the plain-text header notify; let _resend_clarification run for
    # real so it lands a row we can assert on.
    monkeypatch.setattr(sweep_mod, "_notify", _noop)
    await sweep_mod.sweep_queue()


@pytest.mark.asyncio
async def test_sweep_48h_tier_resends_clarification(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    import src.infra.db as db_module
    _reset_db(db_module, db_path)
    from src.infra.db import init_db
    await init_db()

    # escalation_count=1 + waited 49h → tier 2 (48h urgent).
    tid = await _insert_waiting(db_path, hours_ago=49, escalation_count=1)
    await _run_sweep(monkeypatch, db_module, db_path)

    resends = await _resend_tasks_for(db_path, tid)
    assert len(resends) == 1, "48h tier must enqueue exactly one re-send"


@pytest.mark.asyncio
async def test_sweep_24h_tier_resends_clarification(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    import src.infra.db as db_module
    _reset_db(db_module, db_path)
    from src.infra.db import init_db
    await init_db()

    # escalation_count=0 + waited 25h → tier 1 (24h reminder).
    tid = await _insert_waiting(db_path, hours_ago=25, escalation_count=0)
    await _run_sweep(monkeypatch, db_module, db_path)

    resends = await _resend_tasks_for(db_path, tid)
    assert len(resends) == 1, "24h tier must enqueue exactly one re-send"


@pytest.mark.asyncio
async def test_sweep_4h_nudge_resends_clarification(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    import src.infra.db as db_module
    _reset_db(db_module, db_path)
    from src.infra.db import init_db
    await init_db()

    # waited 5h (between 4h and 24h), no nudge yet → tier 0 (4h gentle nudge).
    tid = await _insert_waiting(db_path, hours_ago=5, escalation_count=0)
    await _run_sweep(monkeypatch, db_module, db_path)

    resends = await _resend_tasks_for(db_path, tid)
    assert len(resends) == 1, "4h nudge must enqueue exactly one re-send"
