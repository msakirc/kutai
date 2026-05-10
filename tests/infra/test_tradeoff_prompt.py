"""Z10 T3A D5 — tradeoff prompt cron idempotency at 75/25 burn."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "trade.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


async def _stub_post_event(monkeypatch):
    """Replace mission_events.post_event with a recording stub."""
    calls: list[dict] = []

    async def fake_post_event(bot, mission_id, kind, payload, chat_id=None):
        calls.append({
            "mission_id": mission_id,
            "kind": kind,
            "payload": payload,
        })
        return 1000 + len(calls)  # synthetic event id

    from src.app import mission_events
    monkeypatch.setattr(mission_events, "post_event", fake_post_event)
    return calls


@pytest.mark.asyncio
async def test_first_run_posts_one_asking_event(tmp_path, monkeypatch):
    db = await _setup(tmp_path, monkeypatch)
    calls = await _stub_post_event(monkeypatch)

    conn = await db.get_db()
    # Budget=4, elapsed=3.5 → burn=87.5% (> 75%); 2 of 4 tasks pending
    # → scope_remaining 50% (> 25%).
    await conn.execute(
        "INSERT INTO missions (id, title, description, status, "
        "                       time_budget_hours) "
        "VALUES (60, 'late mission', '', 'active', 4.0)",
    )
    # Completed: 3.5h.
    await conn.execute(
        "INSERT INTO tasks (mission_id, title, description, agent_type, "
        " status, started_at, completed_at, estimated_cost_usd) "
        "VALUES (60, 'A', '', 'executor', 'completed', "
        " '2026-05-10 09:00:00', '2026-05-10 12:30:00', 0.50)",
    )
    await conn.execute(
        "INSERT INTO tasks (mission_id, title, description, agent_type, "
        " status, started_at, completed_at, estimated_cost_usd) "
        "VALUES (60, 'B', '', 'executor', 'completed', "
        " '2026-05-10 09:00:00', '2026-05-10 09:00:00', 0.10)",
    )
    # Pending: 2 tasks with cost.
    await conn.execute(
        "INSERT INTO tasks (mission_id, title, description, agent_type, "
        " status, estimated_cost_usd) "
        "VALUES (60, 'C', '', 'executor', 'pending', 1.00)",
    )
    await conn.execute(
        "INSERT INTO tasks (mission_id, title, description, agent_type, "
        " status, estimated_cost_usd) "
        "VALUES (60, 'D', '', 'executor', 'pending', 2.00)",
    )
    await conn.commit()

    from src.infra.mission_pacing_cron import check_and_post_tradeoff_prompts
    n = await check_and_post_tradeoff_prompts()
    assert n == 1
    assert len(calls) == 1
    assert calls[0]["kind"] == "asking"
    assert calls[0]["mission_id"] == 60
    assert "suggested_cut" in calls[0]["payload"]

    # Second invocation today is idempotent.
    n2 = await check_and_post_tradeoff_prompts()
    assert n2 == 0
    assert len(calls) == 1

    # Log row exists.
    cur = await conn.execute(
        "SELECT COUNT(*) FROM mission_tradeoff_prompts WHERE mission_id = 60",
    )
    cnt = (await cur.fetchone())[0]
    assert cnt == 1


@pytest.mark.asyncio
async def test_no_prompt_when_burn_below_threshold(tmp_path, monkeypatch):
    db = await _setup(tmp_path, monkeypatch)
    calls = await _stub_post_event(monkeypatch)
    conn = await db.get_db()
    await conn.execute(
        "INSERT INTO missions (id, title, description, status, "
        "                       time_budget_hours) "
        "VALUES (61, 'pacing fine', '', 'active', 10.0)",
    )
    await conn.execute(
        "INSERT INTO tasks (mission_id, title, description, agent_type, "
        " status, started_at, completed_at) "
        "VALUES (61, 'A', '', 'executor', 'completed', "
        " '2026-05-10 09:00:00', '2026-05-10 10:00:00')",
    )
    await conn.commit()
    from src.infra.mission_pacing_cron import check_and_post_tradeoff_prompts
    n = await check_and_post_tradeoff_prompts()
    assert n == 0
    assert calls == []
