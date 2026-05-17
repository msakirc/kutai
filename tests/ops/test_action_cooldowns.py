"""Z8 T4B — action_cooldowns + Mr. Roboto pre-execute enforcement."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "cooldowns.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.mark.asyncio
async def test_table_exists(tmp_path, monkeypatch):
    db_mod = await _setup(tmp_path, monkeypatch)
    conn = await db_mod.get_db()
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='action_cooldowns'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None, "action_cooldowns migration didn't run"


@pytest.mark.asyncio
async def test_check_passes_when_no_history(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from src.ops.action_cooldowns import check

    assert await check(mission_id=1, verb="rollback_to_last_green") is True


@pytest.mark.asyncio
async def test_rollback_blocked_after_2_per_hour(tmp_path, monkeypatch):
    """rollback_to_last_green has max_per_hour=2; the 3rd must block."""
    await _setup(tmp_path, monkeypatch)
    from src.ops.action_cooldowns import check, record

    for _ in range(2):
        assert await check(mission_id=1, verb="rollback_to_last_green") is True
        await record(1, "rollback_to_last_green", "ok")
    # 3rd call within the hour must block.
    assert await check(mission_id=1, verb="rollback_to_last_green") is False


@pytest.mark.asyncio
async def test_cooldown_scoped_per_mission(tmp_path, monkeypatch):
    """Hitting rollback ceiling on mission 1 must not block mission 2."""
    await _setup(tmp_path, monkeypatch)
    from src.ops.action_cooldowns import check, record

    for _ in range(2):
        await record(1, "rollback_to_last_green", "ok")
    assert await check(1, "rollback_to_last_green") is False
    assert await check(2, "rollback_to_last_green") is True


@pytest.mark.asyncio
async def test_rotate_key_daily_cap(tmp_path, monkeypatch):
    """rotate_failed_key has max_per_day=1 — second attempt must block."""
    await _setup(tmp_path, monkeypatch)
    from src.ops.action_cooldowns import check, record

    assert await check(1, "rotate_failed_key") is True
    await record(1, "rotate_failed_key", "ok")
    assert await check(1, "rotate_failed_key") is False


@pytest.mark.asyncio
async def test_unknown_verb_does_not_block(tmp_path, monkeypatch):
    """Verbs without explicit policy fall through (max_per_hour=999)."""
    await _setup(tmp_path, monkeypatch)
    from src.ops.action_cooldowns import check, record

    for _ in range(5):
        assert await check(1, "novel_verb_no_policy") is True
        await record(1, "novel_verb_no_policy", "ok")


# ─────────────────────── oncall_action executor wiring ────────────────


def _make_task(mission_id, verb, params=None):
    return {
        "id": 1,
        "mission_id": mission_id,
        "payload": {
            "action": "oncall_action",
            "verb": verb,
            "params": params or {},
        },
    }


@pytest.mark.asyncio
async def test_oncall_action_blocks_when_cooldown_exhausted(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from mr_roboto.executors.oncall_action import run

    for _ in range(2):
        res = await run(_make_task(1, "rollback_to_last_green"))
        # stub verbs return not_implemented (honest failure, not ok)
        assert res["status"] == "not_implemented"
    res = await run(_make_task(1, "rollback_to_last_green"))
    assert res["status"] == "blocked_by_cooldown"
    assert res["verb"] == "rollback_to_last_green"


@pytest.mark.asyncio
async def test_oncall_action_refuses_non_whitelisted(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from mr_roboto.executors.oncall_action import run

    res = await run(_make_task(1, "delete_production_database"))
    assert res["status"] == "refused_not_whitelisted"
    assert "delete_production_database" in res["verb"]


@pytest.mark.asyncio
async def test_oncall_action_records_after_invocation(tmp_path, monkeypatch):
    """Cooldown counter increments even for not_implemented stubs — the verb
    was attempted (whitelist + cooldown passed); the cooldown must still fire."""
    db_mod = await _setup(tmp_path, monkeypatch)
    from mr_roboto.executors.oncall_action import run

    res = await run(_make_task(7, "restart_service", {"svc": "api"}))
    # stub verbs return not_implemented, not ok
    assert res["status"] == "not_implemented"
    assert res["verb"] == "restart_service"
    assert "not implemented" in res["error"]
    conn = await db_mod.get_db()
    async with conn.execute(
        "SELECT COUNT(*) FROM action_cooldowns "
        "WHERE mission_id=? AND verb=?",
        (7, "restart_service"),
    ) as cur:
        (n,) = await cur.fetchone()
    assert n == 1


@pytest.mark.asyncio
async def test_oncall_action_block_does_not_record(tmp_path, monkeypatch):
    """Blocked invocations must not bump the counter (would deadlock)."""
    db_mod = await _setup(tmp_path, monkeypatch)
    from mr_roboto.executors.oncall_action import run

    for _ in range(2):
        await run(_make_task(2, "rollback_to_last_green"))
    # blocked
    await run(_make_task(2, "rollback_to_last_green"))
    conn = await db_mod.get_db()
    async with conn.execute(
        "SELECT COUNT(*) FROM action_cooldowns "
        "WHERE mission_id=? AND verb=?",
        (2, "rollback_to_last_green"),
    ) as cur:
        (n,) = await cur.fetchone()
    assert n == 2  # not 3 — block doesn't record


@pytest.mark.asyncio
async def test_oncall_action_missing_verb_fails(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from mr_roboto.executors.oncall_action import run

    res = await run({"id": 1, "mission_id": 1, "payload": {"action": "oncall_action"}})
    assert res["status"] == "failed"
    assert "verb" in res["error"]


@pytest.mark.asyncio
async def test_oncall_action_routes_through_mr_roboto_dispatch(tmp_path, monkeypatch):
    """End-to-end: mr_roboto.run() dispatches oncall_action correctly."""
    await _setup(tmp_path, monkeypatch)
    import mr_roboto

    task = {
        "id": 10,
        "mission_id": 5,
        "payload": {
            "action": "oncall_action",
            "verb": "scale_up",
            "params": {"replicas": 3},
        },
    }
    action = await mr_roboto.run(task)
    # not_implemented stubs propagate as failed so the workflow engine
    # and on-call agent see a genuine failure and escalate.
    assert action.status == "failed"
    assert action.result["status"] == "not_implemented"
    assert action.result["verb"] == "scale_up"
