import pytest
import dabidabi
import fatih_hoca  # registers schema
from fatih_hoca import db as fdb


@pytest.mark.asyncio
async def test_record_and_get_model_stats(tmp_path):
    dabidabi.configure(str(tmp_path / "s.db"))
    await dabidabi.init_db()
    await fdb.record_model_call("m1", "coder", True, cost=0.01, latency=1.0, grade=0.8)
    rows = await fdb.get_model_stats(model="m1")
    assert rows and rows[0]["model"] == "m1"
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_get_recent_picks_read_api(tmp_path):
    dabidabi.configure(str(tmp_path / "p.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    await db.execute(
        "INSERT INTO model_pick_log (task_name, picked_model, picked_score, candidates_json) "
        "VALUES ('t', 'm1', 0.9, '[]')"
    )
    await db.commit()
    picks = await fdb.get_recent_picks(limit=10)
    assert any(p["picked_model"] == "m1" for p in picks)
    await dabidabi.close_db()


def test_update_model_stats_is_gone():
    assert not hasattr(dabidabi, "update_model_stats")


@pytest.mark.asyncio
async def test_get_pick_summary(tmp_path):
    dabidabi.configure(str(tmp_path / "ps.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    for m in ("m1", "m1", "m2"):
        await db.execute(
            "INSERT INTO model_pick_log (task_name, picked_model, picked_score, candidates_json) "
            "VALUES ('t', ?, 0.8, '[]')", (m,))
    await db.commit()
    rows = await fdb.get_pick_summary(since_days=7)
    by_model = {r["picked_model"]: r for r in rows}
    assert by_model["m1"]["picks"] == 2
    await dabidabi.close_db()


# ── §2 read-ownership helpers (registry-table READs owned by fatih_hoca) ──


@pytest.mark.asyncio
async def test_get_latest_pick_for_task_tier0_task_id(tmp_path):
    dabidabi.configure(str(tmp_path / "lpt0.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    await db.execute(
        "INSERT INTO model_pick_log (task_name, task_id, picked_model, "
        "picked_score, candidates_json, timestamp) "
        "VALUES ('T', 42, 'mm0', 0.9, '[]', '2026-06-16 10:00:00')")
    await db.commit()
    model, at = await fdb.get_latest_pick_for_task(42, "ignored-title")
    assert model == "mm0" and at == "2026-06-16 10:00:00"
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_get_latest_pick_for_task_tier1_title_fallback(tmp_path):
    dabidabi.configure(str(tmp_path / "lpt1.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    # task_id IS NULL → only the legacy title join can resolve it.
    await db.execute(
        "INSERT INTO model_pick_log (task_name, picked_model, picked_score, "
        "candidates_json) VALUES ('legacy-title', 'mm1', 0.8, '[]')")
    await db.commit()
    model, _ = await fdb.get_latest_pick_for_task(999, "legacy-title")
    assert model == "mm1"
    # No match at all → (None, None).
    none_model, none_at = await fdb.get_latest_pick_for_task(999, "nope")
    assert none_model is None and none_at is None
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_get_latest_model_for_mission_tier0_and_reinforce_excluded(tmp_path):
    dabidabi.configure(str(tmp_path / "lmm.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    await db.execute(
        "INSERT INTO tasks (id, mission_id, title) VALUES (10, 7, 'task A')")
    await db.execute(
        "INSERT INTO model_pick_log (task_name, task_id, picked_model, provider, "
        "call_category, picked_score, candidates_json, timestamp) "
        "VALUES ('task A', 10, 'mm', 'cloud', 'main', 0.9, '[]', '2026-06-16 09:00:00')")
    # Newer row, but it's a reinforce nudge → must be excluded.
    await db.execute(
        "INSERT INTO model_pick_log (task_name, task_id, picked_model, provider, "
        "call_category, picked_score, candidates_json, timestamp) "
        "VALUES ('task A', 10, 'reinf', 'cloud', 'reinforce', 0.5, '[]', '2026-06-16 11:00:00')")
    await db.commit()
    model, provider = await fdb.get_latest_model_for_mission(7)
    assert model == "mm" and provider == "cloud"
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_get_latest_model_for_mission_global_fallback(tmp_path):
    dabidabi.configure(str(tmp_path / "lmm2.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    await db.execute(
        "INSERT INTO model_pick_log (task_name, picked_model, provider, "
        "call_category, picked_score, candidates_json) "
        "VALUES ('t', 'glob', NULL, 'main', 0.7, '[]')")
    await db.commit()
    # mission_id=None → tier-2 global; NULL provider defaults to 'local'.
    model, provider = await fdb.get_latest_model_for_mission(None)
    assert model == "glob" and provider == "local"
    # Empty DB → (None, 'local').
    await db.execute("DELETE FROM model_pick_log")
    await db.commit()
    assert await fdb.get_latest_model_for_mission(None) == (None, "local")
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_get_action_events(tmp_path):
    dabidabi.configure(str(tmp_path / "ae.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    await db.execute(
        "INSERT INTO registry_events (scope, target, event, mission_id, verb, "
        "reversibility, payload_json) VALUES ('action', 'task:1', 'executed', "
        "3, 'git_commit', 'reversible', '{\"status\": \"ok\"}')")
    # Different mission + non-action scope must NOT leak in.
    await db.execute(
        "INSERT INTO registry_events (scope, target, event, mission_id, verb) "
        "VALUES ('action', 'task:2', 'executed', 99, 'other')")
    await db.execute(
        "INSERT INTO registry_events (scope, target, event, mission_id, verb) "
        "VALUES ('model', 'm1', 'register', 3, 'register')")
    await db.commit()
    rows = await fdb.get_action_events(3, limit=20)
    assert len(rows) == 1
    verb, rev, payload_json, _ts = rows[0]
    assert verb == "git_commit" and rev == "reversible"
    assert '"status"' in payload_json
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_record_confidence_claim_uses_fatih_read_helper(tmp_path):
    # End-to-end proof that the dabidabi engine's record_confidence_claim now
    # resolves the model via fatih_hoca.db.get_latest_pick_for_task (lazy
    # import across the dabidabi→fatih_hoca boundary). §2 read-ownership.
    dabidabi.configure(str(tmp_path / "rcc.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    await db.execute(
        "INSERT INTO tasks (id, mission_id, title, agent_type) "
        "VALUES (5, 7, 'taskA', 'coder')")
    await db.execute(
        "UPDATE tasks SET confidence_numeric=0.8, confidence_categorical='high' "
        "WHERE id=5")
    await db.execute(
        "INSERT INTO model_pick_log (task_name, task_id, picked_model, provider, "
        "call_category, picked_score, candidates_json) "
        "VALUES ('taskA', 5, 'mm', 'cloud', 'main', 0.9, '[]')")
    await db.commit()
    rid = await dabidabi.record_confidence_claim(5)
    assert rid is not None
    cur = await db.execute(
        "SELECT model_id FROM confidence_outcomes WHERE id = ?", (rid,))
    row = await cur.fetchone()
    assert row[0] == "mm"  # resolved via fatih helper, not the unknown:: fallback
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_dabidabi_shim_still_works(tmp_path):
    # back-compat: `from dabidabi import record_model_call` must still function (delegates to fatih)
    dabidabi.configure(str(tmp_path / "shim.db"))
    await dabidabi.init_db()
    await dabidabi.record_model_call("m2", "coder", True, cost=0.0, latency=0.5, grade=0.7)
    rows = await dabidabi.get_model_stats(model="m2")
    assert rows and rows[0]["model"] == "m2"
    await dabidabi.close_db()
