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
async def test_dabidabi_shim_still_works(tmp_path):
    # back-compat: `from dabidabi import record_model_call` must still function (delegates to fatih)
    dabidabi.configure(str(tmp_path / "shim.db"))
    await dabidabi.init_db()
    await dabidabi.record_model_call("m2", "coder", True, cost=0.0, latency=0.5, grade=0.7)
    rows = await dabidabi.get_model_stats(model="m2")
    assert rows and rows[0]["model"] == "m2"
    await dabidabi.close_db()
