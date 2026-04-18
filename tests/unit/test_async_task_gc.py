"""Regression tests: fire-and-forget asyncio.create_task must retain strong refs."""
import asyncio
import gc
import logging
import pytest

from fatih_hoca import selector as selector_mod


@pytest.mark.asyncio
async def test_selector_pick_telemetry_task_not_gcd(caplog, tmp_path, monkeypatch):
    """Scheduling _write() must not produce 'Task was destroyed but it is pending'."""
    db_path = tmp_path / "telemetry.db"
    import aiosqlite
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """CREATE TABLE model_pick_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                task_name TEXT, agent_type TEXT, difficulty INTEGER,
                call_category TEXT, picked_model TEXT, picked_score REAL,
                picked_reasons TEXT, candidates_json TEXT,
                failures_json TEXT, snapshot_summary TEXT
            )"""
        )
        await db.commit()
    monkeypatch.setattr(selector_mod, "_telemetry_db_path", str(db_path))

    caplog.set_level(logging.WARNING, logger="asyncio")

    from fatih_hoca.selector import Selector

    class _FakeModel:
        name = "fake-model"
        is_local = False
        is_loaded = True
        load_time_seconds = 0.0

    class _FakeRanked:
        def __init__(self):
            self.model = _FakeModel()
            self.score = 7.5
            self.reasons = ["test"]

    class _FakeReqs:
        effective_task = "test_task"
        agent_type = "coder"
        difficulty = 5

    class _FakeSnapshot:
        vram_available_mb = 7000
        local = None

    sel = Selector.__new__(Selector)
    for _ in range(20):
        sel._persist_pick_telemetry(
            scored=[_FakeRanked()],
            reqs=_FakeReqs(),
            task_name="test_task",
            call_category="main_work",
            failures=[],
            snapshot=_FakeSnapshot(),
        )
        gc.collect()

    for _ in range(5):
        await asyncio.sleep(0.05)

    bad = [r for r in caplog.records if "Task was destroyed" in r.getMessage()]
    assert bad == [], f"GC reaped pending tasks: {bad!r}"
