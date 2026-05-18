import asyncio

import pytest

from src.infra.db import init_db
import mr_roboto


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_yalayut_discovery_executor_reachable(loop, monkeypatch):
    async def _run():
        await init_db()
        import yalayut

        async def _fake_daily():
            return {"sources_scanned": 2, "artifacts_ingested": 4}

        monkeypatch.setattr(yalayut, "daily_discovery",
                            lambda: _fake_daily())
        task = {"id": 1, "agent_type": "mechanical", "context": {},
                "payload": {"action": "yalayut_discovery", "mode": "daily"}}
        r = await mr_roboto.run(task)
        assert r.status == "completed"
        assert r.result["sources_scanned"] == 2
    loop.run_until_complete(_run())


def test_source_scout_executor_reachable(loop, monkeypatch):
    async def _run():
        await init_db()
        import yalayut

        async def _fake_scan():
            return {"candidates_proposed": 3}

        monkeypatch.setattr(yalayut, "source_scout_scan",
                            lambda: _fake_scan())
        task = {"id": 2, "agent_type": "mechanical", "context": {},
                "payload": {"action": "source_scout"}}
        r = await mr_roboto.run(task)
        assert r.status == "completed"
        assert r.result["candidates_proposed"] == 3
    loop.run_until_complete(_run())


def test_capture_hint_executor_reachable(loop, monkeypatch):
    async def _run():
        await init_db()
        import yalayut
        seen = {}

        async def _fake_capture(task, outcome):
            seen["task_id"] = task.get("id")
            seen["iterations"] = outcome.get("iterations")

        monkeypatch.setattr(yalayut, "capture_hint",
                            lambda t, o: _fake_capture(t, o))
        task = {"id": 3, "agent_type": "mechanical", "context": {},
                "payload": {"action": "capture_hint",
                            "source_task": {"id": 77, "title": "t",
                                            "description": "d"},
                            "outcome": {"status": "completed",
                                        "iterations": 3}}}
        r = await mr_roboto.run(task)
        assert r.status == "completed"
        assert seen["task_id"] == 77
        assert seen["iterations"] == 3
    loop.run_until_complete(_run())
