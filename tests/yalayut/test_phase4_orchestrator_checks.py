import asyncio
import time

import pytest

from src.core.orchestrator import Orchestrator


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_yalayut_discovery_check_enqueues_when_due(loop, monkeypatch):
    async def _run():
        enqueued = []

        async def _fake_enqueue(spec, **kw):
            enqueued.append(spec)
            return {"id": 1}

        import general_beckman
        monkeypatch.setattr(general_beckman, "enqueue", _fake_enqueue)

        orch = Orchestrator.__new__(Orchestrator)
        orch._last_yalayut_discovery = 0.0  # never run → due
        await orch._check_yalayut_discovery()
        assert len(enqueued) == 1
        spec = enqueued[0]
        assert spec["agent_type"] == "mechanical"
        # payload MUST be nested under context, never a top-level key —
        # enqueue forwards spec via add_task(**spec) and add_task has no
        # `payload` param (live crash 2026-06-05: "add_task() got an
        # unexpected keyword argument 'payload'").
        assert "payload" not in spec
        assert spec["context"]["payload"]["action"] == "yalayut_discovery"
        # second call right away → NOT due, gated.
        await orch._check_yalayut_discovery()
        assert len(enqueued) == 1
    loop.run_until_complete(_run())


def test_source_scout_check_enqueues_when_due(loop, monkeypatch):
    async def _run():
        enqueued = []

        async def _fake_enqueue(spec, **kw):
            enqueued.append(spec)
            return {"id": 2}

        import general_beckman
        monkeypatch.setattr(general_beckman, "enqueue", _fake_enqueue)

        orch = Orchestrator.__new__(Orchestrator)
        orch._last_source_scout = time.time() - 100000  # long ago → due
        await orch._check_source_scout()
        assert len(enqueued) == 1
        assert "payload" not in enqueued[0]
        assert enqueued[0]["context"]["payload"]["action"] == "source_scout"
    loop.run_until_complete(_run())


def test_check_does_not_crash_on_enqueue_error(loop, monkeypatch):
    async def _run():
        async def _boom(spec, **kw):
            raise RuntimeError("beckman down")

        import general_beckman
        monkeypatch.setattr(general_beckman, "enqueue", _boom)
        orch = Orchestrator.__new__(Orchestrator)
        orch._last_yalayut_discovery = 0.0
        # must swallow the error — pump must never crash on a periodic check.
        await orch._check_yalayut_discovery()
    loop.run_until_complete(_run())
