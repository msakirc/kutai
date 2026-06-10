import asyncio
import time

import pytest

from src.core.periodic_checks import PeriodicChecks


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

        pc = PeriodicChecks()
        pc._last_yalayut_discovery = 0.0  # never run → due
        await pc.check_yalayut_discovery()
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
        await pc.check_yalayut_discovery()
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

        pc = PeriodicChecks()
        pc._last_source_scout = time.time() - 100000  # long ago → due
        await pc.check_source_scout()
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
        pc = PeriodicChecks()
        pc._last_yalayut_discovery = 0.0
        # must swallow the error — pump must never crash on a periodic check.
        await pc.check_yalayut_discovery()
    loop.run_until_complete(_run())


def test_run_due_fires_all_checks_individually_guarded(loop, monkeypatch):
    """run_due must invoke every check and never propagate a single failure."""
    async def _run():
        calls = []

        async def _boom():
            calls.append("founder")
            raise RuntimeError("sweep down")

        async def _ok_mcp():
            calls.append("mcp")

        async def _ok_disc():
            calls.append("disc")

        async def _ok_scout():
            calls.append("scout")

        pc = PeriodicChecks()
        monkeypatch.setattr(pc, "check_founder_sweep", _boom)
        monkeypatch.setattr(pc, "check_mcp_idle_sweep", _ok_mcp)
        monkeypatch.setattr(pc, "check_yalayut_discovery", _ok_disc)
        monkeypatch.setattr(pc, "check_source_scout", _ok_scout)

        # Founder check raises but run_due swallows it and still runs the rest.
        await pc.run_due()
        assert calls == ["founder", "mcp", "disc", "scout"]
    loop.run_until_complete(_run())
