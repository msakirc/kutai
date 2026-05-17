import asyncio

import pytest

from src.infra.db import init_db, get_db
from yalayut.discovery import on_demand
from yalayut.discovery import demand as yal_demand


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_on_demand_ingests_matching_source(loop, monkeypatch):
    async def _run():
        await init_db()
        db = await get_db()
        await db.execute(
            "INSERT OR IGNORE INTO yalayut_sources "
            "(source_id, source_type, endpoint, trusted, enabled, "
            " discovery_mode, min_interval_s) "
            "VALUES ('github:awesome/mcp', 'awesome_list_md', 'x', 0, 1, "
            "'on_demand', 0)")
        await db.commit()

        async def _fake_ingest(source_row, *, artifact_cap=10):
            return {"ingested": 2, "enabled": 0, "quarantined": 2}

        monkeypatch.setattr(on_demand, "_ingest_source_capped", _fake_ingest)
        result = await on_demand.on_demand_discovery({
            "source_step_pattern": "mcp-cloudflare-setup",
            "intent_keywords": ["mcp", "cloudflare"],
            "stacked_confidence": 0.7,
        })
        assert result["artifacts_ingested"] == 2
        assert result["pattern"] == "mcp-cloudflare-setup"
    loop.run_until_complete(_run())


def test_on_demand_marks_pattern_discovered(loop, monkeypatch):
    async def _run():
        await init_db()
        await yal_demand.record_signal(yal_demand.DemandSignal(
            source_step_pattern="pat-done", intent_keywords=["x"],
            signal_type="dlq", confidence=0.5))

        async def _fake_ingest(source_row, *, artifact_cap=10):
            return {"ingested": 0, "enabled": 0, "quarantined": 0}

        monkeypatch.setattr(on_demand, "_ingest_source_capped", _fake_ingest)
        monkeypatch.setattr(on_demand, "_untrusted_sources_for",
                            lambda kw: [])
        await on_demand.on_demand_discovery({
            "source_step_pattern": "pat-done",
            "intent_keywords": ["x"],
            "stacked_confidence": 0.5,
        })
        db = await get_db()
        cur = await db.execute(
            "SELECT resulted_in_discovery FROM yalayut_demand_signals "
            "WHERE source_step_pattern = 'pat-done'")
        row = await cur.fetchone()
        await cur.close()
        assert row[0] == 1
    loop.run_until_complete(_run())
