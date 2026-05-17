import asyncio

import pytest

from src.infra.db import init_db, get_db
from src.infra.times import utc_now, to_db
from yalayut.discovery import cron as yal_cron


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


async def _seed_trusted_source():
    db = await get_db()
    await db.execute(
        "INSERT OR IGNORE INTO yalayut_sources "
        "(source_id, source_type, endpoint, trusted, enabled, "
        " discovery_mode, min_interval_s) "
        "VALUES ('github:test/skills', 'github_path', "
        "'https://example.com', 1, 1, 'cron', 0)")
    await db.commit()


def test_daily_discovery_returns_summary(loop, monkeypatch):
    async def _run():
        await init_db()
        await _seed_trusted_source()

        async def _fake_ingest(source_row):
            return {"ingested": 3, "enabled": 2, "quarantined": 1}

        monkeypatch.setattr(yal_cron, "_ingest_source", _fake_ingest)
        summary = await yal_cron.daily_discovery()
        assert summary["sources_scanned"] == 1
        assert summary["artifacts_ingested"] == 3
        assert summary["artifacts_enabled"] == 2
    loop.run_until_complete(_run())


def test_daily_discovery_respects_min_interval(loop, monkeypatch):
    async def _run():
        await init_db()
        db = await get_db()
        # source ran 30s ago, min_interval 3600 → must be skipped.
        await db.execute(
            "INSERT OR IGNORE INTO yalayut_sources "
            "(source_id, source_type, endpoint, trusted, enabled, "
            " discovery_mode, min_interval_s, last_run_at) "
            "VALUES ('github:recent/skills', 'github_path', 'x', 1, 1, "
            "'cron', 3600, ?)",
            (to_db(utc_now()),))
        await db.commit()

        called = []

        async def _fake_ingest(source_row):
            called.append(source_row["source_id"])
            return {"ingested": 0, "enabled": 0, "quarantined": 0}

        monkeypatch.setattr(yal_cron, "_ingest_source", _fake_ingest)
        summary = await yal_cron.daily_discovery()
        assert "github:recent/skills" not in called
        assert summary["sources_skipped"] >= 1
    loop.run_until_complete(_run())
