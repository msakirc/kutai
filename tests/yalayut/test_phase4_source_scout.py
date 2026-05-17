import asyncio

import pytest

from src.infra.db import init_db, get_db
from yalayut.discovery import source_scout


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


async def _clean_candidates():
    """Wipe yalayut_source_candidates so daily-cap / dedup state doesn't
    leak between tests on the shared singleton DB."""
    db = await get_db()
    await db.execute("DELETE FROM yalayut_source_candidates")
    await db.commit()


def test_source_scout_writes_candidates(loop, monkeypatch):
    async def _run():
        await init_db()
        await _clean_candidates()

        async def _fake_github_trending():
            return [
                {"candidate_source_id": "github:acme/skills",
                 "source_type": "github_path",
                 "endpoint": "https://github.com/acme/skills",
                 "metadata_json": '{"description": "acme skills"}'},
            ]

        async def _empty():
            return []

        monkeypatch.setattr(source_scout, "_scan_github_trending",
                            _fake_github_trending)
        monkeypatch.setattr(source_scout, "_scan_readme_crossrefs", _empty)
        monkeypatch.setattr(source_scout, "_scan_demand_websearch", _empty)
        monkeypatch.setattr(source_scout, "_scan_founder_urls", _empty)

        result = await source_scout.source_scout_scan()
        assert result["candidates_proposed"] == 1
        db = await get_db()
        cur = await db.execute(
            "SELECT candidate_source_id, state FROM yalayut_source_candidates")
        row = await cur.fetchone()
        await cur.close()
        assert row[0] == "github:acme/skills"
        assert row[1] == "pending"
    loop.run_until_complete(_run())


def test_source_scout_respects_daily_cap(loop, monkeypatch):
    async def _run():
        await init_db()
        await _clean_candidates()

        async def _many():
            return [
                {"candidate_source_id": f"github:x/repo{i}",
                 "source_type": "github_path",
                 "endpoint": f"https://github.com/x/repo{i}",
                 "metadata_json": "{}"}
                for i in range(20)
            ]

        async def _empty():
            return []

        monkeypatch.setattr(source_scout, "_scan_github_trending", _many)
        monkeypatch.setattr(source_scout, "_scan_readme_crossrefs", _empty)
        monkeypatch.setattr(source_scout, "_scan_demand_websearch", _empty)
        monkeypatch.setattr(source_scout, "_scan_founder_urls", _empty)

        result = await source_scout.source_scout_scan()
        # default cap = 5
        assert result["candidates_proposed"] == source_scout.DAILY_CANDIDATE_CAP
    loop.run_until_complete(_run())


def test_source_scout_executor_runs_policy_observer(loop, monkeypatch):
    """Fix 2: the source_scout mechanical executor must call
    yalayut.observe_and_propose() after scanning — best-effort, so a failure
    there must not fail the scout run."""
    async def _run():
        await init_db()
        import yalayut
        from mr_roboto.executors import source_scout as scout_exec

        reached = {"observe": False}

        async def _fake_scan():
            return {"candidates_proposed": 0}

        async def _fake_observe():
            reached["observe"] = True
            return 4

        monkeypatch.setattr(yalayut, "source_scout_scan",
                            lambda: _fake_scan())
        monkeypatch.setattr(yalayut, "observe_and_propose",
                            lambda: _fake_observe())

        result = await scout_exec.run({"id": 9, "agent_type": "mechanical",
                                       "payload": {"action": "source_scout"}})
        assert reached["observe"], (
            "source_scout executor must reach observe_and_propose")
        assert result["policy_proposals_written"] == 4
    loop.run_until_complete(_run())


def test_source_scout_executor_survives_observer_failure(loop, monkeypatch):
    """observe_and_propose failure must not fail the scout run."""
    async def _run():
        await init_db()
        import yalayut
        from mr_roboto.executors import source_scout as scout_exec

        async def _fake_scan():
            return {"candidates_proposed": 1}

        async def _boom():
            raise RuntimeError("observer exploded")

        monkeypatch.setattr(yalayut, "source_scout_scan",
                            lambda: _fake_scan())
        monkeypatch.setattr(yalayut, "observe_and_propose",
                            lambda: _boom())

        result = await scout_exec.run({"id": 10, "agent_type": "mechanical",
                                       "payload": {"action": "source_scout"}})
        # scout result still returned, no policy_proposals_written key
        assert result["candidates_proposed"] == 1
        assert "policy_proposals_written" not in result
    loop.run_until_complete(_run())


def test_source_scout_dedupes_existing(loop, monkeypatch):
    async def _run():
        await init_db()
        await _clean_candidates()
        db = await get_db()
        await db.execute(
            "INSERT INTO yalayut_source_candidates "
            "(candidate_source_id, source_type, endpoint, state, proposed_at) "
            "VALUES ('github:dup/repo', 'github_path', 'x', 'pending', "
            "datetime('now'))")
        # Also already an approved source — must not be re-proposed.
        await db.execute(
            "INSERT OR IGNORE INTO yalayut_sources (source_id, source_type, endpoint, "
            "trusted, enabled, discovery_mode) "
            "VALUES ('github:known/repo', 'github_path', 'x', 1, 1, 'cron')")
        await db.commit()

        async def _candidates():
            return [
                {"candidate_source_id": "github:dup/repo",
                 "source_type": "github_path", "endpoint": "x",
                 "metadata_json": "{}"},
                {"candidate_source_id": "github:known/repo",
                 "source_type": "github_path", "endpoint": "x",
                 "metadata_json": "{}"},
                {"candidate_source_id": "github:fresh/repo",
                 "source_type": "github_path", "endpoint": "x",
                 "metadata_json": "{}"},
            ]

        async def _empty():
            return []

        monkeypatch.setattr(source_scout, "_scan_github_trending",
                            _candidates)
        monkeypatch.setattr(source_scout, "_scan_readme_crossrefs", _empty)
        monkeypatch.setattr(source_scout, "_scan_demand_websearch", _empty)
        monkeypatch.setattr(source_scout, "_scan_founder_urls", _empty)

        result = await source_scout.source_scout_scan()
        assert result["candidates_proposed"] == 1  # only github:fresh/repo
    loop.run_until_complete(_run())
