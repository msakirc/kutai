import asyncio
import json

import pytest

from src.infra.db import init_db, get_db
from yalayut import admin


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


async def _seed_t2_artifact():
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO yalayut_index "
        "(artifact_type, kind, source, owner, name, version, vet_tier, "
        " exposure_class, enabled, created_at) "
        "VALUES ('skill', 'prompt_skill', 'github:x/y', 'x', 'a-skill', "
        " '1.0.0', 2, 'quarantine', 0, datetime('now'))")
    await db.commit()
    return cur.lastrowid


def test_pending_artifacts_lists_t2(loop, clean_yalayut_index):
    async def _run():
        await init_db()
        aid = await _seed_t2_artifact()
        pend = await admin.pending_artifacts()
        assert any(p["id"] == aid for p in pend)
    loop.run_until_complete(_run())


def test_approve_artifact_enables(loop, clean_yalayut_index):
    async def _run():
        await init_db()
        aid = await _seed_t2_artifact()
        await admin.approve_artifact(aid)
        db = await get_db()
        cur = await db.execute(
            "SELECT enabled FROM yalayut_index WHERE id = ?", (aid,))
        assert (await cur.fetchone())[0] == 1
        await cur.close()
    loop.run_until_complete(_run())


def test_reject_artifact_disables(loop, clean_yalayut_index):
    async def _run():
        await init_db()
        aid = await _seed_t2_artifact()
        await admin.reject_artifact(aid)
        db = await get_db()
        cur = await db.execute(
            "SELECT enabled, vet_tier FROM yalayut_index WHERE id = ?", (aid,))
        row = await cur.fetchone()
        await cur.close()
        assert row[0] == 0
    loop.run_until_complete(_run())


def test_approve_source_creates_source_row(loop, clean_yalayut_index):
    async def _run():
        await init_db()
        db = await get_db()
        cur = await db.execute(
            "INSERT INTO yalayut_source_candidates "
            "(candidate_source_id, source_type, endpoint, state, proposed_at) "
            "VALUES ('github:new/src', 'github_path', 'https://x', "
            "'pending', datetime('now'))")
        await db.commit()
        cand_id = cur.lastrowid
        await admin.approve_source(cand_id, trusted=True)
        cur = await db.execute(
            "SELECT trusted, enabled FROM yalayut_sources "
            "WHERE source_id = 'github:new/src'")
        row = await cur.fetchone()
        await cur.close()
        assert tuple(row) == (1, 1)
    loop.run_until_complete(_run())


def test_decide_policy_approve_writes_policy_row(loop, clean_yalayut_index):
    async def _run():
        await init_db()
        db = await get_db()
        cur = await db.execute(
            "INSERT INTO yalayut_policy_proposals "
            "(check_name, key, proposed_value, state, proposed_at) "
            "VALUES ('shell_allowlist', 'wasp', 'allow', 'pending', "
            "datetime('now'))")
        await db.commit()
        pid = cur.lastrowid
        await admin.decide_policy(pid, approve=True)
        cur = await db.execute(
            "SELECT value FROM yalayut_policy "
            "WHERE check_name = 'shell_allowlist' AND key = 'wasp'")
        assert (await cur.fetchone())[0] == "allow"
        await cur.close()
    loop.run_until_complete(_run())


def test_set_secret_encrypts(loop, clean_yalayut_index):
    async def _run():
        await init_db()
        await admin.set_secret("TEST_API_KEY", "supersecret")
        db = await get_db()
        cur = await db.execute(
            "SELECT encrypted_value FROM yalayut_secrets "
            "WHERE key_name = 'TEST_API_KEY'")
        row = await cur.fetchone()
        await cur.close()
        # stored encrypted — never the plaintext.
        assert row is not None
        assert b"supersecret" not in (row[0] or b"")
    loop.run_until_complete(_run())


def test_stats_returns_counts(loop, clean_yalayut_index):
    async def _run():
        await init_db()
        await _seed_t2_artifact()
        stats = await admin.stats()
        assert "tier_counts" in stats
        assert "exposure_class_counts" in stats
    loop.run_until_complete(_run())
