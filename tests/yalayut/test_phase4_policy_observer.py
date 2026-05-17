import asyncio
import json

import pytest

from src.infra.db import init_db, get_db
from yalayut import policy_observer


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_observe_proposes_repeated_unknown_token(loop):
    async def _run():
        await init_db()
        db = await get_db()
        # 3 artifacts capped at T2 by the same unknown shell token 'wasp'.
        for i in range(3):
            await db.execute(
                "INSERT INTO yalayut_index "
                "(artifact_type, kind, source, name, version, vet_tier, "
                " exposure_class, check_max_json, enabled, created_at) "
                "VALUES ('skill', 'shell_recipe', 'github:x/y', ?, '1.0.0', "
                " 2, 'inject', ?, 1, datetime('now'))",
                (f"artifact-{i}",
                 json.dumps({"shell_allowlist": {"tier": 2,
                                                 "unknown_token": "wasp"}})))
        await db.commit()
        n = await policy_observer.observe_and_propose()
        assert n >= 1
        cur = await db.execute(
            "SELECT check_name, key, state FROM yalayut_policy_proposals "
            "WHERE key = 'wasp'")
        row = await cur.fetchone()
        await cur.close()
        assert row[0] == "shell_allowlist"
        assert row[2] == "pending"
    loop.run_until_complete(_run())


def test_observe_skips_below_threshold(loop):
    async def _run():
        await init_db()
        db = await get_db()
        # only 1 occurrence — below the propose threshold (3).
        await db.execute(
            "INSERT INTO yalayut_index "
            "(artifact_type, kind, source, name, version, vet_tier, "
            " exposure_class, check_max_json, enabled, created_at) "
            "VALUES ('skill', 'shell_recipe', 'github:x/y', 'lone', '1.0.0', "
            " 2, 'inject', ?, 1, datetime('now'))",
            (json.dumps({"shell_allowlist": {"tier": 2,
                                             "unknown_token": "rare"}}),))
        await db.commit()
        await policy_observer.observe_and_propose()
        cur = await db.execute(
            "SELECT COUNT(*) FROM yalayut_policy_proposals WHERE key = 'rare'")
        assert (await cur.fetchone())[0] == 0
        await cur.close()
    loop.run_until_complete(_run())


def test_observe_idempotent(loop):
    async def _run():
        await init_db()
        db = await get_db()
        for i in range(3):
            await db.execute(
                "INSERT INTO yalayut_index "
                "(artifact_type, kind, source, name, version, vet_tier, "
                " exposure_class, check_max_json, enabled, created_at) "
                "VALUES ('skill', 'shell_recipe', 'github:x/y', ?, '1.0.0', "
                " 2, 'inject', ?, 1, datetime('now'))",
                (f"idem-{i}",
                 json.dumps({"shell_allowlist": {"tier": 2,
                                                 "unknown_token": "idem"}})))
        await db.commit()
        await policy_observer.observe_and_propose()
        await policy_observer.observe_and_propose()
        cur = await db.execute(
            "SELECT COUNT(*) FROM yalayut_policy_proposals WHERE key = 'idem'")
        assert (await cur.fetchone())[0] == 1  # no duplicate proposal
        await cur.close()
    loop.run_until_complete(_run())
