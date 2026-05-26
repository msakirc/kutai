"""Mechanical tasks are EXEMPT from the lane concurrency cap.

Bug (2026-05-26): mr_roboto mechanical tasks (git commit, workspace
snapshot, notify_user, clarify) — CPU-only, no LLM/GPU/cloud — were
stalling in the queue behind LLM tasks. Root cause: next_task()'s lane
cap gate (``if count_in_flight >= cap: return None``) fired BEFORE the
candidate loop and counted mechanical tasks against the same ceiling as
LLM work, so a lane saturated by 4 slow LLM tasks rejected admission for
everything — including the mechanical exemption logic downstream that was
supposed to admit them unbounded.

Fix: (1) count_in_flight excludes mechanical; (2) the cap gate only
short-circuits when no ready mechanical is waiting; (3) a per-task cap
guard skips LLM candidates over cap so a mechanical behind them is still
reached and admitted.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import aiosqlite


# ─── SQL helpers against an isolated in-memory DB (NOT the live kutai.db) ──

async def _seed_db():
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute(
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, lane TEXT, status TEXT, "
        "agent_type TEXT, runner TEXT, mission_id INTEGER, next_retry_at TEXT, "
        "priority INTEGER, created_at TEXT, depends_on TEXT)"
    )
    await conn.execute(
        "CREATE TABLE missions (id INTEGER PRIMARY KEY, lifecycle_state TEXT)"
    )
    return conn


@pytest.mark.asyncio
async def test_count_in_flight_excludes_mechanical():
    from general_beckman.lanes import count_in_flight
    conn = await _seed_db()
    # 2 LLM in-flight + 3 mechanical in-flight, all on oneshot
    await conn.execute("INSERT INTO tasks (id, lane, status, agent_type, runner) "
                       "VALUES (1,'oneshot','processing','coder','react')")
    await conn.execute("INSERT INTO tasks (id, lane, status, agent_type, runner) "
                       "VALUES (2,'oneshot','in_progress','implementer','react')")
    await conn.execute("INSERT INTO tasks (id, lane, status, agent_type, runner) "
                       "VALUES (3,'oneshot','processing','mechanical','mechanical')")
    await conn.execute("INSERT INTO tasks (id, lane, status, agent_type, runner) "
                       "VALUES (4,'oneshot','processing','mr_roboto','mechanical')")
    await conn.execute("INSERT INTO tasks (id, lane, status, agent_type, runner) "
                       "VALUES (5,'oneshot','assigned','mechanical','mechanical')")
    await conn.commit()

    n = await count_in_flight(conn, "oneshot")
    await conn.close()
    assert n == 2, "mechanical in-flight tasks must not count toward the LLM cap"


@pytest.mark.asyncio
async def test_has_ready_mechanical_true_for_pending_mechanical():
    from general_beckman.lanes import has_ready_mechanical
    conn = await _seed_db()
    await conn.execute("INSERT INTO missions (id, lifecycle_state) VALUES (10,'active')")
    await conn.execute(
        "INSERT INTO tasks (id, lane, status, agent_type, runner, mission_id) "
        "VALUES (1,'oneshot','pending','mechanical','mechanical',10)"
    )
    await conn.commit()
    assert await has_ready_mechanical(conn, "oneshot") is True
    await conn.close()


@pytest.mark.asyncio
async def test_has_ready_mechanical_false_when_only_llm_pending():
    from general_beckman.lanes import has_ready_mechanical
    conn = await _seed_db()
    await conn.execute(
        "INSERT INTO tasks (id, lane, status, agent_type, runner) "
        "VALUES (1,'oneshot','pending','coder','react')"
    )
    await conn.commit()
    assert await has_ready_mechanical(conn, "oneshot") is False
    await conn.close()


@pytest.mark.asyncio
async def test_has_ready_mechanical_false_when_mechanical_not_pending():
    from general_beckman.lanes import has_ready_mechanical
    conn = await _seed_db()
    await conn.execute(
        "INSERT INTO tasks (id, lane, status, agent_type, runner) "
        "VALUES (1,'oneshot','processing','mechanical','mechanical')"
    )
    await conn.commit()
    assert await has_ready_mechanical(conn, "oneshot") is False
    await conn.close()


# ─── next_task admission (mocked queue layer) ──────────────────────────────

def _mech_task(tid=1, lane="oneshot"):
    return {
        "id": tid, "priority": 5, "difficulty": 5,
        "agent_type": "mechanical", "runner": "mechanical",
        "created_at": time.time(), "status": "pending", "lane": lane,
        "mission_id": None, "worker_attempts": 0,
    }


def _llm_task(tid=9, lane="oneshot"):
    return {
        "id": tid, "priority": 9, "difficulty": 5,
        "agent_type": "coder", "runner": "react",
        "created_at": time.time(), "status": "pending", "lane": lane,
        "mission_id": None, "worker_attempts": 0,
    }


@pytest.mark.asyncio
async def test_next_task_admits_mechanical_when_lane_saturated():
    """Lane at cap with LLM work, a mechanical is ready → admit it anyway."""
    import general_beckman
    from general_beckman.lanes import ONESHOT_CONCURRENCY

    snap = MagicMock()
    mech = _mech_task(1)

    with patch("general_beckman.queue.pick_ready_top_k",
               new=AsyncMock(return_value=[mech])), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("general_beckman._ceiling_ok", new=AsyncMock(return_value=True)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("general_beckman.lanes.count_in_flight",
               new=AsyncMock(return_value=ONESHOT_CONCURRENCY)), \
         patch("general_beckman.lanes.has_ready_mechanical",
               new=AsyncMock(return_value=True)), \
         patch("nerd_herd.refresh_snapshot",
               new=AsyncMock(return_value=snap), create=True):
        out = await general_beckman.next_task()

    assert out is not None, "mechanical task starved by lane cap"
    assert out["id"] == 1
    assert out["status"] == "processing"


@pytest.mark.asyncio
async def test_next_task_skips_llm_over_cap_but_admits_mechanical_behind_it():
    """At cap, an LLM candidate ahead of a mechanical must be skipped (cap
    respected for LLM) while the mechanical behind it is admitted."""
    import general_beckman
    from general_beckman.lanes import ONESHOT_CONCURRENCY

    snap = MagicMock()
    llm = _llm_task(9)      # higher priority, first in candidates
    mech = _mech_task(1)
    select_mock = MagicMock()

    with patch("general_beckman.queue.pick_ready_top_k",
               new=AsyncMock(return_value=[llm, mech])), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("general_beckman._ceiling_ok", new=AsyncMock(return_value=True)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("general_beckman.lanes.count_in_flight",
               new=AsyncMock(return_value=ONESHOT_CONCURRENCY)), \
         patch("general_beckman.lanes.has_ready_mechanical",
               new=AsyncMock(return_value=True)), \
         patch("fatih_hoca.select", select_mock), \
         patch("nerd_herd.refresh_snapshot",
               new=AsyncMock(return_value=snap), create=True):
        out = await general_beckman.next_task()

    assert out is not None
    assert out["id"] == 1, "should skip the over-cap LLM task and admit the mechanical"
    select_mock.assert_not_called(), "LLM task must not even reach selection when over cap"
