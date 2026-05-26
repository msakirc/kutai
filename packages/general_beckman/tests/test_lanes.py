"""Z8 T1B — ongoing-lane admission tests."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _task(tid=1, lane="oneshot", priority=5, agent_type="coder"):
    return {
        "id": tid, "priority": priority, "difficulty": 5,
        "agent_type": agent_type, "created_at": time.time(),
        "status": "ready", "lane": lane, "mission_id": 100 + tid,
    }


def _mock_pick():
    fake_model = MagicMock(is_local=False, provider="anthropic")
    fake_model.name = "claude-sonnet-4-6"
    return MagicMock(model=fake_model, composite=0.6)


def _breakdown(scalar: float):
    bd = MagicMock()
    bd.scalar = scalar
    return bd


# ── Pure unit tests on lanes module ──────────────────────────────────────


def test_pick_lane_alert_triage_is_ongoing():
    from general_beckman.lanes import pick_lane, LANE_ONGOING
    assert pick_lane("alert_triage") == LANE_ONGOING


def test_pick_lane_cron_types_are_ongoing():
    from general_beckman.lanes import pick_lane, LANE_ONGOING
    for t in (
        "cron_backup_verify", "cron_dep_hygiene", "cron_cve_scan",
        "cron_secret_scan", "cron_cost_pull", "cron_synthetic_check",
        "support_ticket",
    ):
        assert pick_lane(t) == LANE_ONGOING, t


def test_pick_lane_default_is_oneshot():
    from general_beckman.lanes import pick_lane, LANE_ONESHOT
    assert pick_lane("apply") == LANE_ONESHOT
    assert pick_lane("coder") == LANE_ONESHOT
    assert pick_lane("") == LANE_ONESHOT


@pytest.mark.asyncio
async def test_cap_for_lanes():
    from general_beckman.lanes import (
        cap_for, LANE_ONESHOT, LANE_ONGOING,
        ONESHOT_CONCURRENCY, ONGOING_CONCURRENCY,
    )
    assert await cap_for(LANE_ONESHOT) == ONESHOT_CONCURRENCY
    assert await cap_for(LANE_ONGOING) == ONGOING_CONCURRENCY


@pytest.mark.asyncio
async def test_count_in_flight_queries_lane_column():
    from general_beckman.lanes import count_in_flight, LANE_ONGOING

    class _Cur:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return None
        async def fetchone(self):
            return (3,)

    class _Conn:
        def __init__(self):
            self.captured = None
        def execute(self, sql, params):
            self.captured = (sql, params)
            return _Cur()

    conn = _Conn()
    n = await count_in_flight(conn, LANE_ONGOING)
    assert n == 3
    assert "lane=?" in conn.captured[0]
    assert conn.captured[1] == (LANE_ONGOING,)


# ── next_task lane filter (mocked queue layer) ───────────────────────────


@pytest.mark.asyncio
async def test_next_task_default_lane_returns_oneshot():
    import general_beckman
    from general_beckman.lanes import LANE_ONESHOT

    snap = MagicMock()
    snap.pressure_for = MagicMock(return_value=_breakdown(0.7))
    one = _task(1, lane="oneshot", priority=7)

    captured = {}

    async def _picker(k=5, lane=LANE_ONESHOT):
        captured["lane"] = lane
        return [one]

    with patch("general_beckman.queue.pick_ready_top_k", side_effect=_picker), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("general_beckman.lanes.count_in_flight",
               new=AsyncMock(return_value=0)), \
         patch("fatih_hoca.select", return_value=_mock_pick()), \
         patch("nerd_herd.refresh_snapshot",
               new=AsyncMock(return_value=snap), create=True):
        out = await general_beckman.next_task()
    assert out is not None
    assert out["id"] == 1
    assert captured["lane"] == LANE_ONESHOT


@pytest.mark.asyncio
async def test_next_task_ongoing_lane_filters():
    import general_beckman
    from general_beckman.lanes import LANE_ONGOING

    snap = MagicMock()
    snap.pressure_for = MagicMock(return_value=_breakdown(0.7))
    captured = {}

    async def _picker(k=5, lane=LANE_ONGOING):
        captured["lane"] = lane
        return [_task(2, lane="ongoing", agent_type="mechanical")]

    with patch("general_beckman.queue.pick_ready_top_k", side_effect=_picker), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("general_beckman.lanes.count_in_flight",
               new=AsyncMock(return_value=0)), \
         patch("nerd_herd.refresh_snapshot",
               new=AsyncMock(return_value=snap), create=True):
        out = await general_beckman.next_task(lane=LANE_ONGOING)
    assert out is not None
    assert out["id"] == 2
    assert captured["lane"] == LANE_ONGOING


@pytest.mark.asyncio
async def test_ongoing_lane_cap_enforced():
    import general_beckman
    from general_beckman.lanes import LANE_ONGOING, ONGOING_CONCURRENCY

    snap = MagicMock()
    snap.pressure_for = MagicMock(return_value=_breakdown(0.7))
    # Cap reached AND no mechanical waiting: pick_ready_top_k never called.
    pick_mock = AsyncMock(return_value=[])
    with patch("general_beckman.queue.pick_ready_top_k", pick_mock), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("general_beckman.lanes.count_in_flight",
               new=AsyncMock(return_value=ONGOING_CONCURRENCY)), \
         patch("general_beckman.lanes.has_ready_mechanical",
               new=AsyncMock(return_value=False)), \
         patch("nerd_herd.refresh_snapshot",
               new=AsyncMock(return_value=snap), create=True):
        out = await general_beckman.next_task(lane=LANE_ONGOING)
    assert out is None
    pick_mock.assert_not_called()


@pytest.mark.asyncio
async def test_enqueue_defaults_lane_from_task_type():
    """``enqueue`` should pass a derived lane to add_task when none given."""
    from general_beckman import enqueue
    from general_beckman.lanes import LANE_ONGOING

    captured = {}

    async def _fake_add_task(**kw):
        captured.update(kw)
        return 99

    with patch("src.infra.db.add_task", new=_fake_add_task), \
         patch("general_beckman.queue_profile_push.build_and_push",
               new=AsyncMock(return_value=None)):
        await enqueue({"title": "t", "description": "d",
                       "agent_type": "alert_triage"})
    assert captured.get("lane") == LANE_ONGOING


@pytest.mark.asyncio
async def test_enqueue_explicit_lane_wins():
    from general_beckman import enqueue
    from general_beckman.lanes import LANE_ONGOING

    captured = {}

    async def _fake_add_task(**kw):
        captured.update(kw)
        return 100

    with patch("src.infra.db.add_task", new=_fake_add_task), \
         patch("general_beckman.queue_profile_push.build_and_push",
               new=AsyncMock(return_value=None)):
        await enqueue({"title": "t", "description": "d", "agent_type": "coder"},
                      lane=LANE_ONGOING)
    assert captured.get("lane") == LANE_ONGOING
