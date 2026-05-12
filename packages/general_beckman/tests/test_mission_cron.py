"""Z8 T5-prep — mission_cron scheduler tests.

The cron loop body calls ``await enqueue(...)`` then ``await asyncio.sleep(N)``.
Tests stub enqueue to count calls, patch asyncio.sleep to yield ~immediately,
and assert behaviour over a bounded number of iterations.
"""
from __future__ import annotations

import asyncio
import pytest

from general_beckman import mission_cron


@pytest.fixture(autouse=True)
def _reset_cron_state():
    mission_cron._reset_for_tests()
    yield
    mission_cron._reset_for_tests()


class _FakeMission:
    def __init__(self, mid: int, cursor: dict):
        self.id = mid
        self.cursor = cursor
        self.title = f"m{mid}"


async def _wait_for(predicate, *, timeout: float = 2.0):
    """Poll predicate every 10 ms until it's truthy or we time out."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        if predicate():
            return
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("timed out waiting for predicate")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_arm_enqueues_on_each_tick(monkeypatch):
    calls: list[dict] = []

    async def fake_enqueue(spec, **kwargs):
        calls.append({"spec": spec, "kwargs": kwargs})
        return 42

    monkeypatch.setattr(
        "general_beckman.mission_cron.enqueue"
        if hasattr(mission_cron, "enqueue")
        else "general_beckman.enqueue",
        fake_enqueue,
    )
    # Defensive: also patch on the package since _loop imports lazily.
    import general_beckman as gb
    monkeypatch.setattr(gb, "enqueue", fake_enqueue)

    real_sleep = asyncio.sleep

    async def tiny_sleep(_secs):
        # Yield several times so other coros can progress.
        await real_sleep(0)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", tiny_sleep)

    await mission_cron.arm(mission_id=1, action="cron_backup_verify", interval_seconds=3600)
    await _wait_for(lambda: len(calls) >= 3, timeout=2.0)
    await mission_cron.disarm(1)

    # Each call must be a mechanical task on the ongoing lane with the
    # canonical mechanical context shape.
    first = calls[0]
    spec = first["spec"]
    assert spec["agent_type"] == "mechanical"
    assert spec["context"]["executor"] == "mechanical"
    assert spec["context"]["payload"]["action"] == "cron_backup_verify"
    assert spec["mission_id"] == 1
    assert first["kwargs"].get("lane") == "ongoing"


@pytest.mark.asyncio
async def test_arm_is_idempotent_replaces_existing(monkeypatch):
    calls: list[str] = []

    async def fake_enqueue(spec, **kwargs):
        calls.append(spec["context"]["payload"]["action"])
        return 1

    import general_beckman as gb
    monkeypatch.setattr(gb, "enqueue", fake_enqueue)
    real_sleep = asyncio.sleep

    async def tiny_sleep(_secs):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", tiny_sleep)

    await mission_cron.arm(mission_id=5, action="cron_dep_hygiene", interval_seconds=60)
    await mission_cron.arm(mission_id=5, action="cron_dep_hygiene", interval_seconds=60)
    await _wait_for(lambda: len(calls) >= 2)
    # Only one slot exists for this (action, interval) key.
    slot = mission_cron._TASKS.get(5) or {}
    keys_for_action = [k for k in slot if k.startswith("cron_dep_hygiene@")]
    assert len(keys_for_action) == 1
    await mission_cron.disarm(5)


@pytest.mark.asyncio
async def test_disarm_cancels_all_for_mission(monkeypatch):
    async def fake_enqueue(spec, **kwargs):
        return 1

    import general_beckman as gb
    monkeypatch.setattr(gb, "enqueue", fake_enqueue)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda _s: real_sleep(0))

    await mission_cron.arm(mission_id=7, action="cron_backup_verify", interval_seconds=10)
    await mission_cron.arm(mission_id=7, action="cron_cve_scan", interval_seconds=20)
    assert mission_cron.is_armed(7)

    n = await mission_cron.disarm(7)
    assert n == 2
    # Slot is fully gone.
    assert not mission_cron.is_armed(7)


@pytest.mark.asyncio
async def test_disarm_unknown_mission_is_zero():
    n = await mission_cron.disarm(99999)
    assert n == 0


@pytest.mark.asyncio
async def test_arm_from_cursor_arms_each_entry(monkeypatch):
    armed_calls: list[tuple] = []

    async def fake_arm(mid, action, interval):
        armed_calls.append((mid, action, interval))

    monkeypatch.setattr(mission_cron, "arm", fake_arm)
    m = _FakeMission(
        42,
        {
            "cron": [
                {"action": "cron_backup_verify", "interval_seconds": 86400},
                {"action": "cron_cve_scan", "interval_seconds": 86400},
            ]
        },
    )
    n = await mission_cron.arm_from_cursor(m)
    assert n == 2
    assert (42, "cron_backup_verify", 86400) in armed_calls
    assert (42, "cron_cve_scan", 86400) in armed_calls


@pytest.mark.asyncio
async def test_arm_from_cursor_skips_bad_entries(monkeypatch):
    armed_calls: list[tuple] = []

    async def fake_arm(mid, action, interval):
        armed_calls.append((mid, action, interval))

    monkeypatch.setattr(mission_cron, "arm", fake_arm)
    m = _FakeMission(
        9,
        {
            "cron": [
                {"action": "ok_one", "interval_seconds": 60},
                {"action": ""},  # missing interval
                "not a dict",
                {"interval_seconds": 30},  # missing action
            ]
        },
    )
    n = await mission_cron.arm_from_cursor(m)
    assert n == 1
    assert armed_calls == [(9, "ok_one", 60)]


@pytest.mark.asyncio
async def test_arm_from_cursor_no_cursor_returns_zero():
    m = _FakeMission(3, {})
    n = await mission_cron.arm_from_cursor(m)
    assert n == 0
