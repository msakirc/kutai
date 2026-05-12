"""Z8 T4D — escalation_policy + quiet-hours channel routing."""
from __future__ import annotations

from datetime import datetime

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "escalation.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.mark.asyncio
async def test_table_exists(tmp_path, monkeypatch):
    db_mod = await _setup(tmp_path, monkeypatch)
    conn = await db_mod.get_db()
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='escalation_policy'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None, "escalation_policy migration didn't run"


@pytest.mark.asyncio
async def test_load_default_when_no_row(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from src.ops.escalation_policy import load_policy

    policy = await load_policy(123)
    assert policy.tier1_channel == "telegram"
    assert policy.tier2_channel == "telegram"
    assert policy.tier3_channel == "sms"


@pytest.mark.asyncio
async def test_set_and_load_roundtrip(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from src.ops.escalation_policy import Policy, load_policy, set_policy

    await set_policy(
        Policy(
            mission_id=7,
            quiet_hours_start="22:00",
            quiet_hours_end="06:00",
            tier1_channel="telegram",
            tier2_channel="email",
            tier3_channel="sms",
            tz="UTC",
        )
    )
    policy = await load_policy(7)
    assert policy.tier2_channel == "email"
    assert policy.quiet_hours_start == "22:00"


def test_tier_of():
    from src.ops.escalation_policy import tier_of

    assert tier_of("low") == 1
    assert tier_of("medium") == 1
    assert tier_of("high") == 2
    assert tier_of("critical") == 3
    assert tier_of("sec_critical") == 3


def test_in_quiet_hours_no_window():
    """No window configured → never quiet."""
    from src.ops.escalation_policy import Policy, in_quiet_hours

    p = Policy(mission_id=1)
    assert in_quiet_hours(p, datetime(2026, 5, 12, 3, 0)) is False


def test_in_quiet_hours_overnight_window():
    """22:00→06:00 covers 03:00 (next day) and 23:00 (same day)."""
    from src.ops.escalation_policy import Policy, in_quiet_hours

    p = Policy(
        mission_id=1, quiet_hours_start="22:00", quiet_hours_end="06:00"
    )
    assert in_quiet_hours(p, datetime(2026, 5, 12, 3, 0)) is True
    assert in_quiet_hours(p, datetime(2026, 5, 12, 23, 0)) is True
    assert in_quiet_hours(p, datetime(2026, 5, 12, 10, 0)) is False


def test_channel_for_critical_ignores_quiet_hours():
    """Tier-3 (critical) must page on sms even during quiet hours."""
    from src.ops.escalation_policy import Policy, channel_for

    p = Policy(
        mission_id=1, quiet_hours_start="22:00", quiet_hours_end="06:00"
    )
    ch = channel_for(p, "critical", datetime(2026, 5, 12, 3, 0))
    assert ch == "sms"


def test_channel_for_non_critical_during_quiet_hours():
    """Tier 1/2 collapse to telegram_log_only during quiet hours."""
    from src.ops.escalation_policy import Policy, channel_for

    p = Policy(
        mission_id=1,
        quiet_hours_start="22:00",
        quiet_hours_end="06:00",
        tier1_channel="telegram",
        tier2_channel="email",
    )
    assert channel_for(p, "high", datetime(2026, 5, 12, 3, 0)) == "telegram_log_only"
    assert channel_for(p, "low", datetime(2026, 5, 12, 3, 0)) == "telegram_log_only"
    # Outside quiet → normal
    assert channel_for(p, "high", datetime(2026, 5, 12, 10, 0)) == "email"


@pytest.mark.asyncio
async def test_resolve_channel_helper(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from src.ops.escalation_policy import Policy, resolve_channel, set_policy

    await set_policy(
        Policy(
            mission_id=10,
            quiet_hours_start="22:00",
            quiet_hours_end="06:00",
            tier3_channel="sms",
        )
    )
    res = await resolve_channel(10, "critical", datetime(2026, 5, 12, 3, 0))
    assert res["channel"] == "sms"
    assert res["tier"] == 3
    assert res["quiet"] is True
