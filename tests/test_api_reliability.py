import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import aiosqlite


@pytest_asyncio.fixture
async def mem_db():
    """In-memory SQLite for reliability tests."""
    db = await aiosqlite.connect(":memory:")
    await db.execute("""
        CREATE TABLE api_reliability (
            api_name TEXT PRIMARY KEY,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            consecutive_failures INTEGER DEFAULT 0,
            last_success TEXT,
            last_failure TEXT,
            status TEXT DEFAULT 'active'
        )
    """)
    await db.commit()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_single_failure_does_not_demote(mem_db):
    """An API with 9 successes and 1 failure should stay active."""
    from src.infra.db import record_api_call
    with patch("src.infra.db.get_db", new_callable=AsyncMock, return_value=mem_db):
        for _ in range(9):
            await record_api_call("wttr.in", success=True)
        await record_api_call("wttr.in", success=False)

    cur = await mem_db.execute("SELECT status, consecutive_failures FROM api_reliability WHERE api_name = 'wttr.in'")
    row = await cur.fetchone()
    assert row[0] == "active", f"Expected active, got {row[0]}"
    assert row[1] == 1


@pytest.mark.asyncio
async def test_low_sample_does_not_demote(mem_db):
    """Even with 100% failure, fewer than 15 calls should not demote (but consecutive failures trigger warning)."""
    from src.infra.db import record_api_call
    with patch("src.infra.db.get_db", new_callable=AsyncMock, return_value=mem_db):
        for _ in range(14):
            await record_api_call("badapi", success=False)

    cur = await mem_db.execute("SELECT status FROM api_reliability WHERE api_name = 'badapi'")
    row = await cur.fetchone()
    # Not demoted (needs >=15 total), but consecutive failures (14 >= 3) triggers warning
    assert row[0] == "warning", f"Expected warning from consecutive failures, got {row[0]}"


@pytest.mark.asyncio
async def test_high_failure_rate_demotes_after_threshold(mem_db):
    """API with >=15 calls and <25% success rate gets demoted."""
    from src.infra.db import record_api_call
    with patch("src.infra.db.get_db", new_callable=AsyncMock, return_value=mem_db):
        for _ in range(3):
            await record_api_call("badapi", success=True)
        for _ in range(12):
            await record_api_call("badapi", success=False)

    cur = await mem_db.execute("SELECT status FROM api_reliability WHERE api_name = 'badapi'")
    row = await cur.fetchone()
    assert row[0] == "demoted"


@pytest.mark.asyncio
async def test_consecutive_failures_triggers_warning(mem_db):
    """3 consecutive failures should trigger warning even with good overall rate."""
    from src.infra.db import record_api_call
    with patch("src.infra.db.get_db", new_callable=AsyncMock, return_value=mem_db):
        for _ in range(20):
            await record_api_call("flaky", success=True)
        for _ in range(3):
            await record_api_call("flaky", success=False)

    cur = await mem_db.execute("SELECT status FROM api_reliability WHERE api_name = 'flaky'")
    row = await cur.fetchone()
    assert row[0] == "warning"


@pytest.mark.asyncio
async def test_success_resets_consecutive_failures(mem_db):
    """A success should reset the consecutive failure counter."""
    from src.infra.db import record_api_call
    with patch("src.infra.db.get_db", new_callable=AsyncMock, return_value=mem_db):
        for _ in range(2):
            await record_api_call("recov", success=False)
        await record_api_call("recov", success=True)

    cur = await mem_db.execute("SELECT consecutive_failures FROM api_reliability WHERE api_name = 'recov'")
    row = await cur.fetchone()
    assert row[0] == 0


@pytest.mark.asyncio
async def test_suspended_at_very_low_rate(mem_db):
    """>=20 calls and <10% success → suspended."""
    from src.infra.db import record_api_call
    with patch("src.infra.db.get_db", new_callable=AsyncMock, return_value=mem_db):
        await record_api_call("dead", success=True)
        for _ in range(19):
            await record_api_call("dead", success=False)

    cur = await mem_db.execute("SELECT status FROM api_reliability WHERE api_name = 'dead'")
    row = await cur.fetchone()
    assert row[0] == "suspended"
