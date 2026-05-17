"""Tests for demand convenience API and shared threshold constant (Task 1).

Uses init_db() + explicit table clear, matching the pattern used by the
yalayut demand-signal test suite (asyncio_mode=auto from pytest.ini).
"""
import pytest

from src.infra.db import init_db, get_db
from yalayut.discovery import demand as _demand


@pytest.fixture
async def db():
    """Initialise the DB and wipe demand_signals for test isolation."""
    await init_db()
    db = await get_db()
    await db.execute("DELETE FROM yalayut_demand_signals")
    await db.commit()


@pytest.mark.asyncio
async def test_record_helper_inserts_row(db):
    row_id = await _demand.record(
        source_step_pattern="test:helper-pattern",
        intent_keywords=["pdf", "extract"],
        signal_type="tool_call",
        confidence=0.4,
    )
    assert row_id > 0
    stacked = await _demand.stack_confidence("test:helper-pattern")
    assert stacked == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_threshold_constant_is_half():
    assert _demand.DEMAND_DISCOVERY_THRESHOLD == 0.5
