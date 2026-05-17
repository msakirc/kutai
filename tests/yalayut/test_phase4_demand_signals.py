import asyncio
import json

import pytest

from src.infra.db import init_db, get_db
from yalayut.discovery import demand


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_record_signal_inserts_row(loop):
    async def _run():
        await init_db()
        sig = demand.DemandSignal(
            source_step_pattern="auth-wiring-fastapi",
            intent_keywords=["auth", "jwt", "fastapi"],
            signal_type="planning_miss",
            confidence=0.4,
        )
        sid = await demand.record_signal(sig)
        assert sid > 0
        db = await get_db()
        cur = await db.execute(
            "SELECT signal_type, confidence, resulted_in_discovery "
            "FROM yalayut_demand_signals WHERE id = ?", (sid,))
        row = await cur.fetchone()
        await cur.close()
        assert row[0] == "planning_miss"
        assert abs(row[1] - 0.4) < 1e-6
        assert row[2] in (0, None)
    loop.run_until_complete(_run())


def test_confidence_stacks_across_signals(loop):
    async def _run():
        await init_db()
        pat = "rag-pipeline-setup"
        for st, conf in [("planning_miss", 0.4), ("tool_call", 0.3),
                          ("hint_miss", 0.2)]:
            await demand.record_signal(demand.DemandSignal(
                source_step_pattern=pat, intent_keywords=["rag"],
                signal_type=st, confidence=conf))
        stacked = await demand.stack_confidence(pat)
        # 1 - (1-.4)(1-.3)(1-.2) = 1 - .336 = .664
        assert abs(stacked - 0.664) < 1e-3
    loop.run_until_complete(_run())


def test_dedupe_within_cooldown(loop):
    async def _run():
        await init_db()
        pat = "cooldown-pattern-xyz"
        s1 = demand.DemandSignal(source_step_pattern=pat,
                                 intent_keywords=["x"],
                                 signal_type="dlq", confidence=0.5)
        first = await demand.record_signal(s1)
        assert first > 0
        # Same pattern + same type within cooldown → deduped (returns -1).
        second = await demand.record_signal(s1)
        assert second == -1
    loop.run_until_complete(_run())


def test_pending_signals_orders_by_stacked_confidence(loop):
    async def _run():
        await init_db()
        await demand.record_signal(demand.DemandSignal(
            source_step_pattern="low-pat", intent_keywords=["a"],
            signal_type="repeat_pattern", confidence=0.15))
        for st in ("planning_miss", "step_entry_miss", "tool_call"):
            await demand.record_signal(demand.DemandSignal(
                source_step_pattern="high-pat", intent_keywords=["b"],
                signal_type=st, confidence=0.5))
        pend = await demand.pending_signals(limit=10)
        pats = [p["source_step_pattern"] for p in pend]
        assert pats.index("high-pat") < pats.index("low-pat")
    loop.run_until_complete(_run())
