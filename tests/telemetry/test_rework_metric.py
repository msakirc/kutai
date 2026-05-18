"""Tests for B10 mission-level rework metric instrumentation.

Spec: docs/i2p-evolution/01-pre-code-master-synthesis.md §B10
Helper: src/telemetry/rework.py
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.infra.db as _db_mod
from src.infra.db import (
    add_mission, get_mission, init_db, increment_mission_rework_loops,
    get_mission_rework_summary, add_task, update_task,
)
from src.telemetry.rework import (
    record_rollback, is_phase_7_rework, _phase_num,
)


async def _fresh_db(tmp_path, monkeypatch):
    """Reset DB to a fresh temp file for isolation (mirrors tests/test_beckman_apply.py)."""
    db_file = tmp_path / "rework.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    await init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


# ─── Migration ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_migration_adds_column(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    db = await _db_mod.get_db()
    cursor = await db.execute("PRAGMA table_info(missions)")
    cols = [row[1] for row in await cursor.fetchall()]
    assert "phase_7_rework_loops" in cols, f"got cols: {cols}"
    await _close_db()


@pytest.mark.asyncio
async def test_migration_idempotent(tmp_path, monkeypatch):
    """Running init_db twice must not error or duplicate the column."""
    await _fresh_db(tmp_path, monkeypatch)
    # Second init — column already exists; ALTER must swallow.
    await init_db()
    db = await _db_mod.get_db()
    cursor = await db.execute("PRAGMA table_info(missions)")
    cols = [row[1] for row in await cursor.fetchall()]
    assert cols.count("phase_7_rework_loops") == 1
    await _close_db()


@pytest.mark.asyncio
async def test_existing_missions_default_zero(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    mid = await add_mission(title="m", description="d")
    row = await get_mission(mid)
    assert row["phase_7_rework_loops"] == 0
    await _close_db()


# ─── Phase parsing ──────────────────────────────────────────────────────────


def test_phase_num_extracts():
    assert _phase_num("8.3") == 8
    assert _phase_num("phase_8") == 8
    assert _phase_num("8") == 8
    assert _phase_num("12.4") == 12
    assert _phase_num("") is None
    assert _phase_num("garbage") is None


def test_is_phase_7_rework():
    assert is_phase_7_rework("8.3", "4.16") is True
    assert is_phase_7_rework("phase_9", "phase_5") is True
    assert is_phase_7_rework("7", "6") is True
    # Same band — not a rollback
    assert is_phase_7_rework("8.3", "8.4") is False
    # Sub-7 to sub-7 — not a P7 rework
    assert is_phase_7_rework("4.1", "3.2") is False
    # Forward (not a rollback at all)
    assert is_phase_7_rework("4", "8") is False


# ─── record_rollback core behaviour ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_rollback_increments_and_emits(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    mid = await add_mission(title="m", description="d")

    # Spy on the logger
    from src.telemetry import rework as rw_mod
    emitted = []
    monkeypatch.setattr(
        rw_mod.logger, "info",
        lambda msg, **ctx: emitted.append((msg, ctx)),
    )

    await record_rollback(
        mission_id=mid,
        from_phase="8.3",
        to_phase="4.16",
        reason="reviewer_reject",
        triggered_by="code_reviewer",
    )

    row = await get_mission(mid)
    assert row["phase_7_rework_loops"] == 1
    assert len(emitted) == 1
    msg, ctx = emitted[0]
    assert msg == "phase_rollback"
    assert ctx["event"] == "phase_rollback"
    assert ctx["mission_id"] == mid
    assert ctx["from_phase"] == "8.3"
    assert ctx["to_phase"] == "4.16"
    assert ctx["reason"] == "reviewer_reject"
    assert ctx["triggered_by"] == "code_reviewer"
    assert ctx["phase_7_rework"] is True
    assert ctx["new_count"] == 1

    await _close_db()


@pytest.mark.asyncio
async def test_record_rollback_same_band_skips_counter(tmp_path, monkeypatch):
    """Same-step retry: emit event but don't bump counter."""
    await _fresh_db(tmp_path, monkeypatch)
    mid = await add_mission(title="m", description="d")

    from src.telemetry import rework as rw_mod
    emitted = []
    monkeypatch.setattr(
        rw_mod.logger, "info",
        lambda msg, **ctx: emitted.append((msg, ctx)),
    )

    await record_rollback(
        mission_id=mid,
        from_phase="8.3",
        to_phase="8.3",
        reason="schema_failure",
        triggered_by="coder",
    )

    row = await get_mission(mid)
    assert row["phase_7_rework_loops"] == 0  # not bumped
    assert len(emitted) == 1
    assert emitted[0][1]["phase_7_rework"] is False

    await _close_db()


@pytest.mark.asyncio
async def test_record_rollback_unknown_reason_coerced(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    mid = await add_mission(title="m", description="d")

    from src.telemetry import rework as rw_mod
    emitted_info = []
    emitted_warn = []
    monkeypatch.setattr(
        rw_mod.logger, "info",
        lambda msg, **ctx: emitted_info.append((msg, ctx)),
    )
    monkeypatch.setattr(
        rw_mod.logger, "warning",
        lambda msg, **ctx: emitted_warn.append((msg, ctx)),
    )

    await record_rollback(
        mission_id=mid,
        from_phase="9", to_phase="3",
        reason="bogus",
        triggered_by="x",
    )

    assert any("unknown reason" in m for m, _ in emitted_warn)
    assert emitted_info[0][1]["reason"] == "other"

    await _close_db()


@pytest.mark.asyncio
async def test_record_rollback_missing_mission_safe(tmp_path, monkeypatch):
    """No mission row → emit event, don't crash."""
    await _fresh_db(tmp_path, monkeypatch)

    from src.telemetry import rework as rw_mod
    emitted = []
    monkeypatch.setattr(
        rw_mod.logger, "info",
        lambda msg, **ctx: emitted.append((msg, ctx)),
    )

    # mission_id=999 doesn't exist; UPDATE is a no-op, no exception
    await record_rollback(
        mission_id=999,
        from_phase="8", to_phase="4",
        reason="founder_request",
        triggered_by="founder",
    )

    assert len(emitted) == 1
    assert emitted[0][1]["mission_id"] == 999

    await _close_db()


@pytest.mark.asyncio
async def test_increment_helper_atomic(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    mid = await add_mission(title="m", description="d")

    n1 = await increment_mission_rework_loops(mid)
    n2 = await increment_mission_rework_loops(mid)
    n3 = await increment_mission_rework_loops(mid)

    assert n1 == 1
    assert n2 == 2
    assert n3 == 3

    await _close_db()


# ─── Call-site instrumentation ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_or_dlq_quality_calls_record_rollback(
    tmp_path, monkeypatch,
):
    """Beckman _retry_or_dlq with category=quality must call record_rollback."""
    await _fresh_db(tmp_path, monkeypatch)
    mid = await add_mission(title="m", description="d")
    tid = await add_task(
        title="t", description="", agent_type="coder",
        mission_id=mid,
        context=json.dumps({
            "workflow_step_id": "8.3",
            "workflow_phase": "phase_8",
        }),
    )

    calls = []

    async def _spy(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "src.telemetry.rework.record_rollback", _spy,
    )

    from general_beckman.apply import _retry_or_dlq
    from src.infra.db import get_task
    task = await get_task(tid)
    await _retry_or_dlq(task, category="quality", error="schema fail")

    assert len(calls) == 1
    c = calls[0]
    assert c["mission_id"] == mid
    assert c["from_phase"] == "8.3"
    assert c["reason"] == "schema_failure"
    assert c["triggered_by"] == "coder"

    await _close_db()


@pytest.mark.asyncio
async def test_retry_or_dlq_non_quality_skips_telemetry(
    tmp_path, monkeypatch,
):
    """Availability/exhausted retries are NOT rework — must not emit."""
    await _fresh_db(tmp_path, monkeypatch)
    mid = await add_mission(title="m", description="d")
    tid = await add_task(
        title="t", description="", agent_type="coder",
        mission_id=mid,
        context=json.dumps({"workflow_step_id": "8.3"}),
    )

    calls = []

    async def _spy(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "src.telemetry.rework.record_rollback", _spy,
    )

    from general_beckman.apply import _retry_or_dlq
    from src.infra.db import get_task
    task = await get_task(tid)
    await _retry_or_dlq(task, category="availability", error="rate limit")

    assert calls == []

    await _close_db()


@pytest.mark.asyncio
async def test_code_review_verdict_fail_calls_record_rollback(
    tmp_path, monkeypatch,
):
    """Code-review reject path emits reviewer_reject event."""
    await _fresh_db(tmp_path, monkeypatch)
    mid = await add_mission(title="m", description="d")
    tid = await add_task(
        title="t", description="", agent_type="coder",
        mission_id=mid,
        context=json.dumps({"workflow_step_id": "8.3"}),
    )

    calls = []

    async def _spy(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "src.telemetry.rework.record_rollback", _spy,
    )

    from general_beckman.apply import _apply_code_review_verdict
    from general_beckman.result_router import PostHookVerdict
    from src.infra.db import get_task
    source = await get_task(tid)
    verdict = PostHookVerdict(
        source_task_id=tid, kind="code_review",
        passed=False,
        raw={"issues": ["bad indent", "no docstring"]},
    )
    await _apply_code_review_verdict(
        source, ctx={"workflow_step_id": "8.3"},
        pending=["code_review"], verdict=verdict,
    )

    assert len(calls) == 1
    assert calls[0]["reason"] == "reviewer_reject"
    assert calls[0]["triggered_by"] == "code_reviewer"
    assert calls[0]["from_phase"] == "8.3"

    await _close_db()


# ─── Telegram /rework command shape ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_cmd_rework_returns_summary_shape(tmp_path, monkeypatch):
    """The /rework command renders count + per-mission lines + reasons."""
    await _fresh_db(tmp_path, monkeypatch)
    mid1 = await add_mission(title="m1", description="d")
    mid2 = await add_mission(title="m2", description="d")
    await increment_mission_rework_loops(mid1)
    await increment_mission_rework_loops(mid1)  # m1 has 2 loops
    # m2 stays at 0

    # Mock TelegramInterface._reply to capture the rendered text
    from src.app import telegram_bot as tg

    sent = []

    class _FakeBot:
        async def _reply(self, update, text, **kwargs):
            sent.append(text)

        cmd_rework = tg.TelegramInterface.cmd_rework

    fake_update = MagicMock()
    fake_ctx = MagicMock()
    fake_ctx.args = []

    bot = _FakeBot()
    await bot.cmd_rework(fake_update, fake_ctx)

    assert len(sent) == 1
    out = sent[0]
    assert "Rework Metric" in out
    assert "2" in out  # total
    assert "m1" in out
    assert "m2" in out

    await _close_db()
