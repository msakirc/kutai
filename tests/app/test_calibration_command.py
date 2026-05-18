"""Z10 T4B — /calibration Telegram command."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "calibration.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


async def _seed_matrix(db_mod, rows: list[tuple]):
    """rows: (model_id, task_kind, bucket, sample_n, correct_n, reliability)"""
    db = await db_mod.get_db()
    for r in rows:
        await db.execute(
            "INSERT INTO confidence_reliability_scores "
            "(model_id, task_kind, confidence_bucket, sample_n, correct_n,"
            " reliability) VALUES (?, ?, ?, ?, ?, ?)", r,
        )
    await db.commit()


def _bind_cmd_calibration():
    """Pull the unbound method straight off the class. Avoids the
    full TelegramInterface __init__ (which builds the PTB Application)."""
    from src.app.telegram_bot import TelegramInterface
    return TelegramInterface.cmd_calibration


@pytest.mark.asyncio
async def test_command_returns_formatted_matrix(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_matrix(db_mod, [
        ("gpt-oss", "coder", "high", 89, 75, 0.84),
        ("gpt-oss", "coder", "med", 45, 32, 0.71),
        ("gpt-oss", "coder", "low", 12, 10, 0.83),
        ("sonnet", "coder", "high", 61, 55, 0.90),
    ])

    cmd = _bind_cmd_calibration()
    # Stub instance: only _reply is invoked.
    self_stub = MagicMock()
    self_stub._reply = AsyncMock()
    update = MagicMock()
    ctx = MagicMock()
    await cmd(self_stub, update, ctx)

    self_stub._reply.assert_awaited_once()
    msg = self_stub._reply.await_args.args[1]
    assert "Calibration matrix" in msg
    assert "gpt-oss/coder" in msg
    assert "sonnet/coder" in msg
    # Reliability cues — high sonnet ≥0.85 → green, gpt-oss high mid → yellow
    assert "🟢" in msg
    assert "🟡" in msg


@pytest.mark.asyncio
async def test_command_handles_empty_matrix(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    cmd = _bind_cmd_calibration()
    self_stub = MagicMock()
    self_stub._reply = AsyncMock()
    update = MagicMock()
    ctx = MagicMock()
    await cmd(self_stub, update, ctx)

    self_stub._reply.assert_awaited_once()
    msg = self_stub._reply.await_args.args[1]
    assert "No calibration data yet" in msg


@pytest.mark.asyncio
async def test_low_sample_renders_as_em_dash(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_matrix(db_mod, [
        ("m1", "k1", "high", 3, 1, 0.33),  # below visible threshold (5)
        ("m1", "k1", "med", 50, 25, 0.50),
    ])
    cmd = _bind_cmd_calibration()
    self_stub = MagicMock()
    self_stub._reply = AsyncMock()
    await cmd(self_stub, MagicMock(), MagicMock())
    msg = self_stub._reply.await_args.args[1]
    # The high bucket has only 3 samples → '—'
    assert "—" in msg
