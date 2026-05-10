"""Z10 T1C — action_confirmations skeleton tests."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "confirmations.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_path, db_mod


@pytest.mark.asyncio
async def test_request_confirmation_starts_pending(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    cid = await db_mod.request_confirmation(
        task_id=11,
        verb="notify_user",
        reversibility="irreversible",
        payload_summary="ping founder",
    )
    assert cid > 0
    state = await db_mod.check_confirmation(cid)
    assert state["verdict"] == "pending"
    assert state["responded_at"] is None


@pytest.mark.asyncio
async def test_resolve_approved_flips_verdict(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    cid = await db_mod.request_confirmation(
        task_id=1, verb="run_cmd", reversibility="partial",
        payload_summary="run X",
    )
    await db_mod.resolve_confirmation(cid, "approved")
    state = await db_mod.check_confirmation(cid)
    assert state["verdict"] == "approved"
    assert state["responded_at"] is not None


@pytest.mark.asyncio
async def test_resolve_rejected_mirrors(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    cid = await db_mod.request_confirmation(
        task_id=2, verb="run_cmd", reversibility="irreversible",
        payload_summary="rm -rf",
    )
    await db_mod.resolve_confirmation(cid, "rejected")
    state = await db_mod.check_confirmation(cid)
    assert state["verdict"] == "rejected"
    assert state["responded_at"] is not None


@pytest.mark.asyncio
async def test_resolve_rejects_unknown_verdict(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    cid = await db_mod.request_confirmation(
        task_id=3, verb="git_commit", reversibility="full",
        payload_summary="",
    )
    with pytest.raises(ValueError):
        await db_mod.resolve_confirmation(cid, "yolo")


@pytest.mark.asyncio
async def test_check_missing_id(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    state = await db_mod.check_confirmation(99999)
    assert state["verdict"] == "missing"
    assert state["responded_at"] is None
