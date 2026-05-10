"""Z10-T3B — per-mission SANDBOX_MODE=local opt-in.

When a mission requests ``local`` while the system default is
``docker`` (the safe default), the dispatcher must open a
``sandbox_local_mode`` confirmation and block until the founder
approves or rejects.

Confirmation polling logic is shared with mr_roboto's gate (T1C); here
we just drive verdicts directly in the ``action_confirmations`` row.
"""
from __future__ import annotations

import asyncio

import pytest

from src.tools import shell


async def _init_db(tmp_path, monkeypatch):
    db_path = tmp_path / "sandbox_optin.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


async def _set_mission_mode(db_mod, mission_id: int, mode: str) -> None:
    db = await db_mod.get_db()
    await db.execute(
        "UPDATE missions SET sandbox_mode = ? WHERE id = ?",
        (mode, mission_id),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_mission_local_request_opens_confirmation(tmp_path, monkeypatch):
    db_mod = await _init_db(tmp_path, monkeypatch)
    mission_id = await db_mod.add_mission(
        "T3B opt-in", "test", workflow=""
    )
    await _set_mission_mode(db_mod, mission_id, "local")

    # System default = "docker", mission requested "local". We expect
    # _gate_local_mode_optin to OPEN a confirmation row. Approve it
    # mid-flight so the helper resolves quickly.
    monkeypatch.setattr(shell, "SANDBOX_MODE", "docker")

    async def _approve_after_short_delay():
        await asyncio.sleep(0.1)
        db = await db_mod.get_db()
        await db.execute(
            "UPDATE action_confirmations SET verdict = 'approved' "
            "WHERE verb = 'sandbox_local_mode'"
        )
        await db.commit()

    # Race the approver against the gate.
    approver = asyncio.create_task(_approve_after_short_delay())
    approved = await shell._gate_local_mode_optin(mission_id)
    await approver
    assert approved is True

    # Verify a row was inserted with the expected verb + reversibility.
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT verb, reversibility, verdict FROM action_confirmations "
        "WHERE verb = 'sandbox_local_mode'"
    )
    rows = await cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "sandbox_local_mode"
    assert rows[0][1] == "partial"
    assert rows[0][2] == "approved"


@pytest.mark.asyncio
async def test_mission_local_request_rejected_falls_back(tmp_path, monkeypatch):
    db_mod = await _init_db(tmp_path, monkeypatch)
    mission_id = await db_mod.add_mission("T3B reject", "test", workflow="")
    await _set_mission_mode(db_mod, mission_id, "local")
    monkeypatch.setattr(shell, "SANDBOX_MODE", "docker")

    async def _reject():
        await asyncio.sleep(0.1)
        db = await db_mod.get_db()
        await db.execute(
            "UPDATE action_confirmations SET verdict = 'rejected' "
            "WHERE verb = 'sandbox_local_mode'"
        )
        await db.commit()

    rejector = asyncio.create_task(_reject())
    approved = await shell._gate_local_mode_optin(mission_id)
    await rejector
    assert approved is False


@pytest.mark.asyncio
async def test_resolve_sandbox_mode_reads_mission_column(tmp_path, monkeypatch):
    db_mod = await _init_db(tmp_path, monkeypatch)
    mission_id = await db_mod.add_mission("T3B resolve", "test", workflow="")
    # Default at INSERT time is 'docker'.
    assert await shell.resolve_sandbox_mode(mission_id) == "docker"
    # Flip to local and re-resolve.
    await _set_mission_mode(db_mod, mission_id, "local")
    assert await shell.resolve_sandbox_mode(mission_id) == "local"


@pytest.mark.asyncio
async def test_resolve_sandbox_mode_no_mission_falls_back(monkeypatch):
    monkeypatch.setattr(shell, "SANDBOX_MODE", "docker")
    assert await shell.resolve_sandbox_mode(None) == "docker"
    monkeypatch.setattr(shell, "SANDBOX_MODE", "local")
    assert await shell.resolve_sandbox_mode(None) == "local"
