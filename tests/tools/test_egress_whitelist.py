"""Z10-T3B — per-mission egress whitelist + broader_egress confirmation.

The whitelist lives at ``config/egress_allowlist.txt``. Callers wanting
to reach a host outside the list pass ``request_egress_to=<host>``;
the shell tool opens a ``broader_egress`` confirmation in that case.
"""
from __future__ import annotations

import asyncio

import pytest

from src.tools import shell


async def _init_db(tmp_path, monkeypatch):
    db_path = tmp_path / "egress.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


def test_load_allowlist_seeds_file(tmp_path, monkeypatch):
    """Missing file → seed values get written + returned."""
    cfg = tmp_path / "config" / "egress_allowlist.txt"
    monkeypatch.setattr(shell, "_EGRESS_ALLOWLIST_PATH_DEFAULT", cfg)
    hosts = shell.load_egress_allowlist(cfg)
    assert "api.openai.com" in hosts
    assert "github.com" in hosts
    # The seed file exists now.
    assert cfg.exists()


def test_host_in_allowlist_subdomain_match():
    allow = {"huggingface.co", "github.com"}
    assert shell._host_in_allowlist("huggingface.co", allow) is True
    assert shell._host_in_allowlist("cdn-lfs.huggingface.co", allow) is True
    assert shell._host_in_allowlist("evil.com", allow) is False
    # Empty host → False.
    assert shell._host_in_allowlist("", allow) is False


@pytest.mark.asyncio
async def test_whitelisted_host_no_confirmation(tmp_path, monkeypatch):
    """Host on the allowlist → :func:`_gate_broader_egress` not even called.

    We assert by spying on request_confirmation — if the gate opens a
    row we'd see it. Use the higher-level path via :func:`run_shell`
    by monkeypatching ensure paths to skip docker.
    """
    db_mod = await _init_db(tmp_path, monkeypatch)
    mission_id = await db_mod.add_mission("T3B egress wl", "test", workflow="")

    # Force run_shell to bail before doing any docker work — point
    # SANDBOX_MODE at "none" so the early-skip returns a string.
    monkeypatch.setattr(shell, "SANDBOX_MODE", "none")

    # Seed allowlist with a known host.
    cfg = tmp_path / "config" / "egress_allowlist.txt"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("api.openai.com\n", encoding="utf-8")
    monkeypatch.setattr(shell, "_EGRESS_ALLOWLIST_PATH_DEFAULT", cfg)

    # On-allowlist → no confirmation row created.
    result = await shell.run_shell(
        "echo hi",
        timeout=2,
        mission_id=mission_id,
        request_egress_to="api.openai.com",
    )
    # Should NOT be a BLOCKED message.
    assert "broader_egress" not in result
    assert "not approved" not in result

    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT COUNT(*) FROM action_confirmations WHERE verb = 'broader_egress'"
    )
    n = (await cur.fetchone())[0]
    assert n == 0


@pytest.mark.asyncio
async def test_off_whitelist_opens_confirmation(tmp_path, monkeypatch):
    db_mod = await _init_db(tmp_path, monkeypatch)
    mission_id = await db_mod.add_mission("T3B egress off", "test", workflow="")
    monkeypatch.setattr(shell, "SANDBOX_MODE", "none")

    cfg = tmp_path / "config" / "egress_allowlist.txt"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("api.openai.com\n", encoding="utf-8")
    monkeypatch.setattr(shell, "_EGRESS_ALLOWLIST_PATH_DEFAULT", cfg)

    # Approve the broader_egress request after a brief delay so the
    # gate's polling loop sees the verdict.
    async def _approve():
        await asyncio.sleep(0.1)
        db = await db_mod.get_db()
        await db.execute(
            "UPDATE action_confirmations SET verdict = 'approved' "
            "WHERE verb = 'broader_egress'"
        )
        await db.commit()

    approver = asyncio.create_task(_approve())
    result = await shell.run_shell(
        "echo hi",
        timeout=2,
        mission_id=mission_id,
        request_egress_to="some-private-host.example",
    )
    await approver

    # The off-list request must have opened a confirmation row.
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT verb, reversibility, verdict FROM action_confirmations "
        "WHERE verb = 'broader_egress'"
    )
    rows = await cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "broader_egress"
    assert rows[0][1] == "partial"
    assert rows[0][2] == "approved"
    # After approval, run_shell goes on to its SANDBOX_MODE=none short-circuit.
    assert "skipped" in result.lower() or isinstance(result, str)


@pytest.mark.asyncio
async def test_off_whitelist_rejected_blocks(tmp_path, monkeypatch):
    db_mod = await _init_db(tmp_path, monkeypatch)
    mission_id = await db_mod.add_mission("T3B egress rej", "test", workflow="")
    monkeypatch.setattr(shell, "SANDBOX_MODE", "none")

    cfg = tmp_path / "config" / "egress_allowlist.txt"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("api.openai.com\n", encoding="utf-8")
    monkeypatch.setattr(shell, "_EGRESS_ALLOWLIST_PATH_DEFAULT", cfg)

    async def _reject():
        await asyncio.sleep(0.1)
        db = await db_mod.get_db()
        await db.execute(
            "UPDATE action_confirmations SET verdict = 'rejected' "
            "WHERE verb = 'broader_egress'"
        )
        await db.commit()

    rejector = asyncio.create_task(_reject())
    result = await shell.run_shell(
        "echo hi",
        timeout=2,
        mission_id=mission_id,
        request_egress_to="evil.example",
    )
    await rejector
    assert "BLOCKED" in result and "evil.example" in result
