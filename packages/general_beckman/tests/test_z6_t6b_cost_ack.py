"""Z6 T6B — cost_ack founder_action for irreversible + cost-estimated steps.

T1C admission already implements cost_ack handling. This file pins down the
contract from the T6B perspective:

  • irreversible + cost>0  + no prior ack  →  emit cost_ack, block.
  • irreversible + cost>0  + done ack       →  admit.
  • irreversible + cost==0                  →  admit (no ack required).
  • partial      + cost>0                   →  admit (only irreversible gates).
  • duplicate emit guarded by dedup        →  one row, not two.
"""
from __future__ import annotations

import json

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "z6_t6b.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    import src.founder_actions as fa
    fa._reset_lifecycle_cache()
    return db_path, db_mod, fa


def _task(mid: int, *, reversibility: str, cost: float | int | None,
          step_id: str = "13.1") -> dict:
    ctx = {"workflow_step_id": step_id, "real_tool_kind": "vercel"}
    if cost is not None:
        ctx["cost_estimate_usd"] = cost
    return {
        "id": 1,
        "mission_id": mid,
        "needs_real_tools": 1,
        "reversibility": reversibility,
        "context": json.dumps(ctx),
    }


def _stub_resolves_and_creds(monkeypatch):
    monkeypatch.setattr(
        "general_beckman.z6_admission._resolve_adapter",
        lambda kinds: "vercel",
    )
    async def _has_cred(_svc):
        return {"token": "xxx"}
    monkeypatch.setattr(
        "src.security.credential_store.get_credential", _has_cred,
    )


@pytest.mark.asyncio
async def test_irreversible_with_cost_emits_cost_ack(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    _stub_resolves_and_creds(monkeypatch)
    from general_beckman.z6_admission import check_z6_admission

    mid = await db_mod.add_mission("m", "")
    res = await check_z6_admission(
        _task(mid, reversibility="irreversible", cost=99),
        mid,
    )
    assert res.admit is False
    assert "cost_ack" in res.reason
    rows = await fa.list_by_mission(mid)
    cost_acks = [r for r in rows if r.kind == "cost_ack"]
    assert len(cost_acks) == 1
    a = cost_acks[0]
    assert a.cost_estimate_usd == 99.0
    assert a.blocking_step_id == "13.1"
    assert a.reversibility == "irreversible"
    assert a.expected_output_kind == "ack_only"
    # Title carries the cost number prominently.
    assert "$99" in a.title or "99" in a.title


@pytest.mark.asyncio
async def test_done_ack_admits(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    _stub_resolves_and_creds(monkeypatch)
    from general_beckman.z6_admission import check_z6_admission

    mid = await db_mod.add_mission("m", "")
    a = await fa.create(
        mid, "cost_ack", "ack", "why", [],
        blocking_step_id="13.1", cost_estimate_usd=99,
        notify_telegram=False,
    )
    await fa.resolve(a.id)

    res = await check_z6_admission(
        _task(mid, reversibility="irreversible", cost=99),
        mid,
    )
    assert res.admit is True


@pytest.mark.asyncio
async def test_cost_zero_admits_no_ack(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    _stub_resolves_and_creds(monkeypatch)
    from general_beckman.z6_admission import check_z6_admission

    mid = await db_mod.add_mission("m", "")
    res = await check_z6_admission(
        _task(mid, reversibility="irreversible", cost=0),
        mid,
    )
    assert res.admit is True
    rows = await fa.list_by_mission(mid)
    assert not [r for r in rows if r.kind == "cost_ack"]


@pytest.mark.asyncio
async def test_cost_absent_admits_no_ack(tmp_path, monkeypatch):
    _, db_mod, _fa = await _setup(tmp_path, monkeypatch)
    _stub_resolves_and_creds(monkeypatch)
    from general_beckman.z6_admission import check_z6_admission

    mid = await db_mod.add_mission("m", "")
    res = await check_z6_admission(
        _task(mid, reversibility="irreversible", cost=None),
        mid,
    )
    assert res.admit is True


@pytest.mark.asyncio
async def test_partial_reversibility_admits_with_cost(tmp_path, monkeypatch):
    _, db_mod, _fa = await _setup(tmp_path, monkeypatch)
    _stub_resolves_and_creds(monkeypatch)
    from general_beckman.z6_admission import check_z6_admission

    mid = await db_mod.add_mission("m", "")
    res = await check_z6_admission(
        _task(mid, reversibility="partial", cost=200),
        mid,
    )
    assert res.admit is True


@pytest.mark.asyncio
async def test_duplicate_cost_ack_deduped(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    _stub_resolves_and_creds(monkeypatch)
    from general_beckman.z6_admission import check_z6_admission

    mid = await db_mod.add_mission("m", "")
    task = _task(mid, reversibility="irreversible", cost=42)
    r1 = await check_z6_admission(task, mid)
    r2 = await check_z6_admission(task, mid)
    assert r1.admit is False and r2.admit is False
    rows = await fa.list_by_mission(mid)
    cost_acks = [r for r in rows if r.kind == "cost_ack"]
    assert len(cost_acks) == 1, (
        f"expected single cost_ack after two checks, got {len(cost_acks)}"
    )


@pytest.mark.asyncio
async def test_render_card_shows_cost_prominently(tmp_path, monkeypatch):
    """T1D render should call out the dollar amount in the cost_ack card."""
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    a = await fa.create(
        mid, "cost_ack",
        title="Confirm spend $99.00 for step 13.1",
        why="irreversible step has cost",
        instructions=["Tap Confirm if you accept the cost."],
        blocking_step_id="13.1", cost_estimate_usd=99,
        notify_telegram=False,
    )
    from src.app.founder_action_render import render_action_card
    text, kb = render_action_card(a.to_dict())
    assert "99" in text  # cost number visible
    assert "Confirm" in text or "Confirm" in str(kb)
