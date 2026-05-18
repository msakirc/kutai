"""Z6 T1C — admission gate scenarios."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "z6_adm.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    import src.founder_actions as fa
    fa._reset_lifecycle_cache()
    return db_path, db_mod, fa


@pytest.mark.asyncio
async def test_admits_when_not_needs_real_tools(tmp_path, monkeypatch):
    _, db_mod, _ = await _setup(tmp_path, monkeypatch)
    from general_beckman.z6_admission import check_z6_admission
    mid = await db_mod.add_mission("m", "")
    task = {"id": 1, "mission_id": mid, "context": "{}", "needs_real_tools": 0}
    res = await check_z6_admission(task, mid)
    assert res.admit is True


@pytest.mark.asyncio
async def test_blocks_when_real_tool_kind_missing(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    from general_beckman.z6_admission import check_z6_admission
    mid = await db_mod.add_mission("m", "")
    task = {
        "id": 1, "mission_id": mid, "needs_real_tools": 1,
        "context": '{"workflow_step_id": "13.1"}',
    }
    res = await check_z6_admission(task, mid)
    assert res.admit is False
    assert "real_tool_kind missing" in res.reason
    actions = await fa.list_by_mission(mid)
    assert len(actions) == 1
    assert actions[0].kind == "generic"


@pytest.mark.asyncio
async def test_blocks_when_no_adapter(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    from general_beckman.z6_admission import check_z6_admission
    # Stub registry to return None for everything.
    monkeypatch.setattr(
        "general_beckman.z6_admission._resolve_adapter",
        lambda kinds: None,
    )
    mid = await db_mod.add_mission("m", "")
    task = {
        "id": 1, "mission_id": mid, "needs_real_tools": 1,
        "context": (
            '{"workflow_step_id": "13.1", '
            '"real_tool_kind": "fake_vendor"}'
        ),
    }
    res = await check_z6_admission(task, mid)
    assert res.admit is False
    actions = await fa.list_by_mission(mid)
    assert actions[0].kind == "vendor_enroll"


@pytest.mark.asyncio
async def test_blocks_when_credentials_missing(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    from general_beckman.z6_admission import check_z6_admission
    # Pretend an adapter for 'vercel' exists but credentials don't.
    monkeypatch.setattr(
        "general_beckman.z6_admission._resolve_adapter",
        lambda kinds: "vercel" if "vercel" in kinds else None,
    )
    async def _no_cred(_svc):
        return None
    monkeypatch.setattr(
        "src.security.credential_store.get_credential", _no_cred,
    )
    mid = await db_mod.add_mission("m", "")
    task = {
        "id": 1, "mission_id": mid, "needs_real_tools": 1,
        "context": (
            '{"workflow_step_id": "13.1", '
            '"real_tool_kind": "vercel"}'
        ),
    }
    res = await check_z6_admission(task, mid)
    assert res.admit is False
    assert "no credential" in res.reason
    actions = await fa.list_by_mission(mid)
    assert actions[0].kind == "credential_paste"


@pytest.mark.asyncio
async def test_blocks_when_cost_ack_missing(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    from general_beckman.z6_admission import check_z6_admission
    monkeypatch.setattr(
        "general_beckman.z6_admission._resolve_adapter",
        lambda kinds: "vercel",
    )
    async def _has_cred(_svc):
        return {"token": "xxx"}
    monkeypatch.setattr(
        "src.security.credential_store.get_credential", _has_cred,
    )
    mid = await db_mod.add_mission("m", "")
    task = {
        "id": 1, "mission_id": mid, "needs_real_tools": 1,
        "reversibility": "irreversible",
        "context": (
            '{"workflow_step_id": "13.1", '
            '"real_tool_kind": "vercel", "cost_estimate_usd": 50}'
        ),
    }
    res = await check_z6_admission(task, mid)
    assert res.admit is False
    assert "cost_ack" in res.reason
    actions = await fa.list_by_mission(mid)
    assert actions[0].kind == "cost_ack"
    assert actions[0].cost_estimate_usd == 50.0


@pytest.mark.asyncio
async def test_admits_when_all_prereqs_satisfied(tmp_path, monkeypatch):
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    from general_beckman.z6_admission import check_z6_admission
    monkeypatch.setattr(
        "general_beckman.z6_admission._resolve_adapter",
        lambda kinds: "vercel",
    )
    async def _has_cred(_svc):
        return {"token": "xxx"}
    monkeypatch.setattr(
        "src.security.credential_store.get_credential", _has_cred,
    )
    mid = await db_mod.add_mission("m", "")
    # Pre-resolve cost_ack for this mission+step.
    a = await fa.create(
        mid, "cost_ack", "ack", "why", [],
        blocking_step_id="13.1", cost_estimate_usd=50,
        notify_telegram=False,
    )
    await fa.resolve(a.id)
    task = {
        "id": 1, "mission_id": mid, "needs_real_tools": 1,
        "reversibility": "irreversible",
        "context": (
            '{"workflow_step_id": "13.1", '
            '"real_tool_kind": "vercel", "cost_estimate_usd": 50}'
        ),
    }
    res = await check_z6_admission(task, mid)
    assert res.admit is True


@pytest.mark.asyncio
async def test_admits_partial_reversibility_skips_cost_ack(
    tmp_path, monkeypatch,
):
    """Cost-ack only required for irreversible+cost>0 combo."""
    _, db_mod, _ = await _setup(tmp_path, monkeypatch)
    from general_beckman.z6_admission import check_z6_admission
    monkeypatch.setattr(
        "general_beckman.z6_admission._resolve_adapter",
        lambda kinds: "vercel",
    )
    async def _has_cred(_svc):
        return {"token": "xxx"}
    monkeypatch.setattr(
        "src.security.credential_store.get_credential", _has_cred,
    )
    mid = await db_mod.add_mission("m", "")
    task = {
        "id": 1, "mission_id": mid, "needs_real_tools": 1,
        "reversibility": "partial",
        "context": (
            '{"workflow_step_id": "13.1", '
            '"real_tool_kind": "vercel", "cost_estimate_usd": 50}'
        ),
    }
    res = await check_z6_admission(task, mid)
    assert res.admit is True


@pytest.mark.asyncio
async def test_pipe_separated_kinds_resolve_first_match(tmp_path, monkeypatch):
    _, db_mod, _ = await _setup(tmp_path, monkeypatch)
    from general_beckman.z6_admission import check_z6_admission

    def _resolve(kinds):
        # Only railway adapter exists.
        return "railway" if "railway" in kinds else None

    monkeypatch.setattr(
        "general_beckman.z6_admission._resolve_adapter", _resolve,
    )
    async def _has_cred(_svc):
        return {"token": "xxx"} if _svc == "railway" else None
    monkeypatch.setattr(
        "src.security.credential_store.get_credential", _has_cred,
    )
    mid = await db_mod.add_mission("m", "")
    task = {
        "id": 1, "mission_id": mid, "needs_real_tools": 1,
        "context": (
            '{"workflow_step_id": "13.1", '
            '"real_tool_kind": "vercel|railway|supabase"}'
        ),
    }
    res = await check_z6_admission(task, mid)
    assert res.admit is True


@pytest.mark.asyncio
async def test_dedupes_pending_actions(tmp_path, monkeypatch):
    """Same missing prereq twice in a row → only one founder_action."""
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    from general_beckman.z6_admission import check_z6_admission
    mid = await db_mod.add_mission("m", "")
    task = {
        "id": 1, "mission_id": mid, "needs_real_tools": 1,
        "context": '{"workflow_step_id": "13.1"}',
    }
    await check_z6_admission(task, mid)
    await check_z6_admission(task, mid)
    actions = await fa.list_by_mission(mid)
    assert len(actions) == 1
