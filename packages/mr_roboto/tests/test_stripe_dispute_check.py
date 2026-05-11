"""Z6 T5D — tests for mr_roboto.executors.stripe_dispute_check."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from mr_roboto.executors.stripe_dispute_check import (
    _checkpoint_path,
    _load_checkpoint,
    _save_checkpoint,
    run,
)


# ── unit: checkpoint I/O ──────────────────────────────────────────────────


def test_load_checkpoint_missing_returns_empty(tmp_path):
    path = str(tmp_path / "absent.json")
    assert _load_checkpoint(path) == {"seen_ids": []}


def test_save_then_load_roundtrip(tmp_path):
    path = str(tmp_path / "mission_1" / ".stripe" / "ck.json")
    _save_checkpoint(path, {"seen_ids": ["a", "b"]})
    loaded = _load_checkpoint(path)
    assert sorted(loaded["seen_ids"]) == ["a", "b"]


def test_checkpoint_path_layout(tmp_path):
    p = _checkpoint_path(str(tmp_path), 99)
    assert p.endswith("mission_99/.stripe/last_dispute_check.json".replace("/", _sep()))


def _sep():
    import os
    return os.sep


# ── integration: dispute scan paths ───────────────────────────────────────


@pytest.mark.asyncio
async def test_no_disputes_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    async def _fake_vc(task, service, action, params):
        return {"ok": True, "result": {"data": []}}

    fa_calls: list = []

    async def _fake_emit(*args, **kwargs):
        fa_calls.append((args, kwargs))

    with patch(
        "mr_roboto.executors.stripe_dispute_check._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ), patch(
        "mr_roboto.executors.stripe_dispute_check._emit_legal_counsel",
        new=AsyncMock(side_effect=_fake_emit),
    ):
        res = await run({"mission_id": 3})

    assert res["ok"]
    assert res["new_disputes"] == 0
    assert not fa_calls


@pytest.mark.asyncio
async def test_new_dispute_emits_legal_counsel(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    async def _fake_vc(task, service, action, params):
        return {
            "ok": True,
            "result": {
                "data": [
                    {
                        "id": "dp_1",
                        "amount": 5000,
                        "currency": "usd",
                        "reason": "fraudulent",
                    }
                ]
            },
        }

    fa_calls: list = []

    async def _fake_emit(mission_id, dispute_id, amount, currency, detail):
        fa_calls.append((mission_id, dispute_id, amount, currency, detail))

    with patch(
        "mr_roboto.executors.stripe_dispute_check._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ), patch(
        "mr_roboto.executors.stripe_dispute_check._emit_legal_counsel",
        new=AsyncMock(side_effect=_fake_emit),
    ):
        res = await run({"mission_id": 3})

    assert res["ok"]
    assert res["new_disputes"] == 1
    assert len(fa_calls) == 1
    assert fa_calls[0][1] == "dp_1"
    # Checkpoint recorded.
    ckpt_path = tmp_path / "mission_3" / ".stripe" / "last_dispute_check.json"
    assert ckpt_path.is_file()
    data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert "dp_1" in data["seen_ids"]


@pytest.mark.asyncio
async def test_existing_dispute_does_not_re_emit(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    # Pre-seed checkpoint.
    ckpt_dir = tmp_path / "mission_3" / ".stripe"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "last_dispute_check.json").write_text(
        json.dumps({"seen_ids": ["dp_1"]}), encoding="utf-8",
    )

    async def _fake_vc(task, service, action, params):
        return {
            "ok": True,
            "result": {
                "data": [{"id": "dp_1", "amount": 5000, "currency": "usd"}]
            },
        }

    fa_calls: list = []

    async def _fake_emit(*args, **kwargs):
        fa_calls.append((args, kwargs))

    with patch(
        "mr_roboto.executors.stripe_dispute_check._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ), patch(
        "mr_roboto.executors.stripe_dispute_check._emit_legal_counsel",
        new=AsyncMock(side_effect=_fake_emit),
    ):
        res = await run({"mission_id": 3})

    assert res["ok"]
    assert res["new_disputes"] == 0
    assert not fa_calls


@pytest.mark.asyncio
async def test_list_disputes_failure_bubbles_up(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    async def _fake_vc(task, service, action, params):
        return {"ok": False, "reason": "vendor_error", "error": "401"}

    with patch(
        "mr_roboto.executors.stripe_dispute_check._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ):
        res = await run({"mission_id": 3})

    assert res["ok"] is False
    assert res["reason"] == "list_disputes_failed"


@pytest.mark.asyncio
async def test_no_mission_id_uses_system_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    async def _fake_vc(task, service, action, params):
        return {"ok": True, "result": {"data": []}}

    with patch(
        "mr_roboto.executors.stripe_dispute_check._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ):
        res = await run({})  # no mission_id

    assert res["ok"] is True
    # mission_0 checkpoint dir should exist.
    assert (tmp_path / "mission_0" / ".stripe").is_dir()
