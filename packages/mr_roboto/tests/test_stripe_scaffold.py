"""Z6 T5A — tests for mr_roboto.executors.stripe_scaffold."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from mr_roboto.executors.stripe_scaffold import (
    _detect_stack,
    _wants_stripe,
    run,
)


# ── unit: stack detection ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "arch, expected_stack",
    [
        (None, "python-fastapi"),
        ({}, "python-fastapi"),
        ({"components": [{"tech": "fastapi"}]}, "python-fastapi"),
        ({"components": [{"tech": "Django"}]}, "python-fastapi"),
        # TS-leaning blobs still default to python-fastapi today (v1).
        ({"components": [{"tech": "TypeScript Next.js"}]}, "python-fastapi"),
    ],
)
def test_detect_stack(arch, expected_stack):
    stack, ext = _detect_stack(arch)
    assert stack == expected_stack
    assert ext == "py"


# ── unit: wants_stripe ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "monet, expected",
    [
        (None, False),
        ({}, False),
        ({"billing": {"provider": "stripe"}}, True),
        ({"billing": {"provider": "Stripe"}}, True),
        ({"billing": {"provider": "paddle"}}, False),
        ({"products": [{"name": "Pro"}]}, True),
        ({"products": []}, False),
        ({"billing": {"provider": "paddle"}, "products": [{"name": "Pro"}]}, False),
    ],
)
def test_wants_stripe(monet, expected):
    assert _wants_stripe(monet) is expected


# ── integration-ish: run() with mocked artifacts ──────────────────────────


@pytest.mark.asyncio
async def test_run_writes_four_files_python_fastapi_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    monet = {"billing": {"provider": "stripe"}, "products": [{"name": "Pro"}]}
    arch = {"components": [{"tech": "fastapi"}]}

    async def _fake_load(mission_id, name):
        return {"monetization_strategy": monet, "system_architecture": arch}.get(name)

    with patch(
        "mr_roboto.executors.stripe_scaffold._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ):
        res = await run({"mission_id": 42})

    assert res["ok"] is True
    assert res.get("skipped") is not True
    assert res["stack"] == "python-fastapi"
    assert len(res["files_written"]) == 4

    base = tmp_path / "mission_42" / "api"
    assert (base / "checkout" / "create_session.py").is_file()
    assert (base / "webhook" / "stripe.py").is_file()
    assert (tmp_path / "mission_42" / ".env.example").is_file()
    assert (tmp_path / "mission_42" / "README_STRIPE.md").is_file()

    env_text = (tmp_path / "mission_42" / ".env.example").read_text(encoding="utf-8")
    for key in (
        "STRIPE_SECRET_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "STRIPE_PUBLISHABLE_KEY",
        "STRIPE_TAX_ORIGIN_COUNTRY",
    ):
        assert key in env_text

    webhook_text = (base / "webhook" / "stripe.py").read_text(encoding="utf-8")
    assert "Stripe-Signature" in webhook_text
    assert "construct_event" in webhook_text


@pytest.mark.asyncio
async def test_run_skipped_when_no_stripe(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))
    monet = {"billing": {"provider": "paddle"}}

    async def _fake_load(mission_id, name):
        return monet if name == "monetization_strategy" else None

    with patch(
        "mr_roboto.executors.stripe_scaffold._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ):
        res = await run({"mission_id": 7})

    assert res["ok"] is True
    assert res["skipped"] is True
    assert res["files_written"] == []
    # No mission_7 dir should be created.
    assert not (tmp_path / "mission_7").exists()


@pytest.mark.asyncio
async def test_run_skipped_when_no_billing_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    async def _fake_load(mission_id, name):
        return None

    with patch(
        "mr_roboto.executors.stripe_scaffold._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ):
        res = await run({"mission_id": 9})

    assert res["ok"] is True
    assert res["skipped"] is True
    assert res["reason"] == "no_stripe_billing"


@pytest.mark.asyncio
async def test_run_rejects_missing_mission_id():
    res = await run({})
    assert res["ok"] is False
    assert res["reason"] == "missing_mission_id"


@pytest.mark.asyncio
async def test_run_rejects_bad_mission_id():
    res = await run({"mission_id": "not-a-number"})
    assert res["ok"] is False
    assert res["reason"] == "invalid_mission_id"


@pytest.mark.asyncio
async def test_run_idempotent_overwrite(tmp_path, monkeypatch):
    """Re-running over an existing scaffold simply overwrites — no error."""
    monkeypatch.setenv("MISSION_WORKSPACE_ROOT", str(tmp_path))

    monet = {"billing": {"provider": "stripe"}}

    async def _fake_load(mission_id, name):
        return monet if name == "monetization_strategy" else None

    with patch(
        "mr_roboto.executors.stripe_scaffold._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ):
        res1 = await run({"mission_id": 5})
        res2 = await run({"mission_id": 5})

    assert res1["ok"] and res2["ok"]
    assert res1["files_written"] == res2["files_written"]
