"""Tests for z0_preflight_write — Z0 minimal slice."""
from __future__ import annotations

import json
import os

import pytest

from mr_roboto.z0_preflight import (
    VALID_TIERS,
    default_attention_minutes,
    z0_preflight_write,
)


def test_default_attention_by_tier():
    assert default_attention_minutes("prototype") == 120
    assert default_attention_minutes("private_beta") == 240
    assert default_attention_minutes("public_launch") == 480
    assert default_attention_minutes("revenue_product") is None
    # Unknown → private_beta default.
    assert default_attention_minutes("nonsense") == 240
    assert default_attention_minutes(None) == 240


def test_valid_tiers_exposed():
    assert "prototype" in VALID_TIERS
    assert "revenue_product" in VALID_TIERS
    assert len(VALID_TIERS) == 4


@pytest.mark.asyncio
async def test_preflight_write_creates_json(tmp_path):
    res = await z0_preflight_write(
        mission_id=42,
        ambition_tier="private_beta",
        cost_ceiling_usd=100.0,
        jurisdictions=["TR"],
        user_classes=["b2b"],
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    out = res["preflight_path"]
    assert os.path.isfile(out)
    body = json.loads(open(out, encoding="utf-8").read())
    assert body["ambition_tier"] == "private_beta"
    assert body["cost_ceiling_usd"] == 100.0
    # Tier set but no explicit attention → spec'd default applied.
    assert body["attention_budget_minutes"] == 240
    assert body["jurisdictions"] == ["TR"]
    assert body["user_classes"] == ["b2b"]


@pytest.mark.asyncio
async def test_preflight_explicit_attention_overrides_default(tmp_path):
    res = await z0_preflight_write(
        mission_id=1,
        ambition_tier="prototype",
        attention_budget_minutes=999,
        workspace_path=str(tmp_path),
    )
    body = json.loads(open(res["preflight_path"], encoding="utf-8").read())
    assert body["attention_budget_minutes"] == 999  # explicit beats tier default


@pytest.mark.asyncio
async def test_preflight_invalid_tier_fails(tmp_path):
    res = await z0_preflight_write(
        mission_id=1,
        ambition_tier="dragon_mode",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert "ambition_tier" in res["error"]


@pytest.mark.asyncio
async def test_preflight_no_tier_no_default_applied(tmp_path):
    res = await z0_preflight_write(
        mission_id=1,
        cost_ceiling_usd=50.0,
        workspace_path=str(tmp_path),
    )
    body = json.loads(open(res["preflight_path"], encoding="utf-8").read())
    assert body["ambition_tier"] is None
    assert body["attention_budget_minutes"] is None  # no tier → no default


@pytest.mark.asyncio
async def test_preflight_idempotent_rewrite(tmp_path):
    """Second write overwrites with new payload."""
    r1 = await z0_preflight_write(
        mission_id=1,
        ambition_tier="prototype",
        workspace_path=str(tmp_path),
    )
    r2 = await z0_preflight_write(
        mission_id=1,
        ambition_tier="public_launch",
        workspace_path=str(tmp_path),
    )
    assert r1["preflight_path"] == r2["preflight_path"]
    body = json.loads(open(r2["preflight_path"], encoding="utf-8").read())
    assert body["ambition_tier"] == "public_launch"
    assert body["attention_budget_minutes"] == 480


@pytest.mark.asyncio
async def test_dispatch_through_run(tmp_path):
    from mr_roboto import run
    task = {
        "id": 1,
        "mission_id": 7,
        "payload": {
            "action": "z0_preflight_write",
            "ambition_tier": "prototype",
            "workspace_path": str(tmp_path),
        },
    }
    action = await run(task)
    assert action.status == "completed"
    assert action.result["payload"]["ambition_tier"] == "prototype"
