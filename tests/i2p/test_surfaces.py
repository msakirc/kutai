"""Tests for verify_surfaces_shape (Z1 T3B / C12)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mr_roboto.verify_surfaces_shape import verify_surfaces_shape

FIXTURES = Path(__file__).parent / "reviewer_regression" / "fixtures" / "v1" / "5_10"


def _stage(tmp_path: Path, fixture_name: str) -> Path:
    workspace = tmp_path / "ws"
    (workspace / ".charter").mkdir(parents=True)
    shutil.copy(FIXTURES / fixture_name, workspace / ".charter" / "surfaces.json")
    return workspace


@pytest.mark.asyncio
async def test_good_surfaces_passes(tmp_path):
    ws = _stage(tmp_path, "good_surfaces.json")
    res = await verify_surfaces_shape(
        mission_id=99,
        path=".charter/surfaces.json",
        workspace_path=str(ws),
    )
    assert res["ok"], res
    assert res["surfaces"] == ["mobile", "web", "admin"]
    assert res["primary"] == "mobile"


@pytest.mark.asyncio
async def test_bad_surfaces_empty_array_fails(tmp_path):
    ws = _stage(tmp_path, "bad_surfaces.json")
    res = await verify_surfaces_shape(
        mission_id=99,
        path=".charter/surfaces.json",
        workspace_path=str(ws),
    )
    assert not res["ok"]
    joined = " | ".join(res["errors"])
    assert "non-empty" in joined or "not in surfaces" in joined


@pytest.mark.asyncio
async def test_missing_file_fails(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    res = await verify_surfaces_shape(
        mission_id=99,
        path=".charter/surfaces.json",
        workspace_path=str(ws),
    )
    assert not res["ok"]
    assert any("missing" in e for e in res["errors"])


@pytest.mark.asyncio
async def test_invalid_surface_token_fails(tmp_path):
    ws = tmp_path / "ws"
    (ws / ".charter").mkdir(parents=True)
    (ws / ".charter" / "surfaces.json").write_text(
        '{"_schema_version":"1","mission_id":1,"surfaces":["mobile","watch"],'
        '"primary_surface":"mobile","founder_confirmed_at":"x"}',
        encoding="utf-8",
    )
    res = await verify_surfaces_shape(
        mission_id=1,
        path=".charter/surfaces.json",
        workspace_path=str(ws),
    )
    assert not res["ok"]
    assert any("invalid surface token" in e for e in res["errors"])


@pytest.mark.asyncio
async def test_primary_not_in_surfaces_fails(tmp_path):
    ws = tmp_path / "ws"
    (ws / ".charter").mkdir(parents=True)
    (ws / ".charter" / "surfaces.json").write_text(
        '{"_schema_version":"1","mission_id":1,"surfaces":["mobile"],'
        '"primary_surface":"web","founder_confirmed_at":"x"}',
        encoding="utf-8",
    )
    res = await verify_surfaces_shape(
        mission_id=1,
        path=".charter/surfaces.json",
        workspace_path=str(ws),
    )
    assert not res["ok"]
    assert any("not in surfaces" in e for e in res["errors"])
