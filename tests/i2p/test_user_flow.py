"""Tests for verify_user_flow_shape (Z1 T3B / C4+A12)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mr_roboto.verify_user_flow_shape import verify_user_flow_shape

FIXTURES = Path(__file__).parent / "reviewer_regression" / "fixtures" / "v1" / "5_10"


def _stage(tmp_path: Path, fixture_name: str) -> Path:
    workspace = tmp_path / "ws"
    (workspace / ".flow").mkdir(parents=True)
    shutil.copy(FIXTURES / fixture_name, workspace / ".flow" / "user_flow.md")
    return workspace


@pytest.mark.asyncio
async def test_good_user_flow_passes(tmp_path):
    ws = _stage(tmp_path, "good_user_flow.md")
    res = await verify_user_flow_shape(
        mission_id=99,
        path=".flow/user_flow.md",
        surfaces=["mobile", "web", "admin"],
        workspace_path=str(ws),
    )
    assert res["ok"], res
    # 3 declared surfaces, but mobile has 2 mermaid blocks → 4 total.
    assert res["mermaid_count"] >= 3


@pytest.mark.asyncio
async def test_bad_user_flow_missing_per_surface_diagram_fails(tmp_path):
    ws = _stage(tmp_path, "bad_user_flow.md")
    res = await verify_user_flow_shape(
        mission_id=99,
        path=".flow/user_flow.md",
        surfaces=["mobile", "web", "admin"],
        workspace_path=str(ws),
    )
    assert not res["ok"]
    assert any("mermaid block count" in e for e in res["errors"])


@pytest.mark.asyncio
async def test_missing_file_fails(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    res = await verify_user_flow_shape(
        mission_id=99,
        path=".flow/user_flow.md",
        surfaces=["mobile"],
        workspace_path=str(ws),
    )
    assert not res["ok"]


@pytest.mark.asyncio
async def test_no_mermaid_blocks_fails(tmp_path):
    ws = tmp_path / "ws"
    (ws / ".flow").mkdir(parents=True)
    (ws / ".flow" / "user_flow.md").write_text(
        '---\n_schema_version: "1"\nmission_id: 1\nsurfaces: ["mobile"]\n---\n'
        "# User Flow\n\nNo mermaid here.\n",
        encoding="utf-8",
    )
    res = await verify_user_flow_shape(
        mission_id=1, path=".flow/user_flow.md", surfaces=["mobile"],
        workspace_path=str(ws),
    )
    assert not res["ok"]
    assert any("no ```mermaid" in e for e in res["errors"])
