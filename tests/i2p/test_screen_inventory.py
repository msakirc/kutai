"""Tests for verify_screen_inventory_shape (Z1 T3B / C18)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mr_roboto.verify_screen_inventory_shape import verify_screen_inventory_shape

FIXTURES = Path(__file__).parent / "reviewer_regression" / "fixtures" / "v1" / "5_10"


def _stage(tmp_path: Path, fixture_name: str) -> Path:
    workspace = tmp_path / "ws"
    (workspace / ".flow").mkdir(parents=True)
    shutil.copy(FIXTURES / fixture_name, workspace / ".flow" / "screen_inventory.md")
    return workspace


@pytest.mark.asyncio
async def test_good_inventory_passes(tmp_path):
    ws = _stage(tmp_path, "good_screen_inventory.md")
    res = await verify_screen_inventory_shape(
        mission_id=99,
        path=".flow/screen_inventory.md",
        workspace_path=str(ws),
    )
    assert res["ok"], res
    assert res["total"] == 8
    assert res["chunk_size"] == 4


@pytest.mark.asyncio
async def test_bad_inventory_chunks_math_mismatch_and_missing_route_fails(tmp_path):
    ws = _stage(tmp_path, "bad_screen_inventory.md")
    res = await verify_screen_inventory_shape(
        mission_id=99,
        path=".flow/screen_inventory.md",
        workspace_path=str(ws),
    )
    assert not res["ok"]
    joined = " | ".join(res["errors"])
    assert "sum-of-chunks" in joined
    assert "missing route" in joined


@pytest.mark.asyncio
async def test_chunk_size_violation_fails(tmp_path):
    ws = tmp_path / "ws"
    (ws / ".flow").mkdir(parents=True)
    (ws / ".flow" / "screen_inventory.md").write_text(
        '---\n_schema_version: "1"\nmission_id: 1\n'
        "total_screens: 5\nchunk_size: 4\n"
        'chunks: [["A","B","C","D","E"]]\n---\n'
        "# X\n\n## Mobile\n- A (`/a`)\n- B (`/b`)\n- C (`/c`)\n- D (`/d`)\n- E (`/e`)\n",
        encoding="utf-8",
    )
    res = await verify_screen_inventory_shape(
        mission_id=1, path=".flow/screen_inventory.md", workspace_path=str(ws),
    )
    assert not res["ok"]
    assert any("chunk_size=4" in e for e in res["errors"])


@pytest.mark.asyncio
async def test_missing_frontmatter_fails(tmp_path):
    ws = tmp_path / "ws"
    (ws / ".flow").mkdir(parents=True)
    (ws / ".flow" / "screen_inventory.md").write_text(
        "# Screen Inventory\n\n## Mobile\n- Welcome (`/w`)\n", encoding="utf-8",
    )
    res = await verify_screen_inventory_shape(
        mission_id=1, path=".flow/screen_inventory.md", workspace_path=str(ws),
    )
    assert not res["ok"]
