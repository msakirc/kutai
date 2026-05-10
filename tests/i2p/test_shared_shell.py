"""Tests for verify_shared_shell_shape (Z1 T3B / C18)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mr_roboto.verify_shared_shell_shape import verify_shared_shell_shape

FIXTURES = Path(__file__).parent / "reviewer_regression" / "fixtures" / "v1" / "5_10"


def _stage(tmp_path: Path, fixture_name: str) -> Path:
    workspace = tmp_path / "ws"
    (workspace / ".flow").mkdir(parents=True)
    shutil.copy(FIXTURES / fixture_name, workspace / ".flow" / "shared_shell.md")
    return workspace


@pytest.mark.asyncio
async def test_good_shared_shell_passes(tmp_path):
    ws = _stage(tmp_path, "good_shared_shell.md")
    res = await verify_shared_shell_shape(
        mission_id=99, path=".flow/shared_shell.md", workspace_path=str(ws),
    )
    assert res["ok"], res
    for required in ("header", "empty_state", "error_state", "loading_state"):
        assert required in res["shells"], res


@pytest.mark.asyncio
async def test_bad_shared_shell_missing_required_fails(tmp_path):
    ws = _stage(tmp_path, "bad_shared_shell.md")
    res = await verify_shared_shell_shape(
        mission_id=99, path=".flow/shared_shell.md", workspace_path=str(ws),
    )
    assert not res["ok"]
    assert any("missing required shell sections" in e for e in res["errors"])


@pytest.mark.asyncio
async def test_missing_file_fails(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    res = await verify_shared_shell_shape(
        mission_id=99, path=".flow/shared_shell.md", workspace_path=str(ws),
    )
    assert not res["ok"]


@pytest.mark.asyncio
async def test_alias_singular_headings_accepted(tmp_path):
    ws = tmp_path / "ws"
    (ws / ".flow").mkdir(parents=True)
    (ws / ".flow" / "shared_shell.md").write_text(
        '---\n_schema_version: "1"\nmission_id: 1\n---\n'
        "## Header\nx\n## Empty\nx\n## Error\nx\n## Loading\nx\n",
        encoding="utf-8",
    )
    res = await verify_shared_shell_shape(
        mission_id=1, path=".flow/shared_shell.md", workspace_path=str(ws),
    )
    assert res["ok"], res
