"""Tests for `regen_artifact` mechanical action (Z1 Tier 4A — C11+A15).

The regen primitive re-emits a single existing artifact against a founder
change description, preserving the previous version as `{path}.v{N}.md` and
recording a row in the `regen_log` table.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

import mr_roboto


@pytest.mark.asyncio
async def test_regen_artifact_writes_new_version_and_preserves_old(tmp_path):
    """Happy path: existing v1 artifact + change description → v2 alongside."""
    mission_root = tmp_path / "mission_42"
    mission_root.mkdir()
    artifact = mission_root / "charter.md"
    artifact.write_text("# Charter v1\n\nfact-primary content app.\n", encoding="utf-8")

    task = {
        "id": 100,
        "mission_id": 42,
        "payload": {
            "action": "regen_artifact",
            "artifact_path": "mission_42/charter.md",
            "change_description": "make it more clinical",
            "workspace_path": str(tmp_path),
        },
    }

    # Mock coulson.execute to deterministically return a new body.
    new_body = "# Charter v2 (clinical)\n\nclinical content app.\n"
    with patch("mr_roboto.regen._invoke_emitter", new_callable=AsyncMock) as mock_emit:
        mock_emit.return_value = {"text": new_body}
        action = await mr_roboto.run(task)

    assert action.status == "completed", action.error
    res = action.result
    assert res["new_version"].endswith(".v2.md")
    assert res["prev_version"].endswith(".v1.md")

    # Original artifact path now contains the new body.
    assert artifact.read_text(encoding="utf-8") == new_body
    # Old version preserved at .v1.md sibling.
    v1_path = mission_root / "charter.v1.md"
    assert v1_path.exists()
    assert v1_path.read_text(encoding="utf-8").startswith("# Charter v1")
    # New version snapshot also at .v2.md
    v2_path = mission_root / "charter.v2.md"
    assert v2_path.exists()
    assert v2_path.read_text(encoding="utf-8") == new_body


@pytest.mark.asyncio
async def test_regen_artifact_increments_version_on_repeat(tmp_path):
    """Second regen produces v3 (v1 + v2 already on disk)."""
    mission_root = tmp_path / "mission_7"
    mission_root.mkdir()
    art = mission_root / "prd.md"
    art.write_text("body v2", encoding="utf-8")
    (mission_root / "prd.v1.md").write_text("body v1", encoding="utf-8")
    (mission_root / "prd.v2.md").write_text("body v2", encoding="utf-8")

    task = {
        "id": 1,
        "mission_id": 7,
        "payload": {
            "action": "regen_artifact",
            "artifact_path": "mission_7/prd.md",
            "change_description": "tighter scope",
            "workspace_path": str(tmp_path),
        },
    }
    with patch("mr_roboto.regen._invoke_emitter", new_callable=AsyncMock) as mock_emit:
        mock_emit.return_value = {"text": "body v3"}
        action = await mr_roboto.run(task)

    assert action.status == "completed"
    assert action.result["new_version"].endswith(".v3.md")
    assert (mission_root / "prd.v3.md").exists()
    assert art.read_text(encoding="utf-8") == "body v3"


@pytest.mark.asyncio
async def test_regen_artifact_missing_file_fails(tmp_path):
    task = {
        "id": 2,
        "mission_id": 5,
        "payload": {
            "action": "regen_artifact",
            "artifact_path": "mission_5/does_not_exist.md",
            "change_description": "x",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "failed"
    assert "not found" in (action.error or "").lower()


@pytest.mark.asyncio
async def test_regen_artifact_records_log_row(tmp_path, monkeypatch):
    """`regen_log` row written with prev/new version + change description."""
    mission_root = tmp_path / "mission_3"
    mission_root.mkdir()
    art = mission_root / "design_tokens.json"
    art.write_text('{"v": 1}', encoding="utf-8")

    captured = {}

    async def fake_log(**kwargs):
        captured.update(kwargs)
        return 999

    task = {
        "id": 11,
        "mission_id": 3,
        "payload": {
            "action": "regen_artifact",
            "artifact_path": "mission_3/design_tokens.json",
            "change_description": "darker palette",
            "workspace_path": str(tmp_path),
        },
    }

    with patch("mr_roboto.regen._invoke_emitter", new_callable=AsyncMock) as mock_emit, \
         patch("mr_roboto.regen._record_regen_log", side_effect=fake_log):
        mock_emit.return_value = {"text": '{"v": 2, "tone": "dark"}'}
        action = await mr_roboto.run(task)

    assert action.status == "completed"
    assert captured["mission_id"] == 3
    assert captured["change_description"] == "darker palette"
    assert captured["scope"] == "artifact"
    assert captured["prev_version"].endswith(".v1.md") or captured["prev_version"].endswith(".v1.json")
    assert captured["new_version"].endswith(".v2.md") or captured["new_version"].endswith(".v2.json")


@pytest.mark.asyncio
async def test_regen_artifact_emitter_failure_does_not_overwrite(tmp_path):
    """Emitter ok=False ⇒ canonical path untouched + action fails (no silent same-body write)."""
    mission_root = tmp_path / "mission_9"
    mission_root.mkdir()
    art = mission_root / "charter.md"
    original = "# Charter v1\n\noriginal body\n"
    art.write_text(original, encoding="utf-8")

    task = {
        "id": 1,
        "mission_id": 9,
        "payload": {
            "action": "regen_artifact",
            "artifact_path": "mission_9/charter.md",
            "change_description": "tighten",
            "workspace_path": str(tmp_path),
        },
    }
    with patch("mr_roboto.regen._invoke_emitter", new_callable=AsyncMock) as mock_emit:
        mock_emit.return_value = {"ok": False, "text": None, "error": "coulson unavailable"}
        action = await mr_roboto.run(task)

    assert action.status == "failed"
    assert "coulson unavailable" in (action.error or "")
    assert art.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_regen_artifact_empty_emitter_text_fails(tmp_path):
    """Empty/whitespace text from emitter ⇒ failure, not silent no-op write."""
    mission_root = tmp_path / "mission_10"
    mission_root.mkdir()
    art = mission_root / "prd.md"
    art.write_text("body", encoding="utf-8")

    task = {
        "id": 1,
        "mission_id": 10,
        "payload": {
            "action": "regen_artifact",
            "artifact_path": "mission_10/prd.md",
            "change_description": "x",
            "workspace_path": str(tmp_path),
        },
    }
    with patch("mr_roboto.regen._invoke_emitter", new_callable=AsyncMock) as mock_emit:
        mock_emit.return_value = {"ok": True, "text": "   ", "error": None}
        action = await mr_roboto.run(task)

    assert action.status == "failed"
    assert "empty" in (action.error or "").lower()
    assert art.read_text(encoding="utf-8") == "body"


@pytest.mark.asyncio
async def test_regen_artifact_missing_change_description_fails(tmp_path):
    task = {
        "id": 1,
        "mission_id": 1,
        "payload": {
            "action": "regen_artifact",
            "artifact_path": "mission_1/x.md",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "failed"
    assert "change_description" in (action.error or "")
