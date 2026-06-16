"""Stage 3 — deterministic surface_signal inference (i2p 3.5z).

infer_surface_signal reads the founder's words, infers the surface set, projects
it onto target_platform, and persists .charter/surface_signal.json so step 3.6
honors the founder's stated surface instead of re-deriving from PRD prose.
"""
import json
import os

import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_high_confidence_signal_persisted(tmp_path, monkeypatch):
    import mr_roboto.infer_surface_signal as M
    monkeypatch.setattr(M, "_gather_signal_text",
                        AsyncMock(return_value="Build an iOS app for habits"))

    res = await M.infer_surface_signal({"mission_id": 5},
                                       workspace_path=str(tmp_path))

    assert res["status"] == "completed"
    assert res["surfaces"] == ["mobile"]
    assert res["target_platform"] == "mobile"
    assert res["confidence"] == "high"
    assert res["source"] == "intake_inference"

    on_disk = os.path.join(str(tmp_path), ".charter", "surface_signal.json")
    assert os.path.isfile(on_disk)
    with open(on_disk, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["target_platform"] == "mobile"
    assert saved["mission_id"] == 5


@pytest.mark.asyncio
async def test_both_signal(tmp_path, monkeypatch):
    import mr_roboto.infer_surface_signal as M
    monkeypatch.setattr(M, "_gather_signal_text",
                        AsyncMock(return_value="a mobile app plus a web application"))

    res = await M.infer_surface_signal({"mission_id": 9},
                                       workspace_path=str(tmp_path))
    assert res["surfaces"] == ["mobile", "web"]
    assert res["target_platform"] == "both"


@pytest.mark.asyncio
async def test_low_confidence_writes_null_target(tmp_path, monkeypatch):
    """No surface signal → still write, target_platform null, so 3.6 sees the
    absence explicitly and derives from the PRD itself."""
    import mr_roboto.infer_surface_signal as M
    monkeypatch.setattr(M, "_gather_signal_text",
                        AsyncMock(return_value="a product to help people save money"))

    res = await M.infer_surface_signal({"mission_id": 3},
                                       workspace_path=str(tmp_path))
    assert res["surfaces"] == []
    assert res["target_platform"] is None
    assert res["confidence"] == "low"

    on_disk = os.path.join(str(tmp_path), ".charter", "surface_signal.json")
    with open(on_disk, encoding="utf-8") as f:
        assert json.load(f)["target_platform"] is None


@pytest.mark.asyncio
async def test_dispatch_via_mr_roboto_run(tmp_path, monkeypatch):
    """The mechanical dispatcher routes action=infer_surface_signal → completed."""
    import mr_roboto
    import mr_roboto.infer_surface_signal as M
    monkeypatch.setattr(M, "_gather_signal_text",
                        AsyncMock(return_value="a web sitesi for booking"))

    action = await mr_roboto.run({
        "mission_id": 7,
        "payload": {
            "action": "infer_surface_signal",
            "workspace_path": str(tmp_path),
        },
    })
    assert action.status == "completed", action
    assert action.result["target_platform"] == "web"
