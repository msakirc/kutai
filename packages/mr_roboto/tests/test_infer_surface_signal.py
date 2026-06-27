"""Stage 3 — deterministic surface_signal inference (i2p 3.5z).

Founder-words-primacy (2026-06-27): the signal is inferred from the founder's
OWN words (mission title+description) FIRST. LLM-generated enrichment (idea
brief / charter / PRD prose) — which can carry strong platform words like
"dashboard"/"SaaS" — is consulted ONLY when the founder named no surface, so it
can never outrank the founder's own "app". This grounds target_platform (and the
whole build rail that derives from it) in the founder's intent.
"""
import json
import os

import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_founder_words_beat_enrichment_prose(tmp_path, monkeypatch):
    """THE build-rail bug fix: founder said "app" (→ mobile) but the LLM
    charter/PRD prose says "web dashboard SaaS" (→ strong web). The founder's
    words MUST win → target_platform=mobile, not web."""
    import mr_roboto.infer_surface_signal as M
    monkeypatch.setattr(M, "_gather_founder_text",
                        AsyncMock(return_value="I want an app for tracking habits"))
    monkeypatch.setattr(M, "_gather_enrichment_text",
                        AsyncMock(return_value="A web dashboard SaaS platform with a website"))

    res = await M.infer_surface_signal({"mission_id": 5},
                                       workspace_path=str(tmp_path))

    assert res["surfaces"] == ["mobile"]
    assert res["target_platform"] == "mobile"
    assert res["source"] == "founder_words"


@pytest.mark.asyncio
async def test_high_confidence_signal_persisted(tmp_path, monkeypatch):
    import mr_roboto.infer_surface_signal as M
    monkeypatch.setattr(M, "_gather_founder_text",
                        AsyncMock(return_value="Build an iOS app for habits"))
    monkeypatch.setattr(M, "_gather_enrichment_text", AsyncMock(return_value=""))

    res = await M.infer_surface_signal({"mission_id": 5},
                                       workspace_path=str(tmp_path))

    assert res["status"] == "completed"
    assert res["surfaces"] == ["mobile"]
    assert res["target_platform"] == "mobile"
    assert res["confidence"] == "high"
    assert res["source"] == "founder_words"

    on_disk = os.path.join(str(tmp_path), ".charter", "surface_signal.json")
    assert os.path.isfile(on_disk)
    with open(on_disk, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["target_platform"] == "mobile"
    assert saved["mission_id"] == 5


@pytest.mark.asyncio
async def test_both_signal(tmp_path, monkeypatch):
    import mr_roboto.infer_surface_signal as M
    monkeypatch.setattr(M, "_gather_founder_text",
                        AsyncMock(return_value="a mobile app plus a web application"))
    monkeypatch.setattr(M, "_gather_enrichment_text", AsyncMock(return_value=""))

    res = await M.infer_surface_signal({"mission_id": 9},
                                       workspace_path=str(tmp_path))
    assert res["surfaces"] == ["mobile", "web"]
    assert res["target_platform"] == "both"


@pytest.mark.asyncio
async def test_founder_silent_falls_back_to_enrichment(tmp_path, monkeypatch):
    """Founder named no surface → fall back to the enrichment prose for a
    best-effort guess (the pre-2026-06-27 behavior), marked enrichment_fallback."""
    import mr_roboto.infer_surface_signal as M
    monkeypatch.setattr(M, "_gather_founder_text",
                        AsyncMock(return_value="a product to help people save money"))
    monkeypatch.setattr(M, "_gather_enrichment_text",
                        AsyncMock(return_value="an iOS app built with React Native"))

    res = await M.infer_surface_signal({"mission_id": 3},
                                       workspace_path=str(tmp_path))
    assert res["surfaces"] == ["mobile"]
    assert res["target_platform"] == "mobile"
    assert res["source"] == "enrichment_fallback"


@pytest.mark.asyncio
async def test_low_everywhere_writes_null_target(tmp_path, monkeypatch):
    """No signal in founder words OR enrichment → still write, target null, so
    3.6 sees the absence explicitly and derives from the PRD itself."""
    import mr_roboto.infer_surface_signal as M
    monkeypatch.setattr(M, "_gather_founder_text",
                        AsyncMock(return_value="a product to help people save money"))
    monkeypatch.setattr(M, "_gather_enrichment_text",
                        AsyncMock(return_value="helps users be more productive"))

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
    monkeypatch.setattr(M, "_gather_founder_text",
                        AsyncMock(return_value="a web sitesi for booking"))
    monkeypatch.setattr(M, "_gather_enrichment_text", AsyncMock(return_value=""))

    action = await mr_roboto.run({
        "mission_id": 7,
        "payload": {
            "action": "infer_surface_signal",
            "workspace_path": str(tmp_path),
        },
    })
    assert action.status == "completed", action
    assert action.result["target_platform"] == "web"
