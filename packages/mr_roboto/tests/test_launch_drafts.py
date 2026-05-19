import pytest
from unittest.mock import patch, AsyncMock
import mr_roboto.launch_drafts as ld


@pytest.mark.asyncio
async def test_launch_draft_falls_back_to_founder_voice():
    captured = {}

    async def fake_enqueue(spec, **kw):
        captured["desc"] = spec["description"]
        return {"task_id": 1}

    with patch.object(ld, "_enqueue", fake_enqueue), \
         patch.object(ld, "fetch_launch_lessons", AsyncMock(return_value=[])), \
         patch.object(ld, "load_founder_voice", return_value="I write plainly."):
        res = await ld.run("twitter", {"product_id": "p1", "launch_id": 7, "spec": "x"})

    assert res["status"] == "enqueued"
    assert "I write plainly." in captured["desc"]


@pytest.mark.asyncio
async def test_explicit_brand_voice_wins_over_founder_voice():
    captured = {}

    async def fake_enqueue(spec, **kw):
        captured["desc"] = spec["description"]
        return {"task_id": 1}

    with patch.object(ld, "_enqueue", fake_enqueue), \
         patch.object(ld, "fetch_launch_lessons", AsyncMock(return_value=[])), \
         patch.object(ld, "load_founder_voice", return_value="FALLBACK"):
        res = await ld.run("twitter", {"product_id": "p1", "launch_id": 7,
                                       "spec": "x", "brand_voice": "EXPLICIT"})

    assert "EXPLICIT" in captured["desc"]
    assert "FALLBACK" not in captured["desc"]
