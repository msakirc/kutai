import pytest
from unittest.mock import patch
import mr_roboto.outreach_draft as od


@pytest.mark.asyncio
async def test_outreach_draft_includes_founder_voice():
    captured = {}

    async def fake_enqueue(spec, **kw):
        captured["desc"] = spec["description"]
        return {"task_id": 1}

    with patch.object(od, "enqueue", fake_enqueue), \
         patch.object(od, "load_founder_voice", return_value="Dry, direct, no fluff."):
        res = await od.run_outreach_draft(
            product_id="p1", mission_id=3,
            prospect_data={"name": "Sam"}, template_id="cold", list_id="L1",
        )

    assert res["status"] == "enqueued"
    assert "Dry, direct, no fluff." in captured["desc"]


@pytest.mark.asyncio
async def test_outreach_draft_no_voice_when_unfilled():
    captured = {}

    async def fake_enqueue(spec, **kw):
        captured["desc"] = spec["description"]
        return {"task_id": 1}

    with patch.object(od, "enqueue", fake_enqueue), \
         patch.object(od, "load_founder_voice", return_value=""):
        await od.run_outreach_draft(
            product_id="p1", mission_id=3,
            prospect_data={"name": "Sam"}, template_id="cold", list_id="L1",
        )

    # Empty voice → no dangling "Brand voice:" header
    assert "Brand voice:" not in captured["desc"]
