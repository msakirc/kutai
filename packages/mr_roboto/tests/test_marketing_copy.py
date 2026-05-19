import pytest
from unittest.mock import patch
import mr_roboto.marketing_copy as mc


@pytest.mark.asyncio
async def test_marketing_copy_prompt_includes_founder_voice():
    captured = {}

    async def fake_enqueue(spec, **kw):
        captured["spec"] = spec
        return {"result": {"hero": ["H"]}}

    with patch.object(mc, "enqueue", fake_enqueue), \
         patch.object(mc, "load_founder_voice", return_value="Plainspoken, concrete."):
        await mc.run_marketing_copy(
            product_id="p1", mission_id=5,
            product_spec={"name": "Thing"},
        )

    desc = captured["spec"]["description"]
    user_msg = captured["spec"]["context"]["llm_call"]["messages"][1]["content"]
    assert "Plainspoken, concrete." in desc
    assert "Plainspoken, concrete." in user_msg


@pytest.mark.asyncio
async def test_marketing_copy_no_voice_when_unfilled():
    captured = {}

    async def fake_enqueue(spec, **kw):
        captured["spec"] = spec
        return {"result": {"hero": ["H"]}}

    with patch.object(mc, "enqueue", fake_enqueue), \
         patch.object(mc, "load_founder_voice", return_value=""):
        await mc.run_marketing_copy(
            product_id="p1", mission_id=5,
            product_spec={"name": "Thing"},
        )

    # Empty founder voice → prompt unchanged, no dangling brand-voice header.
    assert "Brand voice" not in captured["spec"]["description"]
