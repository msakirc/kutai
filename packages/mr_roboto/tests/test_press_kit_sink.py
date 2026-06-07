import os
import pytest
import mr_roboto.press_kit_assemble as mod
from mr_roboto.press_kit_assemble import run, AUDIENCE_VARIANTS


def _async(val):
    async def _c():
        return val
    return _c()


@pytest.mark.asyncio
async def test_sink_reads_four_onepagers_and_zips(tmp_path, monkeypatch):
    src_dir = tmp_path / "press_kit" / "src"
    src_dir.mkdir(parents=True)
    for aud in AUDIENCE_VARIANTS:
        (src_dir / f"one_pager_{aud}.md").write_text(f"# {aud} one-pager\n\nbody", encoding="utf-8")

    monkeypatch.setattr(mod, "_get_latest_version", lambda product_id: _async(0))
    monkeypatch.setattr(mod, "_emit_founder_action", lambda **kw: _async(None))

    res = await run(
        mission_id=1,
        product_id="prod_x",
        workspace_path=str(tmp_path),
        onepager_dir="press_kit/src",
    )
    assert res["ok"] is True
    assert res["version"] == 1
    for aud in AUDIENCE_VARIANTS:
        assert os.path.isfile(res["manifest"]["variants"][aud]["zip_path"])


@pytest.mark.asyncio
async def test_sink_missing_onepager_fails_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_get_latest_version", lambda product_id: _async(0))
    monkeypatch.setattr(mod, "_emit_founder_action", lambda **kw: _async(None))
    res = await run(
        mission_id=1, product_id="prod_x",
        workspace_path=str(tmp_path), onepager_dir="press_kit/src",
    )
    assert res["ok"] is False
    assert "one_pager" in res["error"]


def test_no_llm_symbols():
    assert not hasattr(mod, "_draft_one_pager_llm"), "LLM draft fn must be deleted"
    assert not hasattr(mod, "_AUDIENCE_PROMPTS"), "prompts moved to press_kit.json"


def test_press_kit_workflow_shapes():
    import json
    wf = json.load(open("src/workflows/press_kit/press_kit.json", encoding="utf-8"))
    steps = {s["id"]: s for s in wf["steps"]}
    for aud in ("investor", "journalist", "partner", "candidate"):
        p = steps[f"1.draft_onepager_{aud}"]
        assert p["agent"] == "planner"
        assert "executor" not in p
        assert p["produces"] == [f"press_kit/src/one_pager_{aud}.md"]
    asm = steps["2.assemble"]
    assert asm["executor"] == "mechanical"
    assert len(asm["depends_on"]) == 4
