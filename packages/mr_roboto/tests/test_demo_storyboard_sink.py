import json
import os
import pytest
from mr_roboto.demo_storyboard import run


@pytest.mark.asyncio
async def test_sink_reads_raw_file_normalizes_and_writes(tmp_path):
    # Producer wrote this raw file (simulated materializer output).
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    raw = {
        "title": "Demo",
        "scenes": [
            {"title": "Intro", "target_seconds": 5, "viewport_state": "home",
             "narrator_text": "Welcome"},
            {"title": "Silent pan", "target_seconds": 4, "viewport_state": "dash",
             "narrator_text": ""},
        ],
    }
    (demo_dir / "storyboard_raw.json").write_text(json.dumps(raw), encoding="utf-8")

    res = await run(
        mission_id=1,
        workspace_path=str(tmp_path),
        raw_filename="demo/storyboard_raw.json",
    )

    assert res["ok"] is True
    out_path = tmp_path / "demo" / "storyboard.json"
    assert out_path.is_file()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["scenes"][0]["id"] == "scene_1"
    assert written["scenes"][0]["visual_only"] is False
    assert written["scenes"][1]["visual_only"] is True
    assert res["scene_count"] == 2


@pytest.mark.asyncio
async def test_sink_missing_raw_file_returns_error(tmp_path):
    res = await run(
        mission_id=1,
        workspace_path=str(tmp_path),
        raw_filename="demo/storyboard_raw.json",
    )
    assert res["ok"] is False
    assert "raw" in res["error"].lower()


def test_sink_makes_no_llm_call():
    import mr_roboto.demo_storyboard as mod
    assert not hasattr(mod, "_enqueue_storyboard_llm"), "LLM enqueue must be deleted"
    assert not hasattr(mod, "_STORYBOARD_SYSTEM"), "LLM prompt must be deleted"
