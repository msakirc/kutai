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


def test_i2p_demo_split_shapes():
    import json
    wf = json.load(open("src/workflows/i2p/i2p_v3.json", encoding="utf-8"))
    steps = {s["id"]: s for s in wf["steps"]}
    draft = steps["13.demo_storyboard_draft"]
    sink = steps["13.demo_storyboard"]
    assert draft["agent"] == "reviewer"
    assert "executor" not in draft  # producer is NOT mechanical
    # mission-prefixed so the materializer writes under the mission workspace,
    # where the (workspace-deriving) sink reads it.
    assert draft["produces"] == ["mission_{mission_id}/demo/storyboard_raw.json"]
    assert sink["executor"] == "mechanical"
    assert sink["depends_on"] == ["13.demo_storyboard_draft"]
    assert "13.demo_storyboard" in steps["13.demo_record"]["depends_on"]


@pytest.mark.asyncio
async def test_sink_derives_mission_workspace_when_path_absent(tmp_path, monkeypatch):
    """Canonical mr_roboto pattern: with no workspace_path, derive it from
    mission_id via get_mission_workspace (cf. verify_artifacts/run_cmd)."""
    import src.tools.workspace as ws
    monkeypatch.setattr(ws, "get_mission_workspace", lambda mid: str(tmp_path))

    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    (demo_dir / "storyboard_raw.json").write_text(
        '{"title":"D","scenes":[{"title":"a","narrator_text":"hi"}]}', encoding="utf-8"
    )

    res = await run(mission_id=7)  # no workspace_path passed

    assert res["ok"] is True
    assert (tmp_path / "demo" / "storyboard.json").is_file()
    assert res["scene_count"] == 1
