"""Router-level regression test for the demo/storyboard workflow dispatch branch.

Drives mr_roboto.run() with the exact payload shape the i2p_v3.json
13.demo_storyboard step produces, ensuring the router branch calls the verb
with the correct keyword arguments and that no TypeError escapes.

(press_kit dispatch is deferred to Plan 3 — see
docs/handoff/2026-06-07-cps-sp4b-plan3-handoff.md.)
"""
import pytest
import mr_roboto


@pytest.mark.asyncio
async def test_demo_storyboard_dispatch_matches_verb_signature(tmp_path):
    """Router branch for demo/storyboard must call the verb without TypeError."""
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    (demo_dir / "storyboard_raw.json").write_text(
        '{"title":"D","scenes":[{"title":"a","narrator_text":"hi"},{"title":"b","narrator_text":""}]}',
        encoding="utf-8",
    )
    task = {
        "mission_id": 1,
        "payload": {
            "action": "demo/storyboard",
            "raw_filename": "demo/storyboard_raw.json",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed", getattr(action, "error", action)
