"""Router-level regression tests for workflow dispatch branches.

These tests drive mr_roboto.run() with the exact payload shape that the
press_kit.json and i2p_v3.json workflow steps produce, ensuring the router
branch calls each verb with the correct keyword arguments and that no
TypeError escapes.
"""
import pytest
import mr_roboto
import mr_roboto.press_kit_assemble as pk
from mr_roboto.press_kit_assemble import AUDIENCE_VARIANTS


def _async(val):
    async def _c():
        return val
    return _c()


@pytest.mark.asyncio
async def test_press_kit_assemble_dispatch_matches_verb_signature(tmp_path, monkeypatch):
    """Router branch must pass onepager_dir (not spec_text) to _pk_assemble.

    This is the regression test for the TypeError that occurred when the
    router called run(spec_text=...) but the verb signature expects
    run(onepager_dir=...) as the optional 4th parameter.
    """
    # Producers wrote the 4 one-pagers into the workspace.
    src_dir = tmp_path / "press_kit" / "src"
    src_dir.mkdir(parents=True)
    for aud in AUDIENCE_VARIANTS:
        (src_dir / f"one_pager_{aud}.md").write_text(f"# {aud}\n\nbody", encoding="utf-8")

    monkeypatch.setattr(pk, "_get_latest_version", lambda product_id: _async(0))
    monkeypatch.setattr(pk, "_emit_founder_action", lambda **kw: _async(None))

    # Payload shape == press_kit.json 2.assemble payload.
    task = {
        "mission_id": 1,
        "payload": {
            "action": "press_kit/assemble",
            "onepager_dir": "press_kit/src",
            "product_id": "prod_x",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed", getattr(action, "error", action)


@pytest.mark.asyncio
async def test_press_kit_assemble_dispatch_no_spec_text_key(tmp_path, monkeypatch):
    """Verify spec_text is no longer passed — sending it must NOT cause a TypeError.

    Even if the workflow JSON accidentally includes spec_text in the payload,
    the router must not forward it to the verb (which would raise TypeError).
    """
    src_dir = tmp_path / "press_kit" / "src"
    src_dir.mkdir(parents=True)
    for aud in AUDIENCE_VARIANTS:
        (src_dir / f"one_pager_{aud}.md").write_text(f"# {aud}\n\nbody", encoding="utf-8")

    monkeypatch.setattr(pk, "_get_latest_version", lambda product_id: _async(0))
    monkeypatch.setattr(pk, "_emit_founder_action", lambda **kw: _async(None))

    # Include spec_text in the payload (as a stale workflow JSON might).
    task = {
        "mission_id": 1,
        "payload": {
            "action": "press_kit/assemble",
            "onepager_dir": "press_kit/src",
            "product_id": "prod_x",
            "workspace_path": str(tmp_path),
            "spec_text": "should be ignored by the router",
        },
    }
    # The router must NOT forward spec_text → verb; it is silently ignored.
    action = await mr_roboto.run(task)
    # spec_text being in the payload should not change the outcome.
    assert action.status == "completed", getattr(action, "error", action)


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
