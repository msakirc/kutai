import pytest

import mr_roboto
from mr_roboto.reversibility import VERB_REVERSIBILITY


def test_swap_verb_registered():
    assert "swap_placeholder_images" in VERB_REVERSIBILITY
    assert VERB_REVERSIBILITY["swap_placeholder_images"] == "full"


def test_verify_verb_registered():
    assert "verify_swap_placeholder_images_shape" in VERB_REVERSIBILITY
    assert VERB_REVERSIBILITY["verify_swap_placeholder_images_shape"] == "full"


def test_module_exports_swap():
    assert hasattr(mr_roboto, "swap_placeholder_images")
    assert "swap_placeholder_images" in mr_roboto.__all__


def test_module_exports_verify():
    assert hasattr(mr_roboto, "verify_swap_placeholder_images_shape")
    assert "verify_swap_placeholder_images_shape" in mr_roboto.__all__


@pytest.mark.asyncio
async def test_dispatch_routes_swap_action(monkeypatch):
    captured = {}

    async def _fake_swap(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "replaced_count": 2, "skipped_count": 1,
                "html_files_seen": 1, "html_files_changed": 1, "errors": []}
    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images.swap_placeholder_images", _fake_swap,
    )
    task = {
        "id": 100, "mission_id": 42,
        "title": "swap_test",
        "context": {"payload": {
            "action": "swap_placeholder_images",
            "design_tokens": {"primary": "#E07A5F"},
            "brand_voice": "warm",
        }},
    }
    res = await mr_roboto.run(task)
    assert res.status == "completed"
    assert res.result["replaced_count"] == 2
    assert captured["mission_id"] == 42
    assert captured["design_tokens"] == {"primary": "#E07A5F"}
    # CPS kickoff threads the 5.35 task id so the prompt_writer child gets a
    # parent_id.
    assert captured["task_id"] == 100


@pytest.mark.asyncio
async def test_dispatch_swap_swallows_unexpected_error(monkeypatch):
    """Best-effort: an unexpected swap exception must not block the mission —
    the dispatch branch degrades to a completed/skipped result."""
    async def _boom(**kwargs):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images.swap_placeholder_images", _boom,
    )
    task = {
        "id": 100, "mission_id": 42, "title": "swap_test",
        "context": {"payload": {"action": "swap_placeholder_images"}},
    }
    res = await mr_roboto.run(task)
    assert res.status == "completed"
    assert res.result["ok"] is True
    assert res.result["replaced_count"] == 0
    assert res.result["chain"] == "none"
    assert any("kaboom" in e for e in res.result["errors"])


@pytest.mark.asyncio
async def test_dispatch_routes_verify_action(monkeypatch):
    """The verify posthook dispatches with action=verify_swap_placeholder_images_shape."""
    captured = {}

    def _fake_verify(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "surviving_placeholders": 0, "expected_replaced": 3}
    monkeypatch.setattr(
        "mr_roboto.verify_swap_placeholder_images_shape."
        "verify_swap_placeholder_images_shape",
        _fake_verify,
    )
    task = {
        "id": 101, "mission_id": 42, "title": "verify_test",
        "context": {"payload": {
            "action": "verify_swap_placeholder_images_shape",
            "workspace_path": "/fake/ws",
            "swap_result": {"replaced_count": 3, "skipped_count": 0, "errors": []},
        }},
    }
    res = await mr_roboto.run(task)
    assert res.status == "completed"
    assert captured["workspace_path"] == "/fake/ws"


@pytest.mark.asyncio
async def test_dispatch_verify_derives_workspace_from_mission_id(
    monkeypatch, tmp_path
):
    """LIVE i2p: the verify step payload carries NO workspace_path. The
    dispatch branch must derive it from task['mission_id'] via
    get_mission_workspace, otherwise the verifier walks nothing and the gate
    is vacuous."""
    captured = {}

    def _fake_verify(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "surviving_placeholders": 0,
                "expected_replaced": 0, "broken_asset_refs": []}
    monkeypatch.setattr(
        "mr_roboto.verify_swap_placeholder_images_shape."
        "verify_swap_placeholder_images_shape",
        _fake_verify,
    )

    derived_ws = str(tmp_path / "mission_42")

    def _fake_get_ws(mission_id):
        assert mission_id == 42
        return derived_ws
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", _fake_get_ws,
    )

    task = {
        "id": 101, "mission_id": 42, "title": "verify_test",
        "context": {"payload": {
            "action": "verify_swap_placeholder_images_shape",
            # NOTE: no workspace_path, no swap_result — the live shape.
        }},
    }
    res = await mr_roboto.run(task)
    assert res.status == "completed"
    assert captured["workspace_path"] == derived_ws
    # Live shape: swap_result is empty, gate rests on self-derived check.
    assert captured["swap_result"] == {}


@pytest.mark.asyncio
async def test_dispatch_verify_action_fails_on_inconsistency(monkeypatch):
    """When the verifier returns ok=False the dispatch branch fails the task
    (gates emit_preview_url), mirroring verify_charter_shape."""
    def _fake_verify(**kwargs):
        return {"ok": False, "error": "inconsistent: x",
                "surviving_placeholders": 1, "expected_replaced": 3}
    monkeypatch.setattr(
        "mr_roboto.verify_swap_placeholder_images_shape."
        "verify_swap_placeholder_images_shape",
        _fake_verify,
    )
    task = {
        "id": 101, "mission_id": 42, "title": "verify_test",
        "context": {"payload": {
            "action": "verify_swap_placeholder_images_shape",
            "workspace_path": "/fake/ws",
            "swap_result": {"replaced_count": 3, "skipped_count": 0, "errors": []},
        }},
    }
    res = await mr_roboto.run(task)
    assert res.status == "failed"
    assert "inconsistent" in (res.error or "")
