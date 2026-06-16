"""5.0b smart surface gate — clarify branch decision logic.

infer-first: high confidence advances silently, medium advances with a
non-blocking "tap to change" notice, low (no signal) falls back to the
original blocking keyboard. Verifies the gate no longer pauses the
pipeline to ask what the mission text already states.
"""
from unittest.mock import AsyncMock

import pytest

OPTIONS = ["mobile only", "web only", "mobile + web"]


def _task():
    return {"id": 99, "mission_id": 7, "payload": {
        "kind": "surface_choice", "options": OPTIONS,
    }}


@pytest.fixture
def patched(monkeypatch):
    """Patch the I/O seams; keep the real infer_surfaces decision logic.

    `_load_target_platform` defaults to None so these tests exercise the
    text-inference FALLBACK path; the derive-from-target_platform path is
    covered separately in test_derive_from_target_platform.
    """
    import mr_roboto.clarify as C
    import mr_roboto.surfaces_persist as P

    write = AsyncMock(return_value={})
    kb = AsyncMock(return_value=True)
    upd = AsyncMock()
    monkeypatch.setattr(P, "write_surfaces_json", write)
    monkeypatch.setattr(C, "send_surface_keyboard", kb)
    monkeypatch.setattr(C, "_resolve_chat_id", AsyncMock(return_value=123))
    monkeypatch.setattr(C, "update_task", upd)
    monkeypatch.setattr(C, "_load_target_platform", AsyncMock(return_value=None))
    monkeypatch.setattr(C, "_load_surface_signal_surfaces", AsyncMock(return_value=[]))
    return C, write, kb, upd


@pytest.mark.asyncio
async def test_derive_layers_desktop_admin_from_signal(patched, monkeypatch):
    """Stage 2 (safe half): desktop/admin from the deterministic surface_signal
    augment the web/mobile derived from target_platform — design lane regains
    them without a pause, primary stays a build surface."""
    C, write, kb, _ = patched
    monkeypatch.setattr(C, "_load_target_platform", AsyncMock(return_value="mobile"))
    monkeypatch.setattr(C, "_load_surface_signal_surfaces",
                        AsyncMock(return_value=["mobile", "desktop", "admin"]))

    res = await C.clarify(_task())

    assert res["status"] == "completed"
    assert res["surfaces"] == ["mobile", "desktop", "admin"]
    assert write.await_args.kwargs["primary_surface"] == "mobile"  # build surface
    kb.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tp,surfaces,primary",
    [
        ("web", ["web"], "web"),
        ("mobile", ["mobile"], "mobile"),
        ("both", ["mobile", "web"], "mobile"),
    ],
)
async def test_derive_from_target_platform(patched, monkeypatch, tp, surfaces, primary):
    """Stage 1: surfaces derived from 3.6's target_platform — never re-ask."""
    C, write, kb, _ = patched
    monkeypatch.setattr(C, "_load_target_platform", AsyncMock(return_value=tp))
    # Mission text would infer mobile, but derivation must win regardless.
    monkeypatch.setattr(C, "_gather_mission_text",
                        AsyncMock(return_value="some app idea"))

    res = await C.clarify(_task())

    assert res["status"] == "completed"
    assert res["derived"] is True
    assert res["surfaces"] == surfaces
    assert res["target_platform"] == tp
    assert write.await_count == 1
    assert write.await_args.kwargs["source"] == "derived"
    assert write.await_args.kwargs["primary_surface"] == primary
    kb.assert_not_awaited()  # canonical signal exists → zero founder pause


@pytest.mark.asyncio
async def test_unknown_target_platform_falls_back_to_inference(patched, monkeypatch):
    """Garbage target_platform → fall back to text inference, not a crash."""
    C, write, kb, _ = patched
    monkeypatch.setattr(C, "_load_target_platform", AsyncMock(return_value="garbage"))
    monkeypatch.setattr(C, "_gather_mission_text",
                        AsyncMock(return_value="Build an iOS app"))

    res = await C.clarify(_task())

    assert res["status"] == "completed"
    assert res.get("inferred") is True  # inference path, not derive path
    assert res["surfaces"] == ["mobile"]


@pytest.mark.asyncio
async def test_high_confidence_advances_silently(patched, monkeypatch):
    C, write, kb, _ = patched
    monkeypatch.setattr(C, "_gather_mission_text",
                        AsyncMock(return_value="Build an iOS app"))

    res = await C.clarify(_task())

    assert res["status"] == "completed"
    assert res["inferred"] is True
    assert res["surfaces"] == ["mobile"]
    assert write.await_count == 1
    assert write.await_args.kwargs["source"] == "inferred"
    kb.assert_not_awaited()  # no founder pause, no notice


@pytest.mark.asyncio
async def test_medium_confidence_advances_with_notice(patched, monkeypatch):
    C, write, kb, _ = patched
    monkeypatch.setattr(C, "_gather_mission_text",
                        AsyncMock(return_value="I want an app for habits"))

    res = await C.clarify(_task())

    assert res["status"] == "completed"      # advances — never parks
    assert res["confidence"] == "medium"
    assert write.await_count == 1
    kb.assert_awaited_once()                 # non-blocking "tap to change"
    assert kb.await_args.kwargs.get("prompt")  # tailored notice text


@pytest.mark.asyncio
async def test_low_confidence_falls_back_to_blocking_keyboard(patched, monkeypatch):
    C, write, kb, upd = patched
    monkeypatch.setattr(C, "_gather_mission_text",
                        AsyncMock(return_value="A product to save money"))

    res = await C.clarify(_task())

    assert res["status"] == "needs_clarification"
    assert res["keyboard_sent"] is True
    write.assert_not_awaited()               # nothing inferred → don't guess
    kb.assert_awaited_once()
    upd.assert_awaited_once()                # task parked waiting_human
