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
    """Patch the I/O seams; keep the real infer_surfaces decision logic."""
    import mr_roboto.clarify as C
    import mr_roboto.surfaces_persist as P

    write = AsyncMock(return_value={})
    kb = AsyncMock(return_value=True)
    upd = AsyncMock()
    monkeypatch.setattr(P, "write_surfaces_json", write)
    monkeypatch.setattr(C, "send_surface_keyboard", kb)
    monkeypatch.setattr(C, "_resolve_chat_id", AsyncMock(return_value=123))
    monkeypatch.setattr(C, "update_task", upd)
    return C, write, kb, upd


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
