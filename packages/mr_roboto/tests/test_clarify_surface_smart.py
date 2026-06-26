"""5.0b smart surface gate — clarify branch decision logic.

Founder-words-win policy (2026-06-26): the founder's own description is the
AUTHORITATIVE surface signal. A high/medium signal in the mission text drives
the surfaces and is NEVER overridden by the LLM tech-analysis (3.6
``target_platform``) — that only fills in when the founder text says nothing
about a platform. Either way a non-blocking "assumed X — tap to change" card is
sent so the founder can override on EVERY mission; only genuinely-ambiguous
missions (no founder signal AND no 3.6 signal) fall back to the original
blocking keyboard.
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

    Defaults: no 3.6 target_platform, no surface_signal, empty mission text
    (low signal). Individual tests override ``_gather_mission_text`` /
    ``_load_target_platform`` to drive a specific branch.
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
    monkeypatch.setattr(C, "_gather_mission_text", AsyncMock(return_value=""))
    return C, write, kb, upd


@pytest.mark.asyncio
async def test_founder_words_win_over_target_platform(patched, monkeypatch):
    """THE bug fix: the founder said "app" (→ mobile) but 3.6's tech-analysis
    derived ``web``. The founder's explicit words MUST win — surfaces=mobile,
    NOT web, and never the silent derive path."""
    C, write, kb, _ = patched
    # 3.6 would say web; founder text says app (mobile, medium).
    monkeypatch.setattr(C, "_load_target_platform", AsyncMock(return_value="web"))
    monkeypatch.setattr(C, "_gather_mission_text",
                        AsyncMock(return_value="I want an app for tracking habits"))

    res = await C.clarify(_task())

    assert res["status"] == "completed"
    assert res["inferred"] is True            # founder-words path, not derive
    assert res.get("derived") is not True
    assert res["surfaces"] == ["mobile"]      # founder won over 3.6's "web"
    assert write.await_args.kwargs["source"] == "inferred"
    kb.assert_awaited_once()                  # correction card always offered
    assert kb.await_args.kwargs.get("prompt")


@pytest.mark.asyncio
async def test_high_confidence_sends_correction_card(patched, monkeypatch):
    """Explicit "iOS app" → mobile. Still sends the non-blocking correction
    card so the founder can override on every mission (was silent before)."""
    C, write, kb, _ = patched
    monkeypatch.setattr(C, "_gather_mission_text",
                        AsyncMock(return_value="Build an iOS app"))

    res = await C.clarify(_task())

    assert res["status"] == "completed"
    assert res["inferred"] is True
    assert res["confidence"] == "high"
    assert res["surfaces"] == ["mobile"]
    assert write.await_args.kwargs["source"] == "inferred"
    kb.assert_awaited_once()
    assert kb.await_args.kwargs.get("prompt")


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
@pytest.mark.parametrize(
    "tp,surfaces,primary",
    [
        ("web", ["web"], "web"),
        ("mobile", ["mobile"], "mobile"),
        ("both", ["mobile", "web"], "mobile"),
    ],
)
async def test_derive_when_no_founder_signal(patched, monkeypatch, tp, surfaces, primary):
    """Founder text says nothing about a platform → derive from 3.6's
    target_platform, and STILL send a correction card so the founder can
    override the machine's guess."""
    C, write, kb, _ = patched
    monkeypatch.setattr(C, "_load_target_platform", AsyncMock(return_value=tp))
    monkeypatch.setattr(C, "_gather_mission_text",
                        AsyncMock(return_value="a tool to manage things"))

    res = await C.clarify(_task())

    assert res["status"] == "completed"
    assert res["derived"] is True
    assert res["surfaces"] == surfaces
    assert res["target_platform"] == tp
    assert write.await_args.kwargs["source"] == "derived"
    assert write.await_args.kwargs["primary_surface"] == primary
    kb.assert_awaited_once()                 # override card on the derive path too
    assert kb.await_args.kwargs.get("prompt")


@pytest.mark.asyncio
async def test_derive_layers_desktop_admin_from_signal(patched, monkeypatch):
    """No founder signal → desktop/admin from the deterministic surface_signal
    augment the web/mobile derived from target_platform; primary stays a build
    surface; correction card still offered."""
    C, write, kb, _ = patched
    monkeypatch.setattr(C, "_load_target_platform", AsyncMock(return_value="mobile"))
    monkeypatch.setattr(C, "_load_surface_signal_surfaces",
                        AsyncMock(return_value=["mobile", "desktop", "admin"]))
    monkeypatch.setattr(C, "_gather_mission_text",
                        AsyncMock(return_value="a tool to manage things"))

    res = await C.clarify(_task())

    assert res["status"] == "completed"
    assert res["surfaces"] == ["mobile", "desktop", "admin"]
    assert write.await_args.kwargs["primary_surface"] == "mobile"  # build surface
    kb.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_runs_before_attention_gate(patched, monkeypatch):
    """An auto-resolved surface (derived or inferred) needs no founder pause —
    it must short-circuit BEFORE the attention-budget gate so an exhausted
    budget can't defer it (which would complete the task without writing
    surfaces.json → DLQ at verify_surfaces_shape)."""
    C, write, kb, _ = patched
    monkeypatch.setattr(C, "_load_target_platform", AsyncMock(return_value="both"))
    monkeypatch.setattr(C, "_gather_mission_text",
                        AsyncMock(return_value="a tool to manage things"))
    import importlib
    A = importlib.import_module("mr_roboto.attention_check")
    deferred = AsyncMock()
    monkeypatch.setattr(A, "attention_check",
                        AsyncMock(return_value={"ok": False, "remaining": 0}))
    monkeypatch.setattr(A, "write_deferred_question", deferred)

    res = await C.clarify(_task())

    assert res["status"] == "completed"
    assert res["derived"] is True
    deferred.assert_not_awaited()            # attention gate never reached
    assert write.await_count == 1            # surfaces.json written regardless


@pytest.mark.asyncio
async def test_inferred_runs_before_attention_gate(patched, monkeypatch):
    """Same guarantee for the founder-words path: a medium inference writes
    surfaces.json even when the attention budget is exhausted (no defer/DLQ)."""
    C, write, kb, _ = patched
    monkeypatch.setattr(C, "_gather_mission_text",
                        AsyncMock(return_value="I want an app for habits"))
    import importlib
    A = importlib.import_module("mr_roboto.attention_check")
    deferred = AsyncMock()
    monkeypatch.setattr(A, "attention_check",
                        AsyncMock(return_value={"ok": False, "remaining": 0}))
    monkeypatch.setattr(A, "write_deferred_question", deferred)

    res = await C.clarify(_task())

    assert res["status"] == "completed"
    assert res["inferred"] is True
    deferred.assert_not_awaited()
    assert write.await_count == 1


@pytest.mark.asyncio
async def test_unknown_target_platform_falls_back_to_inference(patched, monkeypatch):
    """Garbage target_platform + a clear founder signal → inference wins."""
    C, write, kb, _ = patched
    monkeypatch.setattr(C, "_load_target_platform", AsyncMock(return_value="garbage"))
    monkeypatch.setattr(C, "_gather_mission_text",
                        AsyncMock(return_value="Build an iOS app"))

    res = await C.clarify(_task())

    assert res["status"] == "completed"
    assert res.get("inferred") is True
    assert res["surfaces"] == ["mobile"]


@pytest.mark.asyncio
async def test_low_confidence_falls_back_to_blocking_keyboard(patched, monkeypatch):
    """No founder signal AND no 3.6 signal → genuinely ambiguous → the original
    blocking keyboard parks the task waiting_human."""
    C, write, kb, upd = patched
    monkeypatch.setattr(C, "_gather_mission_text",
                        AsyncMock(return_value="A product to save money"))

    res = await C.clarify(_task())

    assert res["status"] == "needs_clarification"
    assert res["keyboard_sent"] is True
    write.assert_not_awaited()               # nothing inferred → don't guess
    kb.assert_awaited_once()
    assert not kb.await_args.kwargs.get("prompt")  # blocking ask, no notice text
    upd.assert_awaited_once()                # task parked waiting_human
