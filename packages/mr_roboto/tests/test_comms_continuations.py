def test_comms_handlers_registered():
    import general_beckman.continuations as C
    import mr_roboto.executors.comms_continuations  # noqa: F401 — triggers register
    for name in (
        "comms.crisis_holding.resume", "comms.crisis_holding.resume_err",
        "comms.incident_update.resume", "comms.incident_update.resume_err",
        "comms.press_kit.resume", "comms.press_kit.resume_err",
    ):
        assert name in C._HANDLERS, f"{name} not registered"


def test_comms_module_in_handler_modules():
    import general_beckman.continuations as C
    assert "mr_roboto.executors.comms_continuations" in C._HANDLER_MODULES


import asyncio


def test_crisis_resume_parses_variants_and_emits(monkeypatch):
    import mr_roboto.executors.comms_continuations as S
    emitted = {}

    async def fake_emit(*, event_id, product_id, tier, variants):
        emitted.update(dict(event_id=event_id, product_id=product_id, tier=tier, variants=variants))

    monkeypatch.setattr(S, "_emit_crisis_card", fake_emit)
    result = {"result": {"content": '["Variant A text here", "Variant B text here"]'}}
    state = {"event_id": 7, "product_id": "p1", "tier": 2,
             "summary": "outage", "playbook_excerpt": ""}
    asyncio.run(S._crisis_resume(1, result, state))
    assert emitted["variants"] == ["Variant A text here", "Variant B text here"]
    assert emitted["event_id"] == 7


def test_crisis_resume_err_uses_canned_fallback(monkeypatch):
    import mr_roboto.executors.comms_continuations as S
    emitted = {}

    async def fake_emit(*, event_id, product_id, tier, variants):
        emitted["variants"] = variants

    monkeypatch.setattr(S, "_emit_crisis_card", fake_emit)
    state = {"event_id": 7, "product_id": "p1", "tier": 2, "summary": "", "playbook_excerpt": ""}
    asyncio.run(S._crisis_resume_err(1, {"error": "exhausted"}, state))
    assert len(emitted["variants"]) == 2  # canned tier fallback
    assert "p1" in emitted["variants"][0]
