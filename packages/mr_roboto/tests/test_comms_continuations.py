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


def test_incident_resume_surfaces_redacted_draft(monkeypatch):
    import mr_roboto.executors.comms_continuations as S
    got = {}
    async def fake_emit(*, incident_id, product_id, draft):
        got.update(dict(incident_id=incident_id, product_id=product_id, draft=draft))
    monkeypatch.setattr(S, "_emit_incident_card", fake_emit)
    res = {"result": {"content": "We are investigating a service issue and will update soon."}}
    state = {"incident_id": 9, "product_id": "p1", "status_kind": "investigating",
             "affected_components": ["api"]}
    asyncio.run(S._incident_resume(1, res, state))
    assert "investigating" in got["draft"].lower()
    assert got["incident_id"] == 9


def test_incident_resume_redacts_internal_leak(monkeypatch):
    import mr_roboto.executors.comms_continuations as S
    got = {}
    async def fake_emit(*, incident_id, product_id, draft):
        got["draft"] = draft
    monkeypatch.setattr(S, "_emit_incident_card", fake_emit)
    # LLM erroneously leaks a private IP — the sink's final redaction must scrub it.
    res = {"result": {"content": "Outage on 10.0.0.5 affecting users."}}
    state = {"incident_id": 9, "product_id": "p1", "status_kind": "investigating",
             "affected_components": []}
    asyncio.run(S._incident_resume(1, res, state))
    assert "10.0.0.5" not in got["draft"]


def test_incident_resume_err_uses_fallback(monkeypatch):
    import mr_roboto.executors.comms_continuations as S
    got = {}
    async def fake_emit(*, incident_id, product_id, draft):
        got["draft"] = draft
    monkeypatch.setattr(S, "_emit_incident_card", fake_emit)
    state = {"incident_id": 9, "product_id": "p1", "status_kind": "monitoring",
             "affected_components": ["billing"]}
    asyncio.run(S._incident_resume_err(1, {"error": "exhausted"}, state))
    assert got["draft"]  # non-empty canned draft
    assert "billing" in got["draft"]


def test_press_kit_resume_chains_to_next_audience(monkeypatch):
    import mr_roboto.executors.comms_continuations as S
    calls = {}
    async def fake_next(*, audience, state):
        calls["audience"] = audience
        calls["staged"] = dict(state["staged"])
        return 123
    monkeypatch.setattr(S, "_enqueue_press_kit_audience", fake_next)
    res = {"result": {"content": "Investor one-pager body."}}
    state = {"product_id": "p1", "mission_id": 1, "version": 2, "workspace_path": "/tmp/ws",
             "spec_text": "spec", "remaining": ["journalist", "partner", "candidate"],
             "current": "investor", "staged": {}, "source": {}}
    asyncio.run(S._press_kit_resume(1, res, state))
    assert calls["audience"] == "journalist"
    assert calls["staged"]["investor"] == "Investor one-pager body."


def test_press_kit_resume_final_audience_assembles(monkeypatch):
    import mr_roboto.executors.comms_continuations as S
    assembled = {}
    async def fake_assemble(**kw):
        assembled.update(kw)
    monkeypatch.setattr(S, "_assemble_press_kit", fake_assemble)
    res = {"result": {"content": "Candidate one-pager body."}}
    state = {"product_id": "p1", "mission_id": 1, "version": 2, "workspace_path": "/tmp/ws",
             "spec_text": "spec", "remaining": [], "current": "candidate",
             "staged": {"investor": "i", "journalist": "j", "partner": "p"}, "source": {}}
    asyncio.run(S._press_kit_resume(1, res, state))
    assert set(assembled["staged"].keys()) == {"investor", "journalist", "partner", "candidate"}


def test_press_kit_resume_err_stubs_and_continues(monkeypatch):
    import mr_roboto.executors.comms_continuations as S
    calls = {}
    async def fake_next(*, audience, state):
        calls["audience"] = audience
        calls["staged"] = dict(state["staged"])
        return 123
    monkeypatch.setattr(S, "_enqueue_press_kit_audience", fake_next)
    state = {"product_id": "p1", "mission_id": 1, "version": 2, "workspace_path": "/tmp/ws",
             "spec_text": "the spec text", "remaining": ["partner", "candidate"],
             "current": "journalist", "staged": {"investor": "i"}, "source": {}}
    asyncio.run(S._press_kit_resume_err(1, {"error": "exhausted"}, state))
    assert calls["audience"] == "partner"
    assert "journalist" in calls["staged"]  # stub was staged for the failed audience
