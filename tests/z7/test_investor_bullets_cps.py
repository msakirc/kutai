"""investor_bullets anomaly hypotheses run as a sequential CPS chain (no
await_inline). Each resume stores one hypothesis and either enqueues the next
anomaly's child or finalizes (render + variants + founder_action)."""
import pytest
import src.app.jobs.investor_bullets as ib


@pytest.mark.asyncio
async def test_resume_enqueues_next_anomaly(monkeypatch):
    enq = {}

    async def fake_enqueue(state):
        enq["state"] = dict(state)
        return 7

    monkeypatch.setattr(ib, "_enqueue_hypothesis_child", fake_enqueue, raising=False)
    finalized = {}

    async def fake_final(state):
        finalized["state"] = state

    monkeypatch.setattr(ib, "_finalize_bullets", fake_final, raising=False)

    state = {
        "product_id": "p1", "mission_id": 0, "metrics": {}, "missing": [],
        "anomalies": [["mrr", 100.0, [50.0, 60.0]], ["churn", 9.0, [3.0, 4.0]]],
        "idx": 0, "hypotheses": {},
    }
    await ib._hypothesis_resume(1, {"content": "MRR jumped on a new enterprise deal."}, state)
    # stored first hypothesis, enqueued the second anomaly, did NOT finalize
    assert enq["state"]["hypotheses"]["mrr"] == "MRR jumped on a new enterprise deal."
    assert enq["state"]["idx"] == 1
    assert "state" not in finalized


@pytest.mark.asyncio
async def test_resume_finalizes_on_last(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("should not enqueue past the last anomaly")

    monkeypatch.setattr(ib, "_enqueue_hypothesis_child", boom, raising=False)
    finalized = {}

    async def fake_final(state):
        finalized["state"] = state

    monkeypatch.setattr(ib, "_finalize_bullets", fake_final, raising=False)

    state = {
        "product_id": "p1", "mission_id": 0, "metrics": {}, "missing": [],
        "anomalies": [["mrr", 100.0, [50.0]]],
        "idx": 0, "hypotheses": {},
    }
    await ib._hypothesis_resume(1, {"content": "One-off annual prepay."}, state)
    assert finalized["state"]["hypotheses"]["mrr"] == "One-off annual prepay."


@pytest.mark.asyncio
async def test_resume_err_skips_and_continues(monkeypatch):
    """A failed hypothesis child must not stall the chain — skip + advance."""
    enq = {}

    async def fake_enqueue(state):
        enq["idx"] = state["idx"]
        return 7

    monkeypatch.setattr(ib, "_enqueue_hypothesis_child", fake_enqueue, raising=False)
    monkeypatch.setattr(ib, "_finalize_bullets",
                        lambda s: (_ for _ in ()).throw(AssertionError("not last")),
                        raising=False)

    state = {
        "product_id": "p1", "mission_id": 0, "metrics": {}, "missing": [],
        "anomalies": [["mrr", 100.0, [50.0]], ["churn", 9.0, [3.0]]],
        "idx": 0, "hypotheses": {},
    }
    await ib._hypothesis_resume_err(1, {"status": "failed", "error": "exhausted"}, state)
    # no hypothesis stored for the failed metric, but the chain advanced
    assert "mrr" not in state["hypotheses"]
    assert enq["idx"] == 1


@pytest.mark.asyncio
async def test_enqueue_hypothesis_child_uses_cps_continuation(monkeypatch):
    captured = {}

    async def fake_overhead(spec, *, lane, **kwargs):
        captured["spec"] = spec
        captured["lane"] = lane
        captured["kwargs"] = kwargs
        return 99

    monkeypatch.setattr(ib, "_enqueue_overhead", fake_overhead, raising=False)
    state = {"anomalies": [["mrr", 180.0, [100.0, 102.0, 98.0]]], "idx": 0, "hypotheses": {}}
    await ib._enqueue_hypothesis_child(state)
    # raw_dispatch overhead call, CPS continuation, NO await_inline
    assert captured["spec"]["context"]["llm_call"]["raw_dispatch"] is True
    assert captured["kwargs"]["on_complete"] == "investor_bullets.hypothesis.resume"
    assert captured["kwargs"]["on_error"] == "investor_bullets.hypothesis.resume_err"
    assert "await_inline" not in captured["kwargs"]
    assert captured["kwargs"]["cont_state"] is state


@pytest.mark.asyncio
async def test_run_no_anomalies_finalizes_immediately(monkeypatch):
    async def fake_collect(pid):
        return ({}, [])

    monkeypatch.setattr(ib, "collect_metrics", fake_collect)
    called = {}

    async def fake_final(state):
        called["yes"] = True
        return {"ok": True, "variants": 0}

    monkeypatch.setattr(ib, "_finalize_bullets", fake_final, raising=False)
    out = await ib.run_investor_bullets("p1")
    assert called.get("yes") is True
    assert out["ok"] is True


@pytest.mark.asyncio
async def test_run_with_anomaly_starts_chain(monkeypatch):
    async def fake_collect(pid):
        # zero-variance history + differing current => guaranteed anomaly
        return ({"mrr": {"current": 9999.0, "history": [100.0, 100.0, 100.0]}}, [])

    monkeypatch.setattr(ib, "collect_metrics", fake_collect)
    started = {}

    async def fake_enqueue(state):
        started["state"] = state
        return 5

    monkeypatch.setattr(ib, "_enqueue_hypothesis_child", fake_enqueue, raising=False)
    out = await ib.run_investor_bullets("p1")
    assert out.get("pending") is True
    assert started["state"]["anomalies"][0][0] == "mrr"


@pytest.mark.asyncio
async def test_cps_handlers_registered():
    from general_beckman.continuations import _HANDLERS
    ib.register_continuations()
    assert "investor_bullets.hypothesis.resume" in _HANDLERS
    assert "investor_bullets.hypothesis.resume_err" in _HANDLERS
