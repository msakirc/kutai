import json
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_enqueue_classify_builds_overhead_child_with_continuation():
    from src.reviews import producers

    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        return 4242

    fake_row = (7, "g2", "Ada", 1, "It crashes on save")

    class _Cur:
        async def fetchone(self):
            return fake_row

    class _DB:
        async def execute(self, *a, **k):
            return _Cur()

    async def fake_get_db():
        return _DB()

    with patch.object(producers, "enqueue", fake_enqueue), \
         patch("src.infra.db.get_db", fake_get_db):
        tid = await producers.enqueue_classify(review_id=7, product_id="prod-x")

    assert tid == 4242
    k = captured["kwargs"]
    assert k["on_complete"] == "reviews.classify.resume"
    assert k["on_error"] == "reviews.classify.resume_err"
    assert k["lane"] == "oneshot"
    st = k["cont_state"]
    assert st["review_id"] == 7 and st["rating"] == 1 and st["body_md"]
    llm = captured["spec"]["context"]["llm_call"]
    assert llm["raw_dispatch"] is True and llm["call_category"] == "overhead"
    assert "crashes" in llm["messages"][0]["content"]


@pytest.mark.asyncio
async def test_classify_resume_persists_and_routes_bug_sideeffect():
    from mr_roboto.executors import reviews_continuations as rc

    updates = []

    class _DB:
        async def execute(self, sql, params=()):
            updates.append((sql, params)); return None
        async def commit(self):
            return None

    async def fake_get_db():
        return _DB()

    emitted = {}
    async def fake_low_star(**kw):
        emitted["low_star"] = kw
    bug = {}
    async def fake_bug(spec, **kw):
        bug["spec"] = spec; return 1

    state = {"review_id": 7, "product_id": "p", "platform": "g2",
             "author": "Ada", "rating": 1, "body_md": "it crashes"}
    result = {"result": {"content": '{"sentiment":"negative","theme_tag":"bug"}'}}

    with patch("src.infra.db.get_db", fake_get_db), \
         patch.object(rc, "_emit_low_star_founder_action", fake_low_star), \
         patch.object(rc, "_enqueue_bug_investigation", fake_bug):
        await rc._classify_resume(99, result, state)

    assert any("UPDATE external_reviews" in s for s, _ in updates)
    assert emitted["low_star"]["theme_tag"] == "bug"
    assert bug["spec"]["title"].startswith("[BUG]")


@pytest.mark.asyncio
async def test_classify_resume_err_uses_heuristic():
    from mr_roboto.executors import reviews_continuations as rc

    updates = []
    class _DB:
        async def execute(self, sql, params=()):
            updates.append(params); return None
        async def commit(self): return None
    async def fake_get_db(): return _DB()

    state = {"review_id": 7, "product_id": "p", "platform": "g2",
             "author": "Ada", "rating": 5, "body_md": "love the UX"}
    with patch("src.infra.db.get_db", fake_get_db), \
         patch.object(rc, "_emit_low_star_founder_action", AsyncMock()), \
         patch.object(rc, "_enqueue_bug_investigation", AsyncMock()):
        await rc._classify_resume_err(99, {"error": "no candidates"}, state)

    # heuristic: rating 5 -> positive
    assert any(p[0] == "positive" for p in updates)


@pytest.mark.asyncio
async def test_draft_reply_resume_surfaces_draft_never_autoposts():
    from mr_roboto.executors import reviews_continuations as rc
    fa = {}
    async def fake_fa(**kw):
        fa.update(kw)
        class _R:
            id = 12
        return _R()
    state = {"review_id": 7, "product_id": "p", "platform": "g2",
             "author": "Ada", "rating": 5}
    result = {"result": {"content": "Thanks so much for the kind words!"}}
    writes = []
    class _DB:
        async def execute(self, sql, params=()):
            writes.append(sql); return None
        async def commit(self): return None
    async def fake_get_db(): return _DB()
    with patch("src.founder_actions.create", fake_fa), \
         patch("src.infra.db.get_db", fake_get_db):
        await rc._draft_reply_resume(99, result, state)
    assert "Thanks so much" in str(fa)
    assert not any("replied_at" in w or "reply_body_md" in w for w in writes)


@pytest.mark.asyncio
async def test_draft_reply_resume_err_uses_fallback():
    from mr_roboto.executors import reviews_continuations as rc
    fa = {}
    async def fake_fa(**kw):
        fa.update(kw)
        class _R:
            id = 13
        return _R()
    state = {"review_id": 7, "product_id": "p", "platform": "g2",
             "author": "Bo", "rating": 2}
    with patch("src.founder_actions.create", fake_fa):
        await rc._draft_reply_resume_err(99, {"error": "no candidates"}, state)
    # fallback draft addresses the author and is surfaced
    assert "Bo" in str(fa)


def test_enqueue_draft_reply_uses_platform_convention():
    # static check: producer holds platform conventions (prompt left mr_roboto)
    from src.reviews import producers
    assert "appstore" in producers._PLATFORM_CONVENTIONS
    assert "g2" in producers._PLATFORM_CONVENTIONS


def test_reviews_continuations_in_handler_modules():
    from general_beckman import continuations as c
    assert "mr_roboto.executors.reviews_continuations" in c._HANDLER_MODULES


@pytest.mark.asyncio
async def test_router_classify_enqueues_producer():
    import mr_roboto
    task = {"id": 1, "payload": {"action": "reviews/classify",
                                 "review_id": 7, "product_id": "p"}}
    with patch("src.reviews.producers.enqueue_classify",
               AsyncMock(return_value=4321)) as m:
        act = await mr_roboto._run_dispatch(task)
    m.assert_awaited_once()
    assert act.status == "completed"
    assert act.result.get("enqueued") == 4321


@pytest.mark.asyncio
async def test_router_draft_reply_enqueues_producer():
    import mr_roboto
    task = {"id": 1, "payload": {"action": "reviews/draft_reply",
                                 "review_id": 7, "product_id": "p"}}
    with patch("src.reviews.producers.enqueue_draft_reply",
               AsyncMock(return_value=999)) as m:
        act = await mr_roboto._run_dispatch(task)
    m.assert_awaited_once()
    assert act.status == "completed"
    assert act.result.get("enqueued") == 999
    assert act.result.get("auto_posted") is False


def test_no_llm_left_in_verb_modules():
    import inspect
    import mr_roboto.reviews_classify as rc
    import mr_roboto.reviews_draft_reply as rd
    assert not hasattr(rc, "_call_llm_classify")
    assert not hasattr(rd, "_call_llm_draft_reply")
    for mod in (rc, rd):
        src = inspect.getsource(mod)
        assert "await_inline=True" not in src
        assert "LLMDispatcher" not in src


@pytest.mark.asyncio
async def test_cron_enqueues_producer_per_review():
    from src.app.jobs import reviews_poll_daily as job
    calls = []

    async def fake_enq(*, review_id, product_id):
        calls.append((review_id, product_id)); return 1

    class _Cur:
        async def fetchall(self):
            return [(7, "p")]

    class _DB:
        async def execute(self, *a, **k):
            return _Cur()

    async def fake_get_db():
        return _DB()

    with patch("src.reviews.producers.enqueue_classify", fake_enq), \
         patch("src.infra.db.get_db", fake_get_db):
        res = await job.run_reviews_poll_daily({"products": []})

    assert (7, "p") in calls
    assert res["total_enqueued"] == 1
