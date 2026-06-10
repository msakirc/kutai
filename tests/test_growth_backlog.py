"""Z9 T3B/T3C — growth signal classifier + backlog scorer tests.

Covers:
  * classify_signals mechanical executor enqueues the signal_classifier
    AGENT via Beckman (never calls LLMDispatcher directly).
  * the classifier on_complete continuation writes classified_signal rows.
  * score_backlog deterministic math — known signals → asserted score
    ordering + inspectable formula breakdown.
  * /backlog lists live candidates.
  * /approve spawns exactly one mission and never auto-spawns others.
"""
import json
import pytest

import src.infra.db as _db_mod
from src.infra.db import (
    init_db, get_db, insert_growth_event, get_growth_events,
)


async def _fresh_db(tmp_path, monkeypatch):
    """Reset DB to a fresh temp file for isolation."""
    db_file = tmp_path / "growth.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    await init_db()


# ───────────────────────── T3B — classifier executor ─────────────────────

@pytest.mark.asyncio
async def test_classify_signals_enqueues_agent_not_dispatcher(
    tmp_path, monkeypatch
):
    """classify_signals must enqueue the signal_classifier AGENT via Beckman.

    Architecture rule: the classifier is LLM work → an agent task through
    Beckman. The mechanical executor must NOT touch LLMDispatcher.
    """
    await _fresh_db(tmp_path, monkeypatch)

    # Seed two raw signals.
    await insert_growth_event(None, "raw_signal", {
        "provider": "intercom", "signal_type": "ticket",
        "content": "Login keeps failing after password reset",
        "external_id": "tk-1", "occurred_at": "2026-05-14 09:00:00",
    })
    await insert_growth_event(None, "raw_signal", {
        "provider": "posthog", "signal_type": "event",
        "content": "Wish there was CSV export",
        "external_id": "ev-2", "occurred_at": "2026-05-14 10:00:00",
    })

    # Spy on beckman.enqueue.
    import general_beckman
    enqueued = []

    async def _fake_enqueue(spec, **kw):
        enqueued.append((spec, kw))
        return 9001

    monkeypatch.setattr(general_beckman, "enqueue", _fake_enqueue)

    # Guard: LLMDispatcher.execute must never be called.
    from src.core import llm_dispatcher as _disp_mod
    dispatcher_calls = []

    async def _boom(*a, **kw):  # pragma: no cover - must not run
        dispatcher_calls.append((a, kw))
        raise AssertionError("classify_signals called the dispatcher directly")

    if hasattr(_disp_mod, "LLMDispatcher"):
        monkeypatch.setattr(
            _disp_mod.LLMDispatcher, "execute", _boom, raising=False
        )

    from mr_roboto.executors.classify_signals import run as classify_run
    res = await classify_run({"id": 1, "payload": {}})

    assert res["ok"] is True
    assert res["enqueued"] is True
    assert res["pending"] == 2
    assert len(enqueued) == 1, "exactly one agent task enqueued"
    spec, kw = enqueued[0]
    assert spec["agent_type"] == "signal_classifier", \
        "must enqueue the signal_classifier agent"
    assert kw.get("on_complete") == "growth.classify_signals_complete"
    # Two signals handed to the agent.
    assert len(spec["context"]["payload"]["signals"]) == 2
    assert dispatcher_calls == [], "dispatcher must not be called"


@pytest.mark.asyncio
async def test_classify_signals_no_signals_skips_enqueue(tmp_path, monkeypatch):
    """No raw signals → no agent enqueued."""
    await _fresh_db(tmp_path, monkeypatch)

    import general_beckman
    enqueued = []

    async def _fake_enqueue(spec, **kw):
        enqueued.append(spec)
        return 1

    monkeypatch.setattr(general_beckman, "enqueue", _fake_enqueue)

    from mr_roboto.executors.classify_signals import run as classify_run
    res = await classify_run({"id": 1, "payload": {}})
    assert res["enqueued"] is False
    assert enqueued == []


@pytest.mark.asyncio
async def test_classifier_continuation_writes_classified_rows(
    tmp_path, monkeypatch
):
    """The on_complete handler persists agent verdicts as classified_signal."""
    await _fresh_db(tmp_path, monkeypatch)

    rs_id = await insert_growth_event(None, "raw_signal", {
        "provider": "intercom", "signal_type": "ticket",
        "content": "App crashes on upload", "external_id": "tk-9",
        "occurred_at": "2026-05-14 09:00:00",
    })

    # Build a child agent task with the ctx the executor would have set.
    from src.infra.db import add_task
    child_id = await add_task(
        title="classify growth signals",
        description="x",
        agent_type="signal_classifier",
        context=json.dumps({
            "payload": {
                "signals": [{
                    "raw_signal_id": rs_id, "external_id": "tk-9",
                    "signal_type": "ticket", "content": "App crashes on upload",
                }],
            },
            "growth_classify": {"mission_id": None},
        }),
    )

    from mr_roboto.executors.classify_signals import _on_classifier_complete
    await _on_classifier_complete(child_id, {
        "status": "completed",
        "result": {
            "classifications": [
                {"external_id": "tk-9", "label": "bug",
                 "domain": "file_upload", "confidence": 0.91},
            ],
        },
    })

    rows = await get_growth_events(kind="classified_signal")
    assert len(rows) == 1
    p = rows[0]["properties"]
    assert p["external_id"] == "tk-9"
    assert p["label"] == "bug"
    assert p["domain"] == "file_upload"
    assert p["raw_signal_id"] == rs_id
    assert "content_excerpt" in p


# ───────────────────────── T3C — scorer math ─────────────────────────────

def test_compute_score_formula_is_deterministic():
    """The score formula is pure arithmetic — same inputs, same output."""
    from mr_roboto.executors.score_backlog import compute_score
    s = compute_score(
        frequency=4, revenue_impact=0.9, north_star_relevance=0.8,
        age_decay=1.0, cost_band_weight=2,
    )
    # 4 * 0.9 * 0.8 * 1.0 / 2 = 1.44
    assert abs(s - 1.44) < 1e-9
    # Higher frequency → strictly higher score, all else equal.
    s2 = compute_score(
        frequency=8, revenue_impact=0.9, north_star_relevance=0.8,
        age_decay=1.0, cost_band_weight=2,
    )
    assert s2 > s


@pytest.mark.asyncio
async def test_score_backlog_orders_candidates_and_shows_formula(
    tmp_path, monkeypatch
):
    """Feed known classified signals → assert ranking + formula breakdown."""
    await _fresh_db(tmp_path, monkeypatch)

    # churn_signal cluster (high revenue impact) — 3 signals.
    for i in range(3):
        await insert_growth_event(None, "classified_signal", {
            "external_id": f"churn-{i}", "label": "churn_signal",
            "domain": "billing", "confidence": 0.9,
            "content_excerpt": "Cancelling, too expensive",
        })
    # praise cluster (~0 revenue impact) — 5 signals; should rank LAST
    # despite higher frequency.
    for i in range(5):
        await insert_growth_event(None, "classified_signal", {
            "external_id": f"praise-{i}", "label": "praise",
            "domain": "general", "confidence": 0.8,
            "content_excerpt": "Love the product",
        })
    # bug cluster — 2 signals.
    for i in range(2):
        await insert_growth_event(None, "classified_signal", {
            "external_id": f"bug-{i}", "label": "bug",
            "domain": "search", "confidence": 0.85,
            "content_excerpt": "Search returns nothing",
        })

    from mr_roboto.executors.score_backlog import run as score_run
    res = await score_run({"id": 1, "payload": {}})

    assert res["ok"] is True
    scored = res["scored"]
    assert len(scored) == 3
    # Scores strictly descending.
    assert scored[0]["score"] >= scored[1]["score"] >= scored[2]["score"]
    # praise must rank last (revenue_impact ~0).
    assert scored[-1]["label"] == "praise"
    # churn outranks praise even though praise is more frequent.
    labels = [c["label"] for c in scored]
    assert labels.index("churn_signal") < labels.index("praise")
    # Formula breakdown present + inspectable on every candidate.
    for c in scored:
        f = c["formula"]
        for key in ("frequency", "revenue_impact", "north_star_relevance",
                    "age_decay", "cost_band", "cost_band_weight",
                    "expression"):
            assert key in f, f"formula missing {key}"
        assert "=" in f["expression"]

    # Candidates persisted as backlog_candidate rows.
    cand_rows = await get_growth_events(kind="backlog_candidate")
    assert len(cand_rows) == 3


@pytest.mark.asyncio
async def test_score_backlog_supersedes_prior_candidates(tmp_path, monkeypatch):
    """Re-running the scorer rewrites — prior live candidates get superseded."""
    await _fresh_db(tmp_path, monkeypatch)
    await insert_growth_event(None, "classified_signal", {
        "external_id": "b-1", "label": "bug", "domain": "auth",
        "confidence": 0.9, "content_excerpt": "broken",
    })
    from mr_roboto.executors.score_backlog import run as score_run
    await score_run({"id": 1, "payload": {}})
    await score_run({"id": 2, "payload": {}})

    rows = await get_growth_events(kind="backlog_candidate")
    live = [r for r in rows
            if not r["properties"].get("superseded")
            and not r["properties"].get("consumed")]
    # Only the latest run's candidate is live.
    assert len(live) == 1


# ───────────────────────── /backlog + /approve ───────────────────────────

class _FakeMsg:
    def __init__(self):
        self.replies = []

        class _Chat:
            id = 12345
        self.chat = _Chat()

    async def reply_text(self, text, **kw):
        self.replies.append(text)
        return self


class _FakeUpdate:
    def __init__(self):
        self.message = _FakeMsg()

    @property
    def effective_chat(self):
        return self.message.chat


class _FakeCtx:
    def __init__(self, args=None):
        self.args = args or []


def _make_tg():
    """Build a TelegramInterface without running its __init__ network setup."""
    from src.app.telegram_bot import TelegramInterface
    tg = TelegramInterface.__new__(TelegramInterface)
    # _reply → _get_current_keyboard needs _kb_state; nothing else is touched.
    tg._kb_state = {}
    return tg


@pytest.mark.asyncio
async def test_backlog_command_lists_candidates(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    await insert_growth_event(None, "backlog_candidate", {
        "label": "bug", "domain": "auth", "score": 3.2, "frequency": 4,
        "formula": {"expression": "4 × 0.70 × 0.80 / 1 = 3.20"},
        "sample_excerpt": "login broken", "consumed": False,
    })
    await insert_growth_event(None, "backlog_candidate", {
        "label": "feature_request", "domain": "search", "score": 1.1,
        "frequency": 2,
        "formula": {"expression": "2 × 0.50 × 0.80 / 3 = 1.10"},
        "sample_excerpt": "want filters", "consumed": False,
    })

    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_backlog(update, _FakeCtx())

    assert update.message.replies, "cmd_backlog produced no reply"
    body = update.message.replies[0]
    assert "Growth backlog" in body
    assert "auth" in body and "search" in body
    assert "/approve" in body


@pytest.mark.asyncio
async def test_approve_spawns_exactly_one_mission(tmp_path, monkeypatch):
    """/approve <id> spawns ONE mission; never auto-spawns the others."""
    await _fresh_db(tmp_path, monkeypatch)

    cid_a = await insert_growth_event(None, "backlog_candidate", {
        "label": "bug", "domain": "auth", "score": 3.2, "frequency": 4,
        "formula": {"expression": "x"}, "sample_excerpt": "login broken",
        "consumed": False,
    })
    cid_b = await insert_growth_event(None, "backlog_candidate", {
        "label": "feature_request", "domain": "search", "score": 1.1,
        "frequency": 2, "formula": {"expression": "y"},
        "sample_excerpt": "want filters", "consumed": False,
    })

    # Spy on beckman.enqueue.
    import general_beckman
    enqueued = []

    async def _fake_enqueue(spec, **kw):
        enqueued.append(spec)
        return 7777

    monkeypatch.setattr(general_beckman, "enqueue", _fake_enqueue)

    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_approve(update, _FakeCtx(args=[str(cid_a)]))

    # Exactly one mission created.
    db = await get_db()
    cur = await db.execute("SELECT COUNT(*) FROM missions")
    (mission_count,) = await cur.fetchone()
    assert mission_count == 1, "approve must spawn exactly one mission"

    # Exactly one planner task enqueued — no auto-spawn for candidate B.
    assert len(enqueued) == 1
    assert enqueued[0]["agent_type"] == "planner"

    # Candidate A is consumed; candidate B untouched (NOT auto-approved).
    rows = await get_growth_events(kind="backlog_candidate")
    by_id = {r["id"]: r["properties"] for r in rows}
    assert by_id[cid_a].get("consumed") is True
    assert by_id[cid_a].get("approved_mission_id") is not None
    assert by_id[cid_b].get("consumed") is not True, \
        "candidate B must NOT be auto-approved"

    # A backlog_approved audit row exists.
    approved = await get_growth_events(kind="backlog_approved")
    assert len(approved) == 1
    assert approved[0]["properties"]["backlog_candidate_id"] == cid_a


@pytest.mark.asyncio
async def test_approve_twice_does_not_double_spawn(tmp_path, monkeypatch):
    """Re-approving a consumed candidate spawns no second mission."""
    await _fresh_db(tmp_path, monkeypatch)
    cid = await insert_growth_event(None, "backlog_candidate", {
        "label": "bug", "domain": "auth", "score": 2.0, "frequency": 3,
        "formula": {"expression": "x"}, "sample_excerpt": "broken",
        "consumed": False,
    })

    import general_beckman
    enqueued = []

    async def _fake_enqueue(spec, **kw):
        enqueued.append(spec)
        return 1

    monkeypatch.setattr(general_beckman, "enqueue", _fake_enqueue)

    tg = _make_tg()
    await tg.cmd_approve(_FakeUpdate(), _FakeCtx(args=[str(cid)]))
    await tg.cmd_approve(_FakeUpdate(), _FakeCtx(args=[str(cid)]))

    db = await get_db()
    cur = await db.execute("SELECT COUNT(*) FROM missions")
    (mission_count,) = await cur.fetchone()
    assert mission_count == 1, "second approve must not spawn another mission"
    assert len(enqueued) == 1
