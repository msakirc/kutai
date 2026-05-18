"""Z9 Growth T5C — feature sunset scorer + roadmap/north-star sync tests.

Covers:
  * score_sunset deterministic math — low-usage + cost feature → candidate;
    high-usage feature → no candidate; threshold boundary behaviour.
  * the weekly sunset_score_recompute + roadmap_northstar_sync cron rows
    are registered in INTERNAL_CADENCES.
  * /sunset lists live candidates with the usage/cost breakdown.
  * /approve_sunset spawns EXACTLY one deprecation mission and never
    auto-executes (no auto-spawn for other candidates).
  * roadmap_sync writes a northstar_review row when the north-star is
    flat / untracked / undefined.
"""
import pytest

import src.infra.db as _db_mod
from src.infra.db import (
    init_db, get_db, insert_growth_event, get_growth_events,
)


async def _fresh_db(tmp_path, monkeypatch):
    """Reset DB to a fresh temp file for isolation."""
    db_file = tmp_path / "sunset.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    await init_db()


def _feature_event(feature, user_id, *, domain="general", cost_band=None):
    """A growth_events properties payload that the scorer counts as usage."""
    props = {"feature": feature, "user_id": user_id, "domain": domain}
    if cost_band:
        props["cost_band"] = cost_band
    return props


# ───────────────────────── pure scoring math ─────────────────────────────

def test_compute_sunset_score_pure():
    from mr_roboto.executors.score_sunset import compute_sunset_score

    # Zero usage, heavy cost → max score (1.0 × 3).
    assert compute_sunset_score(0.0, 3) == pytest.approx(3.0)
    # Full usage → zero score regardless of cost.
    assert compute_sunset_score(1.0, 3) == pytest.approx(0.0)
    # Half usage, moderate cost → 0.5 × 2.
    assert compute_sunset_score(0.5, 2) == pytest.approx(1.0)
    # Over-100% usage cannot drive the score negative.
    assert compute_sunset_score(1.5, 3) == pytest.approx(0.0)


def test_is_sunset_candidate_threshold_boundary():
    from mr_roboto.executors.score_sunset import is_sunset_candidate

    thr = 0.01
    # Strictly below threshold + non-zero cost → candidate.
    assert is_sunset_candidate(0.005, 1, thr) is True
    # Exactly AT threshold → NOT a candidate (strict <).
    assert is_sunset_candidate(0.01, 3, thr) is False
    # Above threshold → not a candidate even with heavy cost.
    assert is_sunset_candidate(0.5, 3, thr) is False


# ───────────────────────── scorer executor ───────────────────────────────

@pytest.mark.asyncio
async def test_low_usage_high_cost_feature_becomes_candidate(
    tmp_path, monkeypatch
):
    """A feature touched by <1% of active users but with non-zero cost is a
    sunset candidate; a heavily-used feature is not."""
    await _fresh_db(tmp_path, monkeypatch)

    # 200 active users exercise the popular feature.
    for uid in range(200):
        await insert_growth_event(
            None, "feature_used",
            _feature_event("dashboard", f"u{uid}", domain="analytics"),
        )
    # The legacy feature is touched by exactly one of those 200 users
    # (0.5% < 1% threshold) and lives in a heavy domain.
    await insert_growth_event(
        None, "feature_used",
        _feature_event("legacy_export", "u0", domain="billing"),
    )

    from mr_roboto.executors.score_sunset import run

    res = await run({"payload": {}})
    assert res["ok"] is True
    assert res["features"] == 2

    candidates = await get_growth_events(kind="sunset_candidate")
    cand_features = {
        (c["properties"] or {})["feature"] for c in candidates
    }
    assert "legacy_export" in cand_features, \
        "low-usage heavy-cost feature must be flagged"
    assert "dashboard" not in cand_features, \
        "heavily-used feature must NOT be flagged"

    legacy = next(
        c for c in candidates
        if (c["properties"] or {})["feature"] == "legacy_export"
    )
    p = legacy["properties"]
    assert p["usage_rate"] < 0.01
    assert p["cost_band"] == "heavy"
    assert p["sunset_score"] > 0
    assert "why" in p and p["why"]
    assert "expression" in (p.get("formula") or {})


@pytest.mark.asyncio
async def test_high_usage_feature_produces_no_candidate(tmp_path, monkeypatch):
    """When every feature is well-used, the scorer writes nothing."""
    await _fresh_db(tmp_path, monkeypatch)
    for uid in range(20):
        await insert_growth_event(
            None, "feature_used", _feature_event("core", f"u{uid}"),
        )

    from mr_roboto.executors.score_sunset import run

    res = await run({"payload": {}})
    assert res["ok"] is True
    assert res["candidates"] == 0
    assert await get_growth_events(kind="sunset_candidate") == []


@pytest.mark.asyncio
async def test_founder_threshold_override(tmp_path, monkeypatch):
    """The 1% threshold is founder-overridable via payload.usage_threshold."""
    await _fresh_db(tmp_path, monkeypatch)
    # 10 active users; 'niche' touched by 1 → 10% usage.
    for uid in range(10):
        await insert_growth_event(
            None, "feature_used", _feature_event("main", f"u{uid}"),
        )
    await insert_growth_event(
        None, "feature_used",
        _feature_event("niche", "u0", domain="billing"),
    )

    from mr_roboto.executors.score_sunset import run

    # At the default 1% threshold, 10% usage is NOT low → no candidate.
    res = await run({"payload": {}})
    assert res["candidates"] == 0

    # Founder raises the threshold to 20% → 10% now counts as low usage.
    res = await run({"payload": {"usage_threshold": 0.20}})
    feats = {
        (c["properties"] or {})["feature"]
        for c in await get_growth_events(kind="sunset_candidate")
        if not (c["properties"] or {}).get("superseded")
    }
    assert "niche" in feats


@pytest.mark.asyncio
async def test_scorer_supersedes_prior_candidates(tmp_path, monkeypatch):
    """Re-running the scorer rewrites rather than accumulates."""
    await _fresh_db(tmp_path, monkeypatch)
    # 150 active users → a feature touched by one of them is 0.67% < 1%.
    for uid in range(150):
        await insert_growth_event(
            None, "feature_used", _feature_event("popular", f"u{uid}"),
        )
    await insert_growth_event(
        None, "feature_used",
        _feature_event("dead", "u0", domain="billing"),
    )

    from mr_roboto.executors.score_sunset import run

    await run({"payload": {}})
    await run({"payload": {}})
    rows = await get_growth_events(kind="sunset_candidate")
    live = [r for r in rows if not (r["properties"] or {}).get("superseded")]
    assert len(live) == 1, "only the latest scoring run is live"


# ───────────────────────── cron registration ─────────────────────────────

def test_weekly_sunset_cron_registered():
    """sunset_score_recompute + roadmap_northstar_sync are weekly cadences."""
    from general_beckman.cron_seed import INTERNAL_CADENCES

    by_title = {c["title"]: c for c in INTERNAL_CADENCES}

    sunset = by_title.get("sunset_score_recompute")
    assert sunset is not None, "sunset_score_recompute cron missing"
    assert sunset["interval_seconds"] == 604800, "must be weekly"
    assert sunset["payload"]["_executor"] == "score_sunset"

    roadmap = by_title.get("roadmap_northstar_sync")
    assert roadmap is not None, "roadmap_northstar_sync cron missing"
    assert roadmap["interval_seconds"] == 604800, "must be weekly"
    assert roadmap["payload"]["_executor"] == "roadmap_sync"


# ───────────────────────── Telegram surface ──────────────────────────────

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
    from src.app.telegram_bot import TelegramInterface
    tg = TelegramInterface.__new__(TelegramInterface)
    tg._kb_state = {}
    return tg


@pytest.mark.asyncio
async def test_sunset_command_lists_candidates(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    await insert_growth_event(None, "sunset_candidate", {
        "feature": "legacy_export", "domain": "billing",
        "usage_rate": 0.004, "distinct_users": 1, "active_users": 250,
        "cost_band": "heavy", "cost_band_weight": 3, "sunset_score": 2.99,
        "why": "1/250 users touched 'legacy_export' in 30d; cost heavy",
        "formula": {"expression": "(1 - 0.004) × 3 = 2.99"},
        "consumed": False,
    })

    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_sunset(update, _FakeCtx())

    assert update.message.replies, "cmd_sunset produced no reply"
    body = update.message.replies[0]
    assert "sunset" in body.lower()
    assert "legacy_export" in body
    assert "/approve_sunset" in body


@pytest.mark.asyncio
async def test_approve_sunset_spawns_exactly_one_mission(
    tmp_path, monkeypatch
):
    """/approve_sunset <id> spawns ONE deprecation mission; never
    auto-executes and never touches the other candidate."""
    await _fresh_db(tmp_path, monkeypatch)

    cid_a = await insert_growth_event(None, "sunset_candidate", {
        "feature": "legacy_export", "domain": "billing",
        "usage_rate": 0.004, "distinct_users": 1, "active_users": 250,
        "cost_band": "heavy", "cost_band_weight": 3, "sunset_score": 2.99,
        "why": "barely used", "formula": {"expression": "x"},
        "consumed": False,
    })
    cid_b = await insert_growth_event(None, "sunset_candidate", {
        "feature": "old_widget", "domain": "general",
        "usage_rate": 0.001, "distinct_users": 1, "active_users": 250,
        "cost_band": "cheap", "cost_band_weight": 1, "sunset_score": 0.99,
        "why": "barely used", "formula": {"expression": "y"},
        "consumed": False,
    })

    # Spy on beckman.enqueue — assert no auto-execution path.
    import general_beckman
    enqueued = []

    async def _fake_enqueue(spec, **kw):
        enqueued.append(spec)
        return 8888

    monkeypatch.setattr(general_beckman, "enqueue", _fake_enqueue)

    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_approve_sunset(update, _FakeCtx(args=[str(cid_a)]))

    # Exactly ONE deprecation mission created.
    db = await get_db()
    cur = await db.execute("SELECT COUNT(*) FROM missions")
    (mission_count,) = await cur.fetchone()
    assert mission_count == 1, "approve_sunset must spawn exactly one mission"

    # Exactly one planner task enqueued — the mission is PLANNED, not
    # auto-executed; candidate B is NOT auto-spawned.
    assert len(enqueued) == 1
    assert enqueued[0]["agent_type"] == "planner", \
        "deprecation goes through planning — never auto-executed"

    # Candidate A consumed + linked to its mission; B untouched.
    rows = await get_growth_events(kind="sunset_candidate")
    by_id = {r["id"]: r["properties"] for r in rows}
    assert by_id[cid_a].get("consumed") is True
    assert by_id[cid_a].get("approved_mission_id") is not None
    assert by_id[cid_b].get("consumed") is not True, \
        "other candidates must NOT be auto-approved"

    # Audit row written.
    approved = await get_growth_events(kind="sunset_approved")
    assert len(approved) == 1


@pytest.mark.asyncio
async def test_approve_sunset_rejects_double_approval(tmp_path, monkeypatch):
    """A consumed candidate cannot be approved twice."""
    await _fresh_db(tmp_path, monkeypatch)
    cid = await insert_growth_event(None, "sunset_candidate", {
        "feature": "legacy_export", "domain": "billing",
        "usage_rate": 0.004, "cost_band": "heavy", "cost_band_weight": 3,
        "sunset_score": 2.99, "why": "x", "consumed": True,
        "approved_mission_id": 99,
    })

    import general_beckman
    enqueued = []

    async def _fake_enqueue(spec, **kw):
        enqueued.append(spec)
        return 1

    monkeypatch.setattr(general_beckman, "enqueue", _fake_enqueue)

    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_approve_sunset(update, _FakeCtx(args=[str(cid)]))

    assert enqueued == [], "already-consumed candidate must not re-spawn"
    body = update.message.replies[0]
    assert "already" in body.lower()


# ───────────────────────── roadmap / north-star sync ─────────────────────

def test_assess_north_star_flat_metric():
    """A north-star whose readings have not moved is flagged flat."""
    from mr_roboto.executors.roadmap_sync import assess_north_star

    sm = {"north_star_metric": {"name": "weekly_active_users"}}
    flat_rows = [
        {"properties": {"metric": "weekly_active_users", "value": 100}},
        {"properties": {"metric": "weekly_active_users", "value": 100}},
        {"properties": {"metric": "weekly_active_users", "value": 100}},
    ]
    verdict = assess_north_star(sm, flat_rows)
    assert verdict["stale"] is True
    assert verdict["status"] == "flat"


def test_assess_north_star_undefined_and_untracked():
    from mr_roboto.executors.roadmap_sync import assess_north_star

    # No north-star declared at all.
    assert assess_north_star({}, [])["status"] == "undefined"

    # Declared but nothing measures it.
    sm = {"north_star_metric": {"name": "activation_rate"}}
    assert assess_north_star(sm, [])["status"] == "untracked"


def test_assess_north_star_current_when_moving():
    from mr_roboto.executors.roadmap_sync import assess_north_star

    sm = {"north_star_metric": {"name": "weekly_active_users"}}
    moving = [
        {"properties": {"metric": "weekly_active_users", "value": 100}},
        {"properties": {"metric": "weekly_active_users", "value": 130}},
        {"properties": {"metric": "weekly_active_users", "value": 165}},
    ]
    verdict = assess_north_star(sm, moving)
    assert verdict["stale"] is False
    assert verdict["status"] == "current"


@pytest.mark.asyncio
async def test_roadmap_sync_fires_review_on_flat_metric(tmp_path, monkeypatch):
    """roadmap_sync writes a northstar_review row when the metric is flat."""
    await _fresh_db(tmp_path, monkeypatch)

    # Stub the success_metrics artifact retrieval.
    import mr_roboto.executors.roadmap_sync as _rs

    async def _fake_load(mission_id):
        return {"north_star_metric": {"name": "weekly_active_users"}}

    monkeypatch.setattr(_rs, "_load_success_metrics", _fake_load)

    # Three identical (flat) readings of the north-star metric.
    for _ in range(3):
        await insert_growth_event(7, "metric_snapshot", {
            "metric": "weekly_active_users", "value": 100,
        })

    res = await _rs.run({"mission_id": 7})
    assert res["ok"] is True
    assert res["stale"] is True
    assert res["status"] == "flat"

    reviews = await get_growth_events(kind="northstar_review")
    assert len(reviews) == 1
    assert reviews[0]["properties"]["metric"] == "weekly_active_users"


@pytest.mark.asyncio
async def test_roadmap_sync_quiet_when_metric_healthy(tmp_path, monkeypatch):
    """A moving north-star writes NO northstar_review row."""
    await _fresh_db(tmp_path, monkeypatch)

    import mr_roboto.executors.roadmap_sync as _rs

    async def _fake_load(mission_id):
        return {"north_star_metric": {"name": "weekly_active_users"}}

    monkeypatch.setattr(_rs, "_load_success_metrics", _fake_load)

    for val in (100, 140, 190):
        await insert_growth_event(7, "metric_snapshot", {
            "metric": "weekly_active_users", "value": val,
        })

    res = await _rs.run({"mission_id": 7})
    assert res["stale"] is False
    assert await get_growth_events(kind="northstar_review") == []
