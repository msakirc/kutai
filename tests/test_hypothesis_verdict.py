# tests/test_hypothesis_verdict.py
"""Z9 T4C/T4D/T4E — hypothesis verdict pipeline.

Covers:
  - the Bayesian verdict helper (src/growth/verdict_stats.py)
  - the T4C verdict_window_sweep due-detection + enqueue
  - the T4D record_verdict executor (confirmed / refuted / inconclusive,
    mock-mode metric pull, mission_lessons mirror, refuted suppression)
  - the T4E reinforce nudge (+0.05, 50%/30d decay)

The whole verdict pipeline is mechanical — these tests assert no
LLMDispatcher.request call is made.

Each test runs on a fresh temp-file SQLite DB with a fresh event loop
(project convention — see test_growth_schema.py / feedback_test_timeouts).
"""
import asyncio
import datetime
import os
import tempfile
from unittest.mock import patch

import aiosqlite


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _fresh_db():
    """Create a fresh DB in a temp dir, resetting module state."""
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    import src.infra.db as db_mod

    db_mod.DB_PATH = db_path
    db_mod._db_connection = None
    os.environ["DB_PATH"] = db_path
    await db_mod.init_db()
    return db_mod, db_path


async def _close_db(db_mod):
    """Close the singleton connection so the temp DB file is released."""
    try:
        conn = getattr(db_mod, "_db_connection", None)
        if conn is not None:
            await conn.close()
        db_mod._db_connection = None
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Bayesian verdict helper
# ---------------------------------------------------------------------------

def test_verdict_confirmed_when_prediction_holds():
    """A measured lift at/above the predicted magnitude → confirmed."""
    from src.growth.verdict_stats import compute_verdict

    # predicted +12%, measured +13% with tight sigma → confident hold.
    vr = compute_verdict(
        baseline=100.0, actual=113.0,
        direction="up", magnitude=0.12, rel_sigma=0.02,
    )
    assert vr.verdict == "confirmed"
    assert vr.p_held >= 0.95
    assert vr.observed_lift > 0


def test_verdict_refuted_when_metric_moves_opposite():
    """A metric moving the opposite way with confidence → refuted."""
    from src.growth.verdict_stats import compute_verdict

    # predicted +12%, measured -10% → opposite direction, confident.
    vr = compute_verdict(
        baseline=100.0, actual=90.0,
        direction="up", magnitude=0.12, rel_sigma=0.02,
    )
    assert vr.verdict == "refuted"
    assert vr.p_opposite >= 0.95


def test_verdict_inconclusive_when_noisy():
    """A small move under wide sigma → inconclusive."""
    from src.growth.verdict_stats import compute_verdict

    # predicted +12%, measured +6% (half), but loose sigma → uncertain.
    vr = compute_verdict(
        baseline=100.0, actual=106.0,
        direction="up", magnitude=0.12, rel_sigma=0.20,
    )
    assert vr.verdict == "inconclusive"
    assert vr.p_held < 0.95
    assert vr.p_opposite < 0.95


def test_verdict_down_direction():
    """A predicted decrease that lands is confirmed."""
    from src.growth.verdict_stats import compute_verdict

    # predict latency down 20%, measured down 25%.
    vr = compute_verdict(
        baseline=200.0, actual=150.0,
        direction="down", magnitude=0.20, rel_sigma=0.02,
    )
    assert vr.verdict == "confirmed"
    assert vr.predicted_lift < 0


def test_verdict_percentage_magnitude_normalised():
    """magnitude > 1 is read as a percentage."""
    from src.growth.verdict_stats import compute_verdict

    vr = compute_verdict(
        baseline=100.0, actual=113.0,
        direction="up", magnitude=12, rel_sigma=0.02,  # 12 → 0.12
    )
    assert abs(vr.predicted_lift - 0.12) < 1e-9
    assert vr.verdict == "confirmed"


# ---------------------------------------------------------------------------
# T4C — verdict_window_sweep due detection
# ---------------------------------------------------------------------------

def test_sweep_is_due_detection():
    """is_due() finds closed windows, skips not-yet-due ones."""
    from mr_roboto.executors.verdict_window_sweep import is_due

    now = datetime.datetime(2026, 5, 15, 12, 0, 0)

    # created 10 days ago, 7-day window → due.
    due_hyp = {
        "created_at": "2026-05-05 12:00:00",
        "window_seconds": 7 * 86400,
    }
    assert is_due(due_hyp, now) is True

    # created 1 day ago, 7-day window → NOT due.
    fresh_hyp = {
        "created_at": "2026-05-14 12:00:00",
        "window_seconds": 7 * 86400,
    }
    assert is_due(fresh_hyp, now) is False

    # missing window → not due.
    assert is_due({"created_at": "2026-05-05 12:00:00"}, now) is False
    # missing created_at → not due.
    assert is_due({"window_seconds": 86400}, now) is False
    # non-positive window → not due.
    assert is_due(
        {"created_at": "2026-05-01 12:00:00", "window_seconds": 0}, now
    ) is False


def test_sweep_enqueues_only_due_hypotheses():
    """run() enqueues a verdict task only for hypotheses past their window."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            # one due (created far in the past), one fresh.
            due_id = await db_mod.insert_hypothesis(
                mission_id=1, feature="checkout",
                predicted={"metric": "conversion", "direction": "up",
                           "magnitude": 0.1},
                window_seconds=1, dedup_key="checkout+conversion",
            )
            fresh_id = await db_mod.insert_hypothesis(
                mission_id=1, feature="onboarding",
                predicted={"metric": "activation", "direction": "up",
                           "magnitude": 0.1},
                window_seconds=10 ** 9,  # ~31 years → never due
                dedup_key="onboarding+activation",
            )
            assert due_id > 0 and fresh_id > 0

            from mr_roboto.executors import verdict_window_sweep as sweep

            enqueued = []

            async def fake_enqueue(spec, **kwargs):
                enqueued.append((spec, kwargs))
                return 999

            with patch.object(sweep, "is_due") as mock_due:
                # only the due_id row reports due
                mock_due.side_effect = lambda h, now=None: int(
                    h.get("id") or 0
                ) == due_id
                with patch(
                    "general_beckman.enqueue", side_effect=fake_enqueue
                ):
                    res = await sweep.run({})

            assert res["ok"] is True
            assert res["due"] == 1
            assert res["enqueued"] == 1
            # exactly one verdict task, on the ongoing lane, for due_id.
            assert len(enqueued) == 1
            spec, kwargs = enqueued[0]
            assert kwargs.get("lane") == "ongoing"
            assert (
                spec["context"]["payload"]["action"] == "record_verdict"
            )
            assert spec["context"]["payload"]["hypothesis_id"] == due_id
        finally:
            await _close_db(db_mod)

    run_async(_test())


# ---------------------------------------------------------------------------
# T4D — record_verdict executor
# ---------------------------------------------------------------------------

def _mock_posthog(series):
    """Return a fake _posthog_metric coroutine yielding `series`."""
    async def _fake(task, metric):
        return {"ok": True, "series": list(series)}
    return _fake


def test_record_verdict_confirmed():
    """record_verdict on a holding prediction → confirmed verdict + event."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            hyp_id = await db_mod.insert_hypothesis(
                mission_id=5, feature="new_checkout",
                predicted={"metric": "conversion", "direction": "up",
                           "magnitude": 0.10, "baseline": 100.0},
                window_seconds=1, dedup_key="new_checkout+conversion",
            )
            from mr_roboto.executors import record_verdict as rv

            # measured series ends at 122 → +22% lift vs baseline 100,
            # comfortably past the +5% credit line under default sigma.
            with patch.object(rv, "_posthog_metric",
                              _mock_posthog([100, 110, 122])):
                res = await rv.run(
                    {"payload": {"hypothesis_id": hyp_id}}
                )

            assert res["ok"] is True
            assert res["verdict"] == "confirmed"

            # hypothesis row updated, measured_at stamped.
            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT verdict, measured_at FROM hypotheses WHERE id=?",
                    (hyp_id,),
                )
                row = await cur.fetchone()
            assert row[0] == "confirmed"
            assert row[1] is not None

            # verdict growth_event written.
            events = await db_mod.get_growth_events(kind="verdict")
            assert len(events) == 1
            assert events[0]["properties"]["verdict"] == "confirmed"

            # confirmed → NOT mirrored to mission_lessons.
            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT COUNT(*) FROM mission_lessons "
                    "WHERE source_kind='hypothesis_verdict'"
                )
                assert (await cur.fetchone())[0] == 0
        finally:
            await _close_db(db_mod)

    run_async(_test())


def test_record_verdict_refuted_mirrors_lesson_and_suppresses():
    """A refuted verdict mirrors a mission_lesson and sets 90d suppression."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            hyp_id = await db_mod.insert_hypothesis(
                mission_id=7, feature="upsell_banner",
                predicted={"metric": "revenue", "direction": "up",
                           "magnitude": 0.15, "baseline": 100.0},
                window_seconds=1, dedup_key="upsell_banner+revenue",
            )
            from mr_roboto.executors import record_verdict as rv

            # measured ends at 85 → -15%, opposite of predicted +15%.
            with patch.object(rv, "_posthog_metric",
                              _mock_posthog([100, 92, 85])):
                res = await rv.run(
                    {"payload": {"hypothesis_id": hyp_id}}
                )

            assert res["verdict"] == "refuted"

            # suppression set on the hypothesis row (now + 90d).
            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT verdict, suppressed_until FROM hypotheses "
                    "WHERE id=?", (hyp_id,),
                )
                row = await cur.fetchone()
            assert row[0] == "refuted"
            assert row[1] is not None  # suppressed_until populated

            # mission_lesson mirrored with source_kind='hypothesis_verdict'.
            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT pattern, severity FROM mission_lessons "
                    "WHERE source_kind='hypothesis_verdict'"
                )
                lessons = await cur.fetchall()
            assert len(lessons) == 1
            assert "refuted" in lessons[0][0]
        finally:
            await _close_db(db_mod)

    run_async(_test())


def test_record_verdict_inconclusive_mirrors_lesson():
    """An inconclusive verdict mirrors an info-severity lesson."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            hyp_id = await db_mod.insert_hypothesis(
                mission_id=9, feature="tooltip_hints",
                predicted={"metric": "activation", "direction": "up",
                           "magnitude": 0.30, "baseline": 100.0},
                window_seconds=1, dedup_key="tooltip_hints+activation",
            )
            from mr_roboto.executors import record_verdict as rv

            # tiny ambiguous move → inconclusive.
            with patch.object(rv, "_posthog_metric",
                              _mock_posthog([100, 100.5, 101])):
                res = await rv.run(
                    {"payload": {"hypothesis_id": hyp_id}}
                )

            assert res["verdict"] == "inconclusive"
            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT severity FROM mission_lessons "
                    "WHERE source_kind='hypothesis_verdict'"
                )
                rows = await cur.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "info"
        finally:
            await _close_db(db_mod)

    run_async(_test())


def test_record_verdict_mock_mode_metric_pull():
    """Without a posthog patch, the real mock-mode vendor_call is used."""
    async def _test():
        os.environ["KUTAI_ENV"] = "test"  # mock mode on
        db_mod, db_path = await _fresh_db()
        try:
            hyp_id = await db_mod.insert_hypothesis(
                mission_id=11, feature="search_v2",
                predicted={"metric": "metric", "direction": "up",
                           "magnitude": 0.05},
                window_seconds=1, dedup_key="search_v2+metric",
            )
            from mr_roboto.executors import record_verdict as rv

            res = await rv.run({"payload": {"hypothesis_id": hyp_id}})
            # mock get_insight returns data [10,12,15,14]; baseline 10,
            # actual 14 → +40% lift. Pipeline must produce a real verdict.
            assert res["ok"] is True
            assert res["verdict"] in (
                "confirmed", "refuted", "inconclusive"
            )
            assert res["baseline"] == 10.0
            assert res["actual"] == 14.0
        finally:
            await _close_db(db_mod)

    run_async(_test())


def test_record_verdict_skips_non_pending():
    """A hypothesis already measured is a safe no-op (sweeper idempotency)."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            hyp_id = await db_mod.insert_hypothesis(
                mission_id=1, feature="x",
                predicted={"metric": "m", "direction": "up",
                           "magnitude": 0.1},
                window_seconds=1, dedup_key="x+m",
            )
            await db_mod.record_hypothesis_verdict(
                hyp_id, {"actual": 1}, "confirmed"
            )
            from mr_roboto.executors import record_verdict as rv

            res = await rv.run({"payload": {"hypothesis_id": hyp_id}})
            assert res["ok"] is True
            assert res.get("skipped") is True
        finally:
            await _close_db(db_mod)

    run_async(_test())


# ---------------------------------------------------------------------------
# T4E — reinforce nudge
# ---------------------------------------------------------------------------

def test_reinforce_nudge_writes_row():
    """record_reinforce_nudge writes a reinforce-tagged model_pick_log row."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            await db_mod.record_reinforce_nudge(
                "qwen2.5-coder", hypothesis_id=42,
            )
            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT picked_model, call_category, reinforce "
                    "FROM model_pick_log WHERE call_category='reinforce'"
                )
                rows = await cur.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "qwen2.5-coder"
            assert rows[0][1] == "reinforce"
            assert abs(rows[0][2] - 0.05) < 1e-9  # +0.05 founder-decided
        finally:
            await _close_db(db_mod)

    run_async(_test())


def test_reinforce_bonus_decays_over_time():
    """reinforce_bonus halves roughly every 30 days (50%/30d decay)."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            from src.infra.db import get_db

            db = await get_db()
            now = datetime.datetime.now()
            fresh = now.strftime("%Y-%m-%d %H:%M:%S")
            old = (now - datetime.timedelta(days=30)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            # fresh win for model A, 30-day-old win for model B.
            for model, ts in (("model_A", fresh), ("model_B", old)):
                await db.execute(
                    "INSERT INTO model_pick_log "
                    "(task_name, picked_model, picked_score, call_category, "
                    " candidates_json, reinforce, timestamp) "
                    "VALUES (?, ?, 0, 'reinforce', '[]', 0.05, ?)",
                    ("hypothesis_verdict", model, ts),
                )
            await db.commit()

            from fatih_hoca.grading import reinforce_bonus

            bonus_fresh = reinforce_bonus("model_A")
            bonus_old = reinforce_bonus("model_B")

            # fresh nudge: 0.05 * 1.0 * 20 = 1.0 perf point.
            assert abs(bonus_fresh - 1.0) < 0.05
            # 30-day-old nudge: decayed by ~0.5 → ~0.5 perf point.
            assert abs(bonus_old - 0.5) < 0.05
            assert bonus_old < bonus_fresh
            # unknown model → no bonus.
            assert reinforce_bonus("never_picked") == 0.0
        finally:
            await _close_db(db_mod)

    run_async(_test())


def test_confirmed_verdict_fires_reinforce():
    """A confirmed verdict writes a reinforce row for the mission's model."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            from src.infra.db import get_db, add_mission

            mid = await add_mission(
                title="Reinforce mission", description="d",
            )
            # a prior model pick logged against a task of this mission.
            db = await get_db()
            await db.execute(
                "INSERT INTO tasks (title, description, agent_type, "
                "mission_id, status) VALUES (?, ?, ?, ?, 'completed')",
                ("build feature", "d", "coder", mid),
            )
            await db.execute(
                "INSERT INTO model_pick_log "
                "(task_name, picked_model, picked_score, call_category, "
                " candidates_json, provider) "
                "VALUES (?, ?, 80.0, 'main_work', '[]', 'local')",
                ("build feature", "winner-model"),
            )
            await db.commit()

            hyp_id = await db_mod.insert_hypothesis(
                mission_id=mid, feature="build feature",
                predicted={"metric": "conversion", "direction": "up",
                           "magnitude": 0.10, "baseline": 100.0},
                window_seconds=1, dedup_key="bf+conversion",
            )
            from mr_roboto.executors import record_verdict as rv

            with patch.object(rv, "_posthog_metric",
                              _mock_posthog([100, 108, 114])):
                res = await rv.run({"payload": {"hypothesis_id": hyp_id}})

            assert res["verdict"] == "confirmed"
            assert res["reinforced_model"] == "winner-model"

            async with aiosqlite.connect(db_path) as db2:
                cur = await db2.execute(
                    "SELECT picked_model, reinforce FROM model_pick_log "
                    "WHERE call_category='reinforce'"
                )
                rows = await cur.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "winner-model"
            assert abs(rows[0][1] - 0.05) < 1e-9
        finally:
            await _close_db(db_mod)

    run_async(_test())


# ---------------------------------------------------------------------------
# Architecture invariant — verdict pipeline is mechanical (no LLM)
# ---------------------------------------------------------------------------

def test_verdict_pipeline_makes_no_dispatcher_call():
    """record_verdict + verdict_window_sweep never call LLMDispatcher.execute."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            hyp_id = await db_mod.insert_hypothesis(
                mission_id=1, feature="x",
                predicted={"metric": "m", "direction": "up",
                           "magnitude": 0.1, "baseline": 100.0},
                window_seconds=1, dedup_key="x+m",
            )
            from mr_roboto.executors import record_verdict as rv
            from mr_roboto.executors import verdict_window_sweep as sweep

            calls = []

            async def boom(*a, **k):
                calls.append((a, k))
                raise AssertionError("verdict pipeline must not call dispatcher")

            from src.core import llm_dispatcher

            with patch.object(
                llm_dispatcher.LLMDispatcher, "execute", side_effect=boom
            ):
                with patch.object(rv, "_posthog_metric",
                                  _mock_posthog([100, 110])):
                    await rv.run({"payload": {"hypothesis_id": hyp_id}})

                async def fake_enqueue(spec, **kwargs):
                    return 1

                with patch(
                    "general_beckman.enqueue", side_effect=fake_enqueue
                ):
                    await sweep.run({})

            assert calls == []  # dispatcher never invoked
        finally:
            await _close_db(db_mod)

    run_async(_test())
