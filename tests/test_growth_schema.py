# tests/test_growth_schema.py
"""Z9 T1A — growth zone schema + CRUD helpers.

Covers the three new tables (hypotheses / experiment_variants /
growth_events) and the async helpers in src/infra/db.py.

Each test runs on a fresh temp-file SQLite DB with a fresh event loop
(project convention — see test_db_migration.py / feedback_test_timeouts).
"""
import asyncio
import os
import tempfile

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
    await db_mod.init_db()
    return db_mod, db_path


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def test_growth_tables_created_by_init_db():
    """init_db creates hypotheses / experiment_variants / growth_events."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            async with aiosqlite.connect(db_path) as db:
                for table in (
                    "hypotheses",
                    "experiment_variants",
                    "growth_events",
                ):
                    cur = await db.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name=?",
                        (table,),
                    )
                    row = await cur.fetchone()
                    assert row is not None, f"Table {table} not created"
        finally:
            await db_mod.close_db()

    run_async(_test())


def test_growth_tables_have_expected_columns():
    """Spot-check column sets so sibling tasks rely on the right shape."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute("PRAGMA table_info(hypotheses)")
                cols = {r[1] for r in await cur.fetchall()}
                for c in (
                    "id", "mission_id", "feature", "predicted_json",
                    "actual_json", "verdict", "window_seconds",
                    "measured_at", "dedup_key", "suppressed_until",
                    "created_at",
                ):
                    assert c in cols, f"hypotheses missing {c}"

                cur = await db.execute(
                    "PRAGMA table_info(experiment_variants)"
                )
                cols = {r[1] for r in await cur.fetchall()}
                for c in (
                    "id", "mission_id", "hypothesis_id", "variant_name",
                    "assignment_rule", "status", "shipped_at",
                    "retired_at", "created_at",
                ):
                    assert c in cols, f"experiment_variants missing {c}"

                cur = await db.execute("PRAGMA table_info(growth_events)")
                cols = {r[1] for r in await cur.fetchall()}
                for c in (
                    "id", "mission_id", "kind", "properties_json",
                    "segment", "occurred_at",
                ):
                    assert c in cols, f"growth_events missing {c}"
        finally:
            await db_mod.close_db()

    run_async(_test())


# ---------------------------------------------------------------------------
# hypotheses helpers
# ---------------------------------------------------------------------------

def test_insert_hypothesis_happy_path():
    """insert_hypothesis returns a row id and stores a pending row."""
    async def _test():
        db_mod, _ = await _fresh_db()
        try:
            hid = await db_mod.insert_hypothesis(
                mission_id=1,
                feature="checkout_redesign",
                predicted={"metric": "conversion", "direction": "up",
                           "magnitude": 0.12},
                window_seconds=1209600,
                dedup_key="checkout_redesign::conversion",
            )
            assert hid > 0
            pending = await db_mod.get_pending_hypotheses(mission_id=1)
            assert len(pending) == 1
            row = pending[0]
            assert row["verdict"] == "pending"
            assert row["feature"] == "checkout_redesign"
            assert row["predicted_json"]["magnitude"] == 0.12
            assert row["measured_at"] is None
        finally:
            await db_mod.close_db()

    run_async(_test())


def test_insert_hypothesis_suppression_refusal():
    """A dedup_key with future suppressed_until refuses re-insert (-1)."""
    async def _test():
        db_mod, _ = await _fresh_db()
        try:
            dk = "feature_x::retention"
            hid = await db_mod.insert_hypothesis(
                mission_id=2, feature="feature_x",
                predicted={"metric": "retention", "direction": "up",
                           "magnitude": 0.05},
                window_seconds=2592000, dedup_key=dk,
            )
            assert hid > 0
            # Refute it -> sets suppressed_until now+90d.
            await db_mod.record_hypothesis_verdict(
                hid,
                actual={"metric": "retention", "direction": "flat",
                        "magnitude": 0.0, "p_value": 0.4},
                verdict="refuted",
            )
            # Same dedup_key is now refused.
            again = await db_mod.insert_hypothesis(
                mission_id=2, feature="feature_x",
                predicted={"metric": "retention", "direction": "up",
                           "magnitude": 0.05},
                window_seconds=2592000, dedup_key=dk,
            )
            assert again == -1
        finally:
            await db_mod.close_db()

    run_async(_test())


def test_record_verdict_sets_measured_at_and_suppression():
    """refuted verdict stamps measured_at + suppressed_until; confirmed only measured_at."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            # Refuted hypothesis -> both fields set.
            hid_r = await db_mod.insert_hypothesis(
                mission_id=3, feature="banner",
                predicted={"metric": "ctr", "direction": "up",
                           "magnitude": 0.2},
                window_seconds=604800, dedup_key="banner::ctr",
            )
            await db_mod.record_hypothesis_verdict(
                hid_r,
                actual={"metric": "ctr", "direction": "down",
                        "magnitude": 0.1, "p_value": 0.02},
                verdict="refuted",
            )
            # Confirmed hypothesis -> measured_at set, no suppression.
            hid_c = await db_mod.insert_hypothesis(
                mission_id=3, feature="onboarding",
                predicted={"metric": "activation", "direction": "up",
                           "magnitude": 0.15},
                window_seconds=604800, dedup_key="onboarding::activation",
            )
            await db_mod.record_hypothesis_verdict(
                hid_c,
                actual={"metric": "activation", "direction": "up",
                        "magnitude": 0.17, "p_value": 0.01},
                verdict="confirmed",
            )

            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT verdict, measured_at, suppressed_until, "
                    "actual_json FROM hypotheses WHERE id=?",
                    (hid_r,),
                )
                verdict, measured_at, suppressed, actual = \
                    await cur.fetchone()
                assert verdict == "refuted"
                assert measured_at is not None
                assert "T" not in measured_at  # space-separated, not ISO
                assert suppressed is not None
                assert "T" not in suppressed
                assert '"p_value"' in actual

                cur = await db.execute(
                    "SELECT verdict, measured_at, suppressed_until "
                    "FROM hypotheses WHERE id=?",
                    (hid_c,),
                )
                verdict, measured_at, suppressed = await cur.fetchone()
                assert verdict == "confirmed"
                assert measured_at is not None
                assert suppressed is None
        finally:
            await db_mod.close_db()

    run_async(_test())


def test_get_pending_hypotheses_excludes_resolved():
    """get_pending_hypotheses only returns verdict='pending' rows."""
    async def _test():
        db_mod, _ = await _fresh_db()
        try:
            h1 = await db_mod.insert_hypothesis(
                mission_id=4, feature="a",
                predicted={"metric": "m", "direction": "up",
                           "magnitude": 0.1},
                window_seconds=600, dedup_key="a::m",
            )
            await db_mod.insert_hypothesis(
                mission_id=4, feature="b",
                predicted={"metric": "m2", "direction": "up",
                           "magnitude": 0.1},
                window_seconds=600, dedup_key="b::m2",
            )
            await db_mod.record_hypothesis_verdict(
                h1, actual={"p_value": 0.5}, verdict="inconclusive"
            )
            pending = await db_mod.get_pending_hypotheses(mission_id=4)
            assert len(pending) == 1
            assert pending[0]["feature"] == "b"
            # mission_id=None returns across all missions.
            all_pending = await db_mod.get_pending_hypotheses()
            assert len(all_pending) == 1
        finally:
            await db_mod.close_db()

    run_async(_test())


# ---------------------------------------------------------------------------
# growth_events helpers
# ---------------------------------------------------------------------------

def test_growth_events_insert_and_filter_by_kind():
    """insert_growth_event + get_growth_events filtered by kind."""
    async def _test():
        db_mod, _ = await _fresh_db()
        try:
            await db_mod.insert_growth_event(
                mission_id=5, kind="metric_emit",
                properties={"event": "signup_completed", "count": 3},
            )
            await db_mod.insert_growth_event(
                mission_id=5, kind="backlog_candidate",
                properties={"score": 8.2},
                segment="paid_users",
            )
            await db_mod.insert_growth_event(
                mission_id=5, kind="metric_emit",
                properties={"event": "checkout_started", "count": 1},
            )

            metric = await db_mod.get_growth_events(
                mission_id=5, kind="metric_emit"
            )
            assert len(metric) == 2
            assert all(e["kind"] == "metric_emit" for e in metric)
            assert all(
                isinstance(e["properties_json"], dict) for e in metric
            )

            backlog = await db_mod.get_growth_events(
                mission_id=5, kind="backlog_candidate"
            )
            assert len(backlog) == 1
            assert backlog[0]["segment"] == "paid_users"
            assert backlog[0]["properties_json"]["score"] == 8.2

            all_events = await db_mod.get_growth_events(mission_id=5)
            assert len(all_events) == 3
        finally:
            await db_mod.close_db()

    run_async(_test())


# ---------------------------------------------------------------------------
# experiment_variants helpers
# ---------------------------------------------------------------------------

def test_variant_insert_and_status_update_sets_retired_at():
    """insert_variant -> active; terminal status stamps retired_at."""
    async def _test():
        db_mod, db_path = await _fresh_db()
        try:
            hid = await db_mod.insert_hypothesis(
                mission_id=6, feature="pricing",
                predicted={"metric": "mrr", "direction": "up",
                           "magnitude": 0.1},
                window_seconds=1209600, dedup_key="pricing::mrr",
            )
            vid = await db_mod.insert_variant(
                mission_id=6, hypothesis_id=hid,
                variant_name="variant_b",
                assignment_rule="hash(user_id) % 2 == 1",
            )
            assert vid > 0

            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT status, retired_at FROM experiment_variants "
                    "WHERE id=?",
                    (vid,),
                )
                status, retired = await cur.fetchone()
                assert status == "active"
                assert retired is None

            # Terminal status -> retired_at stamped.
            await db_mod.update_variant_status(vid, "loser")
            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT status, retired_at FROM experiment_variants "
                    "WHERE id=?",
                    (vid,),
                )
                status, retired = await cur.fetchone()
                assert status == "loser"
                assert retired is not None
                assert "T" not in retired  # space-separated datetime
        finally:
            await db_mod.close_db()

    run_async(_test())
