# tests/test_hypothesis_northstar.py
"""Z9 T4 — record_hypothesis verb + inject_north_star verb.

T4A record_hypothesis: per-metric window resolution, dedup/suppression,
non-measurable skip path, low-confidence best-effort extraction.
T4B inject_north_star: context injection, idempotency, graceful no-artifact.

Each test runs on a fresh temp-file SQLite DB with a fresh event loop
(project convention — see test_growth_schema.py).
"""
import asyncio
import json
import os
import tempfile


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _fresh_db():
    """Create a fresh DB in a temp dir, resetting module state.

    Also resets the ArtifactStore singleton — its in-memory cache keys on
    mission_id, and AUTOINCREMENT restarts at 1 on every fresh DB, so a
    stale cached artifact from a prior test would otherwise leak in.
    """
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    import src.infra.db as db_mod

    db_mod.DB_PATH = db_path
    db_mod._db_connection = None
    await db_mod.init_db()
    try:
        import src.workflows.engine.hooks as hooks_mod

        hooks_mod._artifact_store = None
    except Exception:
        pass
    return db_mod, db_path


async def _make_mission(db_mod, title, description, context=None):
    # add_mission json.dumps() the context itself — pass the raw dict.
    mid = await db_mod.add_mission(
        title=title,
        description=description,
        context=context or {},
    )
    return mid


# ===========================================================================
# T4A — record_hypothesis
# ===========================================================================

# ── window-table resolution per metric type ───────────────────────────────

def test_window_table_resolution():
    """_resolve_window returns the per-metric default window."""
    from mr_roboto.executors.record_hypothesis import _resolve_window

    _DAY = 86400
    assert _resolve_window("activation") == 7 * _DAY
    assert _resolve_window("acquisition") == 7 * _DAY
    assert _resolve_window("retention") == 30 * _DAY
    assert _resolve_window("revenue") == 14 * _DAY
    assert _resolve_window("referral") == 14 * _DAY
    assert _resolve_window("latency") == 3 * _DAY
    assert _resolve_window("error_rate") == 3 * _DAY
    # Unrecognised metric → 14d default.
    assert _resolve_window("bananas") == 14 * _DAY
    assert _resolve_window(None) == 14 * _DAY


def test_extract_prediction_conversion():
    """Spec stating 'checkout conversion +12%' parses metric+direction+mag."""
    from mr_roboto.executors.record_hypothesis import _extract_prediction

    pred = _extract_prediction("We expect checkout conversion +12% after this.")
    assert pred["metric"] == "revenue"
    assert pred["direction"] == "up"
    assert pred["magnitude"] == 12.0
    assert pred["confidence"] == "high"


def test_extract_prediction_latency_negative():
    """'p95 latency -200ms' parses as latency / down / 200."""
    from mr_roboto.executors.record_hypothesis import _extract_prediction

    pred = _extract_prediction("This should reduce p95 latency -200ms.")
    assert pred["metric"] == "latency"
    assert pred["direction"] == "down"
    assert pred["magnitude"] == 200.0


def test_record_hypothesis_revenue_window():
    """A revenue mission records a pending hypothesis with a 14d window."""
    async def _test():
        db_mod, _ = await _fresh_db()
        try:
            mid = await _make_mission(
                db_mod,
                "Checkout redesign",
                "Goal: lift checkout conversion +15%.",
            )
            from mr_roboto.executors.record_hypothesis import run

            res = await run({"mission_id": mid, "payload": {}})
            assert res["ok"] is True
            assert res["recorded"] is True
            assert res["metric"] == "revenue"
            assert res["window_seconds"] == 14 * 86400
            pending = await db_mod.get_pending_hypotheses(mission_id=mid)
            assert len(pending) == 1
            assert pending[0]["window_seconds"] == 14 * 86400
            assert pending[0]["predicted_json"]["direction"] == "up"
        finally:
            await db_mod.close_db()

    run_async(_test())


def test_record_hypothesis_latency_window():
    """A latency mission resolves the 3d window."""
    async def _test():
        db_mod, _ = await _fresh_db()
        try:
            mid = await _make_mission(
                db_mod,
                "Perf pass",
                "Cut p95 latency -150ms on the search endpoint.",
            )
            from mr_roboto.executors.record_hypothesis import run

            res = await run({"mission_id": mid, "payload": {}})
            assert res["recorded"] is True
            assert res["metric"] == "latency"
            assert res["window_seconds"] == 3 * 86400
        finally:
            await db_mod.close_db()

    run_async(_test())


def test_record_hypothesis_skip_non_measurable():
    """A pure refactor mission is flagged (skipped) but does not error."""
    async def _test():
        db_mod, _ = await _fresh_db()
        try:
            mid = await _make_mission(
                db_mod,
                "Refactor auth module",
                "Pure refactor — rename helpers, clean up dead code. "
                "No behaviour change.",
            )
            from mr_roboto.executors.record_hypothesis import run

            res = await run({"mission_id": mid, "payload": {}})
            assert res["ok"] is True
            assert res["recorded"] is False
            assert res["skipped"] is True
            # No hypotheses row, but a hypothesis_skipped growth_event.
            pending = await db_mod.get_pending_hypotheses(mission_id=mid)
            assert pending == []
            events = await db_mod.get_growth_events(
                mission_id=mid, kind="hypothesis_skipped"
            )
            assert len(events) == 1
        finally:
            await db_mod.close_db()

    run_async(_test())


def test_record_hypothesis_dedup_suppression():
    """A suppressed dedup_key (refuted cool-off) is not re-recorded."""
    async def _test():
        db_mod, _ = await _fresh_db()
        try:
            mid = await _make_mission(
                db_mod,
                "Checkout redesign",
                "Lift checkout conversion +12%.",
            )
            from mr_roboto.executors.record_hypothesis import run

            # First record — succeeds.
            res1 = await run({"mission_id": mid, "payload": {}})
            assert res1["recorded"] is True
            hyp_id = res1["hypothesis_id"]
            dedup_key = res1["dedup_key"]

            # Refute it → 90-day suppression on the dedup_key.
            await db_mod.record_hypothesis_verdict(
                hyp_id, {"actual": "flat"}, "refuted"
            )

            # Re-record same mission — dedup_key suppressed → not recorded.
            mid2 = await _make_mission(
                db_mod,
                "Checkout redesign",
                "Lift checkout conversion +12%.",
            )
            res2 = await run({"mission_id": mid2, "payload": {}})
            assert res2["ok"] is True
            assert res2["recorded"] is False
            assert res2["suppressed"] is True
            assert res2["dedup_key"] == dedup_key
            events = await db_mod.get_growth_events(
                mission_id=mid2, kind="hypothesis_suppressed"
            )
            assert len(events) == 1
        finally:
            await db_mod.close_db()

    run_async(_test())


def test_record_hypothesis_low_confidence_best_effort():
    """A measurable-domain spec with no magnitude records at low confidence."""
    async def _test():
        db_mod, _ = await _fresh_db()
        try:
            mid = await _make_mission(
                db_mod,
                "Improve retention",
                "We want better retention for new signups somehow.",
            )
            from mr_roboto.executors.record_hypothesis import run

            res = await run({"mission_id": mid, "payload": {}})
            assert res["ok"] is True
            assert res["recorded"] is True
            assert res["low_confidence"] is True
            assert res["metric"] == "retention"
            assert res["window_seconds"] == 30 * 86400
        finally:
            await db_mod.close_db()

    run_async(_test())


def test_record_hypothesis_payload_override():
    """A payload-supplied window override is honoured."""
    async def _test():
        db_mod, _ = await _fresh_db()
        try:
            mid = await _make_mission(
                db_mod, "Checkout", "Lift checkout conversion +10%."
            )
            from mr_roboto.executors.record_hypothesis import run

            res = await run(
                {"mission_id": mid, "payload": {"window_seconds": 99999}}
            )
            assert res["recorded"] is True
            assert res["window_seconds"] == 99999
        finally:
            await db_mod.close_db()

    run_async(_test())


# ===========================================================================
# T4B — inject_north_star
# ===========================================================================

_SUCCESS_METRICS = {
    "north_star_metric": {
        "name": "Weekly Active Creators",
        "justification": "Captures both activation and retention.",
    },
    "aarrr_metrics": [
        {
            "name": "signup_completed",
            "formula": "count(signups)",
            "data_source": "posthog",
            "target_value": 100,
            "measurement_frequency": "daily",
        },
        {
            "name": "day7_retention",
            "formula": "retained / cohort",
            "data_source": "posthog",
            "target_value": 0.4,
            "measurement_frequency": "weekly",
        },
    ],
}


async def _stash_success_metrics(mission_id, metrics):
    """Put a success_metrics artifact into the ArtifactStore singleton cache."""
    from src.workflows.engine.hooks import get_artifact_store

    store = get_artifact_store()
    await store.store(
        mission_id, "success_metrics", json.dumps(metrics)
    )
    return store


def test_inject_north_star_injects_context():
    """inject_north_star merges north_star + aarrr into mission.context."""
    async def _test():
        db_mod, _ = await _fresh_db()
        try:
            mid = await _make_mission(db_mod, "Build product", "desc")
            await _stash_success_metrics(mid, _SUCCESS_METRICS)

            from mr_roboto.executors.inject_north_star import inject_north_star

            res = await inject_north_star(mid)
            assert res["ok"] is True
            assert res["injected"] is True
            assert res["north_star"] == "Weekly Active Creators"
            assert res["aarrr_count"] == 2

            mission = await db_mod.get_mission(mid)
            ctx = json.loads(mission["context"])
            ns = ctx["north_star"]
            assert ns["north_star_metric"]["name"] == "Weekly Active Creators"
            assert len(ns["aarrr_metrics"]) == 2
            assert ns["aarrr_metrics"][0]["name"] == "signup_completed"
        finally:
            await db_mod.close_db()

    run_async(_test())


def test_inject_north_star_idempotent():
    """Second injection with the same artifact is a no-op write (unchanged)."""
    async def _test():
        db_mod, _ = await _fresh_db()
        try:
            mid = await _make_mission(db_mod, "Build product", "desc")
            await _stash_success_metrics(mid, _SUCCESS_METRICS)

            from mr_roboto.executors.inject_north_star import inject_north_star

            res1 = await inject_north_star(mid)
            assert res1["injected"] is True
            mission1 = await db_mod.get_mission(mid)

            res2 = await inject_north_star(mid)
            assert res2["injected"] is True
            assert res2.get("unchanged") is True
            mission2 = await db_mod.get_mission(mid)
            # Context byte-identical — idempotent.
            assert mission1["context"] == mission2["context"]
        finally:
            await db_mod.close_db()

    run_async(_test())


def test_inject_north_star_graceful_no_artifact():
    """No success_metrics artifact → ok with injected=False, no crash."""
    async def _test():
        db_mod, _ = await _fresh_db()
        try:
            # Create enough missions that this one's id is unique across
            # the file — the ArtifactStore singleton + blackboard both key
            # on mission_id and AUTOINCREMENT restarts at 1 per fresh DB.
            for _ in range(50):
                await _make_mission(db_mod, "filler", "filler")
            mid = await _make_mission(db_mod, "Build product", "desc")
            # Deliberately do NOT stash a success_metrics artifact.
            from mr_roboto.executors.inject_north_star import inject_north_star

            res = await inject_north_star(mid)
            assert res["ok"] is True
            assert res["injected"] is False
            assert res["north_star"] is None
            assert res["aarrr_count"] == 0
        finally:
            await db_mod.close_db()

    run_async(_test())


def test_inject_north_star_run_requires_mission_id():
    """The run(task) entry point rejects a missing mission_id."""
    async def _test():
        from mr_roboto.executors.inject_north_star import run

        res = await run({"payload": {}})
        assert res["ok"] is False
        assert "mission_id" in res["error"]

    run_async(_test())
