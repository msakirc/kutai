"""Z9 T2 — weekly analytics digest pipeline tests.

Covers:
* the three growth anti-pattern detectors (positive + negative cases)
* the ``analytics_digest`` mechanical executor's data pull (mock-mode PostHog)
* that the executor enqueues an *agent* task via Beckman rather than calling
  the LLM dispatcher directly (mechanical→Beckman→agent split)
"""
from __future__ import annotations

import asyncio
import os

import pytest

# Mock mode must be on so the PostHog vendor_call returns deterministic fakes.
os.environ.setdefault("KUTAI_ENV", "test")
os.environ.pop("KUTAI_VENDOR_LIVE", None)

from src.growth.anti_patterns import (
    MIN_EXPERIMENT_N,
    detect_all,
    detect_engagement_vampire,
    detect_insufficient_n,
    detect_vanity_metric,
)


# ─────────────────────────────────────────────────────────────────────────
# T2D — anti-pattern detectors
# ─────────────────────────────────────────────────────────────────────────


class TestVanityMetric:
    def test_dau_absolute_flagged(self):
        f = detect_vanity_metric({"name": "Daily Active Users"})
        assert f is not None
        assert f.code == "vanity_metric"
        assert f.severity == "warn"

    def test_page_views_flagged(self):
        assert detect_vanity_metric({"name": "Total Page Views"}) is not None

    def test_total_signups_flagged(self):
        assert detect_vanity_metric({"name": "total signups"}) is not None

    def test_revenue_metric_not_flagged(self):
        assert detect_vanity_metric({"name": "Weekly Paying Retention"}) is None

    def test_dau_mau_ratio_exempt(self):
        # DAU/MAU ratio is a retention metric — must NOT be flagged.
        assert detect_vanity_metric({"name": "DAU/MAU stickiness ratio"}) is None

    def test_missing_or_empty(self):
        assert detect_vanity_metric(None) is None
        assert detect_vanity_metric({}) is None
        assert detect_vanity_metric({"name": ""}) is None


class TestEngagementVampire:
    def test_high_volume_declining_retention_flagged(self):
        # 1200 events + a declining retention curve.
        f = detect_engagement_vampire(1200, [100, 60, 40, 28, 20, 15])
        assert f is not None
        assert f.code == "engagement_vampire"

    def test_high_volume_flat_retention_flagged(self):
        f = detect_engagement_vampire(900, [100, 50, 50, 50, 50])
        assert f is not None

    def test_high_volume_growing_retention_not_flagged(self):
        # Retention improving day over day — healthy.
        assert detect_engagement_vampire(2000, [100, 40, 50, 62, 75]) is None

    def test_low_volume_not_flagged(self):
        # Below the event-count floor — not a vampire even if retention dips.
        assert detect_engagement_vampire(50, [100, 60, 40, 20]) is None

    def test_short_curve_not_flagged(self):
        assert detect_engagement_vampire(2000, [100, 40]) is None

    def test_missing_curve(self):
        assert detect_engagement_vampire(2000, None) is None


class TestInsufficientN:
    def test_under_powered_experiment_flagged(self):
        findings = detect_insufficient_n(
            [{"name": "checkout_v2", "daily_active_samples": 40}]
        )
        assert len(findings) == 1
        assert findings[0].code == "insufficient_n"

    def test_powered_experiment_not_flagged(self):
        findings = detect_insufficient_n(
            [{"name": "checkout_v2", "daily_active_samples": MIN_EXPERIMENT_N}]
        )
        assert findings == []

    def test_alternate_sample_keys(self):
        # 'samples' and 'n' are also accepted sample-count keys.
        assert len(detect_insufficient_n([{"name": "a", "samples": 10}])) == 1
        assert len(detect_insufficient_n([{"name": "b", "n": 5}])) == 1

    def test_mixed_batch(self):
        findings = detect_insufficient_n(
            [
                {"name": "weak", "daily_active_samples": 12},
                {"name": "strong", "daily_active_samples": 500},
                {"name": "also_weak", "daily_active_samples": 99},
            ]
        )
        assert {f.detail["experiment"] for f in findings} == {"weak", "also_weak"}

    def test_empty_or_missing(self):
        assert detect_insufficient_n(None) == []
        assert detect_insufficient_n([]) == []


class TestDetectAll:
    def test_clean_digest_no_findings(self):
        digest = {
            "north_star": {"name": "Weekly Paying Retention"},
            "event_count": 100,
            "retention_curve": [100, 50, 55, 60],
            "experiments": [{"name": "e", "daily_active_samples": 300}],
        }
        assert detect_all(digest) == []

    def test_all_three_fire_together(self):
        digest = {
            "north_star": {"name": "Daily Active Users"},
            "event_count": 5000,
            "retention_curve": [100, 50, 40, 30, 20],
            "experiments": [{"name": "underpowered", "daily_active_samples": 10}],
        }
        codes = {f["code"] for f in detect_all(digest)}
        assert codes == {"vanity_metric", "engagement_vampire", "insufficient_n"}

    def test_findings_are_dicts(self):
        out = detect_all({"north_star": {"name": "page_views"}})
        assert isinstance(out, list)
        assert all(isinstance(f, dict) for f in out)
        assert all({"code", "severity", "message", "detail"} <= set(f) for f in out)


# ─────────────────────────────────────────────────────────────────────────
# T2B — analytics_digest mechanical executor: data pull + Beckman hand-off
# ─────────────────────────────────────────────────────────────────────────


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_executor_pulls_posthog_in_mock_mode(monkeypatch):
    """The data pull uses mock-mode PostHog — deterministic offline fakes."""
    from src.integrations.registry import get_integration_registry

    registry = get_integration_registry()
    monkeypatch.setattr(registry, "mock_mode", True)

    from mr_roboto.executors import analytics_digest as ad

    # Stub the LLM hand-off + DB writes so we exercise the pull in isolation.
    enqueued: list = []

    async def _fake_enqueue(mission_id, task_id, digest_input):
        enqueued.append(digest_input)
        return 9999

    async def _fake_db_agg(mission_id, since):
        return {
            "growth_events": [],
            "pending_hypotheses": [],
            "mission_lessons": [],
            "model_pick": [],
            "retry_stats": {},
            "recipe_pin_rate": None,
        }

    async def _fake_success_metrics(mission_id):
        return {"north_star_metric": {"name": "Weekly Retention"}}

    async def _fake_insert_event(*a, **k):
        return 1

    monkeypatch.setattr(ad, "_enqueue_synthesis_agent", _fake_enqueue)
    monkeypatch.setattr(ad, "_pull_db_aggregates", _fake_db_agg)
    monkeypatch.setattr(ad, "_load_success_metrics", _fake_success_metrics)
    import src.infra.db as _db

    monkeypatch.setattr(_db, "insert_growth_event", _fake_insert_event)

    res = _run(ad.run({"id": 1, "mission_id": 42, "payload": {"action": "analytics_digest"}}))

    assert res["ok"] is True
    # PostHog mock returns 3 query_events rows → event_count == 3.
    assert res["event_count"] == 3
    assert res["posthog_ok"] is True
    di = res["digest_input"]
    # Funnel + retention curve came through the mock.
    assert len(di["funnel"]) == 3
    assert di["retention_curve"] == [50, 31, 22, 18, 15, 14, 13]
    assert di["north_star"] == {"name": "Weekly Retention"}
    # The synthesis agent received the same bundle.
    assert enqueued and enqueued[0] is di


def test_executor_enqueues_agent_not_dispatcher(monkeypatch):
    """Mechanical→Beckman→agent: the executor enqueues an *agent* task via
    ``general_beckman.enqueue`` and NEVER calls the LLM dispatcher."""
    import general_beckman as gb
    from mr_roboto.executors import analytics_digest as ad

    captured: dict = {}

    async def _fake_enqueue(spec, *, parent_id=None, on_complete=None, **kw):
        captured["spec"] = spec
        captured["on_complete"] = on_complete
        return 7777

    monkeypatch.setattr(gb, "enqueue", _fake_enqueue)

    # If the dispatcher is touched the test fails loudly.
    import src.core.llm_dispatcher as _disp

    async def _boom(*a, **k):  # noqa: ANN001
        raise AssertionError("mechanical executor must NOT call the dispatcher")

    if hasattr(_disp, "LLMDispatcher"):
        monkeypatch.setattr(
            _disp.LLMDispatcher, "request", _boom, raising=False
        )

    new_id = _run(
        ad._enqueue_synthesis_agent(42, 1, {"mission_id": 42, "north_star": {}})
    )

    assert new_id == 7777
    spec = captured["spec"]
    # Enqueued task is an LLM AGENT task — agent_type is the synthesis agent,
    # not "mechanical".
    assert spec["agent_type"] == "growth_digest_synthesizer"
    assert spec["agent_type"] != "mechanical"
    # digest_input rides on the task context for the agent to read.
    assert "digest_input" in spec["context"]
    # on_complete chains the weekly_digest persistence continuation.
    assert captured["on_complete"] == ad._DIGEST_CONTINUATION


def test_store_weekly_digest_continuation(monkeypatch):
    """The on_complete continuation persists the agent's markdown as a
    ``growth_events`` row kind='weekly_digest'."""
    from mr_roboto.executors import analytics_digest as ad
    import src.infra.db as _db

    written: list = []

    async def _fake_insert(mission_id, kind, properties, segment=None):
        written.append({"mission_id": mission_id, "kind": kind, "props": properties})
        return 1

    class _FakeCur:
        async def fetchone(self):
            return (42,)

    class _FakeDB:
        async def execute(self, *a, **k):
            return _FakeCur()

    async def _fake_get_db():
        return _FakeDB()

    monkeypatch.setattr(_db, "insert_growth_event", _fake_insert)
    monkeypatch.setattr(_db, "get_db", _fake_get_db)

    _run(
        ad._store_weekly_digest(
            123, {"status": "completed", "result": "## Weekly Growth Digest\n\nbody"}
        )
    )

    assert len(written) == 1
    assert written[0]["kind"] == "weekly_digest"
    assert written[0]["mission_id"] == 42
    assert "Weekly Growth Digest" in written[0]["props"]["markdown"]


def test_anti_patterns_tool_wrapper():
    """The agent tool wrapper parses a JSON digest_input and narrates."""
    import json

    from src.tools.growth_anti_patterns import growth_anti_patterns

    digest = {"north_star": {"name": "Daily Active Users"}}
    out = _run(growth_anti_patterns(json.dumps(digest)))
    assert "vanity_metric" in out

    clean = _run(growth_anti_patterns(json.dumps({"north_star": {"name": "MRR"}})))
    assert "no anti-patterns" in clean.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
