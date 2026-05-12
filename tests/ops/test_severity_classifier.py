"""Z8 T3D — severity classifier rule coverage."""
from __future__ import annotations

import pytest

from src.ops.severity_classifier import classify


# ─────────────────────────── sentry ──────────────────────────────────────


def test_sentry_event_spike_is_critical():
    payload = {"event_count": 500, "timeframe_minutes": 5}
    assert classify("sentry", "issue_alert", payload) == "critical"


def test_sentry_below_spike_threshold_falls_through():
    payload = {"event_count": 50, "timeframe_minutes": 5, "affected_users": 0}
    assert classify("sentry", "issue_alert", payload) == "low"


def test_sentry_high_affected_users_is_high():
    payload = {"event_count": 1, "affected_users": 10}
    assert classify("sentry", "issue_alert", payload) == "high"


def test_sentry_regression_is_medium():
    payload = {"is_regression": True, "affected_users": 0}
    assert classify("sentry", "issue_alert", payload) == "medium"


def test_sentry_unknown_event_type_is_uncertain():
    assert classify("sentry", "i-made-this-up", {"x": 1}) == "uncertain"


# ─────────────────────────── stripe ──────────────────────────────────────


def test_stripe_live_payment_failure_is_critical():
    payload = {"data": {"object": {"livemode": True}}}
    assert (
        classify("stripe", "payment_intent.payment_failed", payload) == "critical"
    )


def test_stripe_test_payment_failure_falls_through_to_low():
    payload = {"data": {"object": {"livemode": False}}}
    assert (
        classify("stripe", "payment_intent.payment_failed", payload) == "low"
    )


def test_stripe_dispute_is_high():
    assert classify("stripe", "charge.dispute.created", {}) == "high"


def test_stripe_invoice_payment_failed_is_medium():
    assert classify("stripe", "invoice.payment_failed", {}) == "medium"


# ───────────────────────── betterstack ──────────────────────────────────


def test_betterstack_down_is_critical():
    payload = {"monitor": {"status": "down", "url": "https://example.com"}}
    assert classify("betterstack", "incident", payload) == "critical"


def test_betterstack_degraded_is_high():
    payload = {"monitor": {"status": "degraded"}}
    assert classify("betterstack", "incident", payload) == "high"


def test_betterstack_unknown_status_is_low():
    payload = {"monitor": {"status": "weird"}}
    assert classify("betterstack", "incident", payload) == "low"


def test_betterstack_missing_monitor_does_not_crash():
    assert classify("betterstack", "incident", {}) == "low"


# ─────────────────────────── github ──────────────────────────────────────


def test_github_critical_advisory():
    assert (
        classify(
            "github", "repository_advisory",
            {"advisory": {"severity": "critical"}},
        ) == "critical"
    )


def test_github_high_advisory():
    assert (
        classify(
            "github", "repository_advisory",
            {"advisory": {"severity": "high"}},
        ) == "high"
    )


# ────────────────────────── unknown vendor ──────────────────────────────


def test_unknown_integration_returns_uncertain():
    assert classify("not-a-vendor", "anything", {}) == "uncertain"


# ────────────────────────── executor wiring ─────────────────────────────


@pytest.mark.asyncio
async def test_alert_triage_executor_returns_severity():
    from mr_roboto.executors.alert_triage import run

    task = {
        "id": 1,
        "payload": {
            "integration_id": "betterstack",
            "event_id": "evt-1",
            "payload": {"type": "incident", "monitor": {"status": "down"}},
        },
    }
    res = await run(task)
    assert res["severity"] == "critical"
    assert res["integration_id"] == "betterstack"
    assert res["event_id"] == "evt-1"
    assert res["llm_graded"] is False


@pytest.mark.asyncio
async def test_alert_triage_executor_uncertain_falls_back_to_llm_stub():
    from mr_roboto.executors.alert_triage import run

    task = {
        "id": 2,
        "payload": {
            "integration_id": "sentry",
            "event_id": "evt-2",
            "payload": {"type": "totally-unknown"},
        },
    }
    res = await run(task)
    assert res["severity"] == "medium"  # T3 stub default
    assert res["llm_graded"] is True


@pytest.mark.asyncio
async def test_alert_triage_action_dispatches_via_mr_roboto():
    import mr_roboto

    task = {
        "id": 3,
        "mission_id": None,
        "payload": {
            "action": "alert_triage",
            "integration_id": "betterstack",
            "event_id": "evt-3",
            "payload": {"type": "incident", "monitor": {"status": "down"}},
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["severity"] == "critical"
