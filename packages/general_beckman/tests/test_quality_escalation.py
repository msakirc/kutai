"""Every QUALITY failure must follow the retry-escalation path (2026-06-04).

`src.core.retry.get_model_constraints` (read by fatih_hoca at worker_attempts>=3)
escalates two ways: a difficulty bump (keyed on attempt count) AND a model
EXCLUSION (keyed on ctx['failed_models']). Before this fix ONLY the grade-FAIL
branch populated failed_models, so every other quality re-pend could re-draw the
exact model that just produced bad output. Two gaps closed:

  1. `_stamp_retry_feedback` — the single chokepoint every quality re-pend calls
     — now records the failing model, so all of them escalate symmetrically.
  2. `prior_art_min_coverage` (a Z1 blocker that judges LLM PRODUCER output, not a
     deterministic on-disk artifact) is routed off the single-shot-DLQ Z1 rail
     onto the retry-with-escalation rail, so a stronger model gets a shot.
"""
from __future__ import annotations

import json

import pytest


# ── pure helper ──────────────────────────────────────────────────────────────

def test_record_failed_model_adds_generating_model():
    from general_beckman.apply import _record_failed_model
    ctx = {"generating_model": "weak-9b", "failed_models": []}
    _record_failed_model(ctx)
    assert ctx["failed_models"] == ["weak-9b"]


def test_record_failed_model_idempotent():
    from general_beckman.apply import _record_failed_model
    ctx = {"generating_model": "weak-9b", "failed_models": ["weak-9b"]}
    _record_failed_model(ctx)
    assert ctx["failed_models"] == ["weak-9b"]  # no dup


def test_record_failed_model_noop_without_model():
    from general_beckman.apply import _record_failed_model
    ctx = {"failed_models": []}
    _record_failed_model(ctx)
    assert ctx.get("failed_models") == []


def test_stamp_retry_feedback_records_model():
    """The universal chokepoint escalates the model for ANY quality path."""
    from general_beckman.apply import _stamp_retry_feedback
    ctx = {"generating_model": "gemma-31b", "_schema_error": "bad"}
    _stamp_retry_feedback(ctx, next_attempt=2)
    assert "gemma-31b" in ctx["failed_models"]


# ── integration through the locked apply path ───────────────────────────────

def _install_fake_db(monkeypatch, store: dict):
    async def fake_get_task(tid):
        return store.get(int(tid))

    async def fake_update_task(tid, **kwargs):
        row = store.get(int(tid))
        if row is not None:
            row.update(kwargs)
        return None

    monkeypatch.setattr("src.infra.db.get_task", fake_get_task)
    monkeypatch.setattr("src.infra.db.update_task", fake_update_task)


def _source(ctx: dict, **over):
    row = {
        "id": 700,
        "status": "ungraded",
        "result": "some produced artifact",
        "context": json.dumps(ctx),
        "task_state": "{}",
        "worker_attempts": 1,          # < cap → normal retry branch (not DLQ)
        "max_worker_attempts": 15,
        "title": "prior_art_synthesize",
        "mission_id": None,            # avoid workflow-advance side effects
        "model": "weak-9b",
    }
    row.update(over)
    return row


@pytest.mark.asyncio
async def test_prior_art_coverage_fail_retries_and_escalates(monkeypatch):
    """A producer-quality Z1 blocker fail must RE-PEND (not single-shot DLQ) AND
    record the fabricating model for escalation."""
    import general_beckman.apply as apply_mod
    from general_beckman.result_router import PostHookVerdict

    ctx = {
        "generating_model": "weak-9b",
        "failed_models": [],
        "_pending_posthooks": ["prior_art_min_coverage"],
    }
    store = {700: _source(ctx)}
    _install_fake_db(monkeypatch, store)

    verdict = PostHookVerdict(
        source_task_id=700, kind="prior_art_min_coverage", passed=False,
        raw={"ok": False,
             "problems": ["unverifiable dead/dormant solutions (no Wayback + no HN ref): ['Habitica', 'Streaks']"],
             "verdict": "graveyard_thin"},
    )
    await apply_mod._apply_posthook_verdict(store[700], verdict)

    row = store[700]
    # Re-pended for another (escalated) attempt — NOT DLQ'd on first fail.
    assert row["status"] == "pending"
    # The fabricating model is excluded on the next pick (escalation arm).
    new_ctx = json.loads(row["context"])
    assert "weak-9b" in (new_ctx.get("failed_models") or [])


@pytest.mark.asyncio
async def test_shape_check_fail_escalates_model(monkeypatch):
    """A parameterized shape `checks` fail (simple_blocker rail) also escalates."""
    import general_beckman.apply as apply_mod
    from general_beckman.result_router import PostHookVerdict

    ctx = {
        "generating_model": "weak-9b",
        "failed_models": [],
        "_pending_posthooks": ["verify_charter_shape"],
        "checks": [{"kind": "verify_charter_shape", "payload": {}}],
    }
    store = {700: _source(ctx, title="product_charter")}
    _install_fake_db(monkeypatch, store)

    verdict = PostHookVerdict(
        source_task_id=700, kind="verify_charter_shape", passed=False,
        raw={"ok": False, "problems": [{"why": "only 2 solution blocks (min 3)"}]},
    )
    await apply_mod._apply_posthook_verdict(store[700], verdict)

    row = store[700]
    assert row["status"] == "pending"
    new_ctx = json.loads(row["context"])
    assert "weak-9b" in (new_ctx.get("failed_models") or [])
