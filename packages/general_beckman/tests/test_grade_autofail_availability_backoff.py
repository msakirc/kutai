"""Grade-reject branch must distinguish an AVAILABILITY auto-fail from a
genuine quality rejection.

mission_79 (2026-05-30/31, PART 2 of project_grader_verdict_autofail): when the
grade CHILD itself can't get a model (cloud daily-exhausted / no candidates),
posthook_continuations builds an AUTO-FAIL verdict
``{"passed": False, "raw": "auto-fail: grader call failed (No model candidates
available)"}``. The source artifact was NEVER judged — this is an availability
failure surfacing through the grade path, not a quality rejection.

The old grade-reject branch in ``_apply_posthook_verdict_locked`` hardcoded
``error_category="quality"`` for every grade FAIL, so an availability auto-fail
burned the quality-sized worker-attempt cap and fast-DLQ'd against an exhausted
pool — instead of riding the availability backoff ladder (to 24h, past a daily
quota reset) as the founder principle demands ("can't get capacity → WAIT").

Discrimination must be on the auto-fail SHAPE (``raw`` key + ``auto-fail:``
prefix) AND an availability marker — NOT a raw sniff of the grader's free-text
verdict, which can legitimately contain words like "daily" / "quota" / "rate
limit" in an insight about the artifact's content.
"""
from __future__ import annotations

import pytest


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


def _source(**over):
    row = {
        "id": 100,
        "status": "ungraded",
        "result": "some artifact draft",
        "context": "{}",
        "task_state": "{}",
        "worker_attempts": 2,
        "max_worker_attempts": 6,  # quality-sized cap — the fast-DLQ trap
        "title": "direct_competitor_identification",
        "mission_id": None,
        "model": "qwen3.5-9b",
    }
    row.update(over)
    return row


# ── pure discriminator ──────────────────────────────────────────────────────

def test_autofail_no_model_classified_availability():
    from general_beckman.apply import _grade_verdict_is_availability
    from general_beckman.result_router import PostHookVerdict

    v = PostHookVerdict(
        source_task_id=100, kind="grade", passed=False,
        raw={"passed": False,
             "raw": "auto-fail: grader call failed (No model candidates available)"},
    )
    assert _grade_verdict_is_availability(v) is True


def test_genuine_quality_verdict_not_availability():
    from general_beckman.apply import _grade_verdict_is_availability
    from general_beckman.result_router import PostHookVerdict

    # Insight prose mentions 'daily' + 'quota' — a raw text sniff would
    # false-positive; the auto-fail SHAPE guard must not.
    v = PostHookVerdict(
        source_task_id=100, kind="grade", passed=False,
        raw={"passed": False, "complete": False,
             "insight": "missing required daily quota disclosure section"},
    )
    assert _grade_verdict_is_availability(v) is False


def test_autofail_but_not_availability_not_classified():
    # An auto-fail that is NOT an availability reason (e.g. parser gave up on a
    # malformed verdict) must stay on the quality path.
    from general_beckman.apply import _grade_verdict_is_availability
    from general_beckman.result_router import PostHookVerdict

    v = PostHookVerdict(
        source_task_id=100, kind="grade", passed=False,
        raw={"passed": False,
             "raw": "auto-fail: grader call failed (malformed verdict json)"},
    )
    assert _grade_verdict_is_availability(v) is False


# ── integration through the locked apply path ───────────────────────────────

@pytest.mark.asyncio
async def test_availability_autofail_backs_off_not_quality_dlq(monkeypatch):
    import general_beckman.apply as apply_mod
    from general_beckman.result_router import PostHookVerdict

    store = {100: _source(worker_attempts=2)}
    _install_fake_db(monkeypatch, store)

    verdict = PostHookVerdict(
        source_task_id=100, kind="grade", passed=False,
        raw={"passed": False,
             "raw": "auto-fail: grader call failed (No model candidates available)"},
    )
    await apply_mod._apply_posthook_verdict(store[100], verdict)

    row = store[100]
    # Rode the availability path, NOT the quality fast-DLQ.
    assert row["error_category"] == "availability"
    # Re-pended to wait for capacity (attempts well under the 15-step ladder).
    assert row["status"] == "pending"


@pytest.mark.asyncio
async def test_genuine_quality_grade_reject_stays_quality(monkeypatch):
    import general_beckman.apply as apply_mod
    from general_beckman.result_router import PostHookVerdict

    store = {100: _source(worker_attempts=1)}
    _install_fake_db(monkeypatch, store)

    verdict = PostHookVerdict(
        source_task_id=100, kind="grade", passed=False,
        raw={"passed": False, "complete": False,
             "insight": "missing required daily quota disclosure section"},
    )
    await apply_mod._apply_posthook_verdict(store[100], verdict)

    row = store[100]
    assert row["error_category"] == "quality"
    assert row["status"] == "pending"
