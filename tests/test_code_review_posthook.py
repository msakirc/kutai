"""Coverage for the code_review post-hook end-to-end:
- determine_posthooks accepts code_review in ctx.post_hooks
- determine_posthooks suppresses the implicit grade for reviewer tasks (SP3)
- _apply_code_review_verdict pass/fail behaviour (unchanged by SP3)
- parse_code_review_response extracts VERDICT + bullet issues

SP3 note: the code_review post-hook no longer spawns a code_reviewer agent
task nor flows through rewrite Rule 0. It is intercepted in
_apply_request_posthook and enqueued as a raw_dispatch "reviewer" child whose
verdict is applied via the posthook.code_review.resume continuation. That CPS
spawn path is covered by tests/beckman/test_posthook_spawn_cps.py and
tests/core/test_build_code_review_spec.py.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from general_beckman.posthooks import determine_posthooks
from general_beckman.apply import (
    _posthook_title,
    _apply_code_review_verdict,
)
from general_beckman.result_router import (
    RequestPostHook, PostHookVerdict,
)
from src.core.code_review import (
    parse_code_review_response, CodeReviewResult,
)


# ── parse_code_review_response ───────────────────────────────────────────

def test_parse_pass_with_no_issues():
    raw = "ISSUES:\n- NONE\n\nVERDICT: PASS\n"
    res = parse_code_review_response(raw)
    assert res.passed is True
    assert res.issues == []


def test_parse_fail_with_bullet_issues():
    raw = (
        "ISSUES:\n"
        "- backend/app/main.py:7: hardcoded SECRET_KEY default\n"
        "- backend/app/routes.py: missing auth middleware\n"
        "\nVERDICT: FAIL\n"
    )
    res = parse_code_review_response(raw)
    assert res.passed is False
    assert len(res.issues) == 2
    assert "SECRET_KEY" in res.issues[0]
    assert "auth middleware" in res.issues[1]


def test_parse_bare_pass_keyword_fallback():
    raw = "Looked at code. PASS"
    res = parse_code_review_response(raw)
    assert res.passed is True


def test_parse_ambiguous_returns_fail_closed():
    raw = "Some prose with no verdict."
    res = parse_code_review_response(raw)
    assert res.passed is False


def test_parse_empty_returns_fail():
    res = parse_code_review_response("")
    assert res.passed is False
    res = parse_code_review_response("a")
    assert res.passed is False


def test_parse_fail_overrides_when_both_keywords_present():
    raw = "Found PASS-related markers but VERDICT: FAIL"
    res = parse_code_review_response(raw)
    assert res.passed is False


# ── determine_posthooks ──────────────────────────────────────────────────

def test_determine_posthooks_accepts_code_review_kind():
    task = {"id": 1, "agent_type": "coder"}
    ctx = {"post_hooks": ["code_review"]}
    kinds = determine_posthooks(task, ctx, {})
    assert "code_review" in kinds
    assert "grade" in kinds  # default still in


def test_determine_posthooks_combinable_with_verify_artifacts():
    task = {"id": 1, "agent_type": "coder"}
    ctx = {"post_hooks": ["verify_artifacts", "code_review"]}
    kinds = determine_posthooks(task, ctx, {})
    assert kinds == ["grade", "verify_artifacts", "code_review"]


def test_determine_posthooks_excludes_implicit_grade_for_reviewer():
    """A reviewer task gets no IMPLICIT grade hook (judge-of-judge).

    SP3: the code_review post-hook child runs as a raw_dispatch ``reviewer``
    task (the deleted ``code_reviewer`` agent type no longer exists), so the
    judge-of-judge suppression that used to key on ``code_reviewer`` now keys
    on ``reviewer``. Z7 B3 corrected contract: excluded agent types suppress
    only the implicit ``grade`` hook — an explicitly declared post-hook is
    still honoured. A reviewer task that declares no post_hooks therefore
    yields an empty list (the implicit grade is dropped).
    """
    task = {"id": 1, "agent_type": "reviewer"}
    assert determine_posthooks(task, {}, {}) == []
    # requires_grading explicitly True still does not re-add grade.
    assert determine_posthooks(task, {"requires_grading": True}, {}) == []


# ── _posthook_title ───────────────────────────────────────────────────────
# SP3: code_review no longer flows through _posthook_agent_and_payload. The
# kind is intercepted in _apply_request_posthook and spawned as a raw_dispatch
# "reviewer" child via _enqueue_posthook_llm_child (on_complete=
# posthook.code_review.resume). That CPS spawn path is covered by
# tests/beckman/test_posthook_spawn_cps.py and tests/core/test_build_code_review_spec.py;
# the deleted code_reviewer agent-routing test was removed here.

def test_posthook_title_code_review():
    a = RequestPostHook(source_task_id=42, kind="code_review", source_ctx={})
    assert _posthook_title(a, {}) == "Code review for #42"


# ── rewrite Rule 0 (REMOVED in SP3) ──────────────────────────────────────
# SP3 deleted the old rewrite Rule 0 that translated a code_reviewer agent
# task's posthook_verdict payload into a PostHookVerdict. The code_reviewer
# wrapper agent is gone; code_review now runs as a raw_dispatch "reviewer"
# child whose verdict is built and applied via the durable
# posthook.code_review.resume continuation (see posthook_continuations.py),
# not via this rewrite-layer translation. The three tests that pinned the
# removed Rule 0 code_reviewer path were deleted here.


# ── _apply_code_review_verdict ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_apply_code_review_pass_completes_when_no_other_pending():
    source = {
        "id": 42, "mission_id": 5, "status": "ungraded", "title": "backend_service",
    }
    ctx = {"_pending_posthooks": ["code_review"]}
    pending = list(ctx["_pending_posthooks"])
    verdict = PostHookVerdict(
        source_task_id=42, kind="code_review", passed=True,
        raw={"passed": True, "issues": []},
    )
    update_calls = []

    async def _update(task_id, **kw):
        update_calls.append((task_id, kw))

    with patch("src.infra.db.update_task", _update), \
         patch("general_beckman.apply._spawn_workflow_advance_if_mission",
               new_callable=AsyncMock):
        await _apply_code_review_verdict(source, ctx, pending, verdict)
    assert update_calls
    last_kwargs = update_calls[-1][1]
    assert last_kwargs["status"] == "completed"


@pytest.mark.asyncio
async def test_apply_code_review_pass_with_other_pending_keeps_ungraded():
    source = {"id": 42, "mission_id": 5, "status": "ungraded", "title": "x"}
    ctx = {"_pending_posthooks": ["grade", "code_review"]}
    pending = list(ctx["_pending_posthooks"])
    verdict = PostHookVerdict(
        source_task_id=42, kind="code_review", passed=True, raw={},
    )
    update_calls = []

    async def _update(task_id, **kw):
        update_calls.append((task_id, kw))

    with patch("src.infra.db.update_task", _update):
        await _apply_code_review_verdict(source, ctx, pending, verdict)
    assert update_calls
    last_kwargs = update_calls[-1][1]
    assert "status" not in last_kwargs
    assert json.loads(last_kwargs["context"])["_pending_posthooks"] == ["grade"]


@pytest.mark.asyncio
async def test_apply_code_review_fail_retries_with_issue_list_feedback():
    source = {
        "id": 42, "mission_id": 5, "status": "ungraded", "title": "backend_service",
        "worker_attempts": 0, "max_worker_attempts": 5,
        "result": "{\"file_path\": \"x.py\", \"content\": \"...\"}",
    }
    ctx = {"_pending_posthooks": ["code_review"]}
    pending = list(ctx["_pending_posthooks"])
    issues = [
        "backend/app/main.py:7: hardcoded SECRET_KEY",
        "backend/app/routes.py:5: missing auth middleware",
    ]
    verdict = PostHookVerdict(
        source_task_id=42, kind="code_review", passed=False,
        raw={"passed": False, "issues": issues},
    )
    update_calls = []

    async def _update(task_id, **kw):
        update_calls.append((task_id, kw))

    with patch("src.infra.db.update_task", _update):
        await _apply_code_review_verdict(source, ctx, pending, verdict)
    assert len(update_calls) == 1
    tid, kw = update_calls[0]
    assert tid == 42
    assert kw["status"] == "pending"
    assert kw["worker_attempts"] == 1
    assert kw["error_category"] == "quality"
    saved_ctx = json.loads(kw["context"])
    feedback = saved_ctx["_schema_error"]
    assert "SECRET_KEY" in feedback
    assert "auth middleware" in feedback


@pytest.mark.asyncio
async def test_apply_code_review_fail_terminal_writes_dlq():
    source = {
        "id": 42, "mission_id": 5, "status": "ungraded", "title": "x",
        "worker_attempts": 4, "max_worker_attempts": 5,
        "result": "",
    }
    ctx = {"_pending_posthooks": ["code_review"]}
    pending = list(ctx["_pending_posthooks"])
    verdict = PostHookVerdict(
        source_task_id=42, kind="code_review", passed=False,
        raw={"passed": False, "issues": ["a", "b"]},
    )
    dlq_calls = []

    async def _dlq(source, error, category, attempts):
        dlq_calls.append((source["id"], error, category, attempts))

    async def _update(task_id, **kw):
        pass

    with patch("src.infra.db.update_task", _update), \
         patch("general_beckman.apply._dlq_write", _dlq), \
         patch("general_beckman.apply._parse_progress", return_value=0.0):
        await _apply_code_review_verdict(source, ctx, pending, verdict)
    assert dlq_calls
    assert dlq_calls[0][0] == 42
    assert dlq_calls[0][2] == "quality"
