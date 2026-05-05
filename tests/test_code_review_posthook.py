"""Coverage for the code_review post-hook end-to-end:
- determine_posthooks accepts code_review in ctx.post_hooks
- _posthook_agent_and_payload routes code_review to code_reviewer agent
- rewrite Rule 0 translates code_reviewer Complete to PostHookVerdict
- _apply_code_review_verdict pass/fail behaviour
- DLQ cascade gate widens to code_review post-hooks
- parse_code_review_response extracts VERDICT + bullet issues
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from general_beckman.posthooks import determine_posthooks
from general_beckman.apply import (
    _posthook_agent_and_payload,
    _posthook_title,
    _apply_code_review_verdict,
)
from general_beckman.result_router import (
    Complete, MissionAdvance, RequestPostHook, PostHookVerdict,
)
from general_beckman.rewrite import rewrite_actions
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


def test_determine_posthooks_excludes_code_reviewer_itself():
    task = {"id": 1, "agent_type": "code_reviewer"}
    ctx = {"post_hooks": ["code_review"]}
    kinds = determine_posthooks(task, ctx, {})
    assert kinds == []


# ── _posthook_agent_and_payload ──────────────────────────────────────────

def test_agent_and_payload_code_review_uses_code_reviewer_agent():
    a = RequestPostHook(source_task_id=42, kind="code_review", source_ctx={})
    source = {"id": 42, "mission_id": 7}
    source_ctx = {
        "produces": ["backend/app/main.py", "backend/app/routes.py"],
        "review_excluded_models": ["model-x"],
    }
    agent, payload = _posthook_agent_and_payload(a, source, source_ctx)
    assert agent == "code_reviewer"
    assert payload["source_task_id"] == 42
    assert payload["posthook_kind"] == "code_review"
    assert payload["produces"] == source_ctx["produces"]
    assert payload["review_excluded_models"] == ["model-x"]


def test_posthook_title_code_review():
    a = RequestPostHook(source_task_id=42, kind="code_review", source_ctx={})
    assert _posthook_title(a, {}) == "Code review for #42"


# ── rewrite Rule 0: code_reviewer Complete → PostHookVerdict ─────────────

def test_rewrite_code_reviewer_complete_synthesises_verdict_pass():
    task = {
        "id": 99, "agent_type": "code_reviewer",
        "context": json.dumps({
            "source_task_id": 42, "posthook_kind": "code_review",
        }),
        "mission_id": 7,
    }
    raw = {
        "status": "completed",
        "result": json.dumps({"passed": True, "issues": []}),
        "posthook_verdict": {
            "kind": "code_review", "source_task_id": 42, "passed": True,
            "raw": {"passed": True, "issues": []},
        },
    }
    actions = [Complete(task_id=99, result=raw["result"], raw=raw)]
    ctx = {"source_task_id": 42, "posthook_kind": "code_review"}
    out = rewrite_actions(task, ctx, actions)
    verdicts = [a for a in out if isinstance(a, PostHookVerdict)]
    assert len(verdicts) == 1
    assert verdicts[0].kind == "code_review"
    assert verdicts[0].passed is True


def test_rewrite_code_reviewer_complete_synthesises_verdict_fail_with_issues():
    issues = ["app/x.py:1: hardcoded secret", "app/y.py: no auth"]
    task = {
        "id": 99, "agent_type": "code_reviewer",
        "context": json.dumps({
            "source_task_id": 42, "posthook_kind": "code_review",
        }),
        "mission_id": 7,
    }
    raw = {
        "status": "completed",
        "result": json.dumps({"passed": False, "issues": issues}),
        "posthook_verdict": {
            "kind": "code_review", "source_task_id": 42, "passed": False,
            "raw": {"passed": False, "issues": issues},
        },
    }
    actions = [Complete(task_id=99, result=raw["result"], raw=raw)]
    ctx = {"source_task_id": 42, "posthook_kind": "code_review"}
    out = rewrite_actions(task, ctx, actions)
    verdicts = [a for a in out if isinstance(a, PostHookVerdict)]
    assert verdicts[0].passed is False
    assert verdicts[0].raw["issues"] == issues


def test_rewrite_code_reviewer_does_not_emit_mission_advance():
    task = {
        "id": 99, "agent_type": "code_reviewer",
        "context": json.dumps({
            "source_task_id": 42, "posthook_kind": "code_review",
        }),
        "mission_id": 7,
    }
    raw = {
        "status": "completed",
        "result": "{}",
        "posthook_verdict": {
            "kind": "code_review", "source_task_id": 42, "passed": True, "raw": {},
        },
    }
    actions = [Complete(task_id=99, result="{}", raw=raw)]
    ctx = {"source_task_id": 42, "posthook_kind": "code_review"}
    out = rewrite_actions(task, ctx, actions)
    assert not any(isinstance(a, MissionAdvance) for a in out)
    assert not any(isinstance(a, RequestPostHook) for a in out)


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
