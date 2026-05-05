"""Coverage for the verify_artifacts post-hook end-to-end:
- determine_posthooks reads `post_hooks` from ctx
- _posthook_agent_and_payload routes verify_artifacts to mechanical
- rewrite synthesises PostHookVerdict from mechanical posthook completion
- _apply_verify_artifacts_verdict pass/fail behaviour
- DLQ cascade gate widens to verify_artifacts posthooks
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from general_beckman.posthooks import determine_posthooks
from general_beckman.apply import (
    _posthook_agent_and_payload,
    _posthook_title,
    _apply_verify_artifacts_verdict,
)
from general_beckman.result_router import (
    Complete, Failed, MissionAdvance, RequestPostHook, PostHookVerdict,
)
from general_beckman.rewrite import rewrite_actions


# ── determine_posthooks ──────────────────────────────────────────────────

def test_determine_posthooks_default_grade_only():
    task = {"id": 1, "agent_type": "coder"}
    kinds = determine_posthooks(task, {}, {})
    assert kinds == ["grade"]


def test_determine_posthooks_appends_verify_artifacts_from_ctx():
    task = {"id": 1, "agent_type": "coder"}
    ctx = {"post_hooks": ["verify_artifacts"]}
    kinds = determine_posthooks(task, ctx, {})
    assert kinds == ["grade", "verify_artifacts"]


def test_determine_posthooks_filters_unknown_kinds():
    task = {"id": 1, "agent_type": "coder"}
    ctx = {"post_hooks": ["verify_artifacts", "totally_made_up"]}
    kinds = determine_posthooks(task, ctx, {})
    assert "totally_made_up" not in kinds
    assert "verify_artifacts" in kinds


def test_determine_posthooks_skips_grade_when_disabled_but_keeps_extras():
    task = {"id": 1, "agent_type": "coder"}
    ctx = {"requires_grading": False, "post_hooks": ["verify_artifacts"]}
    kinds = determine_posthooks(task, ctx, {})
    assert "grade" not in kinds
    assert kinds == ["verify_artifacts"]


def test_determine_posthooks_excluded_agent_returns_empty():
    task = {"id": 1, "agent_type": "mechanical"}
    ctx = {"post_hooks": ["verify_artifacts"]}
    kinds = determine_posthooks(task, ctx, {})
    assert kinds == []


# ── _posthook_agent_and_payload ──────────────────────────────────────────

def test_agent_and_payload_verify_artifacts_uses_mechanical():
    a = RequestPostHook(source_task_id=42, kind="verify_artifacts", source_ctx={})
    source = {"id": 42, "mission_id": 7}
    source_ctx = {"produces": ["backend/app/main.py", "backend/app/routes.py"]}
    agent, payload = _posthook_agent_and_payload(a, source, source_ctx)
    assert agent == "mechanical"
    assert payload["source_task_id"] == 42
    assert payload["posthook_kind"] == "verify_artifacts"
    assert payload["executor"] == "mechanical"
    assert payload["payload"]["action"] == "verify_artifacts"
    assert payload["payload"]["paths"] == ["backend/app/main.py", "backend/app/routes.py"]
    assert payload["payload"]["compile_check"] is True


def test_agent_and_payload_verify_artifacts_handles_missing_produces():
    a = RequestPostHook(source_task_id=42, kind="verify_artifacts", source_ctx={})
    agent, payload = _posthook_agent_and_payload(a, {"id": 42}, {})
    assert payload["payload"]["paths"] == []


def test_posthook_title_verify_artifacts():
    a = RequestPostHook(source_task_id=42, kind="verify_artifacts", source_ctx={})
    assert _posthook_title(a, {}) == "Verify artifacts for #42"


# ── rewrite synthesis: mechanical Complete -> PostHookVerdict ────────────

def test_rewrite_mechanical_posthook_complete_synthesises_verdict_pass():
    task = {
        "id": 99, "agent_type": "mechanical",
        "context": json.dumps({
            "source_task_id": 42, "posthook_kind": "verify_artifacts",
        }),
        "mission_id": 7,
    }
    inner = {"verified": [{"path": "x.py"}], "missing": [], "failed": [], "all_ok": True}
    raw = {"status": "completed", "result": json.dumps(inner)}
    actions = [Complete(task_id=99, result=raw["result"], raw=raw)]
    ctx = {"source_task_id": 42, "posthook_kind": "verify_artifacts"}
    out = rewrite_actions(task, ctx, actions)
    verdicts = [a for a in out if isinstance(a, PostHookVerdict)]
    assert len(verdicts) == 1
    assert verdicts[0].source_task_id == 42
    assert verdicts[0].kind == "verify_artifacts"
    assert verdicts[0].passed is True
    assert verdicts[0].raw["all_ok"] is True


def test_rewrite_mechanical_posthook_complete_synthesises_verdict_fail():
    task = {
        "id": 99, "agent_type": "mechanical",
        "context": json.dumps({
            "source_task_id": 42, "posthook_kind": "verify_artifacts",
        }),
        "mission_id": 7,
    }
    inner = {"verified": [], "missing": ["x.py"], "failed": [], "all_ok": False}
    raw = {"status": "completed", "result": json.dumps(inner)}
    actions = [Complete(task_id=99, result=raw["result"], raw=raw)]
    ctx = {"source_task_id": 42, "posthook_kind": "verify_artifacts"}
    out = rewrite_actions(task, ctx, actions)
    verdicts = [a for a in out if isinstance(a, PostHookVerdict)]
    assert verdicts[0].passed is False
    assert verdicts[0].raw["missing"] == ["x.py"]


def test_rewrite_mechanical_posthook_failed_action_synthesises_verdict_fail():
    task = {
        "id": 99, "agent_type": "mechanical",
        "context": json.dumps({
            "source_task_id": 42, "posthook_kind": "verify_artifacts",
        }),
        "mission_id": 7,
    }
    actions = [Failed(task_id=99, error="cwd rejected", raw={})]
    ctx = {"source_task_id": 42, "posthook_kind": "verify_artifacts"}
    out = rewrite_actions(task, ctx, actions)
    verdicts = [a for a in out if isinstance(a, PostHookVerdict)]
    assert len(verdicts) == 1
    assert verdicts[0].passed is False
    assert verdicts[0].raw["error"] == "cwd rejected"


def test_rewrite_posthook_task_does_not_emit_mission_advance_or_request_posthook():
    """Posthook tasks must not recurse — no MissionAdvance, no nested RequestPostHook."""
    task = {
        "id": 99, "agent_type": "mechanical",
        "context": json.dumps({
            "source_task_id": 42, "posthook_kind": "verify_artifacts",
        }),
        "mission_id": 7,
    }
    inner = {"all_ok": True, "verified": [], "missing": [], "failed": []}
    raw = {"status": "completed", "result": json.dumps(inner)}
    actions = [Complete(task_id=99, result=raw["result"], raw=raw)]
    ctx = {"source_task_id": 42, "posthook_kind": "verify_artifacts"}
    out = rewrite_actions(task, ctx, actions)
    assert not any(isinstance(a, MissionAdvance) for a in out)
    assert not any(isinstance(a, RequestPostHook) for a in out)


def test_rewrite_unrelated_mechanical_complete_unchanged():
    """A mechanical task that is NOT a posthook (no source_task_id) must not be
    converted to a PostHookVerdict."""
    task = {"id": 7, "agent_type": "mechanical", "context": "{}", "mission_id": 5}
    actions = [Complete(task_id=7, result="ok", raw={"status": "completed"})]
    out = rewrite_actions(task, {}, actions)
    assert not any(isinstance(a, PostHookVerdict) for a in out)


# ── _apply_verify_artifacts_verdict ─────────────────────────────────────

@pytest.mark.asyncio
async def test_apply_verify_pass_with_no_other_pending_completes_source():
    source = {
        "id": 42, "mission_id": 5, "status": "ungraded",
        "title": "backend_scaffold",
    }
    ctx = {"_pending_posthooks": ["verify_artifacts"]}
    pending = list(ctx["_pending_posthooks"])
    verdict = PostHookVerdict(
        source_task_id=42, kind="verify_artifacts", passed=True,
        raw={"all_ok": True, "verified": [{"path": "x.py"}]},
    )
    update_calls = []

    async def _update(task_id, **kw):
        update_calls.append((task_id, kw))

    with patch("src.infra.db.update_task", _update), \
         patch("general_beckman.apply._spawn_workflow_advance_if_mission",
               new_callable=AsyncMock):
        await _apply_verify_artifacts_verdict(source, ctx, pending, verdict)
    assert update_calls
    last_kwargs = update_calls[-1][1]
    assert last_kwargs["status"] == "completed"
    assert json.loads(last_kwargs["context"])["_pending_posthooks"] == []


@pytest.mark.asyncio
async def test_apply_verify_pass_with_other_pending_keeps_source_ungraded():
    source = {"id": 42, "mission_id": 5, "status": "ungraded", "title": "x"}
    ctx = {"_pending_posthooks": ["grade", "verify_artifacts"]}
    pending = list(ctx["_pending_posthooks"])
    verdict = PostHookVerdict(
        source_task_id=42, kind="verify_artifacts", passed=True, raw={"all_ok": True},
    )
    update_calls = []

    async def _update(task_id, **kw):
        update_calls.append((task_id, kw))

    with patch("src.infra.db.update_task", _update):
        await _apply_verify_artifacts_verdict(source, ctx, pending, verdict)
    assert update_calls
    last_kwargs = update_calls[-1][1]
    assert "status" not in last_kwargs  # source not flipped to completed
    assert json.loads(last_kwargs["context"])["_pending_posthooks"] == ["grade"]


@pytest.mark.asyncio
async def test_apply_verify_fail_retries_source_with_feedback():
    source = {
        "id": 42, "mission_id": 5, "status": "ungraded",
        "title": "backend_scaffold",
        "worker_attempts": 0, "max_worker_attempts": 5,
        "result": "{\"file_path\": \"x.py\", \"content\": \"...\"}",
    }
    ctx = {"_pending_posthooks": ["verify_artifacts"]}
    pending = list(ctx["_pending_posthooks"])
    verdict = PostHookVerdict(
        source_task_id=42, kind="verify_artifacts", passed=False,
        raw={"all_ok": False, "missing": ["x.py"], "failed": []},
    )
    update_calls = []

    async def _update(task_id, **kw):
        update_calls.append((task_id, kw))

    with patch("src.infra.db.update_task", _update):
        await _apply_verify_artifacts_verdict(source, ctx, pending, verdict)
    assert len(update_calls) == 1
    tid, kw = update_calls[0]
    assert tid == 42
    assert kw["status"] == "pending"
    assert kw["worker_attempts"] == 1
    assert kw["error_category"] == "quality"
    saved_ctx = json.loads(kw["context"])
    assert "missing" in saved_ctx["_schema_error"]
    assert saved_ctx["_pending_posthooks"] == []
    assert "x.py" in saved_ctx["_prev_output"] or "x.py" in saved_ctx["_schema_error"]


@pytest.mark.asyncio
async def test_apply_verify_fail_terminal_writes_dlq():
    source = {
        "id": 42, "mission_id": 5, "status": "ungraded",
        "title": "backend_scaffold",
        "worker_attempts": 4, "max_worker_attempts": 5,
        "result": "",
    }
    ctx = {"_pending_posthooks": ["verify_artifacts"], "progress_pct": 0.0}
    pending = list(ctx["_pending_posthooks"])
    verdict = PostHookVerdict(
        source_task_id=42, kind="verify_artifacts", passed=False,
        raw={"all_ok": False, "missing": ["x.py"], "failed": []},
    )

    dlq_calls = []

    async def _dlq(source, error, category, attempts):
        dlq_calls.append((source["id"], error, category, attempts))

    async def _update(task_id, **kw):
        pass

    with patch("src.infra.db.update_task", _update), \
         patch("general_beckman.apply._dlq_write", _dlq), \
         patch("general_beckman.apply._parse_progress", return_value=0.0):
        await _apply_verify_artifacts_verdict(source, ctx, pending, verdict)
    assert dlq_calls
    assert dlq_calls[0][0] == 42
    assert dlq_calls[0][2] == "quality"
    assert dlq_calls[0][3] == 5  # attempts incremented to 5


@pytest.mark.asyncio
async def test_apply_verify_fail_at_cap_with_progress_grants_bonus():
    source = {
        "id": 42, "mission_id": 5, "status": "ungraded",
        "title": "backend_scaffold",
        "worker_attempts": 4, "max_worker_attempts": 5,
        "result": "{}",
    }
    ctx = {"_pending_posthooks": ["verify_artifacts"], "_bonus_count": 0}
    pending = list(ctx["_pending_posthooks"])
    verdict = PostHookVerdict(
        source_task_id=42, kind="verify_artifacts", passed=False,
        raw={"all_ok": False, "missing": ["x.py"], "failed": []},
    )

    update_calls = []

    async def _update(task_id, **kw):
        update_calls.append((task_id, kw))

    with patch("src.infra.db.update_task", _update), \
         patch("general_beckman.apply._dlq_write", new_callable=AsyncMock) as dlq, \
         patch("general_beckman.apply._parse_progress", return_value=0.7):
        await _apply_verify_artifacts_verdict(source, ctx, pending, verdict)
    dlq.assert_not_called()
    assert update_calls
    kw = update_calls[0][1]
    assert kw["status"] == "pending"
    assert kw["max_worker_attempts"] == 6  # bonus applied
    saved_ctx = json.loads(kw["context"])
    assert saved_ctx["_bonus_count"] == 1
