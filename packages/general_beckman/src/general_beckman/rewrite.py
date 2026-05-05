"""Pure action-rewriting rules. Replaces the old result_guards.py.

Runs between result_router.route_result() and apply._apply_actions().
No I/O — pure transformation of the action list given (task, task_ctx).

Rules (order matters — earlier rules can short-circuit):
  1. Mission-task completion → inject MissionAdvance
  2. Workflow step emitted subtasks → replace with Failed (quality)
  3. Silent task requested clarification → replace with Failed
  4. may_need_clarification=False + clarify request → Failed
  5. Existing clarification_history + clarify request → CompleteWithReusedAnswer
"""
from __future__ import annotations

from typing import Iterable

from general_beckman.result_router import (
    Action, Complete, SpawnSubtasks, RequestClarification,
    Failed, MissionAdvance, CompleteWithReusedAnswer,
    RequestPostHook, PostHookVerdict,
)
from general_beckman.posthooks import determine_posthooks


def _is_workflow_step(task_ctx: dict) -> bool:
    return bool(task_ctx.get("workflow_step") or task_ctx.get("is_workflow_step"))


def _format_history(history: list) -> str:
    parts = []
    for entry in history:
        if isinstance(entry, dict):
            q = entry.get("question", "")
            a = entry.get("answer", "")
        else:
            q, a = "", str(entry)
        if q or a:
            parts.append(f"**Q:** {q}\n**A:** {a}")
    return "\n\n".join(parts)


def rewrite_actions(
    task: dict, task_ctx: dict, actions: Iterable[Action]
) -> list[Action]:
    out: list[Action] = []
    for a in actions:
        out.extend(_rewrite_one(task, task_ctx, a))
    return out


def _rewrite_one(task: dict, task_ctx: dict, a: Action) -> list[Action]:
    # Rule 0: post-hook task (grader/artifact_summarizer) completion
    # → translate its posthook_verdict payload into a PostHookVerdict action.
    # Bookkeeping tasks never fire MissionAdvance or RequestPostHook.
    if isinstance(a, Complete) and task.get("agent_type") in (
        "grader", "artifact_summarizer",
    ):
        raw = a.raw or {}
        verdict_payload = raw.get("posthook_verdict") if isinstance(raw, dict) else None
        if isinstance(verdict_payload, dict):
            return [
                a,
                PostHookVerdict(
                    source_task_id=verdict_payload["source_task_id"],
                    kind=verdict_payload["kind"],
                    passed=bool(verdict_payload.get("passed")),
                    raw=verdict_payload.get("raw") or {},
                ),
            ]
        # No posthook_verdict payload — treat as regular Complete (bookkeeping),
        # fall through to existing logic which won't emit MissionAdvance because
        # agent_type in the skip list.

    # Rule 0b: mechanical post-hook completion → synthesise PostHookVerdict.
    # Mechanical executors (salako) don't emit a posthook_verdict field —
    # their result is shaped by the salako verb (verify_artifacts returns
    # {verified, missing, failed, all_ok}). Detect this via context fields
    # ``source_task_id`` + ``posthook_kind`` placed by
    # _posthook_agent_and_payload, and translate Complete -> PostHookVerdict.
    # The orchestrator wraps salako.run's Action.result into
    # ``{"status": "completed", "result": json.dumps(action.result)}`` —
    # so a.raw["result"] arrives as a JSON string we have to parse here.
    if (
        isinstance(a, Complete)
        and task.get("agent_type") == "mechanical"
        and task_ctx.get("source_task_id") is not None
        and task_ctx.get("posthook_kind")
    ):
        import json as _json_rw
        raw = a.raw if isinstance(a.raw, dict) else {}
        inner = raw.get("result")
        if isinstance(inner, str):
            try:
                inner = _json_rw.loads(inner)
            except (ValueError, TypeError):
                inner = {}
        if not isinstance(inner, dict):
            inner = {}
        passed = bool(inner.get("all_ok"))
        return [
            a,
            PostHookVerdict(
                source_task_id=int(task_ctx["source_task_id"]),
                kind=str(task_ctx["posthook_kind"]),
                passed=passed,
                raw=inner,
            ),
        ]

    # Rule 0c: mechanical post-hook FAILED (e.g. salako returned status=failed,
    # such as "no paths supplied" or workspace resolution error). Surfaces
    # as Failed action; we still want a PostHookVerdict with passed=False so
    # the source advances down the retry-with-feedback path rather than
    # waiting on a verdict that will never arrive. The Failed action remains
    # in the action list so the verifier task itself goes through normal DLQ
    # / retry handling.
    if (
        isinstance(a, Failed)
        and task.get("agent_type") == "mechanical"
        and task_ctx.get("source_task_id") is not None
        and task_ctx.get("posthook_kind")
    ):
        return [
            a,
            PostHookVerdict(
                source_task_id=int(task_ctx["source_task_id"]),
                kind=str(task_ctx["posthook_kind"]),
                passed=False,
                raw={"error": a.error, "missing": [], "failed": []},
            ),
        ]

    # Rule 1: mission-task clean completion → emit MissionAdvance (unless
    # bookkeeping) and RequestPostHook (unless policy says no).
    payload_action = (task_ctx.get("payload") or {}).get("action")
    agent_type = task.get("agent_type", "")
    is_posthook_task = (
        task_ctx.get("source_task_id") is not None
        and bool(task_ctx.get("posthook_kind"))
    )
    is_bookkeeping = (
        payload_action == "workflow_advance"
        or agent_type in {"grader", "artifact_summarizer"}
        or is_posthook_task  # mechanical/reviewer posthook tasks shouldn't recurse
    )

    if isinstance(a, Complete) and task.get("mission_id") and not is_bookkeeping:
        result_actions: list[Action] = [a]
        result_actions.append(
            MissionAdvance(
                task_id=a.task_id,
                mission_id=task["mission_id"],
                completed_task_id=a.task_id,
                raw=a.raw,
            )
        )
        for kind in determine_posthooks(task, task_ctx, a.raw):
            result_actions.append(
                RequestPostHook(
                    source_task_id=a.task_id,
                    kind=kind,
                    source_ctx=dict(task_ctx),
                )
            )
        return result_actions
    # Rule 2: workflow step tried to decompose
    if isinstance(a, SpawnSubtasks) and _is_workflow_step(task_ctx):
        return [Failed(
            task_id=a.parent_task_id,
            error="Workflow step tried to decompose instead of producing artifact",
            raw=a.raw,
        )]
    # Rules 3–5: clarification rewrites
    if isinstance(a, RequestClarification):
        if task_ctx.get("silent"):
            return [Failed(
                task_id=a.task_id,
                error="Insufficient info (silent task, no clarification)",
                raw=a.raw,
            )]
        if task_ctx.get("may_need_clarification") is False:
            return [Failed(
                task_id=a.task_id,
                error="Agent requested clarification on no-clarification step",
                raw=a.raw,
            )]
        history = task_ctx.get("clarification_history")
        if history:
            body = _format_history(history) or task_ctx.get("user_clarification", "")
            return [CompleteWithReusedAnswer(
                task_id=a.task_id, result=body, raw=a.raw,
            )]
    return [a]
