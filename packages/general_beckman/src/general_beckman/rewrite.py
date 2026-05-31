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

import json as _json
from typing import Iterable

from general_beckman.result_router import (
    Action, Complete, SpawnSubtasks, RequestClarification, RequestReview,
    Failed, Exhausted, MissionAdvance, CompleteWithReusedAnswer,
    RequestPostHook, PostHookVerdict,
)
from general_beckman.posthooks import determine_posthooks


#: SP3b Task 6 — the ordered RESULT-REWRITE → GATE chain. determine_posthooks
#: returns these (when applicable) in this exact order; rewrite.py sequences the
#: subset as a single cursor-headed RequestPostHook so the apply layer walks
#: them one at a time (rewriters land BEFORE grade) instead of fanning out in
#: parallel. ``grade`` is the terminal gate and always last.
_POSTHOOK_CHAIN_ORDER: tuple[str, ...] = (
    "constrained_emit", "self_reflect", "grade",
)


def _is_workflow_step(task_ctx: dict) -> bool:
    return bool(task_ctx.get("workflow_step") or task_ctx.get("is_workflow_step"))


# Config-only LLM reviewer agents that run as post-hooks but — unlike
# GraderAgent / CodeReviewerAgent — have no ``execute`` override to build a
# ``posthook_verdict`` payload. They emit a plain ``{"verdict": ..., "findings":
# [...]}`` JSON string in ``final_answer.result``. Rule 0d below parses that
# into a PostHookVerdict so the source task is not stranded in ``ungraded``.
_CONFIG_ONLY_REVIEWER_AGENTS: frozenset[str] = frozenset({
    "integration_reviewer",  # Z3 T2C
    "adr_drift_judge",       # Z3 R3
})


def _parse_reviewer_verdict(a: Complete) -> dict:
    """Extract {passed, raw} from a config-only reviewer's Complete.

    The reviewer emits ``{"verdict": "pass"|"fail", "findings": [...]}`` as a
    JSON string in ``Complete.result``. Fall back to ``a.raw`` keys. When the
    payload cannot be parsed at all, default to ``passed=False`` so a garbled
    reviewer output retries the source rather than silently passing it.
    """
    import json as _json

    parsed: dict = {}
    result = a.result
    if isinstance(result, dict):
        parsed = result
    elif isinstance(result, str) and result.strip():
        try:
            loaded = _json.loads(result)
            if isinstance(loaded, dict):
                parsed = loaded
        except (ValueError, TypeError):
            parsed = {}

    # Fallback: some dispatch paths put the structured dict on raw.
    if not parsed and isinstance(a.raw, dict):
        inner = a.raw.get("result")
        if isinstance(inner, dict):
            parsed = inner

    verdict = str(parsed.get("verdict") or "").strip().lower()
    # Accept the explicit "pass"; anything else (fail / blank / garbled) is a
    # fail so the source is retried with the findings as feedback.
    passed = verdict == "pass"
    return {"passed": passed, "raw": parsed}


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
    # SP3: the old Rule 0 (grader/artifact_summarizer/code_reviewer agent
    # tasks → translate their posthook_verdict payload into a PostHookVerdict)
    # is gone. Those wrapper agent classes are deleted; grade / code_review /
    # summary now run as raw_dispatch reviewer/summarizer CHILDREN whose
    # verdict is applied via the durable posthook.*.resume continuation, not
    # via this rewrite-layer translation.

    # Rule 0d: config-only LLM reviewer post-hook completion → synthesise a
    # PostHookVerdict from the {verdict, findings} JSON the reviewer emits.
    # integration_reviewer (Z3 T2C) and adr_drift_judge (Z3 R3) have no
    # execute() override to build a posthook_verdict payload the way grader /
    # code_reviewer do — without this rule their source task is stranded in
    # 'ungraded' forever (the kind is never dropped from _pending_posthooks).
    if (
        isinstance(a, Complete)
        and task.get("agent_type") in _CONFIG_ONLY_REVIEWER_AGENTS
        and task_ctx.get("source_task_id") is not None
        and task_ctx.get("posthook_kind")
    ):
        parsed = _parse_reviewer_verdict(a)
        return [
            a,
            PostHookVerdict(
                source_task_id=int(task_ctx["source_task_id"]),
                kind=str(task_ctx["posthook_kind"]),
                passed=parsed["passed"],
                raw=parsed["raw"],
            ),
        ]

    # Rule 0b: mechanical post-hook completion → synthesise PostHookVerdict.
    # Mechanical executors (mr_roboto) don't emit a posthook_verdict field —
    # their result is shaped by the mr_roboto verb (verify_artifacts returns
    # {verified, missing, failed, all_ok}). Detect this via context fields
    # ``source_task_id`` + ``posthook_kind`` placed by
    # _posthook_agent_and_payload, and translate Complete -> PostHookVerdict.
    # The orchestrator wraps mr_roboto.run's Action.result into
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
        # `all_ok` is the verify_artifacts convention; Z1 + Z2 mechanical
        # post-hooks (compliance_template_present, prior_art_min_coverage,
        # etc.) emit `ok`; check_grounding emits `passed`. Accept all three
        # so future hooks don't have to rename their result keys.
        #
        # The missing `passed` alias caused an infinite retry loop: the
        # grounding post-hook returned {"passed": true, "missing": []}, but
        # `inner.get("all_ok") or inner.get("ok")` evaluated to None → the
        # synthesised PostHookVerdict was always passed=False → the source
        # task retried forever, spawning a fresh grade post-hook each cycle
        # (production 2026-05-15 mission 70 writer #43432: 4 loops, error
        # text literally said "0 produces slot(s) ungrounded. missing=[]").
        passed = bool(
            inner.get("all_ok")
            or inner.get("ok")
            or inner.get("passed")
        )
        return [
            a,
            PostHookVerdict(
                source_task_id=int(task_ctx["source_task_id"]),
                kind=str(task_ctx["posthook_kind"]),
                passed=passed,
                raw=inner,
            ),
        ]

    # Rule 0c': mechanical post-hook returned needs_review (e.g.
    # find_similar_missions found matches; spec_consistency_check found
    # drift). Spawn the review row so the founder is surfaced, AND
    # synthesise PostHookVerdict(passed=False) so the source isn't stuck
    # in 'ungraded' forever waiting for a verdict that the review channel
    # would never produce. Z1 verdict handler decides whether passed=False
    # means "block source" (blocker kind) or "soft-drop and advance"
    # (warning kind, e.g. find_similar_missions).
    if (
        isinstance(a, RequestReview)
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
                raw={"needs_review": True, "summary": a.summary or ""},
            ),
        ]

    # Rule 0c: post-hook FAILED / EXHAUSTED (mechanical mr_roboto status=failed,
    # such as "no paths supplied"; or a config-only reviewer that exhausted its
    # iteration budget). Surfaces as Failed/Exhausted; we still want a
    # PostHookVerdict with passed=False so the source advances down the
    # retry-with-feedback path rather than waiting on a verdict that will never
    # arrive. The Failed/Exhausted action remains in the list so the post-hook
    # task itself goes through normal DLQ / retry handling.
    if (
        isinstance(a, (Failed, Exhausted))
        and (
            task.get("agent_type") == "mechanical"
            or task.get("agent_type") in _CONFIG_ONLY_REVIEWER_AGENTS
        )
        and task_ctx.get("source_task_id") is not None
        and task_ctx.get("posthook_kind")
    ):
        verdict = PostHookVerdict(
            source_task_id=int(task_ctx["source_task_id"]),
            kind=str(task_ctx["posthook_kind"]),
            passed=False,
            raw={"error": a.error, "missing": [], "failed": []},
        )
        # A MECHANICAL validator that actually RAN its check and returned a
        # negative verdict ({"ok"/"passed"/"all_ok": False, ...}) has DONE its
        # job — the fail is a VERDICT for the producer (carried above), not a
        # failure of the validator TASK. Keeping the Failed action sent the
        # validator down normal retry/DLQ handling: it re-ran the same
        # deterministic check against the same artifact 5× and DLQ'd as
        # "Worker attempts exceeded: 5/6" noise (mission_79 #225576
        # interview_script_shape, #227677 prior_art_min_coverage, 2026-05-31).
        # Convert it to a terminal Complete so only the producer re-runs.
        #
        # An EXECUTOR error (exception / "no paths supplied") returns
        # status=failed with NO verdict-shaped result — that stays a retryable
        # Failed (transient input/IO can clear). Config-only reviewers that
        # EXHAUSTED also stay retryable (iteration-budget, not a verdict). The
        # producer verdict above is byte-identical to before either way —
        # only the validator's own action changes.
        inner = (a.raw or {}).get("result") if isinstance(a.raw, dict) else None
        if isinstance(inner, str):
            try:
                inner = _json.loads(inner)
            except (ValueError, TypeError):
                inner = None
        is_mechanical_checkfail = (
            task.get("agent_type") == "mechanical"
            and isinstance(inner, dict)
            and any(k in inner for k in ("ok", "passed", "all_ok"))
        )
        if is_mechanical_checkfail:
            return [
                Complete(
                    task_id=a.task_id,
                    result=inner,
                    iterations=0,
                    metadata={},
                    raw=a.raw if isinstance(a.raw, dict) else {},
                ),
                verdict,
            ]
        return [a, verdict]

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
        # SP3: reviewer = grade/code_review child; summarizer = summary child.
        # Both are bookkeeping — a mission-bearing edge case must not let a
        # summary/grade child spawn MissionAdvance or a recursive grade hook.
        or agent_type in {"reviewer", "summarizer"}
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
        determined = determine_posthooks(task, task_ctx, a.raw)
        # Build the result-scalar view once, shared by every posthook kind.
        # incident/draft_update returns {"draft": ..., "incident_id": ...,
        # "status_kind": ...} in its result dict but never writes these to the
        # task context, so apply.py's _posthook_agent_and_payload would find
        # source_ctx.get("draft") empty — silently bypassing the B3 founder-
        # review gate.  The result scalars must be merged into source_ctx.
        #
        # Mechanical executors (mr_roboto) return an Action whose .result is
        # JSON-encoded by the orchestrator into
        # ``{"status": "completed", "result": "<json-string>"}`` — so the
        # interesting fields (draft, incident_id, ...) are NOT top-level keys
        # of a.raw, they are nested inside the a.raw["result"] JSON string.
        # Unwrap that string so the scalars actually reach the posthook ctx.
        _result_scalars: dict = {}
        if isinstance(a.raw, dict):
            for _rk, _rv in a.raw.items():
                if _rk in ("status", "ok", "result"):
                    continue
                if isinstance(_rv, (str, int, float, bool)):
                    _result_scalars[_rk] = _rv
            _inner = a.raw.get("result")
            if isinstance(_inner, str) and _inner.strip():
                try:
                    _inner = _json.loads(_inner)
                except (ValueError, TypeError):
                    _inner = None
            if isinstance(_inner, dict):
                for _rk, _rv in _inner.items():
                    if _rk in ("status", "ok"):
                        continue
                    if isinstance(_rv, (str, int, float, bool)):
                        # top-level a.raw scalars (rare) take precedence over
                        # the nested executor-result scalars.
                        _result_scalars.setdefault(_rk, _rv)
        # SP3b Task 6 — partition the rewrite→grade chain from the independent
        # (mechanical) post-hooks. The chain (constrained_emit / self_reflect /
        # grade) must run SEQUENTIALLY so the rewriters land on the source
        # before grade reads it; emit it as a SINGLE cursor-headed
        # RequestPostHook. Everything else keeps the existing parallel fan-out.
        chain = [k for k in _POSTHOOK_CHAIN_ORDER if k in determined]
        # A chain that's only ["grade"] (no rewriter ahead of it) is the legacy
        # grade-only case — emit grade as a plain post-hook (no queue) so its
        # behaviour is byte-identical to pre-Task-6. Sequencing only matters
        # when at least one rewriter precedes grade.
        has_rewriter = any(k != "grade" for k in chain)

        def _build_ctx() -> dict:
            # Using setdefault means task_ctx values win (they were set
            # intentionally); result scalars fill only fields the task didn't
            # set.
            _c = dict(task_ctx)
            for _rk, _rv in _result_scalars.items():
                _c.setdefault(_rk, _rv)
            return _c

        if has_rewriter:
            # One RequestPostHook for the chain HEAD, carrying the ordered queue.
            # The apply layer (_apply_request_posthook) reads _posthook_queue,
            # parks only the gating kinds (grade) in _pending_posthooks, and
            # walks the cursor (skip-draining no-op rewriters).
            head_ctx = _build_ctx()
            head_ctx["_posthook_queue"] = list(chain)
            result_actions.append(
                RequestPostHook(
                    source_task_id=a.task_id,
                    kind=chain[0],
                    source_ctx=head_ctx,
                )
            )

        # Independent / mechanical post-hooks (verify_artifacts, grounding, ...)
        # and — when there is no rewriter — a plain grade keep the parallel path.
        for kind in determined:
            if kind in chain and has_rewriter:
                continue  # owned by the sequential chain above
            result_actions.append(
                RequestPostHook(
                    source_task_id=a.task_id,
                    kind=kind,
                    source_ctx=_build_ctx(),
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
