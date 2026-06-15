"""Workflow engine: advance one mission by consuming a completed step's result.

Delegates to src/workflows/engine primitives until/unless they are migrated
wholesale into this package. Minimal surface: one function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdvanceResult:
    status: str = "completed"   # 'completed' | 'needs_clarification' | 'failed'
    error: str = ""
    next_subtasks: list[dict] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


async def advance(mission_id: int, completed_task_id: int,
                  previous_result: dict) -> AdvanceResult:
    """Post-step hook + artifact capture + next-phase subtask emission."""
    from src.workflows.engine.hooks import (
        is_workflow_step, post_execute_workflow_step, get_artifact_store,
    )
    from src.workflows.engine.pipeline_artifacts import extract_pipeline_artifacts
    from src.tools.workspace import get_mission_workspace
    from dabidabi import get_task

    out = AdvanceResult()
    task = await get_task(completed_task_id)
    if task is None:
        out.status = "failed"
        out.error = f"completed_task_id {completed_task_id} not found"
        return out
    task_ctx = _parse_ctx(task)
    if not is_workflow_step(task_ctx):
        # Not a workflow step; nothing to advance. Callers should guard,
        # but we defend here too.
        return out

    # 1. Artifact capture (from guard_pipeline_artifacts).
    try:
        ws = None
        if task.get("mission_id"):
            try:
                ws = get_mission_workspace(task["mission_id"])
            except Exception:
                ws = None
        extra = await extract_pipeline_artifacts(task, previous_result, ws)
        if extra:
            store = get_artifact_store()
            for name, content in extra.items():
                await store.store(mission_id, name, content)
            out.artifacts = dict(extra)
    except Exception:
        pass

    # 2. Post-hook: may flip status.
    # The mechanical workflow_advance task carries `previous_result` as a
    # condensed payload (often `{"summary": "..."}` from the artifact
    # summarizer post-hook chain) rather than the LLM task's full
    # `{"result": "<markdown>", ...}` shape. The post-hook reads
    # `result.get("result", "")` and an empty value tripped the schema-
    # validation gate (commit 2d6d82c) on producer steps — DLQ'ing
    # mission 46 tasks 3797/3804 with "schema requires X" even though
    # the original LLM siblings (2923, 2924) had completed cleanly with
    # multi-thousand-char artifacts in tasks.result. Inject the full
    # result string from the DB row when the carried payload doesn't
    # already carry one, so the hook validates real content.
    hook_result = dict(previous_result) if isinstance(previous_result, dict) else {}
    if not hook_result.get("result"):
        _full_result = task.get("result") or ""
        if _full_result:
            hook_result["result"] = _full_result
    if "status" not in hook_result:
        hook_result["status"] = task.get("status") or "completed"

    try:
        await post_execute_workflow_step(task, hook_result)
    except Exception as e:
        out.status = "failed"
        out.error = str(e)[:300]
        return out

    flipped = hook_result.get("status")
    if flipped == "needs_clarification":
        out.status = "needs_clarification"
        out.error = hook_result.get("question", "")
        return out
    if flipped == "failed":
        out.status = "failed"
        out.error = hook_result.get("error", "Post-hook failed")
        return out

    # 3. Next-phase subtasks: there are none to emit here, by design.
    # Every workflow step is expanded into a concrete, dependency-gated task
    # at mission creation (`expander.expand_steps_to_tasks` — "all steps are
    # expanded upfront before any file is written"). Later-phase tasks already
    # exist and unblock as their `depends_on_steps` complete; there is no
    # runtime phase-emission primitive. `next_subtasks` therefore stays empty.
    # It is kept on AdvanceResult as a reserved, always-empty field so the
    # mr_roboto/result_router `needs_subtasks` consumer contract is stable.
    #
    # Historic note: the Task-13 plan scaffolded a
    # `src.workflows.engine.recipe.advance_recipe` import at this point as a
    # possible future migration target. That primitive was never implemented,
    # so the import raised ImportError on every call and the branch was a
    # permanent no-op (the swallowed ImportError also masked which). Removed
    # 2026-06-05 — see docs/handoff/2026-06-05-readme-sweep-bug-findings.md #1.

    # 4. Mission completion — if no next subtasks queued AND every other
    # mission task is terminal, mark the mission done. Replaces
    # _check_mission_completion that used to live in the old orchestrator's
    # _handle_complete (deleted in Task 13).
    if not out.next_subtasks and out.status == "completed":
        try:
            await _maybe_complete_mission(mission_id, completed_task_id)
        except Exception as e:
            # Non-fatal: mission stays open, but advance result stands.
            import logging
            logging.getLogger("workflow_engine.advance").warning(
                "mission completion check failed: %s", e,
            )
    return out


async def _maybe_complete_mission(mission_id: int, completed_task_id: int) -> None:
    """Mark a mission 'completed' when no non-terminal tasks remain.

    Excludes the workflow_advance task currently running (caller's
    completed_task_id points at the source task, not the advance task, so
    we also skip any pending workflow_advance mechanical rows on the same
    mission — they are bookkeeping, not work.)
    """
    from dabidabi import get_tasks_for_mission, get_mission
    from general_beckman import update_mission
    from dabidabi.times import db_now
    import json as _json

    mission = await get_mission(mission_id)
    if mission is None:
        return
    if mission.get("status") in ("completed", "cancelled", "failed"):
        return

    tasks = await get_tasks_for_mission(mission_id)
    terminal = {"completed", "skipped", "cancelled", "failed"}
    for t in tasks:
        status = t.get("status")
        if status in terminal:
            continue
        # Skip pending/in_progress workflow_advance rows — they are the
        # machinery that drove us here, not pending work.
        ctx_raw = t.get("context") or "{}"
        try:
            ctx = _json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
        except Exception:
            ctx = {}
        payload_action = (ctx.get("payload") or {}).get("action") if isinstance(ctx, dict) else None
        if payload_action == "workflow_advance":
            continue
        # Some non-terminal, non-advance task still pending — mission not done.
        return

    await update_mission(mission_id, status="completed", completed_at=db_now())

    # Z10-T3B: tear down the per-mission sandbox container + network.
    # Best-effort — failures must not break the completion path.
    try:
        from src.tools.shell import teardown_mission_container
        await teardown_mission_container(mission_id)
    except Exception as e:
        import logging
        logging.getLogger("workflow_engine.advance").debug(
            "mission container teardown failed (non-fatal): %s", e,
        )

    # Best-effort Telegram delivery — get_telegram() may raise if the bot
    # isn't initialised yet; failing here must not break advance().
    try:
        from src.app.telegram_bot import get_telegram
        tg = get_telegram()
        if tg is None:
            return

        # Find the "delivery" task: the last completed non-mechanical task
        # with a non-empty result. For shopping/product_research workflows
        # this is [2.1] deliver_product_research / [1.1] format_and_deliver.
        # For workflows without an explicit delivery step, falls through to
        # the summary-only message.
        final_result = ""
        # Bookkeeping agents run AFTER the real final step (the reviewer
        # grades it, the summarizer condenses artifacts). Their results are
        # JSON verdicts / structural summaries, not user-facing content —
        # skipping them prevents the verdict JSON from leaking out as
        # the mission's delivered message. (reviewer/summarizer children are
        # mission-less so they don't normally reach here — kept for defence.)
        _bookkeeping = {"mechanical", "reviewer", "summarizer"}
        for t in sorted(tasks, key=lambda x: x.get("id") or 0, reverse=True):
            if t.get("agent_type") in _bookkeeping:
                continue
            if t.get("status") != "completed":
                continue
            r = t.get("result") or ""
            if len(r.strip()) < 10:
                continue
            # A skip_when short-circuit completes with result
            # {"skipped": true, "reason": ...}. Such a row is NOT the delivered
            # output — without this guard a late conditional step that skips
            # (e.g. shopping_v3 2.3z compare_assemble on the variant path)
            # outranks the real delivery step by id and ships its skip JSON.
            try:
                _pj = _json.loads(r)
                if isinstance(_pj, dict) and _pj.get("skipped") is True:
                    continue
            except Exception:
                pass
            final_result = r
            break

        title = mission.get("title") or f"mission #{mission_id}"
        n_completed = sum(1 for t in tasks if t.get("status") == "completed")
        n_failed = sum(1 for t in tasks if t.get("status") == "failed")

        if final_result:
            # Primary: deliver the workflow's final payload. Keep it terse
            # — the completion summary goes below.
            # If the result is a JSON dict with a `formatted_text` key
            # (shopping_pipeline_v2 format_response step), unwrap it so
            # the user sees markdown, not raw JSON.
            body = final_result
            try:
                parsed = _json.loads(body)
                if isinstance(parsed, dict) and "formatted_text" in parsed:
                    body = parsed["formatted_text"]
            except Exception:
                pass
            if len(body) > 3800:
                body = body[:3800] + "\n...(truncated)"
            await tg.send_notification(body)

        summary = (
            f"\u2705 Mission #{mission_id} complete\n"
            f"**{title[:80]}**\n"
            f"{n_completed} done, {n_failed} failed"
        )
        await tg.send_notification(summary)
    except Exception:
        pass


def _parse_ctx(task: dict) -> dict:
    import json
    raw = task.get("context") or "{}"
    if isinstance(raw, dict):
        return dict(raw)
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}
