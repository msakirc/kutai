"""Dispatch-time metrics push.

Captures the side-effect telemetry that used to live inline in the deleted
Orchestrator._handle_complete (mission cost tracking, episodic memory,
model_stats, preference feedback, skill A/B metrics). Runs after
beckman.on_task_finished so the task row is already terminal when we record.

Any subroutine failure is logged and swallowed — telemetry must not
block the dispatch pump.
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("core.metrics_push")


async def push_metrics(task: dict, result: dict | None) -> None:
    """Best-effort: record model_stats, mission cost, episodic memory,
    metrics counters, and skill injection outcomes.

    All failures are caught and logged at DEBUG level.
    """
    for fn in (
        _push_model_stats,
        _push_mission_cost,
        _push_episodic_memory,
        _push_metrics_counter,
        _push_preference_feedback,
        _push_skill_injection,
    ):
        try:
            await fn(task, result)
        except Exception as e:
            logger.debug(f"{fn.__name__} failed: {e}")


# ── Subroutines ──────────────────────────────────────────────────────────────

async def _push_model_stats(task: dict, result: dict | None) -> None:
    """Record task-outcome signal for model_stats aggregation.

    Failures matter as much as successes for selector scoring — without
    them success_rate stays pinned at 1.0 and grading_perf_score collapses
    to a constant. We accept any terminal status, classify success vs
    failure via the result.status, and route through ``record_model_call``
    (the rich aggregator that updates avg_grade / success_rate /
    total_calls). The previously-used ``update_model_stats`` writes to a
    different schema and silently no-ops against the live model_stats
    table, which is why every row shows success_rate=1.00 — the per-LLM
    call writes from base.py never see failures.
    """
    if result is None:
        return
    status = result.get("status", "")
    if status not in ("completed", "ungraded", "failed", "timed_out", "dlq"):
        return
    model = result.get("model") or ""
    if not model:
        return
    try:
        from src.infra.db import record_model_call
        agent_type = task.get("agent_type", "executor")
        # success: completed/ungraded count as success (task finished —
        # ungraded means the grader couldn't run, not that work failed).
        # failed/timed_out/dlq count as failure.
        success = status in ("completed", "ungraded")
        cost = float(result.get("cost", 0.0) or 0.0)
        # latency proxy: iterations × avg_ms. record_model_call expects
        # seconds, not ms — convert explicitly.
        iterations = result.get("iterations") or 1
        latency_seconds = float(iterations) * 2.0
        # grade: only pass when the result actually carries it. With
        # grade=None record_model_call leaves avg_grade untouched. Most
        # call sites today don't propagate the grader verdict — fixing
        # that is a separate concern (see grading.py result wireup).
        raw_grade = result.get("grade")
        grade = float(raw_grade) if isinstance(raw_grade, (int, float)) else None
        await record_model_call(
            model=model,
            agent_type=agent_type,
            success=success,
            cost=cost,
            latency=latency_seconds,
            grade=grade,
        )
    except ImportError:
        pass


async def _push_mission_cost(task: dict, result: dict | None) -> None:
    """Update the mission cost blackboard — mirrors old Fix #8 logic."""
    mission_id = task.get("mission_id")
    if not mission_id or result is None:
        return
    cost = result.get("cost", 0.0)
    if cost <= 0:
        return
    try:
        from src.collaboration.blackboard import read_blackboard, write_blackboard
        from src.core.task_context import parse_context

        current = await read_blackboard(mission_id, "cost_tracking")
        if not isinstance(current, dict):
            current = {"total_cost": 0.0, "task_count": 0, "by_phase": {}}
        current["total_cost"] = current.get("total_cost", 0.0) + cost
        current["task_count"] = current.get("task_count", 0) + 1
        task_ctx = parse_context(task)
        phase = task_ctx.get("workflow_phase", "unknown") if isinstance(task_ctx, dict) else "unknown"
        phase_costs = current.get("by_phase", {})
        phase_costs[phase] = phase_costs.get(phase, 0.0) + cost
        current["by_phase"] = phase_costs
        await write_blackboard(mission_id, "cost_tracking", current)
    except (ImportError, TypeError):
        pass


async def _push_episodic_memory(task: dict, result: dict | None) -> None:
    """Store task outcome in episodic memory for future RAG retrieval."""
    if result is None or result.get("status") != "completed":
        return
    try:
        from src.memory.episodic import store_task_result
        result_text = result.get("result") or result.get("content") or ""
        model = result.get("model") or "unknown"
        cost = result.get("cost", 0.0)
        await store_task_result(
            task=task,
            result=result_text,
            model=model,
            cost=cost,
            duration=0.0,
            success=True,
        )
    except ImportError:
        pass


async def _push_metrics_counter(task: dict, result: dict | None) -> None:
    """Record in-memory metrics counters (record_task_complete)."""
    if result is None:
        return
    status = result.get("status", "")
    if status not in ("completed", "ungraded"):
        return
    try:
        from src.infra.metrics import record_task_complete
        model = result.get("model") or ""
        cost = result.get("cost", 0.0)
        record_task_complete(model=model, cost=cost)
    except ImportError:
        pass


async def _push_preference_feedback(task: dict, result: dict | None) -> None:
    """Record implicit acceptance feedback for the completed task."""
    if result is None or result.get("status") != "completed":
        return
    try:
        from src.memory.preferences import record_feedback
        await record_feedback(task, "accepted")
    except (ImportError, TypeError):
        pass


async def _push_skill_injection(task: dict, result: dict | None) -> None:
    """Record which injected skills were successfully used."""
    if result is None or result.get("status") == "ungraded":
        return
    try:
        from src.core.task_context import parse_context
        from src.memory.skills import record_injection_success
        task_ctx = parse_context(task)
        injected = task_ctx.get("injected_skills", [])
        if injected:
            await record_injection_success(injected)
    except ImportError:
        pass
