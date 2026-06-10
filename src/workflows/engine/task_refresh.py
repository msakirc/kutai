# task_refresh.py — dispatch-time workflow-step refresh helpers.
#
# Task rows freeze their config at expansion time. Edits to a workflow's
# JSON do not propagate to already-expanded rows. coulson's
# `_refresh_workflow_step_config` re-syncs description + context fields at
# execution time, but it DELIBERATELY excludes ``agent_type`` (see its
# docstring) because the agent CLASS is selected in the orchestrator's
# `_dispatch` — BEFORE the runtime executes — so a refresh inside the agent
# would be too late. This module owns that complementary piece: re-resolving
# a workflow step's ``agent`` at dispatch entry, before ``get_agent()`` picks
# the class.
import json

from src.infra.logging_config import get_logger

logger = get_logger("core.orchestrator")


async def refresh_workflow_agent_type(task: dict, agent_type: str) -> str:
    """Re-resolve a workflow step's agent_type from live workflow JSON.

    Task rows freeze agent_type at expansion time. When a workflow JSON edit
    changes a step's agent (e.g. the 2026-04-25 sweep moved 24 planner→
    array/object steps to analyst), existing rows keep dispatching the old
    agent. Mission 46 tasks 2939, 2942 burned 4+ retries each as planner
    emitting subtask plans for array/object schemas — the schema validator
    kept failing the same shape it could never produce.

    Returns the (possibly updated) agent_type. On change, persists the new
    value to the task row and mutates ``task["agent_type"]`` in place so the
    rest of dispatch reads the same value. Never raises — a refresh failure
    must not break dispatch; the frozen agent_type is returned unchanged.
    """
    try:
        ctx_raw_for_agent = task.get("context") or "{}"
        _tctx = json.loads(ctx_raw_for_agent) if isinstance(ctx_raw_for_agent, str) else ctx_raw_for_agent
        if isinstance(_tctx, str):
            _tctx = json.loads(_tctx)
        if not (isinstance(_tctx, dict) and _tctx.get("is_workflow_step")):
            return agent_type
        _step_id = _tctx.get("workflow_step_id")
        _mid = task.get("mission_id")
        if not (_step_id and _mid):
            return agent_type
        from src.infra.db import get_db
        _mdb = await get_db()
        _mcur = await _mdb.execute(
            "SELECT context FROM missions WHERE id = ?", (_mid,),
        )
        _mrow = await _mcur.fetchone()
        await _mcur.close()
        _mctx = {}
        if _mrow and _mrow[0]:
            try:
                _mctx = json.loads(_mrow[0])
                if isinstance(_mctx, str):
                    _mctx = json.loads(_mctx)
            except (json.JSONDecodeError, TypeError):
                _mctx = {}
        _wf_name = (
            _mctx.get("workflow_name") if isinstance(_mctx, dict) else None
        ) or "i2p_v3"
        from src.workflows.engine.loader import load_workflow
        _wf = load_workflow(_wf_name)
        _step = _wf.get_step(_step_id)
        if not _step:
            return agent_type
        _live_agent = _step.get("agent")
        if (_live_agent
                and _live_agent != agent_type
                and _live_agent != "mechanical"
                and agent_type != "mechanical"):
            from src.infra.db import update_task
            await update_task(task["id"], agent_type=_live_agent)
            logger.info(
                f"[Task #{task['id']}] agent_type refresh: "
                f"{agent_type} → {_live_agent} "
                f"(step={_step_id}, wf={_wf_name})"
            )
            task["agent_type"] = _live_agent
            return _live_agent
    except Exception as _e:
        logger.debug(f"agent_type refresh failed #{task.get('id')}: {_e}")
    return agent_type
