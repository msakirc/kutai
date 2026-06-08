# dispatch_prep.py — pre-dispatch task-context preparation.
#
# Seam between agent-CLASS knowledge (src.agents) and Beckman's post-hook
# determination (general_beckman.posthooks.determine_posthooks). Lives in
# src/core because that layer already depends on BOTH; placing it in
# general_beckman would invert layering (task-master → agent registry) and
# coulson is the wrong phase (it runs at execution, this runs pre-dispatch).
import json

from src.infra.logging_config import get_logger
from .task_context import parse_context

logger = get_logger("core.orchestrator")


async def bridge_self_reflection(task: dict, agent_type: str, get_agent) -> None:
    """Persist ``profile.enable_self_reflection`` (class attr) into task context.

    SP3b Task 7 moved per-agent self-reflection from coulson's inline path
    (which fired on the agent CLASS attr ``profile.enable_self_reflection``)
    to a Beckman post-hook gated by ``determine_posthooks()``. But that gate
    reads the DB task row + parsed context — it has NO access to the agent
    profile object. The flag lives ONLY on the class (e.g.
    ``CoderAgent.enable_self_reflection = True``), so it was never visible to
    the completion path → ``self_reflect`` NEVER spawned (reflection silently
    dead for every code-emitting agent).

    Bridge it here: ``get_agent()`` has the resolved profile, and
    ``on_task_finished`` re-reads the row via ``get_task()`` AFTER dispatch —
    so PERSIST it to the context column now. Only stamp ``True``; never
    clutter context with ``False``. Skip mechanical tasks entirely — they go
    through mr_roboto (no agent profile, never self-reflect), so touching
    get_agent for them is pointless work. Mutates ``task["context"]`` in
    place so the agent execution path reads the same context. Never raises.

    ``get_agent`` is injected by the caller (the orchestrator's module-level
    ``get_agent``) rather than imported here so that tests patching
    ``src.core.orchestrator.get_agent`` keep covering this path — otherwise a
    bridge with its own import would resolve the REAL profile and fire a live
    ``update_task`` against the prod DB during those tests.
    """
    try:
        _refl_ctx = parse_context(task)
        _is_mech_dispatch = (
            task.get("runner") == "mechanical"
            or task.get("executor") == "mechanical"
            or _refl_ctx.get("executor") == "mechanical"
            or agent_type == "mechanical"
        )
        if _is_mech_dispatch:
            return
        _profile = get_agent(agent_type)
        if getattr(_profile, "enable_self_reflection", False):
            if _refl_ctx.get("enable_self_reflection") is not True:
                _refl_ctx["enable_self_reflection"] = True
                from src.infra.db import update_task as _ut_refl
                await _ut_refl(task["id"], context=json.dumps(_refl_ctx))
                # Keep the in-memory task dict consistent so the agent
                # execution path below reads the same context.
                task["context"] = json.dumps(_refl_ctx)
    except Exception as _e:
        logger.warning(
            "self_reflect bridge failed task #%s (agent_type=%s) — "
            "self-reflection will be skipped for this task: %s",
            task.get("id"), agent_type, _e,
        )
