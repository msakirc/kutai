"""SP3b Task 7 critical: profile.enable_self_reflection → task context bridge.

SP3b moved per-agent self-reflection from coulson's inline path (which fired
on the agent CLASS attr ``profile.enable_self_reflection``) to a Beckman
post-hook gated by ``determine_posthooks()``. That gate reads the DB task row +
its parsed context — it has NO access to the agent profile object. The flag
lives ONLY on the class (e.g. ``CoderAgent.enable_self_reflection = True``), so
without a bridge it is never visible to the completion path and ``self_reflect``
NEVER spawns — reflection silently dead for every code-emitting agent.

These tests drive the REAL bridge in ``Orchestrator._dispatch`` against a real
DB task row with a real agent profile (NOT a hand-stamped context dict — that
hand-stamping is exactly what masked the bug). They assert:
  1. After dispatch, the PERSISTED task context carries
     ``enable_self_reflection: True`` for an agent whose profile has the flag
     (coder), and ``determine_posthooks`` on that persisted context includes
     ``self_reflect``.
  2. An agent WHOSE PROFILE LACKS the flag (assistant — inherits False) does
     NOT get ``enable_self_reflection`` stamped, and ``determine_posthooks``
     does NOT include ``self_reflect``.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from src.core.orchestrator import Orchestrator
from general_beckman.posthooks import determine_posthooks
from general_beckman.task_context import parse_context


def _make_orch() -> Orchestrator:
    """Build an Orchestrator without starting any background loops."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.telegram = None
    orch.shutdown_event = None
    orch._shutting_down = False
    orch._running_futures = []
    orch.running = False
    return orch


async def _reset_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    from src.infra.db import init_db
    await init_db()


async def _dispatch_real_bridge(orch, task_row: dict):
    """Drive _dispatch up to (but not through) on_task_finished.

    Patches:
      - coulson.execute so no LLM runs (we only exercise the bridge). Task 5.5
        relocated agent execution off ``profile.execute`` to
        ``coulson.execute(get_agent(agent_type), task)``; the profile no longer
        carries an ``.execute`` attribute, so the execution seam to mock is now
        ``src.core.orchestrator.coulson.execute``.
      - on_task_finished so the test stops once the bridge has persisted;
      - inject_chain_context as an identity no-op (it never persists context
        and would otherwise read sibling rows from the DB).
    get_agent is NOT patched — the bridge must read the REAL profile class
    attribute. We only patch the execution call, never the bridge.
    """
    async def _fake_execute(_profile, _task):
        return {"status": "completed", "result": "ok"}

    async def _identity_inject(t):
        return t

    with patch("src.core.orchestrator.coulson.execute",
               new=AsyncMock(side_effect=_fake_execute)), \
         patch("general_beckman.on_task_finished", new_callable=AsyncMock), \
         patch("src.core.orchestrator.inject_chain_context",
               new_callable=AsyncMock, side_effect=_identity_inject), \
         patch("src.core.orchestrator.release_task_locks", new_callable=AsyncMock):
        await orch._dispatch(task_row)


@pytest.mark.asyncio
async def test_bridge_persists_flag_and_self_reflect_fires_for_coder(tmp_path, monkeypatch):
    """coder profile has enable_self_reflection=True → bridge persists it →
    determine_posthooks(persisted_ctx) includes self_reflect.

    No manual injection: the task row is created WITHOUT the flag; the bridge
    must put it there from CoderAgent.enable_self_reflection.
    """
    await _reset_db(tmp_path, monkeypatch)
    from src.infra.db import add_task, get_task

    # Sanity: the real profile carries the flag (not a stub).
    from src.agents import get_agent
    assert getattr(get_agent("coder"), "enable_self_reflection", False) is True

    task_id = await add_task(
        title="build feature",
        description="",
        agent_type="coder",
        mission_id=1,
        # Deliberately NO enable_self_reflection here — the bridge must add it.
        context={"requires_grading": True},
    )
    row = await get_task(task_id)
    assert "enable_self_reflection" not in parse_context(row)  # RED-before precondition

    orch = _make_orch()
    await _dispatch_real_bridge(orch, dict(row))

    # The bridge PERSISTED the flag to the DB row that on_task_finished re-reads.
    refreshed = await get_task(task_id)
    ctx = parse_context(refreshed)
    assert ctx.get("enable_self_reflection") is True

    # And determine_posthooks — which runs in on_task_finished against exactly
    # this persisted (task, ctx) pair — now spawns self_reflect.
    kinds = determine_posthooks(refreshed, ctx, {"status": "completed"})
    assert "self_reflect" in kinds


@pytest.mark.asyncio
async def test_bridge_skips_flag_for_agent_without_reflection(tmp_path, monkeypatch):
    """assistant profile inherits enable_self_reflection=False → no stamp →
    determine_posthooks does NOT include self_reflect."""
    await _reset_db(tmp_path, monkeypatch)
    from src.infra.db import add_task, get_task

    # Sanity: the real profile does NOT carry the flag.
    from src.agents import get_agent
    assert getattr(get_agent("assistant"), "enable_self_reflection", False) is False

    task_id = await add_task(
        title="answer question",
        description="",
        agent_type="assistant",
        mission_id=1,
        context={"requires_grading": True},
    )
    row = await get_task(task_id)

    orch = _make_orch()
    await _dispatch_real_bridge(orch, dict(row))

    refreshed = await get_task(task_id)
    ctx = parse_context(refreshed)
    # Flag never stamped (kept out of context when False — no clutter).
    assert "enable_self_reflection" not in ctx

    kinds = determine_posthooks(refreshed, ctx, {"status": "completed"})
    assert "self_reflect" not in kinds
    # Regression guard: a normal LLM agent still gets the terminal grade gate.
    assert "grade" in kinds
