"""End-to-end: a profile dispatches through coulson.execute (Prompt Foundry Task 5.5).

After Task 5.5, the orchestrator no longer calls ``get_agent(type).execute(task)``
— it calls ``coulson.execute(get_agent(type), task)``. This test proves that
contract works through the REAL ``coulson.execute`` path: it runs the full setup
phase (db prompt override, task-ctx parse, tools-hint / auto-strip, needs_real_tools
gate, execution-pattern routing, and the restore-on-finally) and returns a dict —
with the LLM-call seam mocked so no model loads.

The lowest seam that keeps execute's real body intact while skipping the model is
``coulson._react_run`` (the react-loop dispatch target ``coulson.execute`` invokes
for ``execution_pattern == "react_loop"``). Patching it means everything in
``coulson.execute`` before/after the loop is exercised for real; the loop itself
(which would select + load a model and call the dispatcher) is stubbed.
"""
import pytest


@pytest.mark.asyncio
async def test_profile_dispatches_through_coulson(monkeypatch):
    import coulson
    from src.agents import get_agent

    p = get_agent("summarizer")
    # summarizer is now a pure-data Foundry Profile (no .execute()); the
    # orchestrator path calls coulson.execute(profile, task) to drive it.
    assert not hasattr(p, "execute"), "summarizer must be pure data (no .execute)"

    captured = {}

    async def fake_react_run(profile, task, progress_callback=None):
        # Proves coulson.execute reached the dispatch seam with our profile/task
        # after running its real setup phase.
        captured["profile_name"] = profile.name
        captured["task_id"] = task.get("id")
        return {"status": "completed", "result": "ok", "model": "x", "cost": 0.0}

    monkeypatch.setattr(coulson, "_react_run", fake_react_run, raising=True)

    out = await coulson.execute(
        p,
        {
            "id": 1,
            "title": "t",
            "description": "summarize x",
            "context": {},
            "agent_type": "summarizer",
        },
    )

    assert isinstance(out, dict)
    assert out["status"] == "completed"
    assert captured["profile_name"] == "summarizer"
    assert captured["task_id"] == 1
