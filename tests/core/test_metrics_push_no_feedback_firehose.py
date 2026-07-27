import pytest
import src.core.metrics_push as mp


@pytest.mark.asyncio
async def test_completed_task_does_not_fire_preference_feedback(monkeypatch):
    called = {"feedback": False}

    async def _spy(task, result):
        called["feedback"] = True

    monkeypatch.setattr(mp, "_push_preference_feedback", _spy)

    async def _noop(task, result):
        return None

    for name in ("_push_model_stats", "_push_mission_cost",
                 "_push_episodic_memory", "_push_metrics_counter",
                 "_push_skill_injection"):
        monkeypatch.setattr(mp, name, _noop)

    await mp.push_metrics({"id": 1, "agent_type": "coder"},
                          {"status": "completed", "model": "m"})
    assert called["feedback"] is False, "implicit user_feedback firehose must not fire per-task"
