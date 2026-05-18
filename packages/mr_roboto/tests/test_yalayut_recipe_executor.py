"""yalayut_recipe mechanical executor — body + dispatch reachability."""
import pytest

import mr_roboto
from mr_roboto.executors.yalayut_recipe import run as yalayut_recipe_run


@pytest.mark.asyncio
async def test_executor_body_calls_run_recipe(monkeypatch):
    captured = {}

    async def fake_run_recipe(recipe_id, args):
        captured["recipe_id"] = recipe_id
        captured["args"] = args
        return {"ok": True, "recipe_id": recipe_id, "steps": [{"exit": 0}],
                "artifacts_present": ["x"], "artifacts_missing": [], "reason": None}

    monkeypatch.setattr("yalayut.run_recipe", fake_run_recipe)
    task = {"context": {"payload": {"action": "yalayut_recipe",
                                    "recipe_id": 12, "args": {"db": "postgres"}}}}
    res = await yalayut_recipe_run(task)
    assert res["ok"] is True
    assert captured["recipe_id"] == 12
    assert captured["args"]["db"] == "postgres"


@pytest.mark.asyncio
async def test_executor_body_missing_recipe_id():
    task = {"context": {"payload": {"action": "yalayut_recipe", "args": {}}}}
    res = await yalayut_recipe_run(task)
    assert res["ok"] is False
    assert "recipe_id" in res["reason"]


@pytest.mark.asyncio
async def test_dispatch_reaches_executor(monkeypatch):
    """mr_roboto.run() routes action=yalayut_recipe to the executor."""
    async def fake_run_recipe(recipe_id, args):
        return {"ok": True, "recipe_id": recipe_id, "steps": [],
                "artifacts_present": [], "artifacts_missing": [], "reason": None}

    monkeypatch.setattr("yalayut.run_recipe", fake_run_recipe)
    task = {
        "agent_type": "mechanical",
        "context": {"payload": {"action": "yalayut_recipe",
                                 "recipe_id": 4, "args": {}}},
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["ok"] is True
