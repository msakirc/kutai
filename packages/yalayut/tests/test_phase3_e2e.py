"""Phase 3 end-to-end — preempt task flows through mr_roboto to run_recipe."""
import pytest

import mr_roboto


@pytest.mark.asyncio
async def test_preempt_task_runs_recipe_via_mechanical_lane(monkeypatch, tmp_path):
    """intersect-shaped preempt task -> mr_roboto.run -> yalayut.run_recipe."""
    marker = tmp_path / "scaffolded.txt"
    manifest = {
        "name": "cc-probe",
        "kind": "shell_recipe",
        "mechanizable": True,
        "invocation": {
            "steps": [
                {"cmd": f'python -c "open(r\'{marker}\',\'w\').write(\'done\')"'},
            ]
        },
        "artifacts": [str(marker)],
    }

    async def fake_load(recipe_id):
        return {"id": recipe_id, "name": "cc-probe", "manifest": manifest,
                "mechanizable": True, "vet_tier": 0,
                "workspace_path": str(tmp_path)}

    monkeypatch.setattr("yalayut.executor._load_recipe_row", fake_load)

    # Task shaped exactly as intersect routes a preempt to the mechanical lane.
    task = {
        "agent_type": "mechanical",
        "mission_id": None,
        "context": {
            "payload": {
                "action": "yalayut_recipe",
                "recipe_id": 55,
                "args": {"workspace_path": str(tmp_path)},
            }
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed", action.error
    assert action.result["ok"] is True
    assert marker.read_text() == "done"
