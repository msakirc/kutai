import os
import pytest


@pytest.mark.asyncio
async def test_force_push_step_blocked_end_to_end(tmp_path):
    """End-to-end: a mr_roboto task that requests force-push should NOT execute."""
    from mr_roboto import run

    task = {
        "id": 1,
        "agent_type": "mechanical",
        "title": "danger_force_push",
        "payload": {
            "action": "run_cmd",
            # Use "command" key — safety_guard reads both "cmd" and "command".
            "command": "git push --force origin main",
        },
        "reversibility": "partial",
    }

    result = await run(task)
    # safety_guard hook should block before executor runs.
    if isinstance(result, dict):
        status = result.get("status")
    else:
        status = getattr(result, "status", None)
    assert status == "blocked", f"expected blocked, got {status}"


@pytest.mark.asyncio
async def test_innocuous_step_proceeds(tmp_path):
    """Sanity: a non-dangerous mechanical command isn't blocked by safety_guard."""
    from mr_roboto import run

    task = {
        "id": 2,
        "agent_type": "mechanical",
        "title": "safe_step",
        "payload": {
            "action": "run_cmd",
            # Use "cmd" (argv list) — executor reads this key.
            # Pass workspace_path so run_cmd doesn't fail with "no mission_id".
            "cmd": ["python", "--version"],
            "workspace_path": str(tmp_path),
        },
        "reversibility": "full",
    }
    result = await run(task)
    if isinstance(result, dict):
        status = result.get("status")
    else:
        status = getattr(result, "status", None)
    # Should NOT be "blocked"; safety_guard let it through.
    assert status != "blocked"
