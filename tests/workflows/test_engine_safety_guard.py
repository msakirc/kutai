"""Tests: safety_guard.pre_action is wired into mr_roboto.run() mechanical dispatch."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import mr_roboto


@pytest.mark.asyncio
async def test_mr_roboto_blocks_force_push_run_cmd():
    """A run_cmd step whose cmd is `git push --force` is blocked by safety_guard."""
    task = {
        "id": 999,
        "mission_id": None,
        "title": "test_force_push",
        "payload": {
            "action": "run_cmd",
            "cmd": ["git", "push", "--force", "origin", "main"],
            "workspace_path": "/tmp/workspace",
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "blocked", (
        f"Expected 'blocked', got {action.status!r}. "
        "safety_guard.pre_action should have blocked git push --force before the executor ran."
    )
    assert action.error is not None
    assert "force_push" in (action.error or "") or "block" in (action.error or "").lower()


@pytest.mark.asyncio
async def test_mr_roboto_allows_safe_run_cmd():
    """A run_cmd step with a safe command passes the safety guard and reaches the executor."""
    task = {
        "id": 1,
        "mission_id": None,
        "title": "safe_cmd",
        "payload": {
            "action": "run_cmd",
            "cmd": ["echo", "hello"],
            "workspace_path": "/tmp/workspace",
            "require_exit_zero": False,
        },
    }
    # run() does `from mr_roboto.run_cmd import run_cmd as _run_cmd` locally;
    # patch via sys.modules to intercept the local import.
    import sys
    import importlib
    # Ensure the real module (not the name-shadowing top-level attr) is loaded.
    real_mod = importlib.import_module("mr_roboto.run_cmd")
    original_fn = real_mod.run_cmd
    mock_fn = AsyncMock(return_value={
        "exit": 0,
        "stdout_tail": "hello\n",
        "stderr_tail": "",
        "duration_s": 0.01,
        "timed_out": False,
        "ok": True,
    })
    real_mod.run_cmd = mock_fn
    try:
        action = await mr_roboto.run(task)
    finally:
        real_mod.run_cmd = original_fn
    assert action.status == "completed"
    mock_fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_mr_roboto_non_shell_action_not_blocked():
    """Non-shell actions (workspace_snapshot) are not blocked by the safety guard."""
    task = {
        "id": 2,
        "mission_id": 1,
        "payload": {"action": "workspace_snapshot", "workspace_path": "/tmp/ws"},
    }
    with patch("mr_roboto.snapshot_workspace", new_callable=AsyncMock) as mock_snap:
        mock_snap.return_value = {"hashes": {}, "commit_sha": "abc", "branch": "main"}
        action = await mr_roboto.run(task)
    assert action.status == "completed"


@pytest.mark.asyncio
async def test_mr_roboto_blocks_rm_rf_root():
    """rm -rf / is in blocklist; should be blocked."""
    task = {
        "id": 888,
        "mission_id": None,
        "title": "nuke",
        "payload": {
            "action": "run_cmd",
            "cmd": ["rm", "-rf", "/"],
            "workspace_path": "/tmp/workspace",
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "blocked"
