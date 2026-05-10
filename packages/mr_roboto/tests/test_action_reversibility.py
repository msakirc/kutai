"""Z10-T1B — Action.reversibility populated by dispatcher."""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, patch

import pytest

import mr_roboto


def _patch_submodule_func(monkeypatch, dotted: str, replacement) -> AsyncMock:
    """Patch a lazily-imported submodule function by replacing its attr in
    ``sys.modules``. The dispatcher uses ``from mr_roboto.<sub> import
    <fn>`` inside each `if action == ...` block, so the lookup goes
    through ``sys.modules['mr_roboto.<sub>'].<fn>``.
    """
    mod_name, attr = dotted.rsplit(".", 1)
    mod = sys.modules[mod_name]
    monkeypatch.setattr(mod, attr, replacement)
    return replacement


@pytest.mark.asyncio
async def test_git_commit_action_carries_full(monkeypatch) -> None:
    task = {
        "id": 3,
        "mission_id": 4,
        "title": "commit me",
        "payload": {
            "action": "git_commit",
            "result": {"ok": True},
            "critic_gate": False,  # skip the critic gate path
        },
    }
    mock_commit = AsyncMock(return_value={"committed": True, "empty": False, "message": "x"})
    monkeypatch.setattr(mr_roboto, "auto_commit", mock_commit)
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.reversibility == "full"


@pytest.mark.asyncio
async def test_notify_user_action_carries_irreversible(monkeypatch) -> None:
    task = {
        "id": 5,
        "mission_id": 6,
        "payload": {
            "action": "notify_user",
            "message": "hi",
            "critic_gate": False,
        },
    }
    mock_notify = AsyncMock(return_value={"sent": True})
    # Force submodule import then patch.
    import mr_roboto.notify_user as _nu_mod  # noqa
    _patch_submodule_func(monkeypatch, "mr_roboto.notify_user.notify_user", mock_notify)
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.reversibility == "irreversible"


@pytest.mark.asyncio
async def test_run_cmd_default_partial(monkeypatch) -> None:
    task = {
        "id": 7,
        "mission_id": 8,
        "payload": {
            "action": "run_cmd",
            "cmd": ["echo", "hi"],
            "workspace_path": "/tmp",
        },
    }
    mock_run = AsyncMock(return_value={
        "ok": True, "exit": 0, "stdout_tail": "hi", "stderr_tail": "",
        "duration_s": 0.0, "timed_out": False,
    })
    import mr_roboto.run_cmd as _rc_mod  # noqa
    _patch_submodule_func(monkeypatch, "mr_roboto.run_cmd.run_cmd", mock_run)
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.reversibility == "partial"


@pytest.mark.asyncio
async def test_run_cmd_override_irreversible_wins(monkeypatch) -> None:
    task = {
        "id": 9,
        "mission_id": 10,
        "payload": {
            "action": "run_cmd",
            "cmd": ["rm", "-rf", "build"],
            "workspace_path": "/tmp",
            "reversibility_override": "irreversible",
        },
    }
    mock_run = AsyncMock(return_value={
        "ok": True, "exit": 0, "stdout_tail": "", "stderr_tail": "",
        "duration_s": 0.0, "timed_out": False,
    })
    import mr_roboto.run_cmd as _rc_mod  # noqa
    _patch_submodule_func(monkeypatch, "mr_roboto.run_cmd.run_cmd", mock_run)
    action = await mr_roboto.run(task)
    # confirm dispatcher passed the intent through into the executor
    kwargs = mock_run.await_args.kwargs
    assert kwargs.get("reversibility_intent") == "irreversible"
    assert action.reversibility == "irreversible"


@pytest.mark.asyncio
async def test_unknown_action_default_partial() -> None:
    action = await mr_roboto.run({"id": 1, "payload": {"action": "not_a_thing"}})
    assert action.status == "failed"
    assert action.reversibility == "partial"


@pytest.mark.asyncio
async def test_invalid_override_is_ignored(monkeypatch) -> None:
    task = {
        "id": 11,
        "mission_id": 12,
        "payload": {
            "action": "git_commit",
            "result": {},
            "critic_gate": False,
            "reversibility_override": "totally-bogus",
        },
    }
    mock_commit = AsyncMock(return_value={"committed": True, "empty": False})
    monkeypatch.setattr(mr_roboto, "auto_commit", mock_commit)
    action = await mr_roboto.run(task)
    # Bogus override is dropped; registry "full" wins.
    assert action.reversibility == "full"
