"""Salako — mechanical dispatcher: non-LLM task executors."""
from __future__ import annotations

from salako.actions import Action
from salako.workspace_snapshot import snapshot_workspace
from salako.git_commit import auto_commit

__all__ = ["Action", "run", "snapshot_workspace", "auto_commit"]


async def run(task: dict) -> Action:
    """Route a mechanical task to the appropriate executor.

    ``task["payload"]["action"]`` selects the executor:

    - ``"workspace_snapshot"`` → :func:`salako.snapshot_workspace`
    - ``"git_commit"``         → :func:`salako.auto_commit`

    Unknown actions return an ``Action(status="failed", error=...)``; the
    orchestrator is responsible for marking the task failed.
    """
    payload = task.get("payload") or {}
    action = payload.get("action")

    if action == "workspace_snapshot":
        snap = await snapshot_workspace(
            mission_id=task["mission_id"],
            task_id=task["id"],
            workspace_path=payload["workspace_path"],
            repo_path=payload.get("repo_path"),
        )
        if snap is None:
            return Action(status="failed", error="snapshot failed")
        return Action(status="completed", result=snap)

    if action == "git_commit":
        await auto_commit(task, payload.get("result") or {})
        return Action(status="completed")

    if action == "clarify":
        from salako.clarify import clarify
        try:
            res = await clarify(task)
            # variant_choice that successfully sent a keyboard is waiting
            # on a user tap — must return needs_clarification so beckman's
            # result router leaves the row as waiting_human instead of
            # flipping it back to completed (which caused the mission to
            # advance past the clarify step and produce "no results" for
            # every clarify-gated shopping mission). Plain question-clarify
            # stays completed as before.
            if (isinstance(res, dict)
                    and res.get("status") == "needs_clarification"
                    and res.get("keyboard_sent")):
                return Action(status="needs_clarification", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "notify_user":
        from salako.notify_user import notify_user
        try:
            res = await notify_user(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "todo_reminder":
        from salako.todo_reminder import run as todo_reminder_run
        try:
            res = await todo_reminder_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "price_watch_check":
        from salako.price_watch_check import run as price_watch_run
        try:
            res = await price_watch_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "workflow_advance":
        from salako.workflow_advance import run as workflow_advance_run
        try:
            res = await workflow_advance_run(task)
            if res.get("status") == "failed":
                return Action(status="failed", error=res.get("error", ""))
            return Action(status=res.get("status", "completed"), result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    return Action(status="failed", error=f"unknown mechanical action: {action!r}")
