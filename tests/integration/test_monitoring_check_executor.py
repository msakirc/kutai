"""Integration test for the monitoring_check salako executor.

Verifies:
- URL failure enqueues a notify_user sub-task with correct parent_task_id
- No direct tg.send_notification calls are made
- Sub-task spec contains the alert payload
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.mark.integration
class TestMonitoringCheckExecutor:

    def test_url_failure_enqueues_notify_user_subtask(self, temp_db):
        """A failing URL must enqueue a notify_user mechanical sub-task
        in the tasks table with correct parent_task_id and agent_type."""

        async def _run():
            from src.infra.db import add_task, get_db
            import src.infra.db as db_mod

            # Create a parent task to simulate a cron-fired monitoring_check task
            parent_id = await add_task(
                title="monitoring_check",
                description="URL uptime and GitHub repo poll",
                agent_type="mechanical",
                depends_on=[],
            )
            assert parent_id is not None

            task = {"id": parent_id, "payload": {"action": "monitoring_check"}}

            # Mock check_url_uptime to return failure
            async def fake_check_url(url):
                return False, "Connection refused"

            # Ensure _url_statuses treats the URL as previously up so alert fires
            import src.infra.monitoring as mon_mod
            mon_mod._url_statuses.clear()  # start clean — default was_up=True

            with patch.dict(os.environ, {
                "MONITOR_URLS": "http://nope.invalid",
                "MONITOR_GITHUB_REPOS": "",
            }):
                with patch("src.infra.monitoring.check_url_uptime", side_effect=fake_check_url):
                    # Also mock notify_user send_notification to confirm no direct TG calls
                    mock_tg = MagicMock()
                    mock_tg.send_notification = AsyncMock()

                    with patch("salako.notify_user.get_telegram", return_value=mock_tg):
                        from salako.executors.monitoring_check import run as exec_run
                        result = await exec_run(task)

            assert result["urls_checked"] == 1
            assert result["alerts"] >= 1
            assert result["enqueued"] >= 1

            # Verify sub-task was inserted in the DB
            db = await get_db()
            cursor = await db.execute(
                """SELECT id, title, agent_type, context, parent_task_id
                   FROM tasks
                   WHERE parent_task_id = ? AND agent_type = 'mechanical'""",
                (parent_id,),
            )
            rows = await cursor.fetchall()
            assert len(rows) >= 1, "Expected at least one notify_user sub-task"

            sub = dict(rows[0])
            assert sub["parent_task_id"] == parent_id
            assert sub["agent_type"] == "mechanical"

            ctx = json.loads(sub["context"]) if isinstance(sub["context"], str) else sub["context"]
            assert ctx.get("executor") == "mechanical"
            assert ctx["payload"]["action"] == "notify_user"
            msg = ctx["payload"].get("message", "")
            assert "nope.invalid" in msg or "Connection refused" in msg or "DOWN" in msg

            # Confirm no direct Telegram send was called
            mock_tg.send_notification.assert_not_called()

        run_async(_run())

    def test_no_urls_configured_is_noop(self, temp_db):
        """When no MONITOR_URLS or MONITOR_GITHUB_REPOS are set, executor
        returns success with zero alerts and no sub-tasks enqueued."""

        async def _run():
            from src.infra.db import add_task, get_db

            parent_id = await add_task(
                title="monitoring_check",
                description="",
                agent_type="mechanical",
                depends_on=[],
            )
            task = {"id": parent_id, "payload": {"action": "monitoring_check"}}

            with patch.dict(os.environ, {"MONITOR_URLS": "", "MONITOR_GITHUB_REPOS": ""}):
                from salako.executors.monitoring_check import run as exec_run
                result = await exec_run(task)

            assert result["alerts"] == 0
            assert result["enqueued"] == 0

            db = await get_db()
            cursor = await db.execute(
                "SELECT id FROM tasks WHERE parent_task_id = ? AND agent_type = 'mechanical'",
                (parent_id,),
            )
            rows = await cursor.fetchall()
            assert len(rows) == 0

        run_async(_run())

    def test_url_recovery_enqueues_recovered_alert(self, temp_db):
        """When a URL transitions from down→up, a RECOVERED alert sub-task is enqueued."""

        async def _run():
            from src.infra.db import add_task, get_db
            import src.infra.monitoring as mon_mod

            # Pre-seed URL as down
            mon_mod._url_statuses["http://example.local"] = False

            parent_id = await add_task(
                title="monitoring_check",
                description="",
                agent_type="mechanical",
                depends_on=[],
            )
            task = {"id": parent_id, "payload": {"action": "monitoring_check"}}

            async def fake_up(url):
                return True, "HTTP 200"

            with patch.dict(os.environ, {
                "MONITOR_URLS": "http://example.local",
                "MONITOR_GITHUB_REPOS": "",
            }):
                with patch("src.infra.monitoring.check_url_uptime", side_effect=fake_up):
                    from salako.executors.monitoring_check import run as exec_run
                    result = await exec_run(task)

            assert result["alerts"] >= 1
            assert result["enqueued"] >= 1

            db = await get_db()
            cursor = await db.execute(
                "SELECT context FROM tasks WHERE parent_task_id = ? AND agent_type = 'mechanical'",
                (parent_id,),
            )
            rows = await cursor.fetchall()
            assert len(rows) >= 1
            ctx = json.loads(rows[0][0])
            assert "RECOVERED" in ctx["payload"]["message"]

            # Cleanup state
            mon_mod._url_statuses.clear()

        run_async(_run())
