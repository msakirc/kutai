"""Host-path tests for the Z6 P2 fixes from the 2026-05-18 sweep.

Z6 P2 — audit_completeness_check post-hook path was broken. The cron
path worked (mr_roboto/__init__.py handles action == "audit_completeness_check"
via the inline scanner), but the post-hook dispatch was supposed to
route through the general_beckman handler module — and the kind was not
in the post-hook handler tuple. A post-hook-triggered
audit_completeness_check fell through to the cron scanner instead of
the per-source-task handler, so the source task never got a verdict.

Fix: when action == "audit_completeness_check" arrives with
source_task_id set (post-hook always sets it), delegate to
general_beckman.posthook_handlers.audit_completeness_check.handle().
Cron path (no source_task_id) continues to use the inline scanner.

Z6 P2 — vendor_call tool visibility — verified in same sweep:
TOOL_REGISTRY merges _optional_tools via splat (src/tools/__init__.py:1339),
so vendor_call IS visible. Test pins this.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch, AsyncMock


class TestZ6P2VendorCallToolVisible(unittest.TestCase):
    """vendor_call lives in _optional_tools but is merged into TOOL_REGISTRY."""

    def test_vendor_call_resolves_via_tool_registry(self):
        from src.tools import TOOL_REGISTRY
        self.assertIn("vendor_call", TOOL_REGISTRY)
        spec = TOOL_REGISTRY["vendor_call"]
        self.assertTrue(callable(spec["function"]))


class TestZ6P2AuditCompletenessPosthookDispatch(unittest.IsolatedAsyncioTestCase):
    """Post-hook path (with source_task_id) must reach the handler module."""

    async def test_posthook_path_delegates_to_handler_module(self):
        from mr_roboto import run as mr_run
        from mr_roboto.actions import Action

        async def _fake_handle(task, result):
            return {"status": "ok", "gaps_found": 0, "from": "handler"}

        with patch(
            "general_beckman.posthook_handlers.audit_completeness_check.handle",
            _fake_handle,
        ), patch(
            "src.infra.db.get_task",
            AsyncMock(return_value={"id": 42, "agent_type": "coder"}),
        ):
            task = {
                "id": 1, "mission_id": 1,
                "payload": {
                    "action": "audit_completeness_check",
                    "source_task_id": 42,
                },
                "context": {},
            }
            res: Action = await mr_run(task)
            self.assertEqual(res.status, "completed")
            self.assertEqual(res.result.get("from"), "handler")

    async def test_cron_path_still_uses_inline_scanner(self):
        """No source_task_id → falls through to pending_audit_gaps cron path.

        Verifies the post-hook discrimination doesn't break the cron caller.
        """
        from mr_roboto import run as mr_run

        async def _fake_gaps(window_minutes=5):
            return []  # no gaps → fast completed path

        with patch("mr_roboto.audit_log.pending_audit_gaps", _fake_gaps):
            task = {
                "id": 2, "mission_id": 1,
                "payload": {
                    "action": "audit_completeness_check",
                    "window_minutes": 5,
                },
                "context": {},
            }
            res = await mr_run(task)
            self.assertEqual(res.status, "completed")
            self.assertEqual(res.result.get("gaps_found"), 0)
            self.assertTrue(res.result.get("ok"))


if __name__ == "__main__":
    unittest.main()
