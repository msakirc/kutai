"""Host-path tests for the 2026-05-18 wiring-sweep P0/P1 fixes.

Each test asserts the dead-pipeline symptom flagged in the sweep handoff is
actually closed end-to-end — not just unit-shape green. The unit suites all
passed *with* these bugs; that is the whole reason the sweep was needed.

Coverage:
- Z10 P0 — /mission_cost has a single working definition + handler row.
- Z10 P1 — set_task_confidence persists numeric+derived categorical to the
           tasks row so record_confidence_claim has something to read.
- Z4  P1 — apply.py reads PostHookVerdict.raw (not .result) for visual_review.
- Z2  P1 — inject_lessons present in POST_HOOK_REGISTRY and dispatched via
           _posthook_agent_and_payload as a mechanical action.
"""

from __future__ import annotations

import asyncio
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestZ10P0MissionCostDup(unittest.TestCase):
    """Z10 P0 — duplicate /mission_cost def shadowed the T2A version with a
    stub that imported a non-existent module. Sweep handoff 2026-05-18."""

    def test_single_cmd_mission_cost_definition(self):
        src = (REPO_ROOT / "src/app/telegram_bot.py").read_text(encoding="utf-8")
        defs = len(re.findall(r"async def cmd_mission_cost\b", src))
        self.assertEqual(defs, 1, "expected exactly one cmd_mission_cost def")

    def test_single_mission_cost_command_handler(self):
        src = (REPO_ROOT / "src/app/telegram_bot.py").read_text(encoding="utf-8")
        handlers = len(
            re.findall(r"CommandHandler\(\"mission_cost\"", src)
        )
        self.assertEqual(handlers, 1, "expected exactly one mission_cost handler row")

    def test_surviving_def_uses_real_formatter(self):
        """The kept definition must import format_mission_cost from the real
        module, not the placeholder src.app.mission_cost stub path."""
        src = (REPO_ROOT / "src/app/telegram_bot.py").read_text(encoding="utf-8")
        idx = src.find("async def cmd_mission_cost")
        body = src[idx: idx + 2000]
        self.assertIn("src.infra.cost_wiring", body)
        self.assertNotIn("Stub response.", body)


class TestZ10P1ConfidencePersistence(unittest.IsolatedAsyncioTestCase):
    """Z10 P1 — confidence_categorical/_numeric columns existed but no
    production code wrote to them, so record_confidence_claim was always
    returning None and confidence_outcomes stayed empty."""

    async def test_set_task_confidence_writes_numeric_and_derives_categorical(self):
        from src.infra.db import get_db, set_task_confidence
        db = await get_db()
        cur = await db.execute(
            "INSERT INTO tasks (title, agent_type, status) "
            "VALUES ('z10-p1-host-path', 'coder', 'completed')"
        )
        tid = cur.lastrowid
        await db.commit()
        try:
            await set_task_confidence(tid, numeric=4.2)
            row = await (await db.execute(
                "SELECT confidence_numeric, confidence_categorical "
                "FROM tasks WHERE id = ?",
                (tid,),
            )).fetchone()
            self.assertEqual(row[0], 4.2)
            self.assertEqual(row[1], "high")
        finally:
            await db.execute("DELETE FROM tasks WHERE id = ?", (tid,))
            await db.commit()

    async def test_record_confidence_claim_now_sees_the_write(self):
        """Round-trip: write via set_task_confidence → record_confidence_claim
        returns a non-None claim id (previously always None)."""
        from src.infra.db import (
            get_db, set_task_confidence, record_confidence_claim,
        )
        db = await get_db()
        cur = await db.execute(
            "INSERT INTO tasks (title, agent_type, status) "
            "VALUES ('z10-p1-roundtrip', 'coder', 'completed')"
        )
        tid = cur.lastrowid
        await db.commit()
        try:
            self.assertIsNone(
                await record_confidence_claim(tid),
                "before write: claim should be None",
            )
            await set_task_confidence(tid, numeric=3.1)
            claim_id = await record_confidence_claim(tid)
            self.assertIsNotNone(claim_id, "after write: claim should exist")
        finally:
            await db.execute(
                "DELETE FROM confidence_outcomes WHERE task_id = ?", (tid,)
            )
            await db.execute("DELETE FROM tasks WHERE id = ?", (tid,))
            await db.commit()


class TestZ4P1VisualReviewPayloadField(unittest.TestCase):
    """Z4 P1 — apply.py was reading getattr(a, "result", ...) but
    PostHookVerdict only carries `raw`, so captured_paths was always [] and
    the founder Telegram album never sent."""

    def test_apply_visual_review_uses_a_raw(self):
        src = (
            REPO_ROOT
            / "packages/general_beckman/src/general_beckman/apply.py"
        ).read_text(encoding="utf-8")
        idx = src.find("enqueue_visual_review_notice")
        self.assertGreater(idx, 0)
        block = src[idx: idx + 800]
        self.assertIn("a.raw", block)
        self.assertNotIn('getattr(a, "result"', block)


class TestZ2P1InjectLessonsRegistry(unittest.TestCase):
    """Z2 P1 — inject_lessons had a verb + mr_roboto dispatch but no entry in
    POST_HOOK_REGISTRY, so determine_posthooks() filtered it out and the
    cross-mission learning READ side never ran."""

    def test_inject_lessons_registered(self):
        from general_beckman.posthooks import (
            POST_HOOK_KINDS, POST_HOOK_REGISTRY,
        )
        self.assertIn("inject_lessons", POST_HOOK_REGISTRY)
        self.assertIn("inject_lessons", POST_HOOK_KINDS)
        spec = POST_HOOK_REGISTRY["inject_lessons"]
        self.assertEqual(spec.verb, "inject_lessons")
        self.assertEqual(
            spec.default_severity, "warning",
            "advisory only — must never block source task",
        )
        # Not glob-auto-wired: expander prepends explicitly on phase-0.
        self.assertEqual(spec.resolve_triggers(), [])

    def test_determine_posthooks_retains_inject_lessons(self):
        from general_beckman.posthooks import determine_posthooks
        hooks = determine_posthooks(
            {"agent_type": "coder"},
            {"post_hooks": ["inject_lessons"]},
            {},
        )
        self.assertIn("inject_lessons", hooks)

    def test_apply_payload_routing_is_mechanical(self):
        """_posthook_agent_and_payload must return ('mechanical', payload)
        with action == 'inject_lessons' so mr_roboto dispatch matches."""
        from general_beckman.apply import _posthook_agent_and_payload
        from general_beckman.result_router import PostHookVerdict

        verdict = PostHookVerdict(
            source_task_id=42, kind="inject_lessons",
            passed=True, raw={},
        )
        source = {"id": 42, "mission_id": 7, "context": "{}"}
        ctx = {}
        agent_type, payload = _posthook_agent_and_payload(verdict, source, ctx)
        self.assertEqual(agent_type, "mechanical")
        self.assertEqual(payload["posthook_kind"], "inject_lessons")
        self.assertEqual(payload["payload"]["action"], "inject_lessons")
        self.assertEqual(payload["payload"]["mission_id"], 7)


if __name__ == "__main__":
    unittest.main()
