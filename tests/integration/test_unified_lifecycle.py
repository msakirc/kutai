"""Integration tests for the unified task lifecycle."""
import pytest
import asyncio
import json


@pytest.mark.integration
class TestUnifiedLifecycle:
    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_ungraded_blocks_dependents(self, temp_db):
        async def _test():
            from src.infra.db import add_task, update_task, get_ready_tasks

            parent = await add_task("parent", "work")
            await update_task(parent, status="ungraded",
                              context=json.dumps({"generating_model": "model_a"}))
            child = await add_task("child", "depends", depends_on=[parent])

            ready = await get_ready_tasks()
            ready_ids = [t["id"] for t in ready]
            assert child not in ready_ids

            # Complete parent → child unblocks
            await update_task(parent, status="completed")
            ready = await get_ready_tasks()
            ready_ids = [t["id"] for t in ready]
            assert child in ready_ids

        self._run(_test())

    def test_next_retry_at_filters_ready_tasks(self, temp_db):
        async def _test():
            from src.infra.db import add_task, get_db, get_ready_tasks

            tid = await add_task("delayed", "desc")
            db = await get_db()
            await db.execute(
                "UPDATE tasks SET next_retry_at = datetime('now', '+1 hour') WHERE id = ?",
                (tid,),
            )
            await db.commit()

            ready = await get_ready_tasks()
            assert tid not in [t["id"] for t in ready]

        self._run(_test())

    def test_past_retry_at_included(self, temp_db):
        async def _test():
            from src.infra.db import add_task, get_db, get_ready_tasks

            tid = await add_task("past", "desc")
            db = await get_db()
            await db.execute(
                "UPDATE tasks SET next_retry_at = datetime('now', '-1 hour') WHERE id = ?",
                (tid,),
            )
            await db.commit()

            ready = await get_ready_tasks()
            assert tid in [t["id"] for t in ready]

        self._run(_test())

    def test_new_columns_exist(self, temp_db):
        async def _test():
            from src.infra.db import get_db
            db = await get_db()
            cursor = await db.execute("PRAGMA table_info(tasks)")
            columns = {row[1] for row in await cursor.fetchall()}
            for col in ["attempts", "max_attempts", "grade_attempts",
                        "max_grade_attempts", "next_retry_at",
                        "retry_reason", "failed_in_phase"]:
                assert col in columns, f"Missing column: {col}"

        self._run(_test())

    def test_accelerate_retries(self, temp_db):
        async def _test():
            from src.infra.db import add_task, get_db, accelerate_retries, get_ready_tasks

            tid = await add_task("avail", "desc")
            db = await get_db()
            await db.execute(
                """UPDATE tasks SET
                   next_retry_at = datetime('now', '+1 hour'),
                   retry_reason = 'availability'
                   WHERE id = ?""",
                (tid,),
            )
            await db.commit()

            # Not ready yet
            ready = await get_ready_tasks()
            assert tid not in [t["id"] for t in ready]

            # Accelerate
            woken = await accelerate_retries("test_signal")
            assert woken == 1

            # Now ready
            ready = await get_ready_tasks()
            assert tid in [t["id"] for t in ready]

        self._run(_test())
