"""Integration test: RetryContext survives DB roundtrip."""
from __future__ import annotations

import json

import pytest

from tests.integration.conftest import run_async


@pytest.mark.integration
class TestRetryContextDBRoundtrip:
    def test_quality_failure_persists(self, temp_db):
        async def _test():
            from src.infra.db import add_task, get_task, update_task
            from src.core.retry import RetryContext

            tid = await add_task(title="Test quality", description="test", agent_type="coder")
            task = await get_task(tid)
            ctx = RetryContext.from_task(task)
            assert ctx.worker_attempts == 0

            # First quality failure
            decision = ctx.record_failure("quality", model="model-a")
            assert ctx.worker_attempts == 1
            assert decision.action == "immediate"

            task_ctx = json.loads(task.get("context") or "{}")
            task_ctx.update(ctx.to_context_patch())
            await update_task(tid, context=json.dumps(task_ctx), **ctx.to_db_fields())

            # Verify roundtrip
            task2 = await get_task(tid)
            ctx2 = RetryContext.from_task(task2)
            assert ctx2.worker_attempts == 1
            assert "model-a" in ctx2.failed_models
            assert ctx2.failed_in_phase == "worker"

        run_async(_test())

    def test_infra_failure_independent(self, temp_db):
        async def _test():
            from src.infra.db import add_task, get_task, update_task
            from src.core.retry import RetryContext

            tid = await add_task(title="Test infra", description="test", agent_type="executor")
            task = await get_task(tid)
            ctx = RetryContext.from_task(task)

            # Infra failure doesn't touch worker_attempts
            decision = ctx.record_failure("infrastructure")
            assert ctx.infra_resets == 1
            assert ctx.worker_attempts == 0
            assert decision.action == "immediate"

            await update_task(tid, **ctx.to_db_fields())

            task2 = await get_task(tid)
            ctx2 = RetryContext.from_task(task2)
            assert ctx2.infra_resets == 1
            assert ctx2.worker_attempts == 0

        run_async(_test())

    def test_exhaustion_reason_persists(self, temp_db):
        async def _test():
            from src.infra.db import add_task, get_task, update_task
            from src.core.retry import RetryContext

            tid = await add_task(title="Test exhaust", description="test", agent_type="analyst")
            task = await get_task(tid)
            ctx = RetryContext.from_task(task)
            ctx.guard_burns = 5
            ctx.max_iterations = 8

            ctx.record_failure("exhaustion", model="m")
            assert ctx.exhaustion_reason == "guards"

            await update_task(tid, **ctx.to_db_fields())

            task2 = await get_task(tid)
            ctx2 = RetryContext.from_task(task2)
            assert ctx2.exhaustion_reason == "guards"
            assert ctx2.worker_attempts == 1

        run_async(_test())

    def test_multiple_failures_accumulate(self, temp_db):
        async def _test():
            from src.infra.db import add_task, get_task, update_task
            from src.core.retry import RetryContext

            tid = await add_task(title="Test multi", description="test", agent_type="coder")

            for i in range(3):
                task = await get_task(tid)
                ctx = RetryContext.from_task(task)
                decision = ctx.record_failure("quality", model=f"model-{i}")
                task_ctx = json.loads(task.get("context") or "{}")
                task_ctx.update(ctx.to_context_patch())
                await update_task(tid, context=json.dumps(task_ctx), **ctx.to_db_fields())

            task_final = await get_task(tid)
            ctx_final = RetryContext.from_task(task_final)
            assert ctx_final.worker_attempts == 3
            assert len(ctx_final.failed_models) == 3
            assert ctx_final.excluded_models == ctx_final.failed_models  # at 3, exclusion kicks in

        run_async(_test())
