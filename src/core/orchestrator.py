# orchestrator.py
import asyncio
import json
import os
from pathlib import Path
from ..app.config import MAX_CONTEXT_CHAIN_LENGTH, TASK_PRIORITY
from ..infra.db import (
    init_db, get_db, close_db, add_task, release_task_locks,
)
from src.infra.logging_config import get_logger
from .router import ModelCallFailed
from .task_context import parse_context, set_context
import salako
from ..agents import get_agent
from ..tools.workspace import (
    get_file_tree, get_mission_workspace, get_mission_workspace_relative,
)
from ..tools.git_ops import ensure_git_repo, create_mission_branch
from ..app.telegram_bot import TelegramInterface

logger = get_logger("core.orchestrator")

# Default timeouts per agent type (seconds).
# Override via tasks.timeout_seconds column for per-task control.
AGENT_TIMEOUTS: dict[str, int] = {
    "planner": 420, "architect": 420, "coder": 420, "implementer": 420,
    "fixer": 420, "test_generator": 300, "reviewer": 300, "visual_reviewer": 180,
    "researcher": 420, "analyst": 420, "writer": 420, "summarizer": 180,
    "assistant": 180, "executor": 300, "pipeline": 600, "workflow": 900,
    "shopping_advisor": 600, "product_researcher": 300, "deal_analyst": 240,
    "shopping_pipeline": 60, "shopping_clarifier": 120,
}


class Orchestrator:
    def __init__(self, shutdown_event=None):
        self.telegram = TelegramInterface(self)
        self.running = False
        self._shutting_down = False
        self.shutdown_event = shutdown_event or asyncio.Event()
        self.requested_exit_code: int | None = None
        self._current_task_future = None
        self._running_futures: list[asyncio.Task] = []

    # ─── Context Chaining ────────────────────────────────────────────────

    async def _inject_chain_context(self, task: dict) -> dict:
        """Inject completed sibling results + workspace snapshot into task context."""
        task_context = parse_context(task)
        parent_id = task.get("parent_task_id")
        prior_steps = []

        if parent_id:
            db = await get_db()
            cursor = await db.execute(
                """SELECT id, title, result, agent_type, status
                   FROM tasks WHERE parent_task_id = ? AND status = 'completed'
                   AND id != ? ORDER BY completed_at ASC""",
                (parent_id, task["id"])
            )
            for sib in [dict(r) for r in await cursor.fetchall()]:
                rt = sib.get("result", "")
                prior_steps.append({
                    "title": sib["title"], "agent_type": sib.get("agent_type", "?"),
                    "status": sib["status"],
                    "result": rt[:1500] + "\n... [truncated]" if len(rt) > 1500 else rt,
                })

        total = sum(len(s["result"]) for s in prior_steps)
        while total > MAX_CONTEXT_CHAIN_LENGTH and prior_steps:
            i = max(range(len(prior_steps)), key=lambda x: len(prior_steps[x]["result"]))
            prior_steps[i]["result"] = prior_steps[i]["result"][:500] + "\n... [heavily truncated]"
            total = sum(len(s["result"]) for s in prior_steps)
        if prior_steps:
            task_context["prior_steps"] = prior_steps

        agent_type = task.get("agent_type", "executor")
        mission_id = task.get("mission_id")
        if agent_type in ("coder", "reviewer", "writer", "planner"):
            try:
                tree_path = get_mission_workspace_relative(mission_id) if mission_id else ""
                tree = await get_file_tree(path=tree_path, max_depth=3)
                if tree and "File not found" not in tree and len(tree.split("\n")) > 1:
                    task_context["workspace_snapshot"] = tree
                    if mission_id:
                        task_context["workspace_path"] = get_mission_workspace_relative(mission_id)
            except Exception as e:
                logger.debug(f"workspace snapshot failed: {e}")

        return set_context(task, task_context)

    # ─── Dispatch ────────────────────────────────────────────────────────

    def _timeout_for(self, task: dict) -> int:
        return task.get("timeout_seconds") or AGENT_TIMEOUTS.get(task.get("agent_type", "executor"), 240)

    async def _dispatch(self, task: dict) -> None:
        """Inject context → run agent/salako → on_task_finished → push_metrics."""
        import general_beckman
        task_id = task["id"]
        agent_type = task.get("agent_type", "executor")

        try:
            task = await self._inject_chain_context(task)
        except Exception as e:
            logger.debug(f"context injection failed #{task_id}: {e}")

        async def _run() -> dict:
            ctx = parse_context(task)
            is_mech = (task.get("executor") == "mechanical"
                       or ctx.get("executor") == "mechanical"
                       or agent_type == "mechanical")
            if is_mech:
                t = dict(task)
                if "payload" not in t and "payload" in ctx:
                    t["payload"] = ctx["payload"]
                r = await salako.run(t)
                return ({"status": "completed", "result": json.dumps(r.result)}
                        if r.status == "completed"
                        else {"status": "failed", "error": r.error or "mechanical failed"})
            return await get_agent(agent_type).execute(task)

        try:
            result: dict = await asyncio.wait_for(_run(), timeout=self._timeout_for(task))
        except asyncio.TimeoutError:
            result = {"status": "failed", "error": f"dispatch timeout after {self._timeout_for(task)}s"}
            logger.error("timeout task #%s (%ss)", task_id, self._timeout_for(task))
        except ModelCallFailed as mcf:
            result = {"status": "failed", "error": str(mcf)[:500], "error_category": "availability"}
            logger.warning("ModelCallFailed task #%s: %s", task_id, mcf)
        except Exception as e:
            result = {"status": "failed", "error": f"{type(e).__name__}: {str(e)[:300]}"}
            logger.exception("dispatch failed task #%s: %s", task_id, e)

        try:
            await general_beckman.on_task_finished(task_id, result)
        except Exception as e:
            logger.exception("on_task_finished raised #%s: %s", task_id, e)

        try:
            from src.core.metrics_push import push_metrics
            await push_metrics(task, result)
        except Exception as e:
            logger.debug("push_metrics failed: %s", e)

        try:
            await release_task_locks(task_id)
        except Exception:
            pass

    # ─── Main Loop ───────────────────────────────────────────────────────

    async def run_loop(self):
        """Dispatch pump: check shutdown signal → next_task() → fire-and-forget."""
        self.running = True
        logger.info("Autonomous orchestrator started")
        try:
            from src.app.config import WORKSPACE_ROOT
            os.makedirs(WORKSPACE_ROOT, exist_ok=True)
            await ensure_git_repo()
        except Exception as e:
            logger.warning(f"Workspace/git init: {e}")

        shutdown_signal = Path("logs") / "shutdown.signal"
        import general_beckman

        while self.running and not self.shutdown_event.is_set():
            try:
                if shutdown_signal.exists():
                    intent = shutdown_signal.read_text().strip()
                    shutdown_signal.unlink()
                    logger.info("External shutdown signal: %s", intent)
                    self.requested_exit_code = 42 if intent == "restart" else 0
                    self.shutdown_event.set()
                    break
                if self._shutting_down:
                    logger.info("Shutdown flag — draining tasks")
                    break
                task = await general_beckman.next_task()
                if task is not None:
                    t = asyncio.create_task(self._dispatch(task))
                    self._running_futures.append(t)
                    t.add_done_callback(
                        lambda f: self._running_futures.remove(f)
                        if f in self._running_futures else None
                    )
                await asyncio.sleep(3)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("run_loop iteration failed: %s", e)
                await asyncio.sleep(3)

        logger.info("Orchestrator main loop exited")

    # ─── Mission Planning ─────────────────────────────────────────────────

    async def plan_mission(self, mission_id: int, title: str, description: str):
        """Create initial planning task for a new mission."""
        try:
            get_mission_workspace(mission_id)
            await ensure_git_repo(get_mission_workspace_relative(mission_id))
            branch = await create_mission_branch(
                mission_id, title, path=get_mission_workspace_relative(mission_id))
            if not branch.startswith("❌"):
                logger.info(f"[Mission #{mission_id}] Created branch: {branch}")
        except Exception as e:
            logger.debug(f"[Mission #{mission_id}] Workspace setup skipped: {e}")

        await add_task(
            title=f"Plan: {title[:40]}",
            description=f"Create an execution plan for this mission:\n\n{title}\n\n{description}",
            mission_id=mission_id, agent_type="planner", priority=TASK_PRIORITY["high"],
        )

    # ─── Background Tasks & Startup ───────────────────────────────────────

    async def _heartbeat_loop(self):
        from yasar_usta import HeartbeatWriter
        await HeartbeatWriter(
            "logs/orchestrator.heartbeat", "logs/heartbeat", interval=15.0).run()

    async def _startup_recovery(self):
        """Post-restart: reset stuck tasks + clear retry backoffs."""
        db = await get_db()
        summary: list[str] = []

        c = await db.execute("SELECT id, infra_resets FROM tasks WHERE status = 'processing'")
        interrupted = [dict(r) for r in await c.fetchall()]
        for t in interrupted:
            ir = (t.get("infra_resets") or 0) + 1
            await db.execute(
                "UPDATE tasks SET status='pending', infra_resets=?, retry_reason='infrastructure' WHERE id=?",
                (ir, t["id"]))
        if interrupted:
            await db.commit()
            summary.append(f"Reset {len(interrupted)} interrupted task(s)")

        try:
            from ..infra.db import accelerate_retries
            if w := await accelerate_retries("startup"):
                summary.append(f"Accelerated {w} task(s)")
        except Exception as e:
            logger.debug(f"accelerate_retries failed: {e}")

        c = await db.execute(
            "SELECT id FROM tasks WHERE status IN ('pending','ungraded') "
            "AND next_retry_at IS NOT NULL AND next_retry_at > datetime('now')")
        delayed = [dict(r) for r in await c.fetchall()]
        for t in delayed:
            await db.execute("UPDATE tasks SET next_retry_at=NULL WHERE id=?", (t["id"],))
        if delayed:
            await db.commit()
            summary.append(f"Cleared backoff for {len(delayed)} task(s)")

        try:
            await db.execute("DELETE FROM file_locks")
            await db.commit()
        except Exception:
            pass

        logger.info(f"[Startup Recovery] {' | '.join(summary) or 'clean start'}")

    async def start(self):
        await init_db()
        try:
            await self._startup_recovery()
        except Exception as e:
            logger.warning(f"Startup recovery failed: {e}")

        from ..models.local_model_manager import get_local_manager
        manager = get_local_manager()
        self._background_tasks: list[asyncio.Task] = [
            asyncio.create_task(manager.run_idle_unloader()),
            asyncio.create_task(manager.run_health_watchdog()),
            asyncio.create_task(self._heartbeat_loop()),
        ]

        try:
            from ..memory.prompt_versions import seed_from_agents
            if n := await seed_from_agents():
                logger.info(f"Seeded {n} prompt versions")
        except Exception as e:
            logger.debug(f"Prompt seeding skipped: {e}")

        try:
            from ..shopping.cache import init_cache_db
            from ..shopping.request_tracker import init_request_db
            from ..shopping.memory import init_memory_db
            await asyncio.gather(init_cache_db(), init_request_db(), init_memory_db())
        except Exception as e:
            logger.warning(f"Shopping DB init failed: {e}")

        async with self.telegram.app:
            await self.telegram.app.start()
            await self.telegram.app.updater.start_polling()
            await self.telegram.set_bot_commands()
            logger.info("System online — Telegram + Orchestrator running")

            try:
                from ..app.config import TELEGRAM_ADMIN_CHAT_ID
                from ..app.telegram_bot import REPLY_KEYBOARD
                if TELEGRAM_ADMIN_CHAT_ID:
                    await self.telegram.app.bot.send_message(
                        chat_id=TELEGRAM_ADMIN_CHAT_ID,
                        text="Kutay online. Buttons ready.", reply_markup=REPLY_KEYBOARD,
                    )
            except Exception as e:
                logger.debug(f"Startup keyboard send failed: {e}")

            try:
                await self.telegram.restore_clarification_state()
            except Exception as e:
                logger.debug(f"Clarification state restore failed: {e}")

            try:
                await self.run_loop()
            finally:
                if self.shutdown_event.is_set():
                    self._shutting_down = True
                    self.running = False
                    for t in self._background_tasks:
                        t.cancel()
                    active = [f for f in self._running_futures if f and not f.done()]
                    if active:
                        try:
                            await asyncio.wait_for(
                                asyncio.gather(*[asyncio.shield(f) for f in active],
                                               return_exceptions=True), timeout=10)
                        except Exception:
                            pass
                    try:
                        db = await get_db()
                        await db.execute("DELETE FROM file_locks")
                        await db.commit()
                    except Exception:
                        pass
                    for attr in ("src.infra.metrics.persist_metrics",):
                        try:
                            import importlib
                            m, fn = attr.rsplit(".", 1)
                            r = getattr(importlib.import_module(m), fn)()
                            if asyncio.iscoroutine(r):
                                await r
                        except Exception:
                            pass
                    try:
                        from src.models.model_registry import get_registry
                        get_registry().flush_speed_cache()
                    except Exception:
                        pass

                try:
                    await manager.shutdown()
                    logger.info("llama-server stopped")
                except Exception as e:
                    logger.warning(f"Error stopping llama-server: {e}")

                await close_db(checkpoint=self.shutdown_event.is_set())

                for coro, name in [(self.telegram.app.updater.stop(), "updater"),
                                   (self.telegram.app.stop(), "app")]:
                    try:
                        await asyncio.wait_for(coro, timeout=5)
                    except asyncio.TimeoutError:
                        logger.warning(f"Telegram {name}.stop() timed out")
