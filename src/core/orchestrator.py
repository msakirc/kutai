# orchestrator.py
import asyncio
import json
import os
from pathlib import Path
from ..app.config import TASK_PRIORITY
from ..infra.db import (
    init_db, get_db, close_db, add_task, release_task_locks,
)
from src.infra.logging_config import get_logger
from .router import ModelCallFailed
from .task_context import parse_context
from .context_injection import inject_chain_context
from .startup_recovery import startup_recovery
import salako
from ..agents import get_agent
from ..tools.workspace import get_mission_workspace, get_mission_workspace_relative
from ..tools.git_ops import ensure_git_repo, create_mission_branch
from ..app.telegram_bot import TelegramInterface

logger = get_logger("core.orchestrator")

# Per-agent wall-clock dispatch timeouts deleted 2026-04-22. Replaced
# with a progress-heartbeat watchdog (src/core/heartbeat.py): agents
# bump() at each iteration boundary; the watchdog kills only when no
# progress arrives within heartbeat.PROGRESS_TIMEOUT_SECONDS. A slow but
# advancing task is no longer killed; a wedged task no longer hides
# behind a still-running budget.


class Orchestrator:
    def __init__(self, shutdown_event=None):
        self.telegram = TelegramInterface(self)
        self.running = False
        self._shutting_down = False
        self.shutdown_event = shutdown_event or asyncio.Event()
        self.requested_exit_code: int | None = None
        self._current_task_future = None
        self._running_futures: list[asyncio.Task] = []

    def _drop_running_future(self, f: asyncio.Task) -> None:
        """done_callback: remove a completed dispatch task from the tracker.
        Tolerant of the task already being absent (e.g. stop() drained it)."""
        try:
            self._running_futures.remove(f)
        except ValueError:
            pass

    # ─── Dispatch ────────────────────────────────────────────────────────

    async def _dispatch(self, task: dict) -> None:
        """Inject context → run agent/salako → on_task_finished → push_metrics."""
        import general_beckman
        task_id = task["id"]
        agent_type = task.get("agent_type", "executor")

        try:
            task = await inject_chain_context(task)
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
                # Legacy-shape rescue: older workflow expansions (pre-
                # 2026-04-24 expander fix) stored context as
                # {"executor": "<action>", ...} instead of
                # {"executor": "mechanical", "payload": {"action": ...}}.
                # Existing DB rows with the old shape dispatch here with
                # no payload and salako returns `unknown mechanical
                # action: None`. Promote ctx.executor to payload.action
                # at runtime so we don't have to migrate rows.
                if "payload" not in t:
                    _legacy_action = ctx.get("executor")
                    if _legacy_action and _legacy_action != "mechanical":
                        _skip = {"executor", "payload"}
                        _extras = {k: v for k, v in ctx.items() if k not in _skip}
                        t["payload"] = {"action": _legacy_action, **_extras}
                r = await salako.run(t)
                return ({"status": "completed", "result": json.dumps(r.result)}
                        if r.status == "completed"
                        else {"status": "failed", "error": r.error or "mechanical failed"})
            if agent_type == "shopping_pipeline":
                from src.workflows.shopping.pipeline import ShoppingPipeline
                return await ShoppingPipeline().run(task)
            if agent_type == "shopping_pipeline_v2":
                from src.workflows.shopping.pipeline_v2 import ShoppingPipelineV2
                return await ShoppingPipelineV2().run(task)
            return await get_agent(agent_type).execute(task)

        try:
            from src.core import heartbeat as _hb
            _hb.current_task_id.set(int(task_id) if task_id else None)
            _hb.bump(task_id)  # initial heartbeat — task is alive
            runner_task = asyncio.create_task(_run())

            async def _watchdog() -> None:
                """Wake periodically; abort runner if heartbeat goes stale."""
                while not runner_task.done():
                    await asyncio.sleep(15)
                    if _hb.stale_seconds(task_id) > _hb.PROGRESS_TIMEOUT_SECONDS:
                        runner_task.cancel()
                        return

            watchdog_task = asyncio.create_task(_watchdog())
            try:
                result = await runner_task
            except asyncio.CancelledError:
                limit = _hb.PROGRESS_TIMEOUT_SECONDS
                result = {
                    "status": "failed",
                    "error": f"no progress for {limit:.0f}s — task wedged",
                }
                logger.error("no-progress watchdog killed task #%s", task_id)
            finally:
                watchdog_task.cancel()
                _hb.clear(task_id)
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

        # Release the dispatcher's per-task in-flight slot so admission sees
        # the lane as free. Safe to call even for mechanical tasks that
        # never entered dispatcher — release_task is a no-op on absent slots.
        try:
            from src.core.llm_dispatcher import release_task as _dispatcher_release_task
            await _dispatcher_release_task(task_id)
        except Exception as e:
            logger.debug("dispatcher.release_task raised #%s: %s", task_id, e)

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
                    t.add_done_callback(self._drop_running_future)
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

    async def start(self):
        await init_db()
        try:
            await startup_recovery()
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
