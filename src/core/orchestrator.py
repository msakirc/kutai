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
            # Skip-when gate: evaluate the workflow step's skip_when
            # expression against loaded artifacts BEFORE dispatching. If
            # the condition is met, short-circuit with an empty
            # completion and requires_grading=false so the post-hook
            # grader doesn't mark the empty output as a quality failure.
            # Fixes shopping_v2's synth_one steps that previously ran
            # against a non-"chosen" gate and produced grader-failing
            # empty cards on every mission.
            try:
                from src.workflows.engine.hooks import should_skip_workflow_step
                _skip, _reason = await should_skip_workflow_step(task)
            except Exception as _e:
                logger.debug(f"should_skip check failed #{task_id}: {_e}")
                _skip, _reason = False, ""
            if _skip:
                logger.info(
                    f"[Task #{task_id}] skipped via skip_when: {_reason}"
                )
                # Inject requires_grading=false into ctx so the post-hook
                # layer doesn't spawn a grader.
                try:
                    ctx["requires_grading"] = False
                    from src.infra.db import update_task
                    await update_task(task_id, context=json.dumps(ctx))
                except Exception:
                    pass
                return {
                    "status": "completed",
                    "result": json.dumps({"skipped": True, "reason": _reason}),
                    "model": "workflow_engine",
                    "cost": 0.0,
                    "iterations": 0,
                    "_skipped": True,
                }

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
                # Deeper rescue: expander also CLOBBERED ctx.executor to
                # "mechanical" on old rows, erasing the original action
                # name. Look the workflow step up by workflow_step_id in
                # the current workflow JSON and pull action from its
                # context.executor. Works for clarify_variant (action
                # "clarify") + any similar legacy mechanical step.
                if "payload" not in t and ctx.get("is_workflow_step"):
                    try:
                        _step_id = ctx.get("workflow_step_id")
                        _mission_id = task.get("mission_id")
                        if _step_id and _mission_id:
                            from src.infra.db import get_db as _get_db
                            _db = await _get_db()
                            _cur = await _db.execute(
                                "SELECT context FROM missions WHERE id = ?",
                                (_mission_id,),
                            )
                            _row = await _cur.fetchone()
                            await _cur.close()
                            _mctx = {}
                            if _row and _row[0]:
                                try:
                                    _mctx = json.loads(_row[0])
                                    if isinstance(_mctx, str):
                                        _mctx = json.loads(_mctx)
                                except (json.JSONDecodeError, TypeError):
                                    _mctx = {}
                            _wf_name = (
                                _mctx.get("workflow_name")
                                if isinstance(_mctx, dict) else None
                            ) or "i2p_v3"
                            from src.workflows.engine.loader import load_workflow
                            _wf = load_workflow(_wf_name)
                            _step = _wf.get_step(_step_id)
                            if _step:
                                _step_ctx = _step.get("context") or {}
                                _step_action = (
                                    _step_ctx.get("executor")
                                    if isinstance(_step_ctx, dict) else None
                                )
                                if _step_action and _step_action != "mechanical":
                                    _skip = {"executor", "payload"}
                                    _extras = {
                                        k: v for k, v in ctx.items()
                                        if k not in _skip
                                    }
                                    t["payload"] = {
                                        "action": _step_action, **_extras,
                                    }
                    except Exception as _e:
                        logger.debug(
                            f"legacy mechanical rescue failed #{task.get('id','?')}: {_e}"
                        )
                r = await salako.run(t)
                if r.status == "completed":
                    return {"status": "completed", "result": json.dumps(r.result)}
                if r.status == "needs_clarification":
                    # Mechanical executor asked the user something
                    # (variant_choice keyboard) and set the task to
                    # waiting_human. Return the same status upstream so
                    # beckman's router leaves the row where salako put
                    # it — no "completed" flip that would advance the
                    # mission past a user-gated step.
                    return {
                        "status": "needs_clarification",
                        "result": json.dumps(r.result),
                    }
                return {"status": "failed", "error": r.error or "mechanical failed"}
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

        # Mechanical clarify (variant_choice keyboard) self-manages state:
        # salako.clarify already called update_task(status="waiting_human")
        # after successfully sending the keyboard, and the user's tap writes
        # clarify_choice + flips the task to completed via
        # _resume_mission_at_step. Running on_task_finished here would route
        # the needs_clarification result through result_router, which would
        # spawn another RequestClarification → infinite clarify tasks. Skip.
        if isinstance(result, dict) and result.get("status") == "needs_clarification" \
                and (task.get("executor") == "mechanical"
                     or (parse_context(task) or {}).get("executor") == "mechanical"
                     or agent_type == "mechanical"):
            logger.info(
                "[Task #%s] mechanical clarify sent keyboard; leaving as waiting_human",
                task_id,
            )
        else:
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
