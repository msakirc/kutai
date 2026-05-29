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
import mr_roboto
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


def _dispatch_exc_to_result(exc: BaseException, task: dict) -> dict:
    """Build a failure result dict from a dispatch exception.

    Bug 2026-05-26: a bare ``asyncio.TimeoutError`` (e.g. a cloud call that
    rode the 600s wall-clock cap) has ``str(exc) == ''``, so the old generic
    handler wrote ``error="TimeoutError: "`` with NO ``error_category``.
    Beckman then defaulted the category to ``worker`` and the DB row's DLQ
    reason was blank — losing both the right retry curve and any forensic
    "where". Tag timeouts explicitly and name the held model so the row is
    actionable. (asyncio.TimeoutError and the builtin TimeoutError are
    distinct classes on Python 3.10 — catch both.)
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        held = task.get("_held_pick") if isinstance(task, dict) else None
        model = getattr(getattr(held, "model", None), "name", "") if held else ""
        msg = "call timed out (wall-clock cap)" + (f" on {model}" if model else "")
        return {"status": "failed", "error": msg, "error_category": "timeout"}
    return {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def _mech_action_to_result(action) -> dict:
    """Map a mr_roboto Action to the orchestrator result dict.

    needs_review (e.g. find_similar_missions: prior missions matched) MUST
    propagate so result_router → RequestReview → rewrite Rule 0c' surfaces
    the founder Continue/Branch/Abort decision and _apply_review marks the
    post-hook task complete. Bug 2026-05-26: the missing needs_review case
    fell through to {"status": "failed"} → category=worker → retried 6× on
    the backoff ladder → DLQ (#166396, even after the _apply_review fix —
    the result never reached result_router as needs_review).
    """
    if action.status == "completed":
        return {"status": "completed", "result": json.dumps(action.result)}
    if action.status == "needs_clarification":
        return {"status": "needs_clarification", "result": json.dumps(action.result)}
    if action.status == "needs_review":
        return {
            "status": "needs_review",
            "result": json.dumps(action.result),
            "error": action.error or "",
        }
    return {"status": "failed", "error": action.error or "mechanical failed"}


async def _check_mcp_idle_sweep() -> None:
    """Periodically shut down idle MCP servers (lazy-start companion).

    Runs on the same cadence as the other orchestrator _check_* jobs. Never
    starts a server — only kills servers idle past their idle_timeout_s.
    """
    try:
        from yalayut.mcp_manager import get_manager
        killed = await get_manager().sweep_idle()
        if killed:
            from src.infra.logging_config import get_logger as _get_logger
            _get_logger("orchestrator.mcp").info("mcp idle sweep", killed=killed)
    except Exception:
        # Sweep failures must never disturb the pump.
        pass


class Orchestrator:
    def __init__(self, shutdown_event=None):
        self.telegram = TelegramInterface(self)
        self.running = False
        self._shutting_down = False
        self.shutdown_event = shutdown_event or asyncio.Event()
        self.requested_exit_code: int | None = None
        self._current_task_future = None
        self._running_futures: list[asyncio.Task] = []
        # Yalayut Phase 4 — periodic-check gates. 0.0 → first pump tick after
        # boot fires both checks immediately, then they self-gate to 24h.
        self._last_yalayut_discovery: float = 0.0
        self._last_source_scout: float = 0.0

    def _drop_running_future(self, f: asyncio.Task) -> None:
        """done_callback: remove a completed dispatch task from the tracker.
        Tolerant of the task already being absent (e.g. stop() drained it)."""
        try:
            self._running_futures.remove(f)
        except ValueError:
            pass

    # ─── Yalayut Phase 4 periodic checks ─────────────────────────────────
    #
    # Mirror the _check_todo_reminders pattern: timestamp-gated, enqueue a
    # plain dict via beckman.enqueue. The orchestrator imports ZERO from
    # yalayut — the mechanical executor (action "yalayut_discovery" /
    # "source_scout") owns the yalayut import. The cron_seed cadence rows
    # are the restart-survivable backstop; these in-process checks give a
    # finer cadence and fire promptly after boot.

    _YALAYUT_DISCOVERY_INTERVAL_S: float = 86400.0   # 24h
    _SOURCE_SCOUT_INTERVAL_S: float = 86400.0        # 24h

    async def _check_yalayut_discovery(self) -> None:
        """Enqueue a yalayut daily-discovery mechanical task when due."""
        import time as _time
        last = getattr(self, "_last_yalayut_discovery", 0.0)
        now = _time.time()
        if now - last < self._YALAYUT_DISCOVERY_INTERVAL_S:
            return
        self._last_yalayut_discovery = now
        try:
            import general_beckman
            await general_beckman.enqueue(
                {
                    "agent_type": "mechanical",
                    "title": "Yalayut daily discovery",
                    "payload": {"action": "yalayut_discovery",
                                "mode": "daily"},
                },
                lane="oneshot",
            )
            logger.info("enqueued yalayut daily discovery task")
        except Exception as e:
            logger.warning("yalayut discovery enqueue failed: %s", e)

    async def _check_source_scout(self) -> None:
        """Enqueue a yalayut source-scout mechanical task when due."""
        import time as _time
        last = getattr(self, "_last_source_scout", 0.0)
        now = _time.time()
        if now - last < self._SOURCE_SCOUT_INTERVAL_S:
            return
        self._last_source_scout = now
        try:
            import general_beckman
            await general_beckman.enqueue(
                {
                    "agent_type": "mechanical",
                    "title": "Yalayut source scout",
                    "payload": {"action": "source_scout"},
                },
                lane="oneshot",
            )
            logger.info("enqueued yalayut source-scout task")
        except Exception as e:
            logger.warning("source-scout enqueue failed: %s", e)

    # ─── Dispatch ────────────────────────────────────────────────────────

    async def _dispatch(self, task: dict) -> None:
        """Inject context → run agent/mr_roboto → on_task_finished → push_metrics."""
        import general_beckman
        task_id = task["id"]
        agent_type = task.get("agent_type", "executor")

        # ── Refresh workflow-step agent_type from live JSON ──
        # Task rows freeze agent_type at expansion time. When a workflow
        # JSON edit changes a step's agent (e.g. the 2026-04-25 sweep
        # moved 24 planner→array/object steps to analyst), existing rows
        # keep dispatching the old agent. Mission 46 tasks 2939, 2942
        # burned 4+ retries each as planner emitting subtask plans for
        # array/object schemas — the schema validator kept failing the
        # same shape it could never produce. The base.py per-task field
        # refresh deliberately excluded agent_type because the agent
        # CLASS is selected here in the orchestrator BEFORE execute()
        # runs; refreshing it inside the agent would be too late. So
        # do it here, at the dispatch entry, before get_agent() picks
        # the class.
        try:
            ctx_raw_for_agent = task.get("context") or "{}"
            _tctx = json.loads(ctx_raw_for_agent) if isinstance(ctx_raw_for_agent, str) else ctx_raw_for_agent
            if isinstance(_tctx, str):
                _tctx = json.loads(_tctx)
            if isinstance(_tctx, dict) and _tctx.get("is_workflow_step"):
                _step_id = _tctx.get("workflow_step_id")
                _mid = task.get("mission_id")
                if _step_id and _mid:
                    from src.infra.db import get_db
                    _mdb = await get_db()
                    _mcur = await _mdb.execute(
                        "SELECT context FROM missions WHERE id = ?", (_mid,),
                    )
                    _mrow = await _mcur.fetchone()
                    await _mcur.close()
                    _mctx = {}
                    if _mrow and _mrow[0]:
                        try:
                            _mctx = json.loads(_mrow[0])
                            if isinstance(_mctx, str):
                                _mctx = json.loads(_mctx)
                        except (json.JSONDecodeError, TypeError):
                            _mctx = {}
                    _wf_name = (
                        _mctx.get("workflow_name") if isinstance(_mctx, dict) else None
                    ) or "i2p_v3"
                    from src.workflows.engine.loader import load_workflow
                    _wf = load_workflow(_wf_name)
                    _step = _wf.get_step(_step_id)
                    if _step:
                        _live_agent = _step.get("agent")
                        if (_live_agent
                                and _live_agent != agent_type
                                and _live_agent != "mechanical"
                                and agent_type != "mechanical"):
                            from src.infra.db import update_task
                            await update_task(task_id, agent_type=_live_agent)
                            logger.info(
                                f"[Task #{task_id}] agent_type refresh: "
                                f"{agent_type} → {_live_agent} "
                                f"(step={_step_id}, wf={_wf_name})"
                            )
                            agent_type = _live_agent
                            task["agent_type"] = _live_agent
        except Exception as _e:
            logger.debug(f"agent_type refresh failed #{task_id}: {_e}")

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

            # Phase D — orchestrator dispatches by task.runner. Legacy
            # rows (pre-D backfill) and ad-hoc context shapes still
            # checked as fallback so we don't regress old missions.
            is_mech = (task.get("runner") == "mechanical"
                       or task.get("executor") == "mechanical"
                       or ctx.get("executor") == "mechanical"
                       or agent_type == "mechanical")
            if is_mech:
                t = dict(task)
                if "payload" not in t and "payload" in ctx:
                    t["payload"] = ctx["payload"]
                r = await mr_roboto.run(t)
                # needs_clarification → beckman's router leaves the row where
                # mr_roboto put it (waiting_human); needs_review → founder
                # decision surface; both must NOT collapse to failed/completed.
                return _mech_action_to_result(r)
            # ── raw_dispatch sentinel: LLM call routed via beckman.enqueue
            # alias (dispatcher.request → beckman.enqueue → pump → here).
            # These tasks have context.llm_call.raw_dispatch == True and no
            # matching agent class — send them straight to husam.run() (the
            # non-agentic single-call worker that owns select → execute → map).
            try:
                _ctx_raw_rd = task.get("context") or "{}"
                _ctx_rd = json.loads(_ctx_raw_rd) if isinstance(_ctx_raw_rd, str) else _ctx_raw_rd
                if isinstance(_ctx_rd, str):
                    _ctx_rd = json.loads(_ctx_rd)
                _llm_call_rd = _ctx_rd.get("llm_call") if isinstance(_ctx_rd, dict) else None
                _is_raw = isinstance(_llm_call_rd, dict) and _llm_call_rd.get("raw_dispatch") is True
            except Exception:
                _is_raw = False
            if _is_raw:
                import husam
                # Forward the in-memory preselected_pick that Beckman attached at
                # admission so husam can skip re-selection. Admission gates
                # (fatih_hoca.select, pool_pressure, reserve_task) already ran in
                # Beckman; husam → dispatcher.execute is pure call-execution here.
                _dispatch_result = await husam.run(
                    {
                        "context": _ctx_rd,
                        "kind": task.get("kind", "main_work"),
                        "preselected_pick": task.get("preselected_pick"),
                    }
                )
                return {
                    "status": "completed",
                    "result": json.dumps(_dispatch_result) if not isinstance(_dispatch_result, str) else _dispatch_result,
                    **{k: v for k, v in _dispatch_result.items() if k != "result"},
                }
            if agent_type == "shopping_pipeline_v2":
                from src.workflows.shopping.pipeline_v2 import ShoppingPipelineV2
                return await ShoppingPipelineV2().run(task)
            return await get_agent(agent_type).execute(task)

        # Z6 W2 — set audit_context for the per-task execution. Any
        # credential vault read fired by the agent or vendor_call tool now
        # gets mission_id/task_id/agent stamped on its credential_access_log
        # row. Without this, the rows have NULL provenance and post-mortem
        # audits can't tie a credential read to a specific step.
        try:
            from src.security._audit_context import audit_context as _audit_cm
        except Exception:  # pragma: no cover - defensive
            _audit_cm = None  # type: ignore[assignment]

        async def _run_with_audit() -> dict:
            if _audit_cm is None:
                return await _run()
            async with _audit_cm(
                mission_id=task.get("mission_id"),
                task_id=task_id,
                agent=agent_type,
            ):
                return await _run()

        try:
            from src.core import heartbeat as _hb
            _hb.current_task_id.set(int(task_id) if task_id else None)
            _hb.bump(task_id)  # initial heartbeat — task is alive
            runner_task = asyncio.create_task(_run_with_audit())

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
            # Preserve the dispatcher's specific category. "no_model"
            # (selector returned None — provider pool transiently
            # empty) gets a different retry curve in beckman than
            # generic "availability" (call rejected by provider).
            # Production 2026-05-02: 50 cloud ids in dead-models +
            # local pool busy → every retry hit "no candidates" within
            # 40s, exhausted worker_attempts, DLQ'd. Keeping the
            # specific category lets retry.py back off longer and let
            # the pool actually recover.
            cat = getattr(mcf, "error_category", "") or "availability"
            result = {"status": "failed", "error": str(mcf)[:500], "error_category": cat}
            logger.warning("ModelCallFailed task #%s: %s (category=%s)", task_id, mcf, cat)
        except Exception as e:
            result = _dispatch_exc_to_result(e, task)
            if result.get("error_category") == "timeout":
                logger.warning("dispatch timeout task #%s: %s", task_id, result["error"])
            else:
                logger.exception("dispatch failed task #%s: %s", task_id, e)

        # Mechanical clarify (variant_choice keyboard) self-manages state:
        # mr_roboto.clarify already called update_task(status="waiting_human")
        # after successfully sending the keyboard, and the user's tap writes
        # clarify_choice + flips the task to completed via
        # _resume_mission_at_step. Running on_task_finished here would route
        # the needs_clarification result through result_router, which would
        # spawn another RequestClarification → infinite clarify tasks. Skip.
        if isinstance(result, dict) and result.get("status") == "needs_clarification" \
                and (task.get("runner") == "mechanical"
                     or task.get("executor") == "mechanical"
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

        # Z6 T1E: throttle counter for the founder_actions unblock sweep.
        # 20 ticks * 3s base sleep ≈ 60s between sweeps. Keep it cheap —
        # the sweep is one indexed SELECT + zero-to-N UPDATEs.
        _z6_sweep_counter = 0

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
                    # Yalayut Phase 2 — match catalog skills for this task
                    # once, before dispatch. intersect.flash() attaches a
                    # task["skills"] envelope (coulson reads it) or, for a
                    # preempt recipe, sets runner=mechanical + payload so
                    # the task routes to mr_roboto. Imported lazily so the
                    # orchestrator module load doesn't pull yalayut. flash
                    # graceful-degrades internally — no try/except needed,
                    # but guard the import itself in case the package is
                    # absent in a stripped deploy.
                    try:
                        import intersect
                        task = await intersect.flash(task)
                    except Exception as _e:
                        logger.debug("intersect.flash skipped #%s: %s",
                                     task.get("id"), _e)
                        task.setdefault("skills", [])
                    t = asyncio.create_task(self._dispatch(task))
                    self._running_futures.append(t)
                    t.add_done_callback(self._drop_running_future)

                # Z6 T1E: periodic mission-unblock sweep. Founder may resolve
                # actions via the Yaşar Usta bot or external tooling without
                # the per-resolve hook firing in this process — sweep is the
                # backstop. Throttle to ~once per minute (20 ticks * 3s).
                _z6_sweep_counter += 1
                if _z6_sweep_counter >= 20:
                    _z6_sweep_counter = 0
                    try:
                        import src.founder_actions as _fa
                        n = await _fa.sweep_unblock_all()
                        if n > 0:
                            logger.info(
                                "z6 lifecycle sweep: unblocked %d mission(s)",
                                n,
                            )
                    except Exception as _e:
                        logger.debug(f"z6 sweep skipped: {_e}")

                # Yalayut Phase 3 — MCP idle sweep: kill servers idle past
                # their idle_timeout_s. Runs every loop tick (cheap select);
                # never starts a server; failures are silenced so the pump
                # is never disturbed.
                await _check_mcp_idle_sweep()

                # Yalayut Phase 4 — periodic discovery + source-scout checks.
                # Both are timestamp-gated internally; calling every tick is
                # cheap (a getattr + time comparison).
                try:
                    await self._check_yalayut_discovery()
                    await self._check_source_scout()
                except Exception as e:
                    logger.debug("yalayut periodic check skipped: %s", e)

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

    async def _rebind_ongoing(self, mission) -> None:
        """Z8 T1C v1 + T5-prep — re-bind an ongoing mission after restart.

        Webhook listener (T3) reads ``mission.cursor`` to skip
        already-processed events. T5-prep wires the cron scheduler:
        ``mission.cursor['cron']`` is a list of
        ``{"action": str, "interval_seconds": int}`` entries; each is
        armed via ``mission_cron.arm()``.
        """
        logger.debug(
            f"_rebind_ongoing: mission #{mission.id} parked for "
            f"handler-side replay (cursor_keys="
            f"{sorted(mission.cursor.keys()) if mission.cursor else []})"
        )
        try:
            from general_beckman.mission_cron import arm_from_cursor
            armed = await arm_from_cursor(mission)
            if armed:
                logger.info(
                    "mission_cron armed",
                    mission_id=mission.id,
                    armed=armed,
                )
        except Exception as e:
            logger.warning(
                f"mission_cron arm failed for mid={mission.id}: {e}"
            )

    async def _heartbeat_loop(self):
        """Run heartbeat + state-snapshot writer.

        State snapshot includes currently-active connect_aux blocks and
        in-flight task IDs so that when Yaşar Usta detects a freeze and
        kills the orchestrator, it can log WHAT was happening at the
        moment of the last heartbeat.
        """
        from yasar_usta import HeartbeatWriter

        def _state_provider() -> dict:
            try:
                from src.infra.db import _aux_active_summary
                aux = _aux_active_summary()
            except Exception:
                aux = "(unavailable)"
            try:
                from src.core.in_flight import in_flight_snapshot
                snap = in_flight_snapshot()
                inflight = [
                    f"task={getattr(e,'task_id','?')}|model={getattr(e,'model','?')}"
                    for e in snap[:8]
                ]
            except Exception:
                inflight = "(unavailable)"
            return {"aux_active": aux, "in_flight": inflight}

        await HeartbeatWriter(
            "logs/orchestrator.heartbeat",
            "logs/heartbeat",
            interval=15.0,
            state_path="logs/orchestrator.state.json",
            state_provider=_state_provider,
        ).run()

    async def start(self):
        await init_db()
        try:
            await startup_recovery()
        except Exception as e:
            logger.warning(f"Startup recovery failed: {e}")

        # Z8 T1C — replay ongoing missions. find_resumable() returns every
        # kind='ongoing' AND lifecycle_state='active' AND not revoked
        # mission. v1 just logs + delegates re-binding to handler-side
        # code (webhook listener wakes via cursor in T3; cron scheduler
        # re-arms in T5 by querying find_resumable() itself).
        try:
            from general_beckman.resumption import find_resumable
            resumed = await find_resumable()
            for m in resumed:
                logger.info(
                    "resuming ongoing mission",
                    mission_id=m.id,
                    title=m.title,
                    cursor_keys=sorted(m.cursor.keys()) if m.cursor else [],
                )
                await self._rebind_ongoing(m)
            if resumed:
                logger.info(
                    f"z8 resumption: {len(resumed)} ongoing mission(s) "
                    "marked for handler-side replay"
                )
        except Exception as e:
            logger.warning(f"Ongoing-mission resumption failed: {e}")

        from ..models.local_model_manager import get_local_manager
        manager = get_local_manager()
        self._background_tasks: list[asyncio.Task] = [
            asyncio.create_task(manager.run_idle_unloader()),
            asyncio.create_task(manager.run_health_watchdog()),
            asyncio.create_task(self._heartbeat_loop()),
        ]

        # Auto-seed at boot was removed 2026-04-25. The DB row IS the source
        # of truth for agent prompts; the hardcoded `get_system_prompt` in
        # each agent class is a frozen reference, not a continuously synced
        # mirror. Auto-seed silently re-derived the DB from possibly-stale
        # code — the exact drift this kill fixes. To bring a new agent
        # online or refresh an existing one, use `/prompt seed <agent>`
        # (manual, audited) rather than triggering it on every restart.

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
