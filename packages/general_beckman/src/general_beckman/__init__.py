"""General Beckman — the task master.

Public API (everything else is internal):
  - next_task() -> Task | None
  - on_task_finished(task_id, result) -> None
  - enqueue(spec, *, parent_id, await_inline, on_complete, next_task_spec) -> int | TaskResult
  - resolve_inline(task_id, result) -> None
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from general_beckman.types import Task, AgentResult

__all__ = [
    "next_task", "on_task_finished", "enqueue", "on_model_swap",
    "resolve_inline", "Task", "AgentResult", "TaskResult",
    "INLINE_TIMEOUT", "_inline_waiters",
    "notify_threshold", "THRESHOLDS_PCT",
]

THRESHOLDS_PCT = (50, 75, 90)


async def notify_threshold(mission_id: int, pct: int, spent: float, ceiling: float):
    """Post threshold notify to mission thread."""
    import json as _json
    from src.infra.logging_config import get_logger
    from src.infra.db import get_db

    logger = get_logger(__name__)

    db = await get_db()
    cur = await db.execute(
        "SELECT telegram_thread_id, context FROM missions WHERE id = ?",
        (mission_id,),
    )
    row = await cur.fetchone()
    if not row:
        return
    thread_id, ctx_raw = row
    try:
        ctx = _json.loads(ctx_raw) if ctx_raw else {}
    except (TypeError, ValueError):
        ctx = {}
    chat_id = ctx.get("chat_id") if isinstance(ctx, dict) else None
    if chat_id is None:
        logger.warning("threshold notify: no chat_id in mission %d context", mission_id)
        return

    text = f"📊 Mission #{mission_id} crossed {pct}% — ${spent:.2f} / ${ceiling:.2f}"
    try:
        from src.app.telegram_bot import get_telegram
        tg = get_telegram()
        if tg is None:
            return
        await tg.bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text=text,
        )
    except Exception as e:
        logger.warning("threshold notify post failed: %s", e)


@dataclass
class TaskResult:
    """Minimal result envelope returned by await_inline enqueue."""
    status: str
    result: dict | None
    error: str | None


# Timeout (seconds) for await_inline futures. Monkeypatchable in tests.
INLINE_TIMEOUT: float = 600.0

# task_id → asyncio.Future[TaskResult] for inline waiters
_inline_waiters: dict[int, asyncio.Future] = {}


def resolve_inline(task_id: int, result: "TaskResult") -> None:
    """Resolve an await_inline waiter. Called from terminal hook (Task 3) or tests."""
    fut = _inline_waiters.pop(task_id, None)
    if fut is not None and not fut.done():
        fut.set_result(result)


# Admission-fingerprint cache. When the input state (in_flight + loaded model
# + cloud rpd + candidate ids) hasn't changed since last tick AND last tick
# admitted nothing, this tick's result is deterministic — skip the per-
# candidate Fatih Hoca scan. Single saturated task #4885 was rejected 601x in
# a 42-min window before this guard, each rejection costing a full 13-model
# filter+score pass (2026-04-27 log analysis). Reset by passing
# ``force=True`` or by any state delta.
_last_admission_fp: tuple | None = None
_last_admission_admitted: bool = True


def _format_breakdown(breakdown) -> str:
    """One-line summary of a PressureBreakdown for log lines.

    Output shape:
        S1=-1.00 S2=0.00 ... S11=-0.10 | M1=2.00 M2=1.00 buckets=B/Q/O:0.0/-0.7/-0.1

    Drops zero-valued signals to keep the line short — operator scans for
    the non-zero ones. Buckets prefix shorthand: B=burden (S2/S3), Q=queue
    (S4/S5), O=other (everything else). Without this, REJECT logs printed
    only the final scalar — no way to see WHICH signal pushed the model
    out, which made every triage session a code-spelunk.
    """
    if breakdown is None:
        return ""

    def _f(x, default=0.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float(default)

    try:
        sigs = breakdown.signals or {}
        nonzero = " ".join(
            f"{k}={_f(v):+.2f}" for k, v in sorted(sigs.items())
            if isinstance(k, str) and abs(_f(v)) >= 0.005
        ) or "(all-zero)"
        mods = breakdown.modifiers or {}
        m1 = _f(mods.get("M1", 1.0), 1.0)
        m2 = _f(mods.get("M2", 1.0), 1.0)
        bt = breakdown.bucket_totals or {}
        buckets = (
            f"{_f(bt.get('burden', 0.0)):+.2f}/{_f(bt.get('queue', 0.0)):+.2f}/"
            f"{_f(bt.get('other', 0.0)):+.2f}"
        )
        return f"{nonzero} | M1={m1:.2f} M2={m2:.2f} BQO={buckets}"
    except Exception:
        # Never let log formatting break admission. Real PressureBreakdown
        # objects always work; tests with Mock objects can fail-soft here.
        return ""


def _admission_fingerprint(snap, candidates) -> tuple:
    """Cheap stable hash of admission-relevant state.

    Captures in_flight (auth source for local saturation), loaded model
    name, idle bucket (10s granularity — matches pressure ladder steps),
    cloud rpd-remaining per provider, candidate task IDs, per-model
    daily_exhausted / rpm_cooldown flags, and a 30s wall-clock bucket so
    the cache forces a re-evaluate at least twice per minute even when
    every other component is stable.

    The time bucket fixes a deadlock pattern (production 2026-05-03):
    when every candidate's pressure scalar lands at -1.0 the selector
    returns None, _last_admission_admitted flips to False, and as long
    as in_flight / loaded / cloud_rpd / candidate IDs stay constant the
    fingerprint matches forever — the pump silently short-circuits at
    the cache check, the dead-set TTL never gets a chance to surface
    fresh capacity, and the queue stalls. Per-model availability flips
    (daily_exhausted=True→False at midnight, rpm_cooldown clearing)
    used to be invisible to the fingerprint; surfacing them plus a
    30s upper bound on cache age dissolves the trap.
    """
    if snap is None:
        # No snapshot → can't trust state; bypass cache.
        return ("nosnap", tuple(c.get("id") for c in candidates))
    in_flight = tuple(sorted(
        (c.task_id, c.model, c.provider, c.is_local)
        for c in (snap.in_flight_calls or [])
    ))
    local = snap.local
    loaded = local.model_name if local else None
    idle_bucket = int((local.idle_seconds if local else 0) // 10)
    is_swapping = bool(local.is_swapping) if local else False
    cloud_rpd = tuple(sorted(
        (p, st.limits.rpd.remaining if st.limits and st.limits.rpd else None)
        for p, st in (snap.cloud or {}).items()
    ))
    # Per-model availability flags. Without these, a model flipping from
    # daily_exhausted=True → False (e.g. gemini at midnight UTC) is
    # invisible to the fingerprint as long as cloud-level rpd stays put.
    cloud_models = tuple(sorted(
        (p, mid,
         bool(getattr(ms, "daily_exhausted", False)),
         bool(getattr(ms, "rpm_cooldown", False)))
        for p, st in (snap.cloud or {}).items()
        for mid, ms in (getattr(st, "models", {}) or {}).items()
    ))
    cand_ids = tuple(sorted(c.get("id", 0) for c in candidates))
    import time as _time
    time_bucket = int(_time.time() // 30)
    return (in_flight, loaded, idle_bucket, is_swapping, cloud_rpd,
            cloud_models, cand_ids, time_bucket)


def _capacity_snapshot():
    """Best-effort capacity snapshot. Returns None if nerd_herd isn't wired."""
    try:
        import nerd_herd
        nh = getattr(nerd_herd, "_singleton", None)
        if nh is None:
            return None
        return nh.snapshot()
    except Exception:
        return None


async def _claim_task(task_id: int) -> bool:
    """Claim a single task by id via the existing DB claim primitive."""
    from src.infra.db import claim_task
    return await claim_task(task_id)


async def _ceiling_ok(task: dict, log) -> bool:
    """Return True if admitting this task does not exceed the mission ceiling.

    Enforced only when the mission has a non-NULL cost_ceiling_usd.
    Standalone tasks (mission_id IS NULL) always pass.

    Formula: spent_usd + SUM(estimated_cost_usd WHERE status='running') + new_est <= ceiling
    """
    mid = task.get("mission_id")
    if mid is None:
        return True  # Standalone task — no ceiling applies.

    new_est = float(task.get("estimated_cost_usd") or 0.0)

    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT cost_ceiling_usd, spent_usd FROM missions WHERE id = ?",
            (mid,),
        )
        mrow = await cur.fetchone()
        if mrow is None or mrow[0] is None:
            return True  # No ceiling set — no enforcement.

        ceiling = float(mrow[0])
        spent = float(mrow[1] or 0.0)

        cur2 = await db.execute(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM tasks "
            "WHERE mission_id = ? AND status = 'running'",
            (mid,),
        )
        in_flight = float((await cur2.fetchone())[0] or 0.0)

        total = spent + in_flight + new_est
        if total > ceiling:
            log.info(
                "ceiling backstop: mission %s spent=%.4f in_flight=%.4f new=%.4f > ceiling=%.4f — skip task #%s",
                mid, spent, in_flight, new_est, ceiling, task.get("id"),
            )
            return False
        return True
    except Exception as e:
        log.debug(f"_ceiling_ok check failed (fail-open): {e}")
        return True  # Fail-open: don't block admission on unexpected errors.


async def next_task(lane: str | None = None):
    """Admission loop: pick one ready task whose pool pressure clears its urgency threshold.

    Called by orchestrator on its pump cycle. Iterates top-K ready tasks
    by urgency *within the given admission lane*; for each, asks Fatih
    Hoca for a Pick, then gates on
    ``snapshot.pressure_for(pick.model) >= threshold(urgency)``. First
    candidate to clear is claimed, tagged with ``preselected_pick``, and
    returned. Non-admitted candidates remain in the queue untouched.

    Z8 T1B: ``lane`` (default ``LANE_ONESHOT``) selects between the
    historical oneshot pool and the ongoing-mission pool. A lane-local
    in-flight cap short-circuits the scan when reached so a lane can't
    drown the other.
    """
    import os
    from general_beckman import queue as _queue
    from general_beckman.admission import compute_urgency
    from general_beckman.cron import fire_due
    from general_beckman.lanes import (
        LANE_ONESHOT, cap_for, count_in_flight,
    )

    if lane is None:
        lane = LANE_ONESHOT

    top_k = int(os.environ.get("BECKMAN_TOP_K", "5"))

    await fire_due()

    # Lane cap: reject early so the snapshot/Hoca scan is skipped when
    # the lane is already saturated. Uses the shared db connection.
    try:
        from src.infra.db import get_db as _get_db
        _conn = await _get_db()
        _cap = await cap_for(lane)
        _inflight = await count_in_flight(_conn, lane)
        if _inflight >= _cap:
            return None
    except Exception:
        # If the lane column hasn't migrated yet (pre-T1B DB), fall through
        # — the lane filter on pick_ready_top_k will still work and the
        # caller can keep dispatching from oneshot.
        pass

    # One fresh snapshot per tick. Admission gates on snapshot.pressure_for(pick.model);
    # the in-flight list in the snapshot — pushed by the dispatcher — is the
    # authoritative "what's running right now" signal. No artificial
    # concurrency cap: local is naturally serial via llama-server --parallel 1
    # (surfaced as -1.0 local_pressure), cloud is bounded by per-pool
    # pressure thresholds.
    #
    # refresh_snapshot() is awaited (not cached read) because ticks fire back-
    # to-back in the orchestrator pump and the background 2s refresh loop is
    # too coarse to catch in-flight pushes that just landed on the sidecar.
    try:
        import nerd_herd
        snap = await nerd_herd.refresh_snapshot()
    except Exception:
        snap = None

    # (overlay block below — moved past the _log import)

    try:
        import fatih_hoca
    except Exception:
        fatih_hoca = None  # type: ignore

    from src.infra.logging_config import get_logger
    _log = get_logger("beckman.admission")

    # Overlay in-process truth onto snapshot:
    #   1. in_flight_calls — from src.core.in_flight._task_slots
    #   2. cloud[provider].models[name].limits — from KDV directly
    # Both bypass the nerd_herd sidecar HTTP round-trip that creates a
    # race window between writers (reserve_task / KDV.record_attempt)
    # and readers (refresh_snapshot via /api/snapshot GET). Production
    # triage 2026-05-01: 9 cloud tasks admitted concurrently to a single
    # 8K-TPM model because each parallel admission tick saw a stale
    # snapshot with full TPM headroom while reservations from prior
    # ticks hadn't propagated to the sidecar yet.
    #
    # Both registries are SAME-PROCESS writers and SYNCHRONOUS — reading
    # them here is authoritative and immune to the HTTP race.
    if snap is not None:
        try:
            from src.core.in_flight import in_flight_snapshot as _local_in_flight
            local_calls = _local_in_flight()
            if local_calls:
                try:
                    snap.in_flight_calls = list(local_calls)
                except (AttributeError, TypeError):
                    # MagicMock with property — skip overlay; tests that
                    # rely on the mock's in_flight_calls keep their setup.
                    pass
        except Exception as e:
            _log.debug(f"local in_flight overlay failed: {e}")
        # Cloud state overlay: KDV's RateLimitState (in-process) is the
        # writer for tpm_remaining / rpm_remaining / rpd_remaining via
        # the property accessors. Rebuild the matrix cells from KDV
        # directly so admission sees post-reservation truth.
        #
        # Iterate KDV's KNOWN providers (KDV._providers), not snap.cloud
        # — production triage 2026-05-01: post-restart snap.cloud was
        # empty (sidecar's cloud cache hadn't been populated yet) so
        # the for-loop iterated nothing and the overlay was effectively
        # disabled. selector then saw empty matrix → S1 fired no
        # depletion → admitted saturated models hundreds of times in
        # a row.
        try:
            from src.core.router import get_kdv as _get_kdv
            from kuleden_donen_var.nerd_herd_adapter import (
                build_cloud_provider_state as _build_cloud,
            )
            kdv = _get_kdv()
            if not isinstance(getattr(snap, "cloud", None), dict):
                snap.cloud = {}
            for prov_name in list(kdv._providers.keys()):
                fresh = _build_cloud(kdv, prov_name)
                if fresh is not None:
                    try:
                        snap.cloud[prov_name] = fresh
                    except (AttributeError, TypeError):
                        pass
        except Exception as e:
            _log.debug(f"cloud state overlay failed: {e}")

    candidates = await _queue.pick_ready_top_k(k=top_k, lane=lane)

    # Skip the entire scan if state hasn't changed since last tick and last
    # tick admitted nothing — result is deterministic. Mechanical tasks are
    # the one exception (no Hoca call, no pressure gate); admit them even on
    # a cache hit so blackboard writes / git commits don't stall behind LLM
    # saturation.
    global _last_admission_fp, _last_admission_admitted
    fp = _admission_fingerprint(snap, candidates)
    if (
        not _last_admission_admitted
        and _last_admission_fp == fp
        and not any((c.get("agent_type") or "") == "mechanical" for c in candidates)
    ):
        return None

    for task in candidates:
        # Defensive guard: a pending row past worker_attempts cap should
        # never have reached this point (sweep section 8 is supposed to
        # catch them) but if a fast retry bumped it between sweep ticks,
        # force DLQ here instead of admitting + immediately failing again.
        # Mechanical tasks have no attempt cap — skip the check for them.
        # (Handoff item A.)
        attempts = int(task.get("worker_attempts") or 0)
        max_att = int(task.get("max_worker_attempts") or 15)
        if (
            task.get("agent_type") != "mechanical"
            and max_att > 0
            and attempts >= max_att
        ):
            try:
                from general_beckman.apply import _dlq_write
                fresh = dict(task)
                fresh["failed_in_phase"] = "worker"
                await _dlq_write(
                    fresh,
                    error=f"Worker attempts exceeded at admission: "
                    f"{attempts}/{max_att}",
                    category=task.get("error_category") or "worker",
                    attempts=attempts,
                )
                _log.warning(
                    f"admission: task #{task['id']} REJECT past cap "
                    f"({attempts}/{max_att}) — forced to DLQ"
                )
            except Exception as e:
                _log.warning(f"admission: cap-guard DLQ write failed #{task['id']}: {e}")
            continue

        agent_type = task.get("agent_type") or ""
        difficulty = task.get("difficulty", 5)

        # ── Z6 T1C: real-world bridge admission gate ─────────────────────
        # If the task is flagged needs_real_tools, check that its vendor
        # adapter, credentials, and (for irreversible+cost) founder cost
        # ack are all in place. Missing prereqs emit founder_action rows
        # and park the task in 'blocked_on_founder_action'. T1E's
        # unblock_mission_if_clear flips it back to pending when the
        # founder resolves the action(s).
        try:
            from general_beckman.z6_admission import check_z6_admission
            _mid = task.get("mission_id")
            if _mid is not None:
                _z6 = await check_z6_admission(task, int(_mid))
                if not _z6.admit:
                    try:
                        from src.infra.db import update_task as _ut
                        await _ut(
                            int(task["id"]),
                            status="blocked_on_founder_action",
                        )
                    except Exception as _e:
                        _log.warning(
                            f"z6 admission: failed to park task "
                            f"#{task['id']}: {_e}"
                        )
                    _log.info(
                        f"z6 admission: task #{task['id']} BLOCKED "
                        f"({_z6.reason}); "
                        f"emitted={_z6.founder_actions_emitted}"
                    )
                    continue
        except Exception as _e:
            _log.debug(
                f"z6 admission skipped #{task.get('id')}: {_e}"
            )

        # Mechanical tasks have no LLM, no Hoca pick, no pressure gate.
        # Per design: mechanical is unbounded — local-only backpressure
        # applies only to LLM tasks. Claim and return directly.
        if agent_type == "mechanical":
            # Ceiling backstop: block if spent + in-flight + new estimated cost
            # would exceed the mission ceiling. NULL ceiling → no enforcement.
            if not await _ceiling_ok(task, _log):
                continue
            if not await _claim_task(task["id"]):
                _log.debug(f"admission: task #{task['id']} (mechanical) claim race lost")
                continue
            _log.info(f"admission: task #{task['id']} ADMIT mechanical")
            task["status"] = "processing"
            _last_admission_fp = fp
            _last_admission_admitted = True
            return task

        pick = None
        # Single selection gate: fatih_hoca.select() owns BOTH ranking AND
        # pool-pressure threshold-checking now. Pre-2026-04-30 this code had
        # its own pressure_for / threshold(urgency) gate that fired AFTER
        # selector returned a pick — three pressure gates total (selector
        # ranking, beckman admission, dispatcher recursion) each with their
        # own threshold logic. They drifted out of sync and let dead models
        # through. User feedback: "There must be a singular selection
        # mechanism for everything. Only one".
        #
        # Now: selector takes urgency as input, applies the threshold
        # internally during ranking, and either returns a pick that already
        # cleared the gate or returns None. Beckman trusts that result.
        urgency = compute_urgency(task)

        # Pass per-task token estimates to selector so the S2 (call
        # burden) + S3 (task burden) signals can fire. Without this,
        # selector defaults to estimated_input_tokens=0 / output=0,
        # the burden signals see "free" calls regardless of how many
        # iterations a task actually needs, and admission lets in
        # tasks that exhaust mid-execution.
        #
        # Production 2026-05-02 post-restart: 5+ planner / implementer
        # / test_generator tasks DLQ'd with "No model candidates
        # available" within 60s of restart. Cause: free-tier gemini
        # quota = 20 reqs/day. Each planner uses ~8 ReAct iterations.
        # Admitting 5 planners in parallel = 40 reqs projected against
        # 20 budget. Without estimates passed, selector saw 0-cost
        # calls and admitted all five. Mid-task: quota wall, retry
        # recursion's select() sees exhausted pool, returns None.
        try:
            from fatih_hoca.estimates import estimate_for
            from general_beckman.btable_cache import get_btable
            import json as _json

            ctx_raw = task.get("context") or "{}"
            try:
                ctx_d = _json.loads(ctx_raw) if isinstance(ctx_raw, str) else dict(ctx_raw)
            except Exception:
                ctx_d = {}

            class _EstShim:
                __slots__ = ("agent_type", "context")

            shim = _EstShim()
            shim.agent_type = agent_type
            shim.context = ctx_d if isinstance(ctx_d, dict) else {}
            est = estimate_for(shim, btable=get_btable())
            est_in = est.in_tokens
            est_out = est.out_tokens
        except Exception:
            est_in = 0
            est_out = 0

        select_err = None
        pick = None
        if fatih_hoca is not None:
            try:
                pick = fatih_hoca.select(
                    task=agent_type,
                    agent_type=agent_type,
                    difficulty=difficulty,
                    urgency=urgency,
                    estimated_input_tokens=est_in,
                    estimated_output_tokens=est_out,
                )
            except Exception as e:
                select_err = repr(e)
                pick = None
        if pick is None:
            _log.debug(
                f"admission: task #{task['id']} agent={agent_type} d={difficulty} "
                f"urgency={urgency:.3f} select=None err={select_err}"
            )
            continue

        # Ceiling backstop for LLM tasks: spent + in-flight + new estimated cost.
        if not await _ceiling_ok(task, _log):
            continue

        if not await _claim_task(task["id"]):
            _log.debug(f"admission: task #{task['id']} claim race lost")
            continue

        # Reserve the in-flight slot at admission. The peer registry
        # (src.core.in_flight) represents three GPU-slot lifecycle states:
        # (1) admitted-not-yet-calling, (2) mid-call, (3) between calls
        # in the ReAct iteration gap. Without this reserve, state (1) has
        # no writer — the dispatcher's begin_call only fires AFTER agent
        # pre-work (RAG + chain-context + file-tree scan, often 10-20s),
        # during which the next Beckman tick reads an empty in-flight list
        # and admits a second local task onto a lane the first will
        # re-occupy. Dispatcher's begin_call UPSERTS the same task-{id}
        # key later, so mid-task model changes remain accurate. Release
        # fires via orchestrator._dispatch finally → release_task.
        #
        # Fail-open: if the peer registry is unreachable, still admit.
        # Nerd_herd being down shouldn't halt the system.
        try:
            from src.core.in_flight import reserve_task
            # Pass projected token consumption so pool-pressure consumers
            # back-pressure parallel admissions in the same window. Without
            # this, several tasks admitted within ~15s on the same cloud
            # model all see fresh tpm_remaining and overshoot the quota.
            await reserve_task(
                task["id"], pick,
                est_tokens=int((est_in or 0) + (est_out or 0)),
            )
        except Exception as e:
            _log.warning(f"admission: reserve_task failed #{task['id']}: {e}")

        _log.info(
            f"admission: task #{task['id']} ADMIT model={pick.model.name} "
            f"urgency={urgency:.3f} (selector cleared pool-pressure gate)"
        )
        task["preselected_pick"] = pick
        task["status"] = "processing"

        # Z10 T2A: stamp estimated_cost_usd on the task. Pulls historical
        # avg cost for (model, agent_type) or falls back to per-kind
        # defaults. Best-effort — never blocks admission.
        try:
            from src.infra.db import estimate_task_cost, set_task_estimated_cost
            _kind = task.get("agent_type")
            _model_id = pick.model.name if pick and getattr(pick, "model", None) else None
            _est = await estimate_task_cost(_model_id, _kind)
            await set_task_estimated_cost(int(task["id"]), float(_est))
        except Exception as _e:
            _log.debug(f"admission: estimate_task_cost skipped #{task['id']}: {_e}")

        # Write selected_model into the in-memory context so the orchestrator
        # can forward it to dispatcher.dispatch() without a DB round-trip.
        # dispatcher.dispatch() reads it back as preselected_pick → skip re-select.
        # Only raw_dispatch tasks (LLM calls enqueued via dispatcher.request) need
        # this; agent tasks never reach dispatcher.dispatch().
        try:
            import json as _json
            _ctx_raw = task.get("context") or "{}"
            _ctx_d = _json.loads(_ctx_raw) if isinstance(_ctx_raw, str) else dict(_ctx_raw or {})
            _llm_call = _ctx_d.get("llm_call") if isinstance(_ctx_d, dict) else None
            if isinstance(_llm_call, dict) and _llm_call.get("raw_dispatch"):
                _llm_call["selected_model"] = pick.model.name
                task["context"] = _json.dumps(_ctx_d)
        except Exception as _e:
            _log.debug(f"admission: selected_model inject failed #{task['id']}: {_e}")
        _last_admission_fp = fp
        _last_admission_admitted = True
        return task

    _last_admission_fp = fp
    _last_admission_admitted = False
    return None


async def on_task_finished(task_id, result: dict = None) -> None:
    """Mark terminal + create any follow-up tasks the result implies.

    Pipeline: route_result -> rewrite_actions -> apply_actions.
    No delegation to Orchestrator. Mission-task completions produce a
    MissionAdvance action which spawns a mr_roboto workflow_advance task.

    Calling conventions:
      on_task_finished(task_id, result)  — production (orchestrator)
      on_task_finished(task_dict)        — test shorthand; task_dict has
                                           all fields incl. cost_usd, and
                                           only the spent/threshold path runs.
    """
    # ── Z0: spent_usd accumulation + threshold notifies ───────────────────
    # Detect test calling convention: single dict with task fields embedded.
    # In this mode we only run the cost-tracking path and return early —
    # the full routing pipeline requires a real DB task row.
    if isinstance(task_id, dict):
        _task_dict = task_id
        _mid = _task_dict.get("mission_id")
        _cost = float(_task_dict.get("cost_usd") or 0.0)
        if _mid is not None and _cost > 0:
            import json as _json
            from src.infra.db import get_db as _get_db
            _db = await _get_db()
            await _db.execute(
                "UPDATE missions SET spent_usd = COALESCE(spent_usd, 0) + ? WHERE id = ?",
                (_cost, _mid),
            )
            _cur = await _db.execute(
                "SELECT cost_ceiling_usd, spent_usd, context FROM missions WHERE id = ?",
                (_mid,),
            )
            _row = await _cur.fetchone()
            await _db.commit()
            if _row:
                _ceiling, _spent, _ctx_raw = _row[0], _row[1], _row[2]
                if _ceiling is not None and _ceiling > 0:
                    _ctx = _json.loads(_ctx_raw) if _ctx_raw else {}
                    _fired = set(_ctx.get("thresholds_fired", []))
                    _pct = (float(_spent) / float(_ceiling)) * 100
                    _new_fires = []
                    for _t in THRESHOLDS_PCT:
                        if _pct >= _t and _t not in _fired:
                            _new_fires.append(_t)
                            _fired.add(_t)
                    if _new_fires:
                        _ctx["thresholds_fired"] = sorted(_fired)
                        await _db.execute(
                            "UPDATE missions SET context = ? WHERE id = ?",
                            (_json.dumps(_ctx), _mid),
                        )
                        await _db.commit()
                        for _t in _new_fires:
                            await notify_threshold(_mid, _t, float(_spent), float(_ceiling))
        return

    from general_beckman.result_router import route_result
    from general_beckman.rewrite import rewrite_actions
    from general_beckman.apply import apply_actions
    from general_beckman.task_context import parse_context
    from src.infra.db import get_task
    from src.infra.logging_config import get_logger

    log = get_logger("beckman.on_task_finished")
    task = await get_task(task_id)
    if task is None:
        log.warning("on_task_finished: missing task", task_id=task_id)
        return
    task_ctx = parse_context(task)

    # Z10 T2A: stamp actual_cost_usd from accumulated model_call_tokens.
    # Run before route_result so downstream readers see the final number.
    try:
        from src.infra.db import finalize_task_actual_cost
        await finalize_task_actual_cost(task_id)
    except Exception as _e:
        log.debug("finalize_task_actual_cost skipped", task_id=task_id, error=str(_e))

    # ── Z0: spent_usd accumulation + threshold notifies ───────────────────
    # Production path: cost comes from result dict.
    import json as _json
    from src.infra.db import get_db as _get_db
    _mid = task.get("mission_id")
    _cost = float((result or {}).get("cost_usd") or 0.0)
    if _mid is not None and _cost > 0:
        _db = await _get_db()
        await _db.execute(
            "UPDATE missions SET spent_usd = COALESCE(spent_usd, 0) + ? WHERE id = ?",
            (_cost, _mid),
        )
        _cur = await _db.execute(
            "SELECT cost_ceiling_usd, spent_usd, context FROM missions WHERE id = ?",
            (_mid,),
        )
        _row = await _cur.fetchone()
        await _db.commit()
        if _row:
            _ceiling, _spent, _ctx_raw = _row[0], _row[1], _row[2]
            if _ceiling is not None and _ceiling > 0:
                _ctx = _json.loads(_ctx_raw) if _ctx_raw else {}
                _fired = set(_ctx.get("thresholds_fired", []))
                _pct = (float(_spent) / float(_ceiling)) * 100
                _new_fires = []
                for _t in THRESHOLDS_PCT:
                    if _pct >= _t and _t not in _fired:
                        _new_fires.append(_t)
                        _fired.add(_t)
                if _new_fires:
                    _ctx["thresholds_fired"] = sorted(_fired)
                    await _db.execute(
                        "UPDATE missions SET context = ? WHERE id = ?",
                        (_json.dumps(_ctx), _mid),
                    )
                    await _db.commit()
                    for _t in _new_fires:
                        await notify_threshold(_mid, _t, float(_spent), float(_ceiling))

    # Persist generating_model + accumulate failed_models. The retry-
    # recovery layers (model exclusion at attempts >= 3, difficulty
    # bump in src/core/retry.py::get_model_constraints) gate on
    # ctx.failed_models. Agent emits ``result.generating_model`` /
    # ``result.model`` but no code path was writing it back to ctx —
    # so failed_models stayed [] across 5+ retries and R1/R2 had
    # nothing to act on. Live signal: mission 57 task 4441 hit DLQ at
    # attempts=5 with empty failed_models. Fix: always record the
    # current run's model in ctx.generating_model; on quality-failure
    # status, append it to failed_models so the next retry's selector
    # can see it. Idempotent (no duplicates).
    _model = (
        (result or {}).get("generating_model")
        or (result or {}).get("model")
        or ""
    )
    _status = (result or {}).get("status") or "completed"
    # Persist tool_calls audit log into ctx so the grounding post-hook can
    # read it without going back to the runtime checkpoint. on_task_finished
    # is the join point: coulson returns ``tool_calls`` in result; ctx is
    # the surface every later step (apply, _posthook_agent_and_payload)
    # already reads.
    _ctx_changed = False
    _tool_calls = (result or {}).get("tool_calls")
    if isinstance(_tool_calls, list) and task_ctx.get("tool_calls") != _tool_calls:
        task_ctx["tool_calls"] = _tool_calls
        _ctx_changed = True
    if _model:
        if task_ctx.get("generating_model") != _model:
            task_ctx["generating_model"] = _model
            _ctx_changed = True
        # Quality-class failures (worker schema-fail, exhausted, timeout,
        # disguised failure) all flow through status="failed" here.
        # needs_clarification is NOT a model failure — agent worked, just
        # needs human input.
        if _status == "failed":
            _failed = list(task_ctx.get("failed_models") or [])
            if _model not in _failed:
                _failed.append(_model)
                task_ctx["failed_models"] = _failed
                _ctx_changed = True
                log.info(
                    "tracked failed_model",
                    task_id=task_id,
                    model=_model,
                    failed_count=len(_failed),
                )
    if _ctx_changed:
        try:
            from src.infra.db import update_task as _ut
            import json as _json
            await _ut(task_id, context=_json.dumps(task_ctx))
            # Refresh local task dict so downstream route_result
            # / apply_actions see the persisted state.
            task["context"] = _json.dumps(task_ctx)
        except Exception as e:
            log.warning("ctx persist failed", task_id=task_id, error=str(e))

    # Workflow-step post-hook runs synchronously before routing — stores
    # artifacts and may flip status (degenerate output, schema validation,
    # disguised failures, human-gate clarifications). Deferring this to
    # the workflow_advance mechanical task caused a race: dependent tasks
    # became ready and picked up empty blackboards before the advance
    # task ran.
    try:
        from src.workflows.engine.hooks import (
            is_workflow_step, post_execute_workflow_step,
        )
        if is_workflow_step(task_ctx):
            await post_execute_workflow_step(task, result)
    except Exception as e:
        log.warning("post_execute_workflow_step raised", task_id=task_id, error=str(e))
    # ── Continuation hooks ────────────────────────────────────────────────
    # Dispatch on_complete handler, chain next_task_spec, and resolve any
    # await_inline waiter — all three are independent (do all that apply).
    #
    # MUST run BEFORE the `route_result is None` early-return below.
    # raw_dispatch child tasks (LLM calls spawned via beckman.enqueue,
    # e.g. grade_task's reviewer child) have route_result return None —
    # they carry no workflow routing. With the hook block placed after the
    # early-return, the parent's inline-waiter future was never resolved,
    # the child completed silently, and the parent blocked the full
    # INLINE_TIMEOUT (600s) then DLQ'd. Production 2026-05-15 mission 69:
    # grader tasks #31203/#31204/#35460 timed out while 10 reviewer
    # children all completed cleanly.
    try:
        import json as _json
        _beckman_sub = (task_ctx or {}).get("beckman") or {}

        _on_complete = _beckman_sub.get("on_complete")
        if _on_complete:
            from general_beckman.continuations import dispatch_on_complete
            _cont_result = dict(result or {})
            asyncio.create_task(dispatch_on_complete(_on_complete, task_id, _cont_result))

        _next_spec = _beckman_sub.get("next_task_spec")
        if _next_spec and isinstance(_next_spec, dict):
            asyncio.create_task(enqueue(_next_spec, parent_id=task_id))

        if task_id in _inline_waiters:
            _status = (result or {}).get("status", "completed")
            _tr = TaskResult(
                status=_status,
                result=(result or {}).get("result"),
                error=(result or {}).get("error"),
            )
            resolve_inline(task_id, _tr)
    except Exception as _ce:
        log.debug("continuation hook failed", task_id=task_id, error=str(_ce))

    actions = route_result(task, result)
    if actions is None:
        return
    if not isinstance(actions, (list, tuple)):
        actions = [actions]
    actions = rewrite_actions(task, task_ctx, actions)
    await apply_actions(task, actions)

    # Progress ping: terse per-step notification for workflow-step tasks so
    # the user sees a mission moving forward rather than 2+ minutes of
    # silence. Bookkeeping tasks (mechanical / grader / artifact_summarizer)
    # are skipped — they're internal machinery, not user progress.
    try:
        _bookkeeping = task.get("agent_type") in (
            "mechanical", "grader", "artifact_summarizer",
        )
        if not _bookkeeping:
            status = (result or {}).get("status", "completed")
            if status in ("completed", "failed", "needs_clarification"):
                if task.get("mission_id"):
                    await _send_step_progress(task, status, result)
                else:
                    await _send_standalone_completion(task, status, result)
    except Exception as e:
        log.debug("progress ping failed", task_id=task_id, error=str(e))

    try:
        from general_beckman.queue_profile_push import invalidate_completed_id_cache
        invalidate_completed_id_cache(task_id)
    except Exception:
        pass

    try:
        from general_beckman.queue_profile_push import build_and_push
        await build_and_push()
    except Exception as e:
        log.debug("queue_profile push failed", task_id=task_id, error=str(e))

    # Z0: DLQ cascade auto-pause trigger
    try:
        if task.get("mission_id") is not None and task.get("status") == "failed":
            from general_beckman.lifecycle_events import dlq_cascade_check
            await dlq_cascade_check(task["mission_id"])
    except Exception as e:
        log.warning("dlq_cascade_check failed: %s", e)


async def _send_standalone_completion(task: dict, status: str, result: dict) -> None:
    """Deliver completion for a mission-less task back to the user.

    Standalone tasks (created from generic messages) carry chat_id in
    context. Without this path, the user sees 'Task #N queued' and then
    radio silence even after the task finishes.
    """
    if task.get("agent_type") in ("mechanical", "grader", "artifact_summarizer"):
        return
    import json as _json
    ctx_raw = task.get("context") or "{}"
    try:
        ctx = _json.loads(ctx_raw) if isinstance(ctx_raw, str) else dict(ctx_raw)
    except Exception:
        ctx = {}
    chat_id = ctx.get("chat_id")
    if not chat_id:
        return
    from src.app.telegram_bot import get_telegram
    tg = get_telegram()
    if tg is None:
        return
    title = (task.get("title") or "").strip() or f"task #{task['id']}"
    icon = {
        "completed": "✅",
        "failed": "❌",
        "needs_clarification": "❓",
    }.get(status, "ℹ️")
    body_parts = [f"{icon} Task #{task['id']} — {title[:80]}"]
    if status == "completed":
        out = (result or {}).get("result") or ""
        if isinstance(out, str) and out.strip():
            excerpt = out.strip()
            if len(excerpt) > 3500:
                excerpt = excerpt[:3500] + "\n...(truncated)"
            body_parts.append(excerpt)
    elif status == "failed":
        err = (result or {}).get("error") or "error"
        body_parts.append(f"Error: {str(err)[:500]}")
    await tg.send_notification("\n\n".join(body_parts))


async def _send_step_progress(task: dict, status: str, result: dict) -> None:
    """Send a one-line Telegram progress update when a mission step finishes.

    Fires from on_task_finished, which runs BEFORE the grader verdict.
    A worker that finished is only "done" from the workflow's POV once
    grading passes. Check the live DB status to avoid premature ticks
    on steps that are queued for re-grade or retry.
    """
    if task.get("agent_type") in ("mechanical", "grader", "artifact_summarizer"):
        return
    # Always compare the raw agent-reported status against the live DB
    # status before pinging. The rewrite layer can flip actions between
    # the two (e.g. RequestClarification → CompleteWithReusedAnswer when
    # clarification_history exists). If the rewrite resolved it, the DB
    # row is already "completed" and the needs_clarification ping would
    # wrongly re-alarm the user.
    if status in ("completed", "needs_clarification", "failed"):
        from src.infra.db import get_task as _get_task
        live = await _get_task(task["id"])
        live_status = (live or {}).get("status", "")
        # Silent when DB already shows the step done from a different
        # path. For "completed" the prior gate rule holds (skip if not
        # yet completed — grader still running). For
        # "needs_clarification"/"failed" we silence when the rewrite
        # short-circuited to completed.
        if status == "completed" and live_status != "completed":
            return
        if status in ("needs_clarification", "failed") and live_status == "completed":
            return
    from src.app.telegram_bot import get_telegram
    tg = get_telegram()
    if tg is None:
        return
    # Title is typically "[1.1] enrich_product_results"; reuse it verbatim.
    title = (task.get("title") or "").strip() or f"task #{task['id']}"
    icon = {"completed": "\u2705", "failed": "\u274c", "needs_clarification": "\u2753"}.get(status, "\u2139\ufe0f")
    msg = f"{icon} {title}"
    if status == "failed":
        err = (result or {}).get("error") or "error"
        msg += f"\n  {str(err)[:140]}"
    await tg.send_notification(msg)


async def enqueue(
    spec: dict,
    *,
    parent_id: int | None = None,
    await_inline: bool = False,
    on_complete: str | None = None,
    next_task_spec: dict | None = None,
    lane: str | None = None,
) -> "int | TaskResult":
    """Single external write path for all Beckman tasks.

    Parameters
    ----------
    spec:
        Task creation kwargs (passed through to add_task). ``spec["kind"]``
        defaults to ``"main_work"`` if absent.
    parent_id:
        ID of a parent task — stored in tasks.parent_task_id.
    await_inline:
        Block until the task reaches a terminal state (resolved via
        ``resolve_inline``).  Returns a ``TaskResult``.
    on_complete:
        Name of a registered continuation handler (see continuations.py).
        Stored inside ``spec["context"]["beckman"]["on_complete"]``.
    next_task_spec:
        Spec dict for a follow-up task to enqueue when this task reaches
        a terminal state.  Stored inside context["beckman"]["next_task_spec"].
    """
    import json as _json
    from src.infra.db import add_task
    from general_beckman.queue_profile_push import build_and_push
    from general_beckman.lanes import pick_lane

    spec = dict(spec)  # shallow copy — don't mutate caller's dict

    # ── kind ──────────────────────────────────────────────────────────────
    kind = spec.pop("kind", "main_work")

    # ── lane ──────────────────────────────────────────────────────────────
    # Z8 T1B: explicit ``lane=`` overrides; otherwise derive from
    # agent_type via ``pick_lane``. spec.lane (rare) wins last so an
    # in-spec value is respected.
    if "lane" in spec:
        _lane = spec.pop("lane")
    elif lane is not None:
        _lane = lane
    else:
        _lane = pick_lane(spec.get("agent_type"))

    # ── parent_id ─────────────────────────────────────────────────────────
    if parent_id is not None:
        spec["parent_task_id"] = parent_id

    # ── continuation envelope ─────────────────────────────────────────────
    if on_complete is not None or next_task_spec is not None:
        raw_ctx = spec.get("context")
        if raw_ctx is None:
            ctx: dict = {}
        elif isinstance(raw_ctx, str):
            try:
                ctx = _json.loads(raw_ctx)
            except Exception:
                ctx = {}
        else:
            ctx = dict(raw_ctx)

        beckman_sub: dict = dict(ctx.get("beckman") or {})
        if on_complete is not None:
            beckman_sub["on_complete"] = on_complete
        if next_task_spec is not None:
            beckman_sub["next_task_spec"] = next_task_spec
        ctx["beckman"] = beckman_sub
        spec["context"] = ctx  # add_task will json.dumps() it

    task_id = await add_task(**spec, kind=kind, lane=_lane)
    await build_and_push()

    if not await_inline:
        return task_id

    # ── inline-wait path ───────────────────────────────────────────────────
    # Keep the PARENT task's heartbeat fresh while we wait on the child.
    # The child's runner sets the heartbeat contextvar to the child id,
    # so the parent goes silent and the orchestrator's no-progress watchdog
    # kills it at 300s even though work is happening. Wrap every inline
    # wait — grade_task and other beckman.enqueue(await_inline=True)
    # callers all suffered from this; only dispatcher.request had its own
    # wrapper (production 2026-05-14 mission 69 grader task #31203:
    # 6× watchdog kills × 5 min = 30 min wasted before DLQ).
    try:
        from src.core import heartbeat as _hb_inline
        _inline_keepalive_cm = _hb_inline.keepalive
    except Exception:
        _inline_keepalive_cm = None
    loop = asyncio.get_event_loop()
    fut: asyncio.Future[TaskResult] = loop.create_future()
    _inline_waiters[task_id] = fut
    try:
        if _inline_keepalive_cm is not None:
            async with _inline_keepalive_cm():
                return await asyncio.wait_for(asyncio.shield(fut), timeout=INLINE_TIMEOUT)
        return await asyncio.wait_for(asyncio.shield(fut), timeout=INLINE_TIMEOUT)
    except asyncio.TimeoutError:
        _inline_waiters.pop(task_id, None)
        raise


async def on_model_swap(old_model: str | None, new_model: str | None) -> None:
    """Called by the local model manager when a model swap completes.

    Wakes tasks whose retries were delayed waiting for *any* model to
    load. Grading is no longer triggered here — it's a regular task
    flowing through next_task().
    """
    try:
        from src.infra.db import accelerate_retries
        await accelerate_retries("model_swap")
    except Exception as e:
        from src.infra.logging_config import get_logger
        get_logger("beckman.on_model_swap").debug(
            f"accelerate_retries failed: {e}",
        )

