"""Z9 T2B — ``analytics_digest`` mechanical data-pull executor.

Routed via ``mr_roboto.run`` when ``payload["action"] == "analytics_digest"``.
Armed weekly per-mission by ``mission_cron`` (interval 604800s) once the
mission completes Phase 14 (launch).

Architecture contract (project memory — non-negotiable)
-------------------------------------------------------
This executor is **mechanical**: it never calls the LLM dispatcher. It only
does the *data pull* — PostHog ``vendor_call`` reads + local DB aggregate
queries — bundles everything into a structured ``digest_input`` dict, then
**enqueues the LLM synthesis agent via Beckman** (``general_beckman.enqueue``).
The mechanical step and the LLM step are separate tasks; a mechanical never
does LLM work itself.

End-to-end flow::

    cron tick → analytics_digest (mechanical, this file)
        ├─ vendor_call posthog (events / funnel / retention / cohorts)
        ├─ DB: growth_events, pending hypotheses, mission_lessons,
        │      model_pick_log aggregates (last 7d)
        ├─ growth_events row  kind="digest_run"
        └─ beckman.enqueue(growth_digest_synthesizer agent task)
                            → agent drafts markdown → growth_events
                              row kind="weekly_digest" (written by agent path)
"""
from __future__ import annotations

import datetime
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.analytics_digest")

SYSTEM_MISSION_ID = 0
_WINDOW_DAYS = 7


# ── time helpers ───────────────────────────────────────────────────────────


def _since_db(now: datetime.datetime | None = None) -> str:
    """SQLite datetime string for 7 days ago ('YYYY-MM-DD HH:MM:SS')."""
    now = now or datetime.datetime.utcnow()
    return (now - datetime.timedelta(days=_WINDOW_DAYS)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _iso_year_week(now: datetime.datetime | None = None) -> str:
    now = now or datetime.datetime.utcnow()
    yr, wk, _ = now.isocalendar()
    return f"{yr}-W{wk:02d}"


# ── posthog vendor_call indirection ────────────────────────────────────────


async def _posthog(task: dict, action: str, params: dict) -> dict:
    """One PostHog read via the vendor_call executor.

    Returns the vendor_call envelope ``{ok, result, ...}``. Mock mode
    (``KUTAI_ENV != prod``) means offline callers get deterministic fakes
    from ``configs/posthog.json`` without a network hop.
    """
    from mr_roboto.executors.vendor_call import run as vendor_call_run

    sub = {
        "mission_id": task.get("mission_id"),
        "id": task.get("id"),
        "context": {
            "post_hook": {
                "service": "posthog",
                "action": action,
                "params": params,
            }
        },
    }
    try:
        return await vendor_call_run(sub)
    except Exception as exc:  # noqa: BLE001
        logger.warning("posthog vendor_call raised", action=action, error=str(exc))
        return {"ok": False, "reason": "vendor_call_raised", "error": str(exc)}


def _posthog_project_id(mission_context: dict | None) -> str:
    """Resolve the PostHog project id for this mission.

    v1 = single product per workspace; ``mission.context.posthog_project_id``
    overrides. Mock mode ignores the value entirely.
    """
    if isinstance(mission_context, dict):
        pid = mission_context.get("posthog_project_id")
        if pid:
            return str(pid)
    import os

    return str(os.getenv("KUTAI_POSTHOG_PROJECT_ID", "default"))


# ── analytics pull (PostHog, read-only) ────────────────────────────────────


async def _pull_posthog(task: dict, project_id: str) -> dict[str, Any]:
    """Pull last-7d events / funnel / retention / cohorts. Never raises."""
    base = {"project_id": project_id}
    events = await _posthog(task, "query_events", dict(base))
    funnel = await _posthog(task, "query_funnel", dict(base))
    retention = await _posthog(task, "query_retention", dict(base))
    cohorts = await _posthog(task, "list_cohorts", dict(base))

    def _result(env: dict) -> Any:
        return env.get("result") if isinstance(env, dict) and env.get("ok") else None

    ev = _result(events) or {}
    event_rows = ev.get("results") if isinstance(ev, dict) else None
    event_count = len(event_rows) if isinstance(event_rows, list) else 0

    fn = _result(funnel) or {}
    funnel_rows = fn.get("result") if isinstance(fn, dict) else None

    rt = _result(retention) or {}
    retention_rows = rt.get("result") if isinstance(rt, dict) else None
    # First cohort's day-by-day retention curve, if present.
    retention_curve: list[Any] = []
    if isinstance(retention_rows, list) and retention_rows:
        first = retention_rows[0]
        if isinstance(first, dict) and isinstance(first.get("values"), list):
            retention_curve = list(first["values"])

    ch = _result(cohorts) or {}
    cohort_rows = ch.get("results") if isinstance(ch, dict) else None

    return {
        "ok": any(
            isinstance(e, dict) and e.get("ok")
            for e in (events, funnel, retention, cohorts)
        ),
        "event_count": event_count,
        "events": event_rows or [],
        "funnel": funnel_rows or [],
        "retention": retention_rows or [],
        "retention_curve": retention_curve,
        "cohorts": cohort_rows or [],
    }


# ── local DB aggregates (read-only) ────────────────────────────────────────


async def _pull_db_aggregates(mission_id: int | None, since: str) -> dict[str, Any]:
    """Query growth_events, pending hypotheses, mission_lessons,
    model_pick_log aggregates for the last 7 days. Never raises."""
    out: dict[str, Any] = {
        "growth_events": [],
        "pending_hypotheses": [],
        "mission_lessons": [],
        "model_pick": [],
        "retry_stats": {},
        "recipe_pin_rate": None,
    }

    # growth_events (last 7d) ------------------------------------------------
    try:
        from dabidabi import get_growth_events

        rows = await get_growth_events(mission_id=mission_id, since=since)
        out["growth_events"] = rows or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("growth_events query failed", error=str(exc))

    # pending hypotheses (verdict window may have closed) -------------------
    try:
        from dabidabi import get_pending_hypotheses

        out["pending_hypotheses"] = await get_pending_hypotheses(
            mission_id=mission_id
        ) or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("pending_hypotheses query failed", error=str(exc))

    # raw aggregate SQL over shared tables ----------------------------------
    try:
        from dabidabi import get_db

        db = await get_db()

        # mission_lessons — verdict-side cross-mission memory.
        try:
            cur = await db.execute(
                "SELECT stack, domain, occurrences, last_seen_at "
                "FROM mission_lessons "
                "WHERE last_seen_at >= ? "
                "ORDER BY occurrences DESC LIMIT 20",
                (since,),
            )
            cols = [d[0] for d in cur.description]
            out["mission_lessons"] = [
                dict(zip(cols, r)) for r in await cur.fetchall()
            ]
        except Exception as exc:  # noqa: BLE001
            logger.debug("mission_lessons query failed", error=str(exc))

        # model_pick_log — internal-health: pick distribution last 7d.
        # Repointed to fatih_hoca's read-API (owns model-registry SQL).
        try:
            from fatih_hoca.db import get_pick_summary
            out["model_pick"] = await get_pick_summary(since_days=_WINDOW_DAYS)
        except Exception as exc:  # noqa: BLE001
            logger.debug("model_pick_log query failed", error=str(exc))

        # retry rates — internal-health from the tasks table.
        try:
            cur = await db.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END) AS retried, "
                "SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed "
                "FROM tasks WHERE created_at >= ?",
                (since,),
            )
            row = await cur.fetchone()
            if row:
                total = int(row[0] or 0)
                retried = int(row[1] or 0)
                failed = int(row[2] or 0)
                out["retry_stats"] = {
                    "total_tasks": total,
                    "retried_tasks": retried,
                    "failed_tasks": failed,
                    "retry_rate": round(retried / total, 3) if total else 0.0,
                    "failure_rate": round(failed / total, 3) if total else 0.0,
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("retry_stats query failed", error=str(exc))

        # recipe pin rate — internal-health (best-effort; table may be absent).
        try:
            cur = await db.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='recipe_pin_log'"
            )
            if await cur.fetchone():
                cur = await db.execute(
                    "SELECT COUNT(*) FROM recipe_pin_log WHERE created_at >= ?",
                    (since,),
                )
                pinned = await cur.fetchone()
                out["recipe_pin_rate"] = int(pinned[0]) if pinned else 0
        except Exception as exc:  # noqa: BLE001
            logger.debug("recipe_pin_log query failed", error=str(exc))

    except Exception as exc:  # noqa: BLE001
        logger.debug("db aggregate pull failed", error=str(exc))

    return out


# ── success_metrics artifact (north-star) ──────────────────────────────────


async def _load_success_metrics(mission_id: int | None) -> dict[str, Any]:
    """Best-effort retrieve the ``success_metrics`` artifact (step 2.9).

    Returns ``{}`` when unreachable — the synthesis agent degrades to a
    "north-star not configured" digest section.
    """
    if not mission_id:
        return {}
    try:
        import json

        from src.workflows.engine.hooks import get_artifact_store

        store = get_artifact_store()
        raw = await store.retrieve(int(mission_id), "success_metrics")
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("success_metrics retrieve failed", error=str(exc))
    return {}


# ── weekly_digest persistence (Beckman on_complete continuation) ───────────

# Continuation name — passed as ``on_complete`` so the synthesised digest
# markdown lands in growth_events when the agent task finishes.
_DIGEST_CONTINUATION = "growth.store_weekly_digest"


async def _store_weekly_digest(task_id: int, result: dict, state: dict | None = None) -> None:
    """Beckman on_complete handler — persist the agent's digest markdown.

    ``result`` is the agent task's terminal result envelope; ``result['result']``
    is the ``growth_digest_synthesizer`` agent's ``final_answer`` payload (the
    Telegram-ready markdown). Writes a ``growth_events`` row ``kind="weekly_digest"``
    so the ``/digest`` Telegram command can surface the latest digest.
    """
    try:
        from dabidabi import get_db
        from general_beckman import record_growth_event

        markdown = result.get("result")
        if isinstance(markdown, dict):
            # final_answer may arrive wrapped — unwrap the inner result.
            markdown = markdown.get("result") or markdown.get("text") or ""
        markdown = str(markdown or "").strip()
        if not markdown:
            logger.warning("store_weekly_digest: empty digest", task_id=task_id)
            return

        # Resolve the mission id from the completed task row.
        mission_id: int | None = None
        try:
            db = await get_db()
            cur = await db.execute(
                "SELECT mission_id FROM tasks WHERE id = ?", (task_id,)
            )
            row = await cur.fetchone()
            if row and row[0] is not None:
                mission_id = int(row[0])
        except Exception as exc:  # noqa: BLE001
            logger.debug("store_weekly_digest mission lookup failed", error=str(exc))

        await record_growth_event(
            mission_id=mission_id,
            kind="weekly_digest",
            properties={
                "markdown": markdown,
                "iso_year_week": _iso_year_week(),
                "source_task_id": task_id,
            },
        )
        logger.info(
            "store_weekly_digest: weekly_digest persisted",
            task_id=task_id,
            mission_id=mission_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("store_weekly_digest failed", task_id=task_id, error=str(exc))


def register_continuations() -> None:
    """Register the digest-persistence continuation (idempotent)."""
    try:
        from general_beckman.continuations import register
        register(_DIGEST_CONTINUATION, _store_weekly_digest)
    except Exception as exc:  # noqa: BLE001
        logger.debug("digest continuation registration deferred", error=str(exc))


# Back-compat name used internally elsewhere in this module.
_register_continuation = register_continuations

# Register at import so the handler is present for restart reconcile.
register_continuations()


# ── synthesis agent enqueue (via Beckman — never the dispatcher) ───────────


async def _enqueue_synthesis_agent(
    mission_id: int | None,
    task_id: int | None,
    digest_input: dict[str, Any],
) -> int | None:
    """Enqueue the ``growth_digest_synthesizer`` LLM agent task via Beckman.

    This is the mechanical→Beckman→agent hand-off. The mechanical executor
    NEVER calls ``LLMDispatcher.request`` — it hands the bundle to Beckman,
    which owns the singular dispatch path. ``on_complete`` chains the
    digest-persistence continuation so the markdown lands in growth_events.
    """
    try:
        from general_beckman import enqueue

        _register_continuation()  # defensive — ensure handler present

        spec = {
            "title": f"Weekly growth digest synthesis (mid={mission_id})",
            "description": (
                "Draft a founder-facing weekly growth digest from the "
                "analytics pull in context.digest_input."
            ),
            "agent_type": "growth_digest_synthesizer",
            "mission_id": mission_id,
            "context": {"digest_input": digest_input},
            "depends_on": [],
        }
        new_id = await enqueue(
            spec, parent_id=task_id, on_complete=_DIGEST_CONTINUATION
        )
        logger.info(
            "analytics_digest enqueued synthesis agent",
            mission_id=mission_id,
            agent_task_id=new_id,
        )
        return int(new_id) if isinstance(new_id, int) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "analytics_digest failed to enqueue synthesis agent",
            mission_id=mission_id,
            error=str(exc),
        )
        return None


# ── main entrypoint ────────────────────────────────────────────────────────


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Execute one weekly analytics data-pull cycle.

    Always returns a dict — never raises into the dispatcher.
    """
    mission_id = task.get("mission_id")
    try:
        mission_id_int: int | None = int(mission_id) if mission_id is not None else None
    except (TypeError, ValueError):
        mission_id_int = None

    payload = task.get("payload") or {}
    ctx = task.get("context") or {}
    mission_context = ctx.get("mission_context") if isinstance(ctx, dict) else None

    since = _since_db()
    iso_yw = _iso_year_week()
    project_id = _posthog_project_id(mission_context)

    # 1. PostHog analytics pull (mock-mode safe).
    posthog = await _pull_posthog(task, project_id)

    # 2. Local DB aggregates.
    db_agg = await _pull_db_aggregates(mission_id_int, since)

    # 3. success_metrics artifact → north-star.
    success_metrics = await _load_success_metrics(mission_id_int)
    north_star = success_metrics.get("north_star_metric") or {}
    aarrr_metrics = success_metrics.get("aarrr_metrics") or []

    # Experiments — derived from active experiment_variants if any are
    # surfaced via growth_events; v1 keeps the slot for the insufficient-N
    # detector and T5's A/B harness fills it.
    experiments = payload.get("experiments") or []

    # 4. Bundle the structured digest_input.
    digest_input: dict[str, Any] = {
        "mission_id": mission_id_int,
        "iso_year_week": iso_yw,
        "window_days": _WINDOW_DAYS,
        "since": since,
        # north-star / success metrics
        "north_star": north_star,
        "aarrr_metrics": aarrr_metrics,
        # posthog analytics
        "posthog_ok": posthog.get("ok", False),
        "event_count": posthog.get("event_count", 0),
        "events": posthog.get("events", []),
        "funnel": posthog.get("funnel", []),
        "retention": posthog.get("retention", []),
        "retention_curve": posthog.get("retention_curve", []),
        "cohorts": posthog.get("cohorts", []),
        # local DB aggregates
        "growth_events": db_agg.get("growth_events", []),
        "pending_hypotheses": db_agg.get("pending_hypotheses", []),
        "mission_lessons": db_agg.get("mission_lessons", []),
        "model_pick": db_agg.get("model_pick", []),
        "retry_stats": db_agg.get("retry_stats", {}),
        "recipe_pin_rate": db_agg.get("recipe_pin_rate"),
        # experiment slot for T2D insufficient-N + T5 A/B
        "experiments": experiments,
    }

    # 5. Write the digest_run growth_events row.
    digest_run_id: int | None = None
    try:
        from general_beckman import record_growth_event

        digest_run_id = await record_growth_event(
            mission_id=mission_id_int,
            kind="digest_run",
            properties={
                "iso_year_week": iso_yw,
                "posthog_ok": posthog.get("ok", False),
                "event_count": posthog.get("event_count", 0),
                "pending_hypotheses": len(db_agg.get("pending_hypotheses", [])),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("analytics_digest digest_run write failed", error=str(exc))

    # 5b. Z7 #7 — emit the metric growth_events investor_bullets reads.
    # investor_bullets' _fetch_z6_metrics / _fetch_review_density query
    # growth_events for kinds 'metric_emit' / 'review_density_metric' but
    # nothing ever wrote them, so the monthly investor card was always
    # empty. analytics_digest is the metric-pull cron — it emits them here.
    try:
        await _emit_investor_metrics(mission_id_int, since, posthog, db_agg)
    except Exception as exc:  # noqa: BLE001 — metric emit must not fail the digest
        logger.warning("analytics_digest investor-metric emit failed",
                        error=str(exc))

    # 6. Hand off to the LLM synthesis agent via Beckman (NOT the dispatcher).
    agent_task_id = await _enqueue_synthesis_agent(
        mission_id_int, task.get("id"), digest_input
    )

    logger.info(
        "analytics_digest complete",
        mission_id=mission_id_int,
        iso_year_week=iso_yw,
        event_count=posthog.get("event_count", 0),
        agent_task_id=agent_task_id,
        digest_run_id=digest_run_id,
    )
    return {
        "ok": True,
        "iso_year_week": iso_yw,
        "event_count": posthog.get("event_count", 0),
        "posthog_ok": posthog.get("ok", False),
        "digest_run_id": digest_run_id,
        "synthesis_agent_task_id": agent_task_id,
        "digest_input": digest_input,
    }


#: agent_type values that count as a shipped unit of code work.
_CODE_AGENT_TYPES: tuple[str, ...] = (
    "coder", "implementer", "fixer", "test_generator",
)


async def _emit_investor_metrics(
    mission_id: int | None,
    since: str,
    posthog: dict,
    db_agg: dict,
) -> None:
    """Write the metric_emit + review_density_metric growth_events rows that
    investor_bullets consumes. One row of each kind per digest run.

    metric_emit carries the usage metrics analytics_digest genuinely has
    (event_count, funnel conversion). review_density_metric carries
    prs_shipped — completed code-emitting tasks for the mission in window.
    Financial keys (mrr/revenue) stay absent until a Stripe producer wires
    them; investor_bullets reads every numeric key, so partial data is fine.
    """
    if mission_id is None:
        return
    from dabidabi import get_db
    from general_beckman import record_growth_event
    db = await get_db()

    # ── metric_emit ──
    metric_props: dict[str, Any] = {
        "event_count": int(posthog.get("event_count", 0) or 0),
    }
    funnel = posthog.get("funnel") or []
    if funnel and isinstance(funnel, list):
        # First→last step conversion, if the funnel has numeric counts.
        try:
            first = float(funnel[0].get("count", 0))
            last = float(funnel[-1].get("count", 0))
            if first > 0:
                metric_props["funnel_conversion"] = round(last / first, 4)
        except (KeyError, TypeError, ValueError, AttributeError):
            pass
    retry = db_agg.get("retry_stats") or {}
    if isinstance(retry, dict) and retry.get("total"):
        metric_props["task_retry_rate"] = round(
            float(retry.get("retried", 0)) / float(retry["total"]), 4)
    await record_growth_event(
        mission_id=mission_id, kind="metric_emit", properties=metric_props)

    # ── review_density_metric ──
    prs_shipped = 0
    try:
        placeholders = ",".join("?" * len(_CODE_AGENT_TYPES))
        cur = await db.execute(
            f"SELECT COUNT(*) FROM tasks "
            f"WHERE mission_id = ? AND status = 'completed' "
            f"  AND agent_type IN ({placeholders}) "
            f"  AND COALESCE(completed_at, created_at) >= ?",
            (mission_id, *_CODE_AGENT_TYPES, since),
        )
        row = await cur.fetchone()
        await cur.close()
        prs_shipped = int(row[0]) if row else 0
    except Exception as exc:  # noqa: BLE001
        logger.debug("prs_shipped count failed", error=str(exc))
    await record_growth_event(
        mission_id=mission_id, kind="review_density_metric",
        properties={"prs_shipped": prs_shipped})


__all__ = ["run"]
