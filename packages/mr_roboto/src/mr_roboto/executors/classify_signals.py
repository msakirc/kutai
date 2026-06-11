"""Z9 T3B — classify_signals mechanical executor.

Mechanical half of the signal-classification loop. This executor never
calls the LLM directly — it:

  1. fetches unclassified ``raw_signal`` growth_events,
  2. enqueues the ``signal_classifier`` agent via ``beckman.enqueue`` with an
     ``on_complete`` continuation handler,
  3. the continuation handler (``_on_classifier_complete``) writes each
     agent verdict back as a ``growth_events`` row ``kind="classified_signal"``.

Architecture rule honored: the classifier is LLM work → it goes through
Beckman as an agent task. The mechanical layer only orchestrates and
persists; it issues zero ``LLMDispatcher.request`` calls.

``classified_signal`` properties shape::

    {
        "raw_signal_id": <growth_events.id of the source raw_signal>,
        "external_id": "<provider external id>",
        "signal_type": "<original signal_type>",
        "label": "bug" | "feature_request" | "churn_signal"
                 | "pricing_feedback" | "praise" | "spam",
        "domain": "<recipe lessons_domain slug or general slug>",
        "confidence": 0.0..1.0,
        "content_excerpt": "<first 280 chars of the raw signal content>"
    }
"""
from __future__ import annotations

import json

from src.infra.logging_config import get_logger

_log = get_logger("mr_roboto.classify_signals")

# How many raw signals to hand the classifier in one batch.
_BATCH_LIMIT = 40


def _recipe_domains() -> list[str]:
    """Distinct recipe lessons_domain slugs — the classifier's domain menu.

    Best-effort: a missing recipes/ dir just yields an empty list, which the
    agent prompt tolerates (it falls back to a general slug).
    """
    try:
        from src.infra.recipes import list_recipes

        domains = sorted(
            {r.lessons_domain for r in list_recipes() if r.lessons_domain}
        )
        return domains
    except Exception as exc:  # noqa: BLE001
        _log.debug("recipe domain scan failed", error=str(exc))
        return []


async def _classified_external_ids() -> set[str]:
    """external_ids that already have a classified_signal row (dedup guard)."""
    from src.infra.db import get_growth_events

    rows = await get_growth_events(kind="classified_signal")
    out: set[str] = set()
    for r in rows:
        eid = (r.get("properties") or {}).get("external_id")
        if eid:
            out.add(str(eid))
    return out


async def run(task: dict) -> dict:
    """Fetch unclassified raw signals and enqueue the classifier agent.

    Returns a dict describing what was dispatched. When there are no
    unclassified signals it returns ``{"ok": True, "enqueued": False}``
    without touching Beckman.
    """
    import general_beckman
    from src.infra.db import get_growth_events

    payload = task.get("payload") or {}
    mission_id = task.get("mission_id") or payload.get("mission_id")

    raw = await get_growth_events(
        mission_id=mission_id, kind="raw_signal", limit=_BATCH_LIMIT * 3
    )
    done = await _classified_external_ids()

    pending = []
    for row in raw:
        props = row.get("properties") or {}
        eid = props.get("external_id")
        if eid is not None and str(eid) in done:
            continue
        pending.append(
            {
                "raw_signal_id": row.get("id"),
                "external_id": props.get("external_id"),
                "signal_type": props.get("signal_type"),
                "content": props.get("content"),
                "provider": props.get("provider"),
                "occurred_at": props.get("occurred_at"),
            }
        )
        if len(pending) >= _BATCH_LIMIT:
            break

    if not pending:
        _log.info("classify_signals: no unclassified raw signals")
        return {"ok": True, "enqueued": False, "pending": 0}

    agent_ctx = {
        "payload": {
            "signals": pending,
            "recipe_domains": _recipe_domains(),
        },
        # Echo the source mission so the continuation can attribute the
        # classified rows correctly.
        "growth_classify": {"mission_id": mission_id},
    }

    child_id = await general_beckman.enqueue(
        {
            "title": "classify growth signals",
            "description": (
                f"Classify {len(pending)} raw growth signal(s) into "
                f"label + recipe domain."
            ),
            "agent_type": "signal_classifier",
            "kind": "overhead",
            "priority": 4,
            "mission_id": mission_id,
            "context": agent_ctx,
        },
        parent_id=task.get("id"),
        on_complete="growth.classify_signals_complete",
    )

    _log.info(
        "classify_signals: enqueued signal_classifier",
        child_id=child_id,
        pending=len(pending),
    )
    return {
        "ok": True,
        "enqueued": True,
        "child_task_id": child_id,
        "pending": len(pending),
    }


async def _on_classifier_complete(task_id: int, result: dict, state: dict | None = None) -> None:
    """Continuation: persist signal_classifier verdicts as classified_signal rows.

    Called by Beckman's ``dispatch_on_complete`` when the classifier agent
    task reaches a terminal state. Errors are swallowed by the dispatcher.
    """
    from src.infra.db import get_growth_events, get_task
    from general_beckman import record_growth_event

    res = (result or {}).get("result")
    # The agent may return result as a JSON string or a dict.
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except Exception:
            res = {}
    classifications = []
    if isinstance(res, dict):
        classifications = res.get("classifications") or []

    if not classifications:
        _log.warning(
            "classify_signals_complete: no classifications in result",
            task_id=task_id,
        )
        return

    # Recover mission_id + the source raw signals so we can attach the
    # raw_signal_id / content excerpt to each classified row.
    mission_id = None
    raw_by_eid: dict[str, dict] = {}
    try:
        child = await get_task(task_id)
        ctx_raw = (child or {}).get("context") or "{}"
        ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else dict(ctx_raw)
        mission_id = (ctx.get("growth_classify") or {}).get("mission_id")
        for s in (ctx.get("payload") or {}).get("signals") or []:
            eid = s.get("external_id")
            if eid is not None:
                raw_by_eid[str(eid)] = s
    except Exception as exc:  # noqa: BLE001
        _log.debug("classify_signals_complete: ctx recovery failed", error=str(exc))

    # Fallback: pull raw signals from the DB if the ctx didn't carry them.
    if not raw_by_eid:
        try:
            for row in await get_growth_events(kind="raw_signal"):
                props = row.get("properties") or {}
                eid = props.get("external_id")
                if eid is not None:
                    raw_by_eid[str(eid)] = {
                        "raw_signal_id": row.get("id"),
                        "external_id": eid,
                        "signal_type": props.get("signal_type"),
                        "content": props.get("content"),
                    }
        except Exception:
            pass

    written = 0
    for c in classifications:
        if not isinstance(c, dict):
            continue
        eid = c.get("external_id")
        if eid is None:
            continue
        src = raw_by_eid.get(str(eid), {})
        content = src.get("content") or ""
        props = {
            "raw_signal_id": src.get("raw_signal_id"),
            "external_id": eid,
            "signal_type": src.get("signal_type"),
            "label": str(c.get("label") or "spam"),
            "domain": str(c.get("domain") or "general"),
            "confidence": float(c.get("confidence") or 0.0),
            "content_excerpt": str(content)[:280],
        }
        try:
            await record_growth_event(
                mission_id, "classified_signal", props
            )
            written += 1
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "classify_signals_complete: insert failed",
                external_id=eid,
                error=str(exc),
            )

    _log.info(
        "classify_signals_complete: persisted classified signals",
        task_id=task_id,
        written=written,
    )

    # Chain into scoring unconditionally. Z9 sweep 2026-05-18 P3: the
    # prior `if written:` gate silently skipped the weekly backlog
    # recompute when the signal_classifier agent failed or classified
    # nothing — no retry, no DLQ, no visibility. score_backlog is
    # idempotent and cheap; running it on an empty input is a no-op
    # rather than a regression. Always enqueue so a classifier outage
    # never silently halts the backlog→roadmap chain.
    try:
        await general_beckman.enqueue(
            {
                "title": "score growth backlog",
                "description": (
                    f"Score {written} newly-classified signal(s) into "
                    f"backlog candidates."
                ),
                "agent_type": "mechanical",
                "kind": "overhead",
                "priority": 4,
                "mission_id": mission_id,
                "context": {"payload": {"action": "score_backlog"}},
            },
            parent_id=task_id,
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "classify_signals_complete: score_backlog enqueue failed",
            error=str(exc),
        )


def register_continuations() -> None:
    """Register the classify-signals continuation (idempotent). Called at import
    so the handler survives a restart for the reconcile pass."""
    try:
        from general_beckman.continuations import register
        register("growth.classify_signals_complete", _on_classifier_complete)
    except Exception as exc:  # noqa: BLE001
        _log.debug("classify_signals continuation registration deferred", error=str(exc))


register_continuations()
