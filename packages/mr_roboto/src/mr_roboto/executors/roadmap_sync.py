"""Z9 Growth T5C — ``roadmap_sync`` mechanical executor (north-star sync).

Pure deterministic check — NO LLM. The roadmap half of T5C: a weekly
sanity-check that the product's declared **north-star metric** is still
current. If the north-star looks **stale** (not defined, or defined but
flat / un-tracked in recent reality) the executor writes a
``northstar_review`` growth_events row prompting the founder to refine it.

Routed via ``mr_roboto.run`` when ``payload["action"] == "roadmap_sync"``.
Fired weekly by the global ``roadmap_northstar_sync`` internal cadence.
Implements the i2p step ``15.14 roadmap_update`` north-star refresh loop.

Staleness detection
-------------------
The check retrieves the ``success_metrics`` artifact (i2p step 2.9, shape
``{aarrr_metrics:[...], north_star_metric:{name,justification}}``) and
compares the declared north-star against recent reality:

1. **undefined**     — no ``north_star_metric.name`` → review (it was never set).
2. **untracked**     — no ``growth_events`` in the last ``_FLAT_WINDOW_DAYS``
   reference the north-star metric name → review (we declare it but never
   measure it).
3. **flat**          — the metric IS referenced but its values have not
   moved across the window (min == max over >= ``_MIN_FLAT_POINTS``
   readings) → review (north-star plateaued; the roadmap should refocus).

A clean north-star (defined, tracked, moving) writes NO row — the check is
quiet by design. ``northstar_review`` is a founder prompt, never a mission.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src.infra.logging_config import get_logger

_log = get_logger("mr_roboto.roadmap_sync")

# How far back "recent reality" looks when judging tracked / flat.
_FLAT_WINDOW_DAYS = 30

# Minimum distinct metric readings needed before a flat verdict is fair —
# one or two points can't tell plateau from "just launched".
_MIN_FLAT_POINTS = 3


def _since_db(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    return (now - timedelta(days=_FLAT_WINDOW_DAYS)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


async def _load_success_metrics(mission_id: int | None) -> dict[str, Any]:
    """Best-effort retrieve the ``success_metrics`` artifact (i2p step 2.9)."""
    if not mission_id:
        return {}
    try:
        import json

        from src.workflows.engine.hooks import get_artifact_store

        raw = await get_artifact_store().retrieve(
            int(mission_id), "success_metrics"
        )
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
    except Exception as exc:  # noqa: BLE001
        _log.debug("roadmap_sync: success_metrics retrieve failed", error=str(exc))
    return {}


def _metric_readings(rows: list[dict], metric_name: str) -> list[float]:
    """Pull numeric readings of ``metric_name`` from growth_events rows.

    A row references the metric when its properties carry the metric name as
    a key, or a ``metric``/``metric_name`` field equal to it. The numeric
    value is read from ``value`` / ``metric_value`` / the named key.
    """
    name = (metric_name or "").strip().lower()
    if not name:
        return []
    readings: list[float] = []
    for row in rows or []:
        props = row.get("properties") or {}
        if not isinstance(props, dict):
            continue
        # Lower-cased key map for tolerant matching.
        lk = {str(k).strip().lower(): v for k, v in props.items()}
        named = (
            str(lk.get("metric") or lk.get("metric_name") or "")
            .strip()
            .lower()
        )
        val = None
        if named == name:
            val = lk.get("value", lk.get("metric_value"))
        elif name in lk:
            val = lk.get(name)
        if val is None:
            continue
        try:
            readings.append(float(val))
        except (TypeError, ValueError):
            continue
    return readings


def assess_north_star(
    success_metrics: dict, growth_rows: list[dict]
) -> dict[str, Any]:
    """Return a staleness verdict for the declared north-star.

    Pure — exposed for deterministic tests. Verdict ``status`` is one of
    ``current`` | ``undefined`` | ``untracked`` | ``flat``; only the last
    three warrant a ``northstar_review`` row.
    """
    ns = (success_metrics or {}).get("north_star_metric") or {}
    name = str(ns.get("name") or "").strip()

    if not name:
        return {
            "stale": True,
            "status": "undefined",
            "metric": None,
            "reason": (
                "No north-star metric is declared in success_metrics. "
                "A product without a single guiding metric drifts — "
                "define one."
            ),
        }

    readings = _metric_readings(growth_rows, name)

    if not readings:
        return {
            "stale": True,
            "status": "untracked",
            "metric": name,
            "reason": (
                f"North-star '{name}' is declared but no growth_events in "
                f"the last {_FLAT_WINDOW_DAYS}d measure it. Wire "
                f"instrumentation or pick a metric you actually track."
            ),
        }

    if len(readings) >= _MIN_FLAT_POINTS and min(readings) == max(readings):
        return {
            "stale": True,
            "status": "flat",
            "metric": name,
            "readings": len(readings),
            "value": readings[0],
            "reason": (
                f"North-star '{name}' has not moved across "
                f"{len(readings)} readings in {_FLAT_WINDOW_DAYS}d "
                f"(flat at {readings[0]}). The roadmap should refocus on "
                f"a metric that can still grow."
            ),
        }

    return {
        "stale": False,
        "status": "current",
        "metric": name,
        "readings": len(readings),
    }


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Check the north-star; write a ``northstar_review`` row when stale.

    Always returns a dict — never raises into the dispatcher.
    """
    from src.infra.db import get_growth_events, insert_growth_event

    payload = task.get("payload") or {}
    mission_id = task.get("mission_id") or payload.get("mission_id")
    since = _since_db()

    success_metrics = await _load_success_metrics(mission_id)
    try:
        growth_rows = await get_growth_events(
            mission_id=mission_id, since=since
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("roadmap_sync: growth_events query failed", error=str(exc))
        growth_rows = []

    verdict = assess_north_star(success_metrics, growth_rows or [])

    if not verdict.get("stale"):
        _log.info(
            "roadmap_sync: north-star current",
            mission_id=mission_id,
            metric=verdict.get("metric"),
        )
        return {
            "ok": True,
            "stale": False,
            "status": verdict.get("status"),
            "metric": verdict.get("metric"),
        }

    # Stale → write ONE northstar_review row prompting founder refinement.
    # Idempotent guard: don't pile up un-consumed reviews for the same
    # status — supersede prior open reviews first.
    try:
        import json as _json

        from src.infra.db import get_db

        prior = await get_growth_events(
            mission_id=mission_id, kind="northstar_review"
        )
        db = await get_db()
        for p in prior or []:
            props = p.get("properties") or {}
            if props.get("consumed") or props.get("superseded"):
                continue
            props["superseded"] = True
            await db.execute(
                "UPDATE growth_events SET properties_json = ? WHERE id = ?",
                (_json.dumps(props), p.get("id")),
            )
        await db.commit()
    except Exception as exc:  # noqa: BLE001
        _log.debug("roadmap_sync: supersede prior reviews failed", error=str(exc))

    review = {
        "status": verdict.get("status"),
        "metric": verdict.get("metric"),
        "reason": verdict.get("reason"),
        "readings": verdict.get("readings"),
        "value": verdict.get("value"),
        "window_days": _FLAT_WINDOW_DAYS,
        "consumed": False,
    }
    review_id = await insert_growth_event(
        mission_id, "northstar_review", review
    )

    _log.info(
        "roadmap_sync: northstar_review written",
        mission_id=mission_id,
        status=verdict.get("status"),
        review_id=review_id,
    )
    return {
        "ok": True,
        "stale": True,
        "status": verdict.get("status"),
        "metric": verdict.get("metric"),
        "review_id": review_id,
        "reason": verdict.get("reason"),
    }


__all__ = ["run", "assess_north_star"]
