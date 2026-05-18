"""Z9 T4B — inject_north_star mechanical executor.

THE central Z9 wiring gap: the ``success_metrics`` artifact (north-star
metric + AARRR metrics) is created at i2p step ``2.9
success_metrics_definition`` and then never read again. This verb closes
the READ side — it retrieves the artifact and merges
``north_star_metric`` + ``aarrr_metrics`` into ``mission.context['north_star']``
so Phase 8+ feature-scoring steps (recipe choice, mission ranking,
implementation backlog) see the product's north-star.

Mirrors the Z2 T4C ``inject_lessons`` pattern exactly:
  - read an artifact / store, merge into ``missions.context`` JSON bucket,
  - idempotent (only write when the bucket actually changes),
  - graceful (no artifact / no mission → ok with ``injected=False``).

Pure mechanical — NO LLM.

Public API
----------
inject_north_star(mission_id) -> dict
    ``{"ok": True, "injected": bool, "mission_id": int,
       "north_star": <name|None>, "aarrr_count": int}``
"""
from __future__ import annotations

import json

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.inject_north_star")


async def _load_success_metrics(mission_id: int) -> dict:
    """Best-effort retrieve the ``success_metrics`` artifact (i2p step 2.9).

    Returns ``{}`` when unreachable — mirrors analytics_digest's loader.
    """
    try:
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
        logger.debug(f"inject_north_star: success_metrics retrieve failed: {exc}")
    return {}


async def inject_north_star(mission_id: int) -> dict:
    """Read the success_metrics artifact and inject the north-star context.

    Parameters
    ----------
    mission_id:
        Target mission row in the ``missions`` table.

    Returns
    -------
    dict
        ``{"ok": True, "injected": bool, "mission_id": mission_id,
           "north_star": <metric name|None>, "aarrr_count": int}``
    """
    # ── 1. Retrieve the success_metrics artifact ─────────────────────────
    success_metrics = await _load_success_metrics(mission_id)
    north_star_metric = (success_metrics or {}).get("north_star_metric") or {}
    aarrr_metrics = (success_metrics or {}).get("aarrr_metrics") or []

    if not north_star_metric and not aarrr_metrics:
        # Artifact missing or 2.9 not run — degrade gracefully.
        logger.debug(
            f"inject_north_star: no success_metrics for mission {mission_id}"
        )
        return {
            "ok": True,
            "injected": False,
            "mission_id": mission_id,
            "north_star": None,
            "aarrr_count": 0,
        }

    # ── 2. Read existing mission context ─────────────────────────────────
    try:
        from src.infra.db import get_db

        db = await get_db()
        async with db.execute(
            "SELECT context FROM missions WHERE id = ?",
            (mission_id,),
        ) as cur:
            row = await cur.fetchone()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"inject_north_star: could not read mission context: {exc}"
        )
        return {
            "ok": True,
            "injected": False,
            "mission_id": mission_id,
            "north_star": None,
            "aarrr_count": 0,
        }

    if row is None:
        logger.warning(f"inject_north_star: mission {mission_id} not found")
        return {
            "ok": True,
            "injected": False,
            "mission_id": mission_id,
            "north_star": None,
            "aarrr_count": 0,
        }

    raw_ctx = row[0] or "{}"
    if isinstance(raw_ctx, str):
        try:
            ctx = json.loads(raw_ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    elif isinstance(raw_ctx, dict):
        ctx = raw_ctx
    else:
        ctx = {}

    # ── 3. Build the north_star bucket — idempotent write ────────────────
    north_star_block = {
        "north_star_metric": {
            "name": north_star_metric.get("name", ""),
            "justification": north_star_metric.get("justification", ""),
        },
        "aarrr_metrics": [
            {
                "name": m.get("name", ""),
                "formula": m.get("formula", ""),
                "data_source": m.get("data_source", ""),
                "target_value": m.get("target_value"),
                "measurement_frequency": m.get("measurement_frequency", ""),
            }
            for m in aarrr_metrics
            if isinstance(m, dict)
        ],
    }

    existing = ctx.get("north_star")
    if existing == north_star_block:
        # Already up-to-date — skip the write.
        return {
            "ok": True,
            "injected": True,
            "mission_id": mission_id,
            "north_star": north_star_block["north_star_metric"]["name"] or None,
            "aarrr_count": len(north_star_block["aarrr_metrics"]),
            "unchanged": True,
        }

    ctx["north_star"] = north_star_block
    new_ctx = json.dumps(ctx, ensure_ascii=False)

    try:
        from src.infra.db import get_db as _get_db

        _db = await _get_db()
        await _db.execute(
            "UPDATE missions SET context = ? WHERE id = ?",
            (new_ctx, mission_id),
        )
        await _db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"inject_north_star: could not write mission context: {exc}"
        )
        return {
            "ok": True,
            "injected": False,
            "mission_id": mission_id,
            "north_star": None,
            "aarrr_count": 0,
        }

    ns_name = north_star_block["north_star_metric"]["name"] or "(unnamed)"
    logger.info(
        f"inject_north_star: injected north-star {ns_name!r} "
        f"+ {len(north_star_block['aarrr_metrics'])} AARRR metrics "
        f"into mission {mission_id}"
    )
    return {
        "ok": True,
        "injected": True,
        "mission_id": mission_id,
        "north_star": north_star_block["north_star_metric"]["name"] or None,
        "aarrr_count": len(north_star_block["aarrr_metrics"]),
    }


async def run(task: dict) -> dict:
    """Dispatcher entry point — mirrors the executors/ run(task) convention."""
    payload = task.get("payload") or {}
    mission_id = task.get("mission_id") or payload.get("mission_id")
    if mission_id is None:
        return {"ok": False, "error": "inject_north_star requires mission_id"}
    return await inject_north_star(int(mission_id))
