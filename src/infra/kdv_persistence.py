"""KDV state persistence: bridges KuledenDonenVar ↔ kutai.db.

KDV itself stays free of sqlite imports; this module reads its public
``snapshot_state()`` / ``restore_state()`` API and shuttles per-row
records to the ``kdv_state`` table.

Persistence model:
- One row per (scope, scope_key) where scope ∈ {"model","provider","breaker",
  "meta"}; scope_key is the litellm_name, provider name, or "global".
- snapshot_json is the JSON-encoded dict from the corresponding snapshot_state().
- last_persisted is unix epoch; loader treats rows older than ``stale_hours``
  as stale and skips them. Default 24h — long enough to survive
  overnight reboots, short enough that header reset_at clocks haven't
  drifted into nonsense.

The loader also restores ``enabled_at`` and ``call_count`` (kept under
the "meta" scope as a single dict-row).
"""
from __future__ import annotations

import json
import sqlite3
import time

import aiosqlite

from src.infra.logging_config import get_logger

logger = get_logger("infra.kdv_persistence")

_STALE_HOURS_DEFAULT = 24.0


async def save(kdv, db_path: str) -> None:
    """Snapshot KDV state and write it to kutai.db. Fire-and-forget — never
    raises to the caller. Saves are upserts keyed on (scope, scope_key)."""
    try:
        snap = kdv.snapshot_state()
    except Exception as e:  # noqa: BLE001
        logger.warning("kdv snapshot_state failed: %s", e)
        return

    now = time.time()
    rows: list[tuple[str, str, str, float]] = []
    for mid, model_snap in snap.get("models", {}).items():
        rows.append(("model", mid, json.dumps(model_snap), now))
    for prov, prov_snap in snap.get("providers", {}).items():
        rows.append(("provider", prov, json.dumps(prov_snap), now))
    for prov, cb_snap in snap.get("breakers", {}).items():
        rows.append(("breaker", prov, json.dumps(cb_snap), now))
    rows.append((
        "meta", "global",
        json.dumps({
            "enabled_at": snap.get("enabled_at", {}),
            "call_count": snap.get("call_count", {}),
            "attempt_count": snap.get("attempt_count", {}),
        }),
        now,
    ))

    try:
        from src.infra.db import connect_aux
        async with connect_aux(db_path, _label="kdv_save") as db:
            await db.executemany(
                "INSERT INTO kdv_state (scope, scope_key, snapshot_json, last_persisted) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(scope, scope_key) DO UPDATE SET "
                "snapshot_json=excluded.snapshot_json, "
                "last_persisted=excluded.last_persisted",
                rows,
            )
            await db.commit()
    except Exception as e:  # noqa: BLE001 — telemetry must never propagate
        logger.warning("kdv state save failed: %s", e)


async def load(kdv, db_path: str, stale_hours: float = _STALE_HOURS_DEFAULT) -> dict:
    """Read the kdv_state table and feed it into ``kdv.restore_state``.

    Returns a small report dict with counters: ``{"models": N, "providers": N,
    "breakers": N, "skipped_stale": N}``. Never raises — if the table is
    missing or DB is unavailable, returns zeros.
    """
    report = {"models": 0, "providers": 0, "breakers": 0, "meta": 0,
              "skipped_stale": 0}
    cutoff = time.time() - stale_hours * 3600.0

    snap_for_kdv: dict = {"models": {}, "providers": {}, "breakers": {}}
    enabled_at: dict = {}
    call_count: dict = {}
    attempt_count: dict = {}

    try:
        from src.infra.db import connect_aux
        async with connect_aux(db_path, _label="kdv_load") as db:
            async with db.execute(
                "SELECT scope, scope_key, snapshot_json, last_persisted "
                "FROM kdv_state"
            ) as cursor:
                async for scope, scope_key, payload, last_persisted in cursor:
                    if last_persisted < cutoff:
                        report["skipped_stale"] += 1
                        continue
                    try:
                        decoded = json.loads(payload)
                    except Exception:  # noqa: BLE001
                        report["skipped_stale"] += 1
                        continue
                    if scope == "model":
                        snap_for_kdv["models"][scope_key] = decoded
                        report["models"] += 1
                    elif scope == "provider":
                        snap_for_kdv["providers"][scope_key] = decoded
                        report["providers"] += 1
                    elif scope == "breaker":
                        snap_for_kdv["breakers"][scope_key] = decoded
                        report["breakers"] += 1
                    elif scope == "meta":
                        enabled_at = dict(decoded.get("enabled_at", {}))
                        call_count = dict(decoded.get("call_count", {}))
                        attempt_count = dict(decoded.get("attempt_count", {}))
                        report["meta"] += 1
    except Exception as e:  # noqa: BLE001
        logger.warning("kdv state load failed: %s", e)
        return report

    snap_for_kdv["enabled_at"] = enabled_at
    snap_for_kdv["call_count"] = call_count
    snap_for_kdv["attempt_count"] = attempt_count

    try:
        kdv.restore_state(snap_for_kdv)
    except Exception as e:  # noqa: BLE001
        logger.warning("kdv restore_state failed: %s", e)
        return report

    if report["models"] or report["providers"] or report["breakers"]:
        logger.info(
            "kdv state restored: models=%d providers=%d breakers=%d "
            "meta=%d stale_skipped=%d",
            report["models"], report["providers"], report["breakers"],
            report["meta"], report["skipped_stale"],
        )
    return report


def load_sync(kdv, db_path: str, stale_hours: float = _STALE_HOURS_DEFAULT) -> dict:
    """Synchronous variant for boot-time use (router.get_kdv()).

    Uses stdlib sqlite3 so it works whether or not an event loop is
    running. Reads only — never writes. Mirrors ``load()`` semantics.
    """
    report = {"models": 0, "providers": 0, "breakers": 0, "meta": 0,
              "skipped_stale": 0}
    cutoff = time.time() - stale_hours * 3600.0
    snap_for_kdv: dict = {"models": {}, "providers": {}, "breakers": {}}
    enabled_at: dict = {}
    call_count: dict = {}
    attempt_count: dict = {}

    try:
        from src.infra.db import connect_aux_sync
        conn = connect_aux_sync(db_path)
        try:
            cur = conn.execute(
                "SELECT scope, scope_key, snapshot_json, last_persisted "
                "FROM kdv_state"
            )
            for scope, scope_key, payload, last_persisted in cur.fetchall():
                if last_persisted < cutoff:
                    report["skipped_stale"] += 1
                    continue
                try:
                    decoded = json.loads(payload)
                except Exception:  # noqa: BLE001
                    report["skipped_stale"] += 1
                    continue
                if scope == "model":
                    snap_for_kdv["models"][scope_key] = decoded
                    report["models"] += 1
                elif scope == "provider":
                    snap_for_kdv["providers"][scope_key] = decoded
                    report["providers"] += 1
                elif scope == "breaker":
                    snap_for_kdv["breakers"][scope_key] = decoded
                    report["breakers"] += 1
                elif scope == "meta":
                    enabled_at = dict(decoded.get("enabled_at", {}))
                    call_count = dict(decoded.get("call_count", {}))
                    attempt_count = dict(decoded.get("attempt_count", {}))
                    report["meta"] += 1
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        logger.warning("kdv state load_sync failed: %s", e)
        return report

    snap_for_kdv["enabled_at"] = enabled_at
    snap_for_kdv["call_count"] = call_count
    snap_for_kdv["attempt_count"] = attempt_count

    try:
        kdv.restore_state(snap_for_kdv)
    except Exception as e:  # noqa: BLE001
        logger.warning("kdv restore_state failed: %s", e)
        return report

    if report["models"] or report["providers"] or report["breakers"]:
        logger.info(
            "kdv state restored (sync): models=%d providers=%d breakers=%d "
            "meta=%d stale_skipped=%d",
            report["models"], report["providers"], report["breakers"],
            report["meta"], report["skipped_stale"],
        )
    return report
