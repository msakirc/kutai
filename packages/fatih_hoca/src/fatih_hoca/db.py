"""Async model-registry query helpers, owned by fatih_hoca. Built on dabidabi's
connection primitives. Schema A is canonical for model_stats."""
import json

from dabidabi import get_db
from src.infra.logging_config import get_logger

logger = get_logger("fatih_hoca.db")

# founder-decided: +0.05 per confirmed verdict
REINFORCE_NUDGE: float = 0.05


async def record_model_call(
    model: str,
    agent_type: str,
    success: bool,
    cost: float = 0.0,
    latency: float = 0.0,
    grade: float | None = None,
) -> None:
    """Record a model call for performance tracking.

    Persists per-(model, agent_type) aggregates to the ``model_stats`` DB
    table. Phase C.5 (2026-05-05): in-memory Prometheus counters are
    emitted by ``hallederiz_kadir.caller`` directly — this function used
    to ALSO emit them which inflated counters ~2.5× because every ReAct
    iter passed through both paths. Audit:
    ``docs/handoff/2026-05-04-record-model-call-audit.md``.
    """
    db = await get_db()

    # Upsert: try to get existing row
    cursor = await db.execute(
        "SELECT * FROM model_stats WHERE model = ? AND agent_type = ?",
        (model, agent_type)
    )
    existing = await cursor.fetchone()

    if existing:
        row = dict(existing)
        total = row["total_calls"] + 1
        successes = row["total_successes"] + (1 if success else 0)
        cost_sum = row["total_cost_sum"] + cost
        latency_sum = row["total_latency_sum"] + latency
        grade_sum = row["total_grade_sum"] + (grade if grade else 0)

        avg_cost = cost_sum / total if total > 0 else 0
        avg_latency = latency_sum / total if total > 0 else 0
        graded_calls = row["total_calls"]  # approximate
        if grade is not None:
            graded_calls += 1
        avg_grade = grade_sum / graded_calls if graded_calls > 0 else 0
        success_rate = successes / total if total > 0 else 0

        await db.execute(
            """UPDATE model_stats SET
                   total_calls = ?, total_successes = ?,
                   total_cost_sum = ?, total_latency_sum = ?,
                   total_grade_sum = ?,
                   avg_cost = ?, avg_latency = ?,
                   avg_grade = ?, success_rate = ?,
                   updated_at = datetime('now')
               WHERE model = ? AND agent_type = ?""",
            (total, successes, cost_sum, latency_sum, grade_sum,
             avg_cost, avg_latency, avg_grade, success_rate,
             model, agent_type)
        )
    else:
        avg_grade = grade if grade else 0.0
        await db.execute(
            """INSERT INTO model_stats
               (model, agent_type, total_calls, total_successes,
                total_cost_sum, total_latency_sum, total_grade_sum,
                avg_cost, avg_latency, avg_grade, success_rate)
               VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (model, agent_type, 1 if success else 0,
             cost, latency, grade if grade else 0,
             cost, latency, avg_grade,
             1.0 if success else 0.0)
        )
    await db.commit()


async def get_model_stats(
    model: str | None = None,
    agent_type: str | None = None,
) -> list[dict]:
    """Query model performance stats.

    Filter by model and/or agent_type.  Returns all if no filters.
    """
    db = await get_db()
    query = "SELECT * FROM model_stats WHERE 1=1"
    params: list = []
    if model:
        query += " AND model = ?"
        params.append(model)
    if agent_type:
        query += " AND agent_type = ?"
        params.append(agent_type)
    query += " ORDER BY avg_grade DESC, success_rate DESC"
    cursor = await db.execute(query, params)
    return [dict(row) for row in await cursor.fetchall()]


async def get_model_performance_ranking(agent_type: str) -> list[dict]:
    """Get models ranked by performance for a specific agent type.

    Returns models sorted by: success_rate * avg_grade (composite score).
    Only includes models with >= 3 calls for statistical significance.
    """
    db = await get_db()
    cursor = await db.execute(
        """SELECT model, avg_grade, avg_cost, avg_latency,
                  success_rate, total_calls
           FROM model_stats
           WHERE agent_type = ? AND total_calls >= 3
           ORDER BY (success_rate * avg_grade) DESC""",
        (agent_type,)
    )
    return [dict(row) for row in await cursor.fetchall()]


async def record_action_event(
    verb: str,
    reversibility: str,
    mission_id: int | None,
    task_id: int | None,
    payload: dict | None,
    status: str,
) -> int:
    """Append a per-action audit row to ``registry_events`` (scope='action').

    The existing ``event`` column carries the verb (mirrors ``verb`` for
    backward-compatible target/event scans); ``payload_json`` carries the
    JSON-serialized payload + status. Returns the new row id.
    """
    db = await get_db()
    try:
        payload_json = json.dumps(
            {"payload": payload or {}, "status": status},
            default=str,
        )
    except Exception:
        payload_json = json.dumps(
            {"payload": str(payload), "status": status}
        )
    cur = await db.execute(
        "INSERT INTO registry_events "
        "(scope, target, event, payload_json, "
        " mission_id, task_id, verb, reversibility) "
        "VALUES ('action', ?, ?, ?, ?, ?, ?, ?)",
        (
            verb,
            verb,
            payload_json,
            mission_id,
            task_id,
            verb,
            reversibility,
        ),
    )
    await db.commit()
    return cur.lastrowid or 0


async def record_reinforce_nudge(
    model: str,
    *,
    task_name: str = "hypothesis_verdict",
    amount: float = REINFORCE_NUDGE,
    provider: str = "local",
    hypothesis_id: int | None = None,
) -> None:
    """Write a confirmed-verdict reinforce nudge for ``model``.

    Fire-and-forget telemetry — never raises into the caller. The row is a
    ``model_pick_log`` entry tagged ``call_category='reinforce'`` with the
    bonus stored in the ``reinforce`` column. Fatih Hoca's grading layer
    sums these with a 50%-per-30-day decay so the influence fades.
    """
    if not model:
        return
    try:
        db = await get_db()
        snapshot = f"hypothesis_id={hypothesis_id}" if hypothesis_id else ""
        await db.execute(
            "INSERT INTO model_pick_log "
            "(task_name, picked_model, picked_score, call_category, "
            " candidates_json, snapshot_summary, success, provider, "
            " outcome, reinforce) "
            "VALUES (?, ?, ?, 'reinforce', '[]', ?, 1, ?, 'reinforce', ?)",
            (
                task_name,
                model,
                0.0,
                snapshot,
                provider,
                float(amount),
            ),
        )
        await db.commit()
        logger.info(
            "reinforce nudge recorded model=%s amount=%.3f hyp=%s",
            model, amount, hypothesis_id,
        )
    except Exception as e:  # noqa: BLE001 — telemetry must never propagate
        logger.warning("record_reinforce_nudge failed: %s", e)


# ── Cross-domain read APIs (fatih_hoca owns model-registry reads) ──────────


async def get_recent_picks(limit: int = 100, since_days: int | None = None):
    db = await get_db()
    if since_days is not None:
        cur = await db.execute(
            "SELECT * FROM model_pick_log WHERE timestamp > datetime('now', ?) "
            "ORDER BY timestamp DESC LIMIT ?", (f"-{since_days} days", limit))
    else:
        cur = await db.execute(
            "SELECT * FROM model_pick_log ORDER BY timestamp DESC LIMIT ?", (limit,))
    return [dict(r) for r in await cur.fetchall()]


async def get_model_stats_rows():
    db = await get_db()
    cur = await db.execute(
        "SELECT model, total_calls, success_rate, avg_cost FROM model_stats")
    return [dict(r) for r in await cur.fetchall()]


async def get_pick_summary(since_days: int = 7, group_by_task: bool = False):
    """Aggregate model_pick_log picks over the last ``since_days``.

    Returns rows with ``picked_model``, ``picks``, ``avg_score`` (+
    ``task_name`` when ``group_by_task``). ``avg_score`` is rounded to 2dp.
    """
    db = await get_db()
    if group_by_task:
        cur = await db.execute(
            "SELECT task_name, picked_model, COUNT(*) AS picks, "
            "ROUND(AVG(picked_score), 2) AS avg_score FROM model_pick_log "
            "WHERE timestamp > datetime('now', ?) "
            "GROUP BY task_name, picked_model ORDER BY task_name, picks DESC",
            (f"-{since_days} days",))
    else:
        cur = await db.execute(
            "SELECT picked_model, COUNT(*) AS picks, "
            "ROUND(AVG(picked_score), 2) AS avg_score "
            "FROM model_pick_log WHERE timestamp > datetime('now', ?) "
            "GROUP BY picked_model ORDER BY picks DESC",
            (f"-{since_days} days",))
    return [dict(r) for r in await cur.fetchall()]


async def insert_pick_log_row(
    *,
    task_name: str,
    agent_type: str | None,
    difficulty: int | None,
    picked_model: str,
    picked_score: float,
    category: str,
    candidates_json: str,
    snapshot_summary: str,
    success: bool,
    error_category: str,
    provider: str,
    outcome: str,
    task_id: int | None,
) -> None:
    """Owns the raw ``INSERT INTO model_pick_log`` SQL.

    ``src.infra.pick_log.write_pick_log_row`` delegates here so the SQL
    string lives with the rest of the model-registry reads/writes. Caller
    (pick_log) keeps the fire-and-forget try/except + logging. Does NOT
    commit — preserves the original write_pick_log_row behavior (no explicit
    commit; relies on the singleton connection's later commit/WAL flush).
    """
    db = await get_db()
    await db.execute(
        "INSERT INTO model_pick_log "
        "(task_name, agent_type, difficulty, picked_model, picked_score, "
        " call_category, candidates_json, snapshot_summary, success, "
        " error_category, provider, outcome, task_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            task_name,
            agent_type,
            difficulty,
            picked_model,
            picked_score,
            category,
            candidates_json,
            snapshot_summary,
            1 if success else 0,
            error_category,
            provider,
            outcome,
            task_id,
        ),
    )
