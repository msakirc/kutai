"""Forensic logger for admission-gate violations.

Captures cases where Beckman admitted a task to a model and a downstream
gate (KDV pre_call, dispatcher pool exhaustion, daily_exhausted at call
time) refused to run it. Per the user design 2026-05-03: the
Beckman→Hoca→KDV pipeline rejecting a task post-admission is a
"serious crime" — the pressure model failed to predict and the task is
now stuck. Don't tighten knobs reactively; collect evidence for offline
tuning instead.

Schema reference: src/infra/db.py :: admission_violations table.

Sites that record:
    1. caller.py — KDV pre_call refused after admission (rate, canary,
       circuit_breaker, daily_exhausted). site="kdv_pre_call_refusal"
    2. dispatcher.py — pick is None during retry recursion. Pool drained
       mid-task; selector returned no candidate. site="dispatcher_pool_empty"
    3. caller.py — daily_exhausted bubbled up at call time despite
       eligibility filter passing. site="daily_exhausted_at_call"

Fire-and-forget. Never raises to caller.
"""
from __future__ import annotations

import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("infra.admission_forensics")


async def record_admission_violation(
    *,
    site: str,
    phase: str = "main_work",
    task_id: int | None = None,
    call_category: str = "",
    agent_type: str = "",
    difficulty: int | None = None,
    model: str = "",
    provider: str = "",
    reason: str = "",
    wait_seconds: float = 0.0,
    scope: str = "",
    error_category: str = "",
    error_message: str = "",
    in_flight_n: int | None = None,
    queue_total: int | None = None,
    queue_hard: int | None = None,
    snapshot_summary: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist a forensic record of an admission-gate violation.

    Telemetry only. Never raises; failures log at WARNING and return.
    """
    try:
        from src.infra.db import get_db
        db = await get_db()
        await db.execute(
            "INSERT INTO admission_violations "
            "(site, phase, task_id, call_category, agent_type, difficulty, "
            " model, provider, reason, wait_seconds, scope, error_category, "
            " error_message, in_flight_n, queue_total, queue_hard, "
            " snapshot_summary, extra_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                site,
                phase,
                task_id,
                call_category,
                agent_type or None,
                difficulty,
                model,
                provider,
                reason,
                wait_seconds,
                scope,
                error_category,
                error_message[:500] if error_message else "",
                in_flight_n,
                queue_total,
                queue_hard,
                snapshot_summary[:1000] if snapshot_summary else "",
                json.dumps(extra) if extra else None,
            ),
        )
    except Exception as e:  # noqa: BLE001 — telemetry must never propagate
        logger.warning("admission_violations write failed: %s", e)
