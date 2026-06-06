# pick_recorder.py
"""Fire-and-forget model_pick_log writer.

Evicted from ``LLMDispatcher._record_pick`` (Modularization Finish Plan
Phase 4) — pick telemetry is not part of the dispatcher's load→call loop.
The dispatcher keeps a thin ``_record_pick`` delegator that calls
``record_pick`` here; this module owns the actual write.

Every model pick (success or failure) lands in ``model_pick_log`` for
offline weight tuning. Errors are swallowed — telemetry must never break
dispatch.
"""

from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("telemetry.pick_recorder")


async def record_pick(
    *,
    pick: Any,
    task: str,
    category: Any,
    success: bool,
    error_category: str = "",
    agent_type: str = "",
    difficulty: int | None = None,
) -> None:
    """Write one model_pick_log row. Never propagates errors.

    ``category`` may be a ``CallCategory`` enum or a raw string — both are
    accepted (``.value`` is read defensively).
    """
    try:
        import os
        from src.infra import pick_log as _pick_log_mod

        db_path = os.getenv("DB_PATH") or "kutai.db"
        model = getattr(pick, "model", None)
        picked_model = getattr(model, "name", "") if model is not None else ""
        # Read score from Pick.score (populated by selector). The legacy
        # `composite` attribute never existed on Pick — every row was getting
        # picked_score=0.0 silently. Now persists ScoredModel.score from the
        # post-utilization rank step.
        picked_score = float(getattr(pick, "score", 0.0) or 0.0)
        # Top-5 candidate summary from the same select() invocation. Persists
        # into model_pick_log.snapshot_summary so offline analysis can see
        # runner-up scores alongside the winner — diagnoses "did we have a
        # clear winner or a near-tie?"
        snapshot_summary = str(getattr(pick, "top_summary", "") or "")
        cat_value = getattr(category, "value", None) or str(category)
        task_name = task or cat_value

        # Resolve the active task id from the heartbeat ContextVar. Defensive:
        # overhead calls and tests that don't set the ContextVar get None.
        _active_task_id: int | None = None
        try:
            from src.core.heartbeat import current_task_id as _ctid
            _active_task_id = _ctid.get()
        except Exception:
            pass

        await _pick_log_mod.write_pick_log_row(
            db_path=db_path,
            task_name=task_name,
            picked_model=picked_model,
            picked_score=picked_score,
            category=cat_value,
            success=success,
            error_category=error_category,
            snapshot_summary=snapshot_summary,
            provider=("local" if getattr(model, "is_local", False) else (getattr(model, "provider", "local") or "local")),
            agent_type=agent_type,
            difficulty=difficulty,
            task_id=_active_task_id,
        )
    except Exception as e:  # noqa: BLE001 — telemetry must never break dispatch
        logger.debug("pick_log record failed: %s", e)
