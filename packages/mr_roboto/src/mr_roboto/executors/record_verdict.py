"""Z9 Growth T4D — record_verdict mechanical executor.

Closes one hypothesis: pulls the actual metric value (PostHog via
``vendor_call``, mock-mode safe offline), computes a verdict by Bayesian
posterior, persists it, and feeds the result back into the growth loop.

Routed via ``mr_roboto.run`` when ``payload["action"] == "record_verdict"``.
The verdict task itself is enqueued by the T4C ``verdict_window_sweep``
cron once a hypothesis's measurement window closes.

Architecture contract
---------------------
**Mechanical, no LLM.** The verdict is deterministic math — a
normal-approximation posterior on the measured metric lift
(``src/growth/verdict_stats.py``). Nothing here calls ``LLMDispatcher``.

What it does
------------
1. Load the hypothesis row (``predicted_json = {metric, direction,
   magnitude, baseline?}``).
2. Pull the actual metric value via PostHog ``get_insight`` (mock mode
   returns a deterministic series when ``KUTAI_ENV != prod``).
3. ``compute_verdict`` → confirmed | refuted | inconclusive.
4. ``record_hypothesis_verdict`` — stamps ``measured_at``; on ``refuted``
   the DB layer sets ``suppressed_until = now + 90d``.
5. Mirror ``refuted`` + ``inconclusive`` verdicts into ``mission_lessons``
   (``source_kind='hypothesis_verdict'``). ``confirmed`` needs no lesson.
6. Write a ``growth_events`` row ``kind='verdict'``.
7. On ``confirmed`` — fire the reinforce nudge (T4E).
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.record_verdict")


# ── metric pull (PostHog get_insight, mock-mode safe) ──────────────────────


async def _posthog_metric(task: dict, metric: str) -> dict:
    """Pull a metric series via the PostHog ``get_insight`` vendor_call.

    Returns ``{ok, series}`` — ``series`` is the numeric data list. Mock
    mode (``KUTAI_ENV != prod``) yields a deterministic fake series from
    ``configs/posthog.json`` with no network hop.
    """
    from mr_roboto.executors.vendor_call import run as vendor_call_run

    sub = {
        "mission_id": task.get("mission_id"),
        "id": task.get("id"),
        "context": {
            "post_hook": {
                "service": "posthog",
                "action": "get_insight",
                "params": {"project_id": "default", "insight_id": 1,
                           "metric": metric},
            }
        },
    }
    try:
        env = await vendor_call_run(sub)
    except Exception as exc:  # noqa: BLE001
        logger.warning("posthog get_insight raised", metric=metric, error=str(exc))
        return {"ok": False, "series": []}

    if not (isinstance(env, dict) and env.get("ok")):
        return {"ok": False, "series": []}
    result = env.get("result") or {}
    # get_insight mock shape: {"result": [{"label": "metric", "data": [...]}]}
    series: list = []
    inner = result.get("result") if isinstance(result, dict) else None
    if isinstance(inner, list) and inner:
        first = inner[0]
        if isinstance(first, dict) and isinstance(first.get("data"), list):
            series = [float(x) for x in first["data"] if _is_num(x)]
    return {"ok": True, "series": series}


def _is_num(x: Any) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def _baseline_and_actual(predicted: dict, series: list[float]) -> tuple[float, float]:
    """Resolve (baseline, actual) for the verdict computation.

    Baseline preference: an explicit ``predicted['baseline']`` (recorded by
    T4A at spec time) wins; else the first point of the measured series.
    Actual = last point of the series. Empty series → (1.0, 1.0) so the
    verdict lands ``inconclusive`` rather than crashing.
    """
    if not series:
        return 1.0, 1.0
    actual = float(series[-1])
    baseline = predicted.get("baseline")
    if baseline is not None and _is_num(baseline):
        return float(baseline), actual
    return float(series[0]), actual


# ── mission_lessons mirror (refuted / inconclusive) ────────────────────────


async def _mirror_to_lessons(
    mission_id: int | None,
    feature: str,
    metric: str,
    verdict: str,
    p_held: float,
    observed_lift: float,
    predicted_lift: float,
) -> None:
    """Mirror a non-confirmed verdict into mission_lessons (idempotent).

    The lesson's ``pattern`` is keyed on feature+metric so re-runs dedup
    via ``upsert_mission_lesson``'s sha256 dedup_key. ``confirmed`` verdicts
    are NOT mirrored — only the cautionary ones (refuted/inconclusive).
    """
    if verdict == "confirmed":
        return
    try:
        from src.infra.mission_lessons import upsert_mission_lesson

        pattern = (
            f"Hypothesis for '{feature}' predicting a {predicted_lift:+.1%} "
            f"move in '{metric}' was {verdict} "
            f"(measured {observed_lift:+.1%}, P(held)={p_held:.2f})."
        )
        if verdict == "refuted":
            fix = (
                f"The '{feature}'/'{metric}' bet did not pay off — suppressed "
                f"for 90 days. Re-validate the assumption before predicting "
                f"this pair again."
            )
            severity = "warning"
        else:  # inconclusive
            fix = (
                f"Measurement of '{feature}'/'{metric}' was inconclusive — "
                f"tighten instrumentation or lengthen the window before "
                f"re-betting."
            )
            severity = "info"
        await upsert_mission_lesson(
            stack="growth",
            domain=metric or "general",
            pattern=pattern,
            fix=fix,
            severity=severity,
            source_kind="hypothesis_verdict",
            source_ref={
                "feature": feature,
                "metric": metric,
                "verdict": verdict,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("verdict lessons mirror failed", error=str(exc))


# ── reinforce loop (confirmed verdict → model_pick_log nudge) ──────────────


async def _reinforce_winning_model(
    mission_id: int | None, hypothesis_id: int | None, feature: str
) -> str | None:
    """On a confirmed verdict, nudge the model that built the winning feature.

    Resolves the model from the mission's most recent ``model_pick_log``
    row, then writes a gentle, decaying reinforce nudge (+0.05, 50%/30d
    decay applied at read time by fatih_hoca.grading.reinforce_bonus()).
    Returns the reinforced model name, or None when it cannot be resolved.
    """
    try:
        from src.infra.db import get_db, record_reinforce_nudge

        model: str | None = None
        provider = "local"
        db = await get_db()
        # Best proxy: the most recent main_work pick logged for this
        # mission's tasks. model_pick_log has no mission_id column, so we
        # join via tasks.title = model_pick_log.task_name.
        if mission_id is not None:
            cur = await db.execute(
                "SELECT mpl.picked_model, mpl.provider "
                "FROM model_pick_log mpl "
                "JOIN tasks t ON t.title = mpl.task_name "
                "WHERE t.mission_id = ? AND mpl.call_category != 'reinforce' "
                "ORDER BY mpl.timestamp DESC LIMIT 1",
                (mission_id,),
            )
            row = await cur.fetchone()
            if row:
                model = row[0]
                provider = row[1] or "local"
        if not model:
            # Fall back to the most recent non-reinforce pick overall.
            cur = await db.execute(
                "SELECT picked_model, provider FROM model_pick_log "
                "WHERE call_category != 'reinforce' "
                "ORDER BY timestamp DESC LIMIT 1"
            )
            row = await cur.fetchone()
            if row:
                model = row[0]
                provider = row[1] or "local"
        if not model:
            logger.info("reinforce: no model_pick_log row to reinforce")
            return None

        await record_reinforce_nudge(
            model,
            task_name=f"hypothesis_verdict:{feature}"[:120],
            provider=provider,
            hypothesis_id=hypothesis_id,
        )
        return model
    except Exception as exc:  # noqa: BLE001
        logger.warning("reinforce loop failed", error=str(exc))
        return None


# ── main entrypoint ────────────────────────────────────────────────────────


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Compute + record a verdict for one hypothesis. Never raises."""
    from src.growth.verdict_stats import compute_verdict
    from src.infra.db import (
        get_pending_hypotheses,
        insert_growth_event,
        record_hypothesis_verdict,
    )

    payload = task.get("payload") or {}
    hyp_id = payload.get("hypothesis_id") or task.get("hypothesis_id")
    if hyp_id is None:
        return {"ok": False, "reason": "missing_hypothesis_id"}
    try:
        hyp_id = int(hyp_id)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "bad_hypothesis_id"}

    # Load the hypothesis row (pending-only — an already-measured one is a
    # no-op, keeping the sweeper idempotent).
    pending = await get_pending_hypotheses() or []
    hyp = next((h for h in pending if int(h.get("id") or 0) == hyp_id), None)
    if hyp is None:
        logger.info("record_verdict: hypothesis not pending", hypothesis_id=hyp_id)
        return {"ok": True, "skipped": True, "reason": "not_pending",
                "hypothesis_id": hyp_id}

    mission_id = hyp.get("mission_id")
    feature = hyp.get("feature") or "feature"
    predicted = hyp.get("predicted_json") or {}
    if not isinstance(predicted, dict):
        predicted = {}
    metric = str(predicted.get("metric") or "metric")
    direction = str(predicted.get("direction") or "up")
    magnitude = predicted.get("magnitude") or 0.0

    # 1. Pull the actual metric value.
    pull = await _posthog_metric(task, metric)
    baseline, actual = _baseline_and_actual(predicted, pull.get("series") or [])

    # 2. Bayesian verdict.
    vr = compute_verdict(
        baseline=baseline,
        actual=actual,
        direction=direction,
        magnitude=magnitude,
    )

    actual_json = {
        "metric": metric,
        "baseline": baseline,
        "actual": actual,
        "observed_lift": vr.observed_lift,
        "predicted_lift": vr.predicted_lift,
        "p_held": vr.p_held,
        "p_opposite": vr.p_opposite,
        "p_value": round(1.0 - vr.p_held, 6),
        "model": vr.model,
        "posthog_ok": pull.get("ok", False),
    }

    # 3. Persist the verdict (DB layer stamps measured_at + suppression).
    await record_hypothesis_verdict(hyp_id, actual_json, vr.verdict)

    # 4. Mirror non-confirmed verdicts into mission_lessons.
    await _mirror_to_lessons(
        mission_id, feature, metric, vr.verdict,
        vr.p_held, vr.observed_lift, vr.predicted_lift,
    )

    # 5. Reinforce loop — confirmed verdict bumps the winning model.
    reinforced_model: str | None = None
    if vr.verdict == "confirmed":
        reinforced_model = await _reinforce_winning_model(
            mission_id, hyp_id, feature
        )

    # 6. growth_events row.
    try:
        await insert_growth_event(
            mission_id=mission_id,
            kind="verdict",
            properties={
                "hypothesis_id": hyp_id,
                "feature": feature,
                "metric": metric,
                "verdict": vr.verdict,
                "observed_lift": vr.observed_lift,
                "predicted_lift": vr.predicted_lift,
                "p_held": vr.p_held,
                "reinforced_model": reinforced_model,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("verdict growth_event write failed", error=str(exc))

    logger.info(
        "record_verdict complete",
        hypothesis_id=hyp_id,
        feature=feature,
        verdict=vr.verdict,
        p_held=vr.p_held,
        reinforced_model=reinforced_model,
    )
    return {
        "ok": True,
        "hypothesis_id": hyp_id,
        "verdict": vr.verdict,
        "p_held": vr.p_held,
        "p_opposite": vr.p_opposite,
        "observed_lift": vr.observed_lift,
        "actual": actual,
        "baseline": baseline,
        "reinforced_model": reinforced_model,
    }


__all__ = ["run"]
