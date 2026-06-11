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
8. Z9 T5D — if the mission ran an A/B split (``experiment_variants`` rows),
   measure each arm and write an ``ab_result`` ``growth_events`` row with
   the Bayesian winner. Auto-rollback of a confident loser still needs a
   founder gate (``/experiment_ship`` / ``/experiment_rollback``) — this
   step only *computes* the winner, it never retires a variant.
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
                "mission_id": mission_id,
                "feature": feature,
                "metric": metric,
                "verdict": verdict,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("verdict lessons mirror failed", error=str(exc))


# ── reinforce loop (confirmed verdict → model_pick_log nudge) ──────────────
#
# Sibling loop: Z10 calibration (general_beckman/apply.py::_record_and_resolve_confidence).
# Kept deliberately separate — different signal, different surface:
#   - Z9 (here): hypothesis verdict = "confirmed" → bumps model SELECTION score
#     in fatih_hoca.grading. Read-time decay (50%/30d). Async fire-forget.
#   - Z10 (apply.py): every post-hook verdict → reliability bucket per
#     (model, task_kind, confidence_bucket) → injects [CALIBRATION NOTE]
#     into the LLM PROMPT. Nightly batch rollup. Sync post-hook critical path.
# Merging requires a null-heavy union schema + couples async telemetry to
# the sync post-hook path. See docs/handoff/2026-05-18-z0-and-backlog-handoff.md §2c.


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
        # Strategy (three tiers, most-precise first):
        #
        # Tier 0 (task_id join) — model_pick_log.task_id = tasks.id, filtered
        #   by tasks.mission_id. Introduced 2026-05-21 to fix the silent
        #   wrong-model failure mode: free-form title matching could resolve
        #   a model from a completely different mission. task_id is populated
        #   by the dispatcher via the heartbeat ContextVar for all picks made
        #   since this fix landed.
        #
        # Tier 1 (title join, legacy) — kept verbatim for rows that pre-date
        #   the task_id column (task_id IS NULL). Title is still a useful
        #   signal when the task_name was set correctly, which it was for all
        #   picks made by the old path.
        #
        # Tier 2 (global fallback) — unchanged safety net for edge cases
        #   (no tasks row, no matching pick, fresh DB). Still cross-mission
        #   risk but only reached if both tier-0 and tier-1 find nothing.
        if mission_id is not None:
            # Tier 0: join by task_id (precise, new rows only).
            cur = await db.execute(
                "SELECT mpl.picked_model, mpl.provider "
                "FROM model_pick_log mpl "
                "JOIN tasks t ON mpl.task_id = t.id "
                "WHERE t.mission_id = ? AND mpl.call_category != 'reinforce' "
                "ORDER BY mpl.timestamp DESC LIMIT 1",
                (mission_id,),
            )
            row = await cur.fetchone()
            if row:
                model = row[0]
                provider = row[1] or "local"

            if not model:
                # Tier 1: title join for legacy rows where task_id IS NULL.
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
            # Tier 2: fall back to the most recent non-reinforce pick overall.
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


# ── A/B result evaluation (T5D — Bayesian winner pick) ─────────────────────


async def _pull_variant_metric(task: dict, metric: str, variant: str) -> float | None:
    """Pull one A/B arm's metric value via PostHog get_insight.

    A real PostHog ``get_insight`` is filtered by the ``variant`` property.
    In mock mode the series is identical across arms — so the verdict task
    (or a test) may inject explicit per-arm values under
    ``payload['variant_metrics'] = {'control': x, 'treatment': y}`` which
    take precedence. Returns None when neither path yields a number.
    """
    payload = task.get("payload") or {}
    injected = payload.get("variant_metrics")
    if isinstance(injected, dict) and variant in injected:
        v = injected[variant]
        if _is_num(v):
            return float(v)
    sub = {
        "mission_id": task.get("mission_id"),
        "id": task.get("id"),
        "context": {
            "post_hook": {
                "service": "posthog",
                "action": "get_insight",
                "params": {"project_id": "default", "insight_id": 1,
                           "metric": metric, "variant": variant},
            }
        },
    }
    try:
        from mr_roboto.executors.vendor_call import run as vendor_call_run
        env = await vendor_call_run(sub)
    except Exception:  # noqa: BLE001
        return None
    if not (isinstance(env, dict) and env.get("ok")):
        return None
    result = env.get("result") or {}
    inner = result.get("result") if isinstance(result, dict) else None
    if isinstance(inner, list) and inner:
        first = inner[0]
        if isinstance(first, dict) and isinstance(first.get("data"), list):
            data = [float(x) for x in first["data"] if _is_num(x)]
            if data:
                return data[-1]
    return None


async def _evaluate_ab(
    task: dict, mission_id: int | None, hyp_id: int, metric: str
) -> dict | None:
    """If the mission ran an A/B split, compute the Bayesian winner.

    Writes an ``ab_result`` growth_event and returns its payload, or None
    when the mission had no variants. Deterministic — no LLM.
    """
    if mission_id is None:
        return None
    try:
        from src.infra.db import get_variants
        from src.growth.ab_result import evaluate_ab
        from general_beckman import record_growth_event
    except Exception as exc:  # noqa: BLE001
        logger.debug("ab eval imports failed", error=str(exc))
        return None

    variants = await get_variants(mission_id=mission_id)
    if not variants:
        return None
    has_control = any(
        str(v.get("variant_name") or "").lower() == "control" for v in variants
    )
    has_treatment = any(
        str(v.get("variant_name") or "").lower() == "treatment"
        for v in variants
    )
    if not (has_control and has_treatment):
        return None

    control_m = await _pull_variant_metric(task, metric, "control")
    treatment_m = await _pull_variant_metric(task, metric, "treatment")
    if control_m is None or treatment_m is None:
        logger.info("ab eval: missing arm metric — skipping winner pick")
        return None

    ab = evaluate_ab(control_metric=control_m, treatment_metric=treatment_m)
    result = {
        "hypothesis_id": hyp_id,
        "metric": metric,
        "winner": ab.winner,
        "confident": ab.confident,
        "p_treatment_better": ab.p_treatment_better,
        "p_control_better": ab.p_control_better,
        "control_metric": ab.control_metric,
        "treatment_metric": ab.treatment_metric,
        "relative_lift": ab.relative_lift,
        "model": ab.model,
        "founder_gate": (
            "confident winner — founder must run /experiment_ship or "
            "/experiment_rollback; no auto-retire"
            if ab.confident else "inconclusive — no winner yet"
        ),
    }
    try:
        await record_growth_event(mission_id, "ab_result", result)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ab_result event failed", error=str(exc))
    logger.info(
        "ab eval complete mission=%s winner=%s confident=%s",
        mission_id, ab.winner, ab.confident,
    )
    return result


# ── main entrypoint ────────────────────────────────────────────────────────


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Compute + record a verdict for one hypothesis. Never raises."""
    from src.growth.verdict_stats import compute_verdict
    from src.infra.db import (
        get_pending_hypotheses,
        record_hypothesis_verdict,
    )
    from general_beckman import record_growth_event

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
        await record_growth_event(
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

    # 8. Z9 T5D — A/B result evaluation. If this mission ran an A/B split,
    # pick the Bayesian winner and write an ab_result event. Mechanical.
    ab_result: dict | None = None
    try:
        ab_result = await _evaluate_ab(task, mission_id, hyp_id, metric)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ab evaluation failed", error=str(exc))

    logger.info(
        "record_verdict complete",
        hypothesis_id=hyp_id,
        feature=feature,
        verdict=vr.verdict,
        p_held=vr.p_held,
        reinforced_model=reinforced_model,
        ab_winner=(ab_result or {}).get("winner"),
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
        "ab_result": ab_result,
    }


__all__ = ["run"]
