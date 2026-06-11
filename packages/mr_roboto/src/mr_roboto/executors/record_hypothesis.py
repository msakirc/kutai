"""Z9 T4A — record_hypothesis mechanical executor.

At mission-spec finalization (i2p Phase 7 review) this verb captures the
mission's *predicted* metric impact as a ``hypotheses`` row so a later
verdict (T4C/T4D) can check whether the prediction held.

Pure deterministic — NO LLM. The prediction is extracted from the mission
spec text (title + description + any spec/brief artifacts on
``mission.context``) with a regex/heuristic that looks for the shape
``<metric> <direction> <magnitude>`` (e.g. "checkout conversion +12%",
"p95 latency -200ms", "error rate -50%"). If nothing parseable is found
the hypothesis is still recorded with a best-effort ``predicted_json`` and
``confidence="low"`` — the executor never enqueues an LLM to parse.

Per-metric default window (resolves ``window_seconds``)::

    activation  = 7d      revenue     = 14d     latency    = 3d
    retention   = 30d     referral    = 14d     error_rate = 3d
    acquisition = 7d      (default)   = 14d

``dedup_key`` = feature-slug + metric-name. ``insert_hypothesis`` returns
-1 when that key is still in its 90-day suppression cool-off — in that
case a ``hypothesis_suppressed`` growth_event is written and the verb
returns ok with a note.

A pure refactor / doc-only mission with no measurable metric is *flagged*
(``hypothesis_skipped`` growth_event) but does not error.

Returns
-------
dict
    ``{"ok": True, "recorded": bool, "hypothesis_id": int|None,
       "skipped": bool, "suppressed": bool, ...}``
"""
from __future__ import annotations

import json
import re

from src.infra.logging_config import get_logger

_log = get_logger("mr_roboto.record_hypothesis")

_DAY = 86400

# Per-metric default measurement window (founder may /override later).
_WINDOW_BY_METRIC = {
    "activation": 7 * _DAY,
    "retention": 30 * _DAY,
    "revenue": 14 * _DAY,
    "referral": 14 * _DAY,
    "acquisition": 7 * _DAY,
    "latency": 3 * _DAY,
    "error_rate": 3 * _DAY,
}
_DEFAULT_WINDOW = 14 * _DAY

# Keyword → canonical metric family. First match wins; order matters
# (error_rate / latency before the broad acquisition/revenue families).
_METRIC_KEYWORDS = [
    ("error_rate", ("error rate", "error_rate", "errors", "crash", "failure rate")),
    ("latency", ("latency", "p95", "p99", "response time", "load time", "ttfb")),
    ("retention", ("retention", "churn", "retain", "day-7", "day 7", "d7", "dau", "mau")),
    ("activation", ("activation", "onboarding", "first value", "aha", "time to value")),
    ("revenue", ("revenue", "conversion", "checkout", "mrr", "arr", "arpu", "upgrade", "purchase")),
    ("referral", ("referral", "invite", "share", "viral", "k-factor", "k factor")),
    ("acquisition", ("acquisition", "signup", "sign-up", "sign up", "landing", "traffic", "install")),
]

# Mission classes with no measurable metric impact — flagged, not blocked.
_NON_MEASURABLE_HINTS = (
    "refactor", "documentation", "doc-only", "docs only", "rename",
    "cleanup", "clean up", "tech debt", "technical debt", "lint",
    "formatting", "typo", "comment",
)

# "+12%", "-200ms", "12 %", "200 ms", "-0.5pp" ...
# A magnitude MUST carry a sign or a unit — a bare integer (e.g. the "95"
# in "p95") is not a prediction. ``finditer`` lets us skip false hits.
_MAGNITUDE_RE = re.compile(
    r"([+\-]\s*\d+(?:\.\d+)?\s*(?:%|pp|ms|s|x|×|points?)?"
    r"|\d+(?:\.\d+)?\s*(?:%|pp|ms|x|×|points?))",
    re.IGNORECASE,
)
_NUM_UNIT_RE = re.compile(
    r"([+\-]?\s*\d+(?:\.\d+)?)\s*(%|pp|ms|s|x|×|points?)?",
    re.IGNORECASE,
)


def _slugify(text: str) -> str:
    """feature-slug: lowercase, alnum, dash-joined, max 60 chars."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s or "mission")[:60]


def _classify_metric(text: str) -> str | None:
    """Return the canonical metric family mentioned in ``text``, else None."""
    low = (text or "").lower()
    for metric, keywords in _METRIC_KEYWORDS:
        if any(kw in low for kw in keywords):
            return metric
    return None


def _extract_prediction(spec_text: str) -> dict:
    """Heuristically parse ``<metric> <direction> <magnitude>`` from a spec.

    Returns a ``predicted_json`` dict. ``confidence`` is ``"high"`` when both
    a metric family and a signed magnitude were found, ``"low"`` otherwise.
    Never raises — a best-effort dict is always returned.
    """
    text = spec_text or ""
    metric = _classify_metric(text)

    # Scan every sentence/line for a magnitude that sits near a metric word.
    best: dict | None = None
    for chunk in re.split(r"[.\n;]", text):
        chunk_metric = _classify_metric(chunk)
        if chunk_metric is None:
            continue
        m = _MAGNITUDE_RE.search(chunk)
        if not m:
            continue
        # Re-split the matched span into number + optional unit.
        nu = _NUM_UNIT_RE.search(m.group(0))
        if not nu:
            continue
        raw_num = nu.group(1).replace(" ", "")
        unit = (nu.group(2) or "").lower()
        try:
            value = float(raw_num)
        except ValueError:
            continue
        # Direction: explicit sign, else infer from wording.
        if raw_num.startswith("-"):
            direction = "down"
        elif raw_num.startswith("+"):
            direction = "up"
        elif re.search(r"\b(reduce|decrease|lower|cut|drop|shave|trim)\b", chunk, re.I):
            direction = "down"
        else:
            direction = "up"
        best = {
            "metric": chunk_metric,
            "direction": direction,
            "magnitude": abs(value),
            "unit": unit or "%",
            "confidence": "high",
        }
        break

    if best is not None:
        return best

    # No magnitude parseable — best-effort, low confidence.
    return {
        "metric": metric,
        "direction": None,
        "magnitude": None,
        "unit": None,
        "confidence": "low",
        "note": "spec stated no parseable metric prediction",
    }


def _is_non_measurable(spec_text: str, predicted: dict) -> bool:
    """A pure refactor / doc-only mission with no metric family found."""
    if predicted.get("metric"):
        return False
    low = (spec_text or "").lower()
    return any(hint in low for hint in _NON_MEASURABLE_HINTS)


def _resolve_window(metric: str | None) -> int:
    return _WINDOW_BY_METRIC.get(metric or "", _DEFAULT_WINDOW)


def _collect_spec_text(mission: dict) -> tuple[str, str]:
    """Return ``(feature_label, spec_text)`` from a mission row.

    ``spec_text`` aggregates the title, description and any spec/brief
    artifacts stashed under ``mission.context`` (mission_brief,
    requirements_spec, charter, feature_decl, blackboard equivalents).
    """
    title = str(mission.get("title") or "")
    desc = str(mission.get("description") or "")

    ctx_raw = mission.get("context") or "{}"
    try:
        ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else dict(ctx_raw)
    except (json.JSONDecodeError, TypeError):
        ctx = {}
    # Defend against a double-encoded context (json.loads → str).
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    if not isinstance(ctx, dict):
        ctx = {}

    parts = [title, desc]
    bb = ctx.get("blackboard") if isinstance(ctx.get("blackboard"), dict) else {}
    for key in ("mission_brief", "requirements_spec", "charter", "feature_decl",
                "spec", "predicted_impact", "expected_impact"):
        for src in (ctx, bb):
            val = src.get(key)
            if isinstance(val, str):
                parts.append(val)
            elif isinstance(val, dict):
                parts.append(json.dumps(val, ensure_ascii=False))
    spec_text = "\n".join(p for p in parts if p)
    feature_label = title or desc[:80] or f"mission-{mission.get('id')}"
    return feature_label, spec_text


async def run(task: dict) -> dict:
    """Record a hypothesis row for the mission identified by ``task``.

    Idempotent at the DB layer via ``dedup_key`` suppression; the verb
    itself always re-extracts the prediction so a re-run picks up an
    edited spec.
    """
    from src.infra.db import (
        get_mission,
        insert_hypothesis,
    )
    from general_beckman import record_growth_event

    payload = task.get("payload") or {}
    mission_id = task.get("mission_id") or payload.get("mission_id")
    if mission_id is None:
        return {"ok": False, "error": "record_hypothesis requires mission_id"}
    mission_id = int(mission_id)

    mission = await get_mission(mission_id)
    if not mission:
        return {"ok": False, "error": f"mission {mission_id} not found"}

    # Founder/payload may pre-supply a prediction or window override.
    override_pred = payload.get("predicted")
    override_window = payload.get("window_seconds")

    feature_label, spec_text = _collect_spec_text(mission)

    if isinstance(override_pred, dict) and override_pred:
        predicted = dict(override_pred)
        predicted.setdefault("confidence", "high")
    else:
        predicted = _extract_prediction(spec_text)

    metric = predicted.get("metric")

    # ── Non-measurable mission: flag but do not error ────────────────────
    if not metric and _is_non_measurable(spec_text, predicted):
        await record_growth_event(
            mission_id,
            "hypothesis_skipped",
            {
                "feature": feature_label,
                "reason": "non-measurable mission (refactor/doc-only) — "
                          "no metric family detected in spec",
            },
        )
        _log.info(
            "record_hypothesis: skipped non-measurable mission",
            mission_id=mission_id,
        )
        return {
            "ok": True,
            "recorded": False,
            "skipped": True,
            "suppressed": False,
            "hypothesis_id": None,
            "reason": "non-measurable mission",
        }

    # Unrecognised metric → still record (best-effort, default window).
    feature_slug = _slugify(feature_label)
    metric_name = metric or "unspecified"
    dedup_key = f"{feature_slug}::{metric_name}"
    window_seconds = (
        int(override_window) if override_window is not None
        else _resolve_window(metric)
    )

    predicted_json = dict(predicted)
    predicted_json["resolved_window_seconds"] = window_seconds

    hyp_id = await insert_hypothesis(
        mission_id=mission_id,
        feature=feature_label,
        predicted=predicted_json,
        window_seconds=window_seconds,
        dedup_key=dedup_key,
    )

    # ── Suppressed (refuted feature/metric still in 90d cool-off) ────────
    if hyp_id == -1:
        await record_growth_event(
            mission_id,
            "hypothesis_suppressed",
            {
                "feature": feature_label,
                "metric": metric_name,
                "dedup_key": dedup_key,
                "reason": "feature/metric pair still in 90-day "
                          "refuted cool-off",
            },
        )
        _log.info(
            "record_hypothesis: suppressed (dedup_key in cool-off)",
            mission_id=mission_id,
            dedup_key=dedup_key,
        )
        return {
            "ok": True,
            "recorded": False,
            "skipped": False,
            "suppressed": True,
            "hypothesis_id": None,
            "dedup_key": dedup_key,
            "note": "dedup_key still suppressed — hypothesis not re-recorded",
        }

    await record_growth_event(
        mission_id,
        "hypothesis_recorded",
        {
            "feature": feature_label,
            "metric": metric_name,
            "dedup_key": dedup_key,
            "hypothesis_id": hyp_id,
            "window_seconds": window_seconds,
            "confidence": predicted_json.get("confidence", "low"),
        },
    )
    _log.info(
        "record_hypothesis: recorded",
        mission_id=mission_id,
        hypothesis_id=hyp_id,
        metric=metric_name,
        confidence=predicted_json.get("confidence"),
    )
    return {
        "ok": True,
        "recorded": True,
        "skipped": False,
        "suppressed": False,
        "hypothesis_id": hyp_id,
        "feature": feature_label,
        "metric": metric_name,
        "window_seconds": window_seconds,
        "predicted": predicted_json,
        "dedup_key": dedup_key,
        "low_confidence": predicted_json.get("confidence") == "low",
    }
