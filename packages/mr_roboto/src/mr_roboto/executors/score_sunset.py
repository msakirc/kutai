"""Z9 Growth T5C — ``score_sunset`` mechanical executor.

Pure deterministic usage math — NO LLM. Closes the feature-lifecycle loop:
a feature that **few users touch** but **still costs money to maintain** is a
*sunset candidate* — the founder reviews it via ``/sunset`` and approves a
deprecation mission via ``/approve_sunset``.

Routed via ``mr_roboto.run`` when ``payload["action"] == "score_sunset"``.
Fired weekly by the global ``sunset_score_recompute`` internal cadence.

Sunset scoring formula (inspectable — the breakdown is stored on every row)
---------------------------------------------------------------------------

For each feature of a product::

    usage_rate     = distinct_active_users_touching_feature / active_users
    cost_band      = cheap | moderate | heavy   (maintenance cost heuristic)
    cost_weight    = cheap=1, moderate=2, heavy=3
    sunset_score   = (1 - usage_rate) × cost_weight

A feature is a **sunset candidate** iff::

    usage_rate < SUNSET_USAGE_THRESHOLD   (default 1% of active users)
    AND cost_weight > 0                   (non-zero maintenance cost)

``sunset_score`` ranks candidates: low usage AND high cost scores highest.
``SUNSET_USAGE_THRESHOLD`` is a named, founder-overridable constant
(``payload.usage_threshold`` overrides per-run; env
``KUTAI_SUNSET_USAGE_THRESHOLD`` overrides globally).

Recipe sunset is **OUT of scope** for v1 (deferred — features only).

Data sources
------------
- ``growth_events`` rows in the last 30d that carry a ``feature`` /
  ``feature_id`` property → per-feature event volume + distinct user set.
- ``recipe_pin_log`` → which recipes/features were pinned to missions
  (a feature-catalog source so a zero-traffic feature is still *seen*).
- Active-user count = distinct ``user_id`` across all feature events in the
  window (the denominator); falls back to a small floor so a brand-new
  product with one tester doesn't flag every feature.

Output
------
``growth_events`` rows ``kind="sunset_candidate"`` with shape::

    {
        "feature": "<feature name/id>",
        "usage_rate": <float 0..1>,
        "event_volume": <int>,
        "distinct_users": <int>,
        "active_users": <int>,
        "cost_band": "cheap" | "moderate" | "heavy",
        "cost_band_weight": <int>,
        "sunset_score": <float>,
        "why": "<human-readable reason>",
        "formula": {"expression": "(1-usage)×cost = score", ...},
        "consumed": false
    }
"""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from src.infra.logging_config import get_logger

_log = get_logger("mr_roboto.score_sunset")

# ── founder-overridable constants ──────────────────────────────────────────

# A feature whose distinct-user reach is below this fraction of active users
# is "low usage". Founder-overridable: payload.usage_threshold (per-run) or
# env KUTAI_SUNSET_USAGE_THRESHOLD (global). Default = 1% per the zone doc.
SUNSET_USAGE_THRESHOLD: float = 0.01

# Usage window — feature event volume is counted over the last N days.
_WINDOW_DAYS = 30

# Active-user denominator floor. A product with only a handful of testers
# would make every feature look "1%" used; the floor keeps the rate honest.
_MIN_ACTIVE_USERS = 5

# Cost-band → maintenance-cost weight. cheap=1 still counts as "non-zero
# cost" (a candidate), so the gate is cost_weight > 0, always true here.
_COST_BAND_WEIGHT = {"cheap": 1, "moderate": 2, "heavy": 3}

# Domain heuristic — features in cross-cutting domains cost more to keep
# alive (security surface, integration upkeep, infra). Used when a feature
# event carries no explicit cost_band.
_HEAVY_DOMAINS = {"auth", "billing", "payments", "search", "integrations"}
_MODERATE_DOMAINS = {"notifications", "analytics", "admin", "export"}


def _usage_threshold(payload: dict) -> float:
    """Resolve the founder-overridable low-usage threshold."""
    raw = payload.get("usage_threshold")
    if raw is None:
        raw = os.getenv("KUTAI_SUNSET_USAGE_THRESHOLD")
    if raw is not None:
        try:
            val = float(raw)
            if 0.0 < val < 1.0:
                return val
        except (TypeError, ValueError):
            pass
    return SUNSET_USAGE_THRESHOLD


def _since_db(now: datetime | None = None) -> str:
    """SQLite datetime string for the start of the usage window."""
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    return (now - timedelta(days=_WINDOW_DAYS)).strftime("%Y-%m-%d %H:%M:%S")


def _cost_band(props: dict, domain: str) -> str:
    """Maintenance-cost band for a feature.

    Explicit ``cost_band`` on the event wins; otherwise estimate from the
    feature's domain (cross-cutting domains cost more to keep alive).
    """
    explicit = str(props.get("cost_band") or "").strip().lower()
    if explicit in _COST_BAND_WEIGHT:
        return explicit
    d = (domain or "").strip().lower()
    if d in _HEAVY_DOMAINS:
        return "heavy"
    if d in _MODERATE_DOMAINS:
        return "moderate"
    return "cheap"


def compute_sunset_score(usage_rate: float, cost_band_weight: int) -> float:
    """The Z9 sunset score. Pure — exposed for deterministic tests.

    ``(1 - usage_rate) × cost_band_weight`` — low usage AND high maintenance
    cost score highest. Clamped so a >100% usage_rate (impossible but cheap
    to guard) cannot drive the score negative.
    """
    inv_usage = max(0.0, 1.0 - max(0.0, usage_rate))
    return inv_usage * max(0, cost_band_weight)


def is_sunset_candidate(
    usage_rate: float, cost_band_weight: int, threshold: float
) -> bool:
    """A feature is a sunset candidate iff usage is below threshold AND it
    still carries non-zero maintenance cost."""
    return usage_rate < threshold and cost_band_weight > 0


# ── feature catalog assembly ───────────────────────────────────────────────


async def _collect_feature_usage(
    mission_id: int | None, since: str
) -> dict[str, dict]:
    """Build a per-feature usage map from growth_events in the window.

    Returns ``{feature: {domain, event_volume, users:set, cost_band_hint}}``.
    A growth_events row contributes to a feature when its properties carry
    a ``feature`` or ``feature_id`` key. ``user_id`` (or ``distinct_id``)
    feeds the distinct-user reach; missing user ids count toward volume but
    not reach.
    """
    from dabidabi import get_growth_events

    features: dict[str, dict] = {}
    try:
        rows = await get_growth_events(mission_id=mission_id, since=since)
    except Exception as exc:  # noqa: BLE001
        _log.warning("score_sunset: growth_events query failed", error=str(exc))
        return features

    # Sunset bookkeeping kinds never count as feature usage.
    _SKIP_KINDS = {"sunset_candidate", "sunset_approved", "northstar_review"}

    for row in rows or []:
        if row.get("kind") in _SKIP_KINDS:
            continue
        props = row.get("properties") or {}
        if not isinstance(props, dict):
            continue
        feature = props.get("feature") or props.get("feature_id")
        if not feature:
            continue
        feature = str(feature)
        domain = str(props.get("domain") or props.get("feature_domain") or "general")
        entry = features.setdefault(
            feature,
            {
                "feature": feature,
                "domain": domain,
                "event_volume": 0,
                "users": set(),
                "cost_band_hint": props,
            },
        )
        entry["event_volume"] += 1
        uid = props.get("user_id") or props.get("distinct_id")
        if uid is not None:
            entry["users"].add(str(uid))
    return features


async def _augment_from_recipe_pins(
    mission_id: int | None, features: dict[str, dict]
) -> None:
    """Add zero-traffic features from ``recipe_pin_log``.

    A pinned recipe represents a shipped capability. If it never produced a
    growth_events row it has *zero* usage — exactly the case sunset scoring
    must surface. Without this a dead feature is invisible (no events) and
    can never be flagged. Best-effort; the table may be absent.
    """
    try:
        from dabidabi import get_db

        db = await get_db()
        cur = await db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='recipe_pin_log'"
        )
        if not await cur.fetchone():
            return
        if mission_id is not None:
            cur = await db.execute(
                "SELECT DISTINCT recipe_name FROM recipe_pin_log "
                "WHERE mission_id = ?",
                (mission_id,),
            )
        else:
            cur = await db.execute(
                "SELECT DISTINCT recipe_name FROM recipe_pin_log"
            )
        for (recipe_name,) in await cur.fetchall():
            if not recipe_name:
                continue
            name = str(recipe_name)
            # Only register as a zero-traffic feature when no growth_events
            # already cover it — never clobber real usage data.
            if name not in features:
                features[name] = {
                    "feature": name,
                    "domain": "general",
                    "event_volume": 0,
                    "users": set(),
                    "cost_band_hint": {},
                    "from_recipe_pin": True,
                }
    except Exception as exc:  # noqa: BLE001
        _log.debug("score_sunset: recipe_pin_log augment failed", error=str(exc))


def _active_users(features: dict[str, dict]) -> int:
    """Distinct active users across ALL feature events in the window.

    This is the usage_rate denominator. Floored at ``_MIN_ACTIVE_USERS`` so
    a tiny tester base doesn't make every feature look unused.
    """
    everyone: set[str] = set()
    for entry in features.values():
        everyone |= entry["users"]
    return max(len(everyone), _MIN_ACTIVE_USERS)


# ── main entrypoint ────────────────────────────────────────────────────────


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Score every feature; write ``sunset_candidate`` growth_events rows.

    Idempotent: tombstones prior un-consumed ``sunset_candidate`` rows for
    the mission before writing the fresh ranking, so ``/sunset`` only ever
    shows the latest scoring. Always returns a dict — never raises.
    """
    from dabidabi import get_growth_events
    from general_beckman import record_growth_event, supersede_growth_event

    payload = task.get("payload") or {}
    mission_id = task.get("mission_id") or payload.get("mission_id")
    threshold = _usage_threshold(payload)
    since = _since_db()

    features = await _collect_feature_usage(mission_id, since)
    await _augment_from_recipe_pins(mission_id, features)

    if not features:
        _log.info("score_sunset: no features found", mission_id=mission_id)
        return {
            "ok": True,
            "features": 0,
            "candidates": 0,
            "threshold": threshold,
            "scored": [],
        }

    active_users = _active_users(features)

    scored: list[dict] = []
    for feature, entry in features.items():
        distinct_users = len(entry["users"])
        usage_rate = distinct_users / active_users if active_users else 0.0
        cost_band = _cost_band(entry.get("cost_band_hint") or {}, entry["domain"])
        cost_weight = _COST_BAND_WEIGHT.get(cost_band, 1)
        sunset_score = compute_sunset_score(usage_rate, cost_weight)
        candidate = is_sunset_candidate(usage_rate, cost_weight, threshold)

        why = (
            f"{distinct_users}/{active_users} users "
            f"({usage_rate * 100:.2f}%) touched '{feature}' in {_WINDOW_DAYS}d; "
            f"maintenance cost {cost_band}"
        )
        if entry.get("from_recipe_pin") and entry["event_volume"] == 0:
            why += " — pinned recipe with zero recorded usage"

        record = {
            "feature": feature,
            "domain": entry["domain"],
            "usage_rate": round(usage_rate, 5),
            "event_volume": entry["event_volume"],
            "distinct_users": distinct_users,
            "active_users": active_users,
            "cost_band": cost_band,
            "cost_band_weight": cost_weight,
            "sunset_score": round(sunset_score, 4),
            "is_candidate": candidate,
            "why": why,
            "formula": {
                "usage_rate": round(usage_rate, 5),
                "cost_band": cost_band,
                "cost_band_weight": cost_weight,
                "threshold": threshold,
                "expression": (
                    f"(1 - {usage_rate:.4f}) × {cost_weight} "
                    f"= {sunset_score:.3f}  "
                    f"[candidate iff usage < {threshold:.4f}]"
                ),
            },
            "consumed": False,
        }
        scored.append(record)

    scored.sort(key=lambda d: d["sunset_score"], reverse=True)
    candidates = [r for r in scored if r["is_candidate"]]

    # Idempotent rewrite: tombstone prior un-consumed candidates so /sunset
    # only ever surfaces the latest scoring. Append-only — mark, never delete.
    try:
        await supersede_growth_event(
            mission_id=mission_id, kind="sunset_candidate"
        )
    except Exception as exc:  # noqa: BLE001
        _log.debug("score_sunset: supersede prior failed", error=str(exc))

    candidate_ids: list[int] = []
    for cand in candidates:
        cid = await record_growth_event(mission_id, "sunset_candidate", cand)
        candidate_ids.append(cid)

    _log.info(
        "score_sunset complete",
        mission_id=mission_id,
        features=len(features),
        candidates=len(candidate_ids),
        active_users=active_users,
        threshold=threshold,
    )
    return {
        "ok": True,
        "features": len(features),
        "candidates": len(candidate_ids),
        "candidate_ids": candidate_ids,
        "active_users": active_users,
        "threshold": threshold,
        "scored": scored,
    }


__all__ = [
    "run",
    "compute_sunset_score",
    "is_sunset_candidate",
    "SUNSET_USAGE_THRESHOLD",
]
