"""Z9 T3C — score_backlog mechanical executor.

Pure deterministic math — NO LLM. Reads ``classified_signal`` growth_events,
computes a priority score per (label, domain) cluster, and writes the top-N
clusters back as ``backlog_candidate`` growth_events rows. The founder sees
the candidates via ``/backlog`` and promotes them to missions via
``/approve`` — this executor never spawns a mission.

Score formula (inspectable — the breakdown is stored on every candidate)::

    score = frequency × revenue_impact × north_star_relevance
            × age_decay / cost_band_weight

  - frequency           = count of signals in the (label, domain) cluster
  - revenue_impact      = heuristic weight from the label
  - north_star_relevance= 0..1 from the mission success_metrics, else 0.5
  - age_decay           = recency weight, 1.0 (today) decaying toward ~0.3
  - cost_band_weight    = cheap=1, moderate=2, heavy=3 (label/domain estimate)

``backlog_candidate`` properties shape::

    {
        "label": "<cluster label>",
        "domain": "<cluster domain>",
        "score": <float>,
        "frequency": <int>,
        "formula": {
            "frequency": <int>,
            "revenue_impact": <float>,
            "north_star_relevance": <float>,
            "age_decay": <float>,
            "cost_band": "cheap" | "moderate" | "heavy",
            "cost_band_weight": <int>,
            "expression": "freq×rev×ns×age/cost = score"
        },
        "sample_external_ids": [<up to 5 external_ids in the cluster>],
        "sample_excerpt": "<one representative content excerpt>",
        "consumed": false
    }
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.infra.logging_config import get_logger

_log = get_logger("mr_roboto.score_backlog")

_DEFAULT_TOP_N = 10

# Revenue-impact heuristic per label. pricing_feedback / churn_signal hit the
# wallet directly; bugs hurt retention; feature requests are upside; praise
# and spam carry ~no backlog value.
_REVENUE_IMPACT = {
    "churn_signal": 1.0,
    "pricing_feedback": 0.9,
    "bug": 0.7,
    "feature_request": 0.5,
    "praise": 0.05,
    "spam": 0.0,
}

# Cost-band estimate per label — how expensive the resulting mission is.
# bugs are usually cheap point-fixes; feature work is heavy.
_COST_BAND_BY_LABEL = {
    "bug": "cheap",
    "pricing_feedback": "moderate",
    "churn_signal": "moderate",
    "feature_request": "heavy",
    "praise": "cheap",
    "spam": "cheap",
}

_COST_BAND_WEIGHT = {"cheap": 1, "moderate": 2, "heavy": 3}

# Domains that tend to inflate cost regardless of label (cross-cutting work).
_HEAVY_DOMAINS = {"auth", "billing", "payments", "search"}


def _parse_ts(value) -> datetime | None:
    """Parse a growth_events occurred_at / signal occurred_at timestamp."""
    if not value:
        return None
    s = str(value).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[: len(fmt) + 6], fmt)
        except Exception:
            continue
    return None


def _age_decay(occurred_at, now: datetime) -> float:
    """Recency weight in [~0.3, 1.0]; halves roughly every 14 days."""
    ts = _parse_ts(occurred_at)
    if ts is None:
        return 0.6  # unknown age — neutral
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    decay = 0.5 ** (age_days / 14.0)
    return max(0.3, min(1.0, decay))


def _cost_band(label: str, domain: str) -> str:
    band = _COST_BAND_BY_LABEL.get(label, "moderate")
    if domain in _HEAVY_DOMAINS and band != "heavy":
        # cross-cutting domain bumps the band one notch
        return {"cheap": "moderate", "moderate": "heavy"}.get(band, band)
    return band


async def _north_star_relevance(mission_id) -> float:
    """0..1 relevance pulled from the mission success_metrics, else 0.5.

    success_metrics lives in mission.context (artifact from step 2.9). When
    a north_star_metric is defined the cluster is treated as more relevant
    (0.8); a bare success_metrics block yields 0.65; nothing → 0.5 default.
    """
    if mission_id is None:
        return 0.5
    try:
        from src.infra.db import get_mission

        mission = await get_mission(mission_id)
        if not mission:
            return 0.5
        import json as _json

        ctx_raw = mission.get("context") or "{}"
        ctx = _json.loads(ctx_raw) if isinstance(ctx_raw, str) else dict(ctx_raw)
        sm = ctx.get("success_metrics") or {}
        bb = ctx.get("blackboard") or {}
        sm = sm or (bb.get("success_metrics") if isinstance(bb, dict) else {}) or {}
        if isinstance(sm, dict) and sm.get("north_star_metric"):
            return 0.8
        if sm:
            return 0.65
    except Exception as exc:  # noqa: BLE001
        _log.debug("north_star lookup failed", error=str(exc))
    return 0.5


def compute_score(
    *,
    frequency: int,
    revenue_impact: float,
    north_star_relevance: float,
    age_decay: float,
    cost_band_weight: int,
) -> float:
    """The Z9 backlog score formula. Pure — exposed for deterministic tests."""
    cbw = cost_band_weight if cost_band_weight else 1
    return (
        frequency
        * revenue_impact
        * north_star_relevance
        * age_decay
    ) / cbw


async def run(task: dict) -> dict:
    """Score classified signals and write top-N backlog_candidate rows.

    Idempotent: clears prior un-consumed backlog_candidate rows for the
    mission before writing the fresh ranking, so re-running at each digest
    cycle rewrites rather than accumulates.
    """
    from src.infra.db import get_growth_events, insert_growth_event

    payload = task.get("payload") or {}
    mission_id = task.get("mission_id") or payload.get("mission_id")
    top_n = int(payload.get("top_n") or _DEFAULT_TOP_N)

    classified = await get_growth_events(
        mission_id=mission_id, kind="classified_signal"
    )
    if not classified:
        _log.info("score_backlog: no classified signals")
        return {"ok": True, "candidates": 0, "scored": []}

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    ns_relevance = await _north_star_relevance(mission_id)

    # Cluster by (label, domain). spam/praise produce ~0 scores naturally
    # via the revenue_impact heuristic — no special-casing needed.
    clusters: dict[tuple[str, str], dict] = {}
    for row in classified:
        props = row.get("properties") or {}
        label = str(props.get("label") or "spam")
        domain = str(props.get("domain") or "general")
        key = (label, domain)
        c = clusters.setdefault(
            key,
            {
                "label": label,
                "domain": domain,
                "external_ids": [],
                "decays": [],
                "excerpt": "",
            },
        )
        eid = props.get("external_id")
        if eid is not None:
            c["external_ids"].append(str(eid))
        c["decays"].append(_age_decay(row.get("occurred_at"), now))
        if not c["excerpt"]:
            c["excerpt"] = str(props.get("content_excerpt") or "")

    scored = []
    for (label, domain), c in clusters.items():
        frequency = len(c["external_ids"]) or len(c["decays"])
        revenue_impact = _REVENUE_IMPACT.get(label, 0.3)
        # Cluster age_decay = mean of member decays (recent cluster scores up).
        age_decay = (
            sum(c["decays"]) / len(c["decays"]) if c["decays"] else 0.6
        )
        cost_band = _cost_band(label, domain)
        cost_band_weight = _COST_BAND_WEIGHT.get(cost_band, 2)
        score = compute_score(
            frequency=frequency,
            revenue_impact=revenue_impact,
            north_star_relevance=ns_relevance,
            age_decay=age_decay,
            cost_band_weight=cost_band_weight,
        )
        formula = {
            "frequency": frequency,
            "revenue_impact": round(revenue_impact, 3),
            "north_star_relevance": round(ns_relevance, 3),
            "age_decay": round(age_decay, 3),
            "cost_band": cost_band,
            "cost_band_weight": cost_band_weight,
            "expression": (
                f"{frequency} × {revenue_impact:.2f} × {ns_relevance:.2f} "
                f"× {age_decay:.2f} / {cost_band_weight} = {score:.3f}"
            ),
        }
        scored.append(
            {
                "label": label,
                "domain": domain,
                "score": round(score, 4),
                "frequency": frequency,
                "formula": formula,
                "sample_external_ids": c["external_ids"][:5],
                "sample_excerpt": c["excerpt"][:280],
                "consumed": False,
            }
        )

    scored.sort(key=lambda d: d["score"], reverse=True)
    top = scored[:top_n]

    # Idempotent rewrite: tombstone prior un-consumed candidates so /backlog
    # only ever shows the latest ranking. We mark, not delete, to keep the
    # append-only invariant — consumers filter on properties.superseded.
    try:
        prior = await get_growth_events(
            mission_id=mission_id, kind="backlog_candidate"
        )
        if prior:
            from src.infra.db import get_db

            db = await get_db()
            import json as _json

            for p in prior:
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
        _log.debug("score_backlog: supersede prior failed", error=str(exc))

    candidate_ids = []
    for cand in top:
        cid = await insert_growth_event(
            mission_id, "backlog_candidate", cand
        )
        candidate_ids.append(cid)

    _log.info(
        "score_backlog: wrote backlog candidates",
        mission_id=mission_id,
        candidates=len(candidate_ids),
    )
    return {
        "ok": True,
        "candidates": len(candidate_ids),
        "candidate_ids": candidate_ids,
        "scored": top,
    }
