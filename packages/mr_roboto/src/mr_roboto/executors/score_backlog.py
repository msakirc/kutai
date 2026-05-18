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
  - revenue_impact      = heuristic weight from the label; Z9 T5B picks a
                          b2b / b2c / hybrid table by mission.business_model
  - north_star_relevance= 0..1 from the mission success_metrics, else 0.5,
                          scaled by a b2b/b2c/hybrid multiplier (Z9 T5B)
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
#
# This is the B2C-default table — kept as the module-level constant so legacy
# callers and tests that don't pass a business_model see unchanged behaviour.
_REVENUE_IMPACT = {
    "churn_signal": 1.0,
    "pricing_feedback": 0.9,
    "bug": 0.7,
    "feature_request": 0.5,
    "praise": 0.05,
    "spam": 0.0,
}

# Z9 T5B — business-model-aware revenue-impact tables. B2B revenue is
# concentrated (one churned account = many lost seats / large MRR), so churn
# and pricing dominate even harder, and a single bug can threaten a renewal.
# B2C revenue is diffuse and volume-driven, so feature requests (top-of-funnel
# growth) carry relatively more weight. "hybrid" sits between the two.
_REVENUE_IMPACT_BY_MODEL = {
    "b2b": {
        "churn_signal": 1.0,
        "pricing_feedback": 0.95,
        "bug": 0.85,
        "feature_request": 0.42,
        "praise": 0.05,
        "spam": 0.0,
    },
    "b2c": dict(_REVENUE_IMPACT),
    "hybrid": {
        "churn_signal": 1.0,
        "pricing_feedback": 0.92,
        "bug": 0.78,
        "feature_request": 0.46,
        "praise": 0.05,
        "spam": 0.0,
    },
}

# Z9 T5B — business-model relevance multiplier applied to north_star_relevance.
# B2B north-stars (MRR / seats / churn) align tightly with the same churn /
# pricing signals the classifier surfaces, so backlog clusters score as more
# north-star-relevant; B2C is the 1.0 baseline. Kept modest (< 1.11) so the
# per-label revenue_impact ordering still dominates the score.
_NORTH_STAR_MODEL_WEIGHT = {
    "b2b": 1.08,
    "b2c": 1.0,
    "hybrid": 1.04,
}


def _normalize_business_model(value) -> str:
    """Coerce a raw business_model value to b2b | b2c | hybrid (default b2c)."""
    bm = str(value or "").strip().lower()
    return bm if bm in ("b2b", "b2c", "hybrid") else "b2c"

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


async def _north_star_and_model(mission_id) -> tuple[float, str]:
    """Return ``(north_star_relevance, business_model)`` for the mission.

    success_metrics lives in mission.context (artifact from step 2.9). When
    a north_star_metric is defined the cluster is treated as more relevant
    (0.8); a bare success_metrics block yields 0.65; nothing → 0.5 default.

    Z9 T5B — business_model is read from mission.context['business_model']
    or success_metrics.business_model, defaulting to 'b2c'.
    """
    if mission_id is None:
        return 0.5, "b2c"
    relevance = 0.5
    business_model = "b2c"
    try:
        from src.infra.db import get_mission

        mission = await get_mission(mission_id)
        if not mission:
            return 0.5, "b2c"
        import json as _json

        ctx_raw = mission.get("context") or "{}"
        ctx = _json.loads(ctx_raw) if isinstance(ctx_raw, str) else dict(ctx_raw)
        sm = ctx.get("success_metrics") or {}
        bb = ctx.get("blackboard") or {}
        sm = sm or (bb.get("success_metrics") if isinstance(bb, dict) else {}) or {}
        if isinstance(sm, dict) and sm.get("north_star_metric"):
            relevance = 0.8
        elif sm:
            relevance = 0.65
        # business_model: context key wins, success_metrics field is fallback.
        raw_bm = ctx.get("business_model")
        if raw_bm is None and isinstance(sm, dict):
            raw_bm = sm.get("business_model")
        business_model = _normalize_business_model(raw_bm)
    except Exception as exc:  # noqa: BLE001
        _log.debug("north_star lookup failed", error=str(exc))
    return relevance, business_model


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
    base_ns_relevance, business_model = await _north_star_and_model(mission_id)
    # Z9 T5B — payload may override the business_model (e.g. ad-hoc /task).
    if payload.get("business_model"):
        business_model = _normalize_business_model(payload.get("business_model"))
    # Apply the business-model relevance multiplier (clamped to 1.0 ceiling).
    ns_relevance = min(
        1.0,
        base_ns_relevance * _NORTH_STAR_MODEL_WEIGHT.get(business_model, 1.0),
    )
    revenue_table = _REVENUE_IMPACT_BY_MODEL.get(
        business_model, _REVENUE_IMPACT
    )

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
        revenue_impact = revenue_table.get(label, 0.3)
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
            "business_model": business_model,
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
