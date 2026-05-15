"""Z7 T5 A9 — Monthly investor bullets + A9.r1 segmented templates.

**CRITICAL FRAMING**: A9 produces bullets ONLY — numbers + anomaly hypotheses.
Auto-generated investor PROSE is an antipattern. The founder writes the prose;
A9 surfaces what the founder hasn't noticed. Output is structured Markdown
bullets, NOT auto-sent (founder copies to clipboard — no send button).

Pipeline
--------
1. Collect metrics from data sources (degrade gracefully when absent):
   - Z6 Stripe: revenue, MRR delta, churn, customer count
   - Z6 cost actuals: burn, runway months
   - Z8 ops: uptime, P95 latency, incident count (reads ``incidents`` table)
   - Z3 review density: PRs shipped, security issues
   - A8 support: volume + escalation rate (reads ``tickets`` table)
   - A11 mention monitor counts (absent until A11 ships → degrade)
2. Anomaly detection: compare each metric vs trailing-3-month median ±2σ.
   Flag outliers. For each, call LLM (OVERHEAD lane) for a one-sentence
   "what changed" hypothesis.
3. Render Markdown bullets — 5 sections:
   - Highlights (top 3 positive outliers)
   - Lowlights (top 3 negative outliers)
   - Numbers (table of all metrics)
   - Anomalies needing founder explanation (3-5, with hypothesis)
   - Suggested asks (mission_lessons flagged needs_external_help)
4. A9.r1 — emit 3 segmented variants:
   - ``pre_investor_pitch_bullets`` (warm intro before raise)
   - ``current_investor_update`` (already-on-cap-table monthly)
   - ``advisor_check_in`` (advisor cadence)
5. Surface as founder_action "review monthly bullets" — copy-to-clipboard only.

Public API
----------
- ``run_investor_bullets(product_id)``          — main entry point (mr_roboto).
- ``collect_metrics(product_id)``               — fetch all sources, degrade.
- ``render_bullets(metrics, hypotheses, gaps)`` — produce Markdown bullets.
- ``emit_segmented_variants(bullets_md, contacts)`` — 3 template variants.
- ``_detect_anomaly(name, current, history)``   — ±2σ anomaly check.
- ``_call_llm_anomaly_hypothesis(name, current, history)`` — OVERHEAD LLM call.
- ``_enqueue_overhead``                          — thin wrapper (monkeypatchable).
- ``_create_founder_action``                     — thin wrapper (monkeypatchable).
- ``_list_contacts``                             — thin wrapper (monkeypatchable).
"""
from __future__ import annotations

import asyncio
import json
import math
import statistics
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("app.jobs.investor_bullets")

# ---------------------------------------------------------------------------
# Segmented template preambles (A9.r1)
# ---------------------------------------------------------------------------

_TEMPLATE_PREAMBLES: dict[str, str] = {
    "pre_investor_pitch_bullets": (
        "_Context: pre-raise warm intro bullets — these numbers speak before "
        "you do. Review what stands out; the prose is yours to write._\n\n"
    ),
    "current_investor_update": (
        "_Context: monthly update for current investors on the cap table. "
        "Flag any numbers they'll ask about before you send._\n\n"
    ),
    "advisor_check_in": (
        "_Context: advisor cadence update. Advisors care about traction + "
        "blockers, not financial detail. Highlight the signal._\n\n"
    ),
}

# investor contacts map to pre-raise + current-investor; advisors map to
# advisor_check_in. All 3 are always emitted when any recipient exists.
_CATEGORY_TO_TEMPLATES: dict[str, list[str]] = {
    "investor": [
        "pre_investor_pitch_bullets",
        "current_investor_update",
    ],
    "advisor": [
        "advisor_check_in",
    ],
}

# ---------------------------------------------------------------------------
# Anomaly detection — ±2σ vs trailing-3-month median
# ---------------------------------------------------------------------------

_SIGMA_THRESHOLD = 2.0


def _detect_anomaly(
    name: str,
    current: float,
    history: list[float],
) -> dict:
    """Compare *current* vs trailing-history ±2σ.

    Returns a dict with keys:
      - ``is_anomaly`` (bool)
      - ``direction`` (str: "up" | "down" | "none")
      - ``sigma`` (float: signed z-score, or 0 when insufficient data)
      - ``median`` (float)
    """
    if len(history) < 2:
        return {"is_anomaly": False, "direction": "none", "sigma": 0.0, "median": 0.0}

    median = statistics.median(history)
    try:
        stdev = statistics.stdev(history)
    except statistics.StatisticsError:
        stdev = 0.0

    if stdev == 0.0:
        # Zero variance — flag only if current differs materially
        if current != median:
            direction = "up" if current > median else "down"
            return {"is_anomaly": True, "direction": direction, "sigma": float("inf"), "median": median}
        return {"is_anomaly": False, "direction": "none", "sigma": 0.0, "median": median}

    sigma = (current - median) / stdev
    is_anomaly = abs(sigma) >= _SIGMA_THRESHOLD
    if is_anomaly:
        direction = "up" if sigma > 0 else "down"
    else:
        direction = "none"

    return {
        "is_anomaly": is_anomaly,
        "direction": direction,
        "sigma": round(sigma, 2),
        "median": round(median, 4),
    }


# ---------------------------------------------------------------------------
# LLM hypothesis — OVERHEAD lane (monkeypatchable entry point)
# ---------------------------------------------------------------------------


async def _enqueue_overhead(spec: dict, *, lane: str) -> Any:
    """Thin wrapper around ``general_beckman.enqueue`` (monkeypatchable)."""
    from general_beckman import enqueue
    return await enqueue(spec, lane=lane)


async def _call_llm_anomaly_hypothesis(
    metric_name: str,
    current: float,
    history: list[float],
) -> str:
    """Enqueue a one-sentence anomaly hypothesis via OVERHEAD lane.

    Returns the hypothesis string, or an empty string on timeout/failure.
    """
    from general_beckman.lanes import LANE_ONESHOT as LANE_OVERHEAD  # OVERHEAD tasks use oneshot lane

    median_val = statistics.median(history) if len(history) >= 2 else 0.0
    direction = "above" if current > median_val else "below"

    prompt = (
        f"You are surfacing a data anomaly for a founder's investor update.\n"
        f"Metric: {metric_name}\n"
        f"This month: {current}\n"
        f"Trailing 3-month median: {round(median_val, 4)}\n"
        f"Direction: {direction} the median (significant deviation).\n\n"
        f"In ONE sentence, state the most plausible business reason for this change. "
        f"Do NOT guess if uncertain — say 'needs founder explanation'. "
        f"No prose, no preamble. Just the hypothesis sentence."
    )

    result_holder: list[str] = []
    done_event = asyncio.Event()

    async def _on_finish(task_result: dict) -> None:
        result_holder.append(
            task_result.get("output") or task_result.get("result") or ""
        )
        done_event.set()

    try:
        await _enqueue_overhead(
            {
                "title": f"investor_bullets:hypothesis:{metric_name}",
                "description": f"One-sentence anomaly hypothesis for {metric_name}.",
                "agent_type": "assistant",
                "kind": "overhead",
                "context": {
                    "prompt": prompt,
                    "_callback": _on_finish,
                },
            },
            lane=LANE_OVERHEAD,
        )
        await asyncio.wait_for(done_event.wait(), timeout=20.0)
    except asyncio.TimeoutError:
        logger.warning(
            "investor_bullets: LLM hypothesis timed out",
            metric=metric_name,
        )
        return ""
    except Exception as exc:
        logger.warning(
            "investor_bullets: LLM hypothesis failed",
            metric=metric_name,
            error=str(exc),
        )
        return ""

    return result_holder[0].strip() if result_holder else ""


# ---------------------------------------------------------------------------
# Data-source fetchers (each degrades independently)
# ---------------------------------------------------------------------------


async def _fetch_z6_metrics(product_id: str) -> dict:
    """Read Z6 Stripe metrics: revenue, MRR, MRR delta, churn, customer count.

    Returns empty dict when Z6 Stripe integration is absent (degrade).
    """
    from src.infra.db import get_db
    db = await get_db()
    # Z6 growth_events: look for metric_emit rows for this product
    try:
        cur = await db.execute(
            "SELECT properties_json, occurred_at FROM growth_events "
            "WHERE kind = 'metric_emit' "
            "ORDER BY occurred_at DESC LIMIT 90",
        )
        rows = await cur.fetchall()
    except Exception:
        return {}

    metrics: dict[str, list[float]] = {}
    for row in rows:
        try:
            props = json.loads(row[0] or "{}")
            for key in ("mrr", "revenue", "churn_rate", "customer_count",
                        "burn", "runway_months", "mrr_delta"):
                if key in props and props[key] is not None:
                    metrics.setdefault(key, []).append(float(props[key]))
        except Exception:
            continue

    # Latest value + trailing history
    result: dict = {}
    for key, vals in metrics.items():
        if vals:
            result[key] = {"current": vals[0], "history": vals[1:4]}
    return result


async def _fetch_ops_metrics(product_id: str) -> dict:
    """Read Z8 ops metrics from the incidents table.

    Returns incident_count for current month vs prior 3 months.
    Degrades to empty dict if incidents table is absent.
    """
    from src.infra.db import get_db
    db = await get_db()
    try:
        # Count incidents opened this month
        cur = await db.execute(
            "SELECT COUNT(*) FROM incidents "
            "WHERE product_id = ? "
            "AND opened_at >= strftime('%Y-%m-01', 'now')",
            (product_id,),
        )
        row = await cur.fetchone()
        current_incidents = int(row[0]) if row else 0

        # Count for previous 3 months (rough trailing history)
        history_counts: list[float] = []
        for offset in (1, 2, 3):
            cur2 = await db.execute(
                "SELECT COUNT(*) FROM incidents "
                "WHERE product_id = ? "
                "AND opened_at >= strftime('%Y-%m-01', 'now', ? || ' months') "
                "AND opened_at < strftime('%Y-%m-01', 'now', ? || ' months')",
                (product_id, f"-{offset + 1}", f"-{offset}"),
            )
            row2 = await cur2.fetchone()
            history_counts.append(float(row2[0]) if row2 else 0.0)

        return {
            "incident_count": {
                "current": float(current_incidents),
                "history": history_counts,
            }
        }
    except Exception as exc:
        logger.debug("investor_bullets: ops metrics unavailable", error=str(exc))
        return {}


async def _fetch_review_density(product_id: str) -> dict:
    """Read Z3 review density: PRs shipped, security issues.

    Degrades to empty dict when Z3 metrics are absent.
    """
    from src.infra.db import get_db
    db = await get_db()
    try:
        # mission_events tagged with 'prs_shipped' or 'security_issues'
        cur = await db.execute(
            "SELECT properties_json FROM growth_events "
            "WHERE kind = 'review_density_metric' "
            "ORDER BY occurred_at DESC LIMIT 12",
        )
        rows = await cur.fetchall()
        if not rows:
            return {}

        prs: list[float] = []
        sec: list[float] = []
        for row in rows:
            try:
                props = json.loads(row[0] or "{}")
                if "prs_shipped" in props:
                    prs.append(float(props["prs_shipped"]))
                if "security_issues" in props:
                    sec.append(float(props["security_issues"]))
            except Exception:
                continue

        result: dict = {}
        if prs:
            result["prs_shipped"] = {"current": prs[0], "history": prs[1:4]}
        if sec:
            result["security_issues"] = {"current": sec[0], "history": sec[1:4]}
        return result
    except Exception as exc:
        logger.debug("investor_bullets: review density unavailable", error=str(exc))
        return {}


async def _fetch_support_metrics(product_id: str) -> dict:
    """Read A8 support metrics from the tickets table.

    Returns support_volume and escalation_rate for current vs trailing 3 months.
    """
    from src.infra.db import get_db
    db = await get_db()
    try:
        # Current month
        cur = await db.execute(
            "SELECT COUNT(*), SUM(escalated_to_founder) FROM tickets "
            "WHERE created_at >= strftime('%Y-%m-01', 'now')",
        )
        row = await cur.fetchone()
        total = int(row[0]) if row else 0
        escalated = int(row[1] or 0) if row else 0

        vol_history: list[float] = []
        esc_history: list[float] = []
        for offset in (1, 2, 3):
            cur2 = await db.execute(
                "SELECT COUNT(*), SUM(escalated_to_founder) FROM tickets "
                "WHERE created_at >= strftime('%Y-%m-01', 'now', ? || ' months') "
                "AND created_at < strftime('%Y-%m-01', 'now', ? || ' months')",
                (f"-{offset + 1}", f"-{offset}"),
            )
            row2 = await cur2.fetchone()
            vol = float(row2[0]) if row2 else 0.0
            esc = float(row2[1] or 0) if row2 else 0.0
            vol_history.append(vol)
            esc_history.append(esc / vol if vol > 0 else 0.0)

        esc_rate = escalated / total if total > 0 else 0.0
        return {
            "support_volume": {"current": float(total), "history": vol_history},
            "escalation_rate": {"current": esc_rate, "history": esc_history},
        }
    except Exception as exc:
        logger.debug("investor_bullets: support metrics unavailable", error=str(exc))
        return {}


async def _fetch_mention_counts(product_id: str) -> dict:
    """Read A11 mention monitor counts.

    A11 is not built yet → always degrade to empty dict (graceful).
    """
    # A11 table (mentions) may not exist or may have no rows yet.
    from src.infra.db import get_db
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT COUNT(*) FROM mentions "
            "WHERE product_id = ? "
            "AND detected_at >= strftime('%Y-%m-01', 'now')",
            (product_id,),
        )
        row = await cur.fetchone()
        count = int(row[0]) if row else 0
        return {"mention_count": {"current": float(count), "history": []}}
    except Exception:
        # Table absent → degrade silently
        return {}


async def collect_metrics(product_id: str) -> tuple[dict, list[str]]:
    """Collect all metrics from all sources; return (metrics_dict, missing_sources).

    Each source is tried independently. Failures are collected in missing_sources
    without raising. metrics_dict is keyed by metric name, values are
    ``{"current": float, "history": [float, ...]}``.
    """
    metrics: dict = {}
    missing: list[str] = []

    async def _safe(name: str, coro) -> dict:
        try:
            return await coro
        except Exception as exc:
            logger.warning(
                "investor_bullets: source unavailable",
                source=name,
                error=str(exc),
            )
            missing.append(name)
            return {}

    results = await asyncio.gather(
        _safe("z6", _fetch_z6_metrics(product_id)),
        _safe("ops", _fetch_ops_metrics(product_id)),
        _safe("review_density", _fetch_review_density(product_id)),
        _safe("support", _fetch_support_metrics(product_id)),
        _safe("mentions", _fetch_mention_counts(product_id)),
        return_exceptions=False,
    )

    for source_metrics in results:
        metrics.update(source_metrics)

    return metrics, missing


# ---------------------------------------------------------------------------
# Render bullets — structured Markdown only
# ---------------------------------------------------------------------------


# Metrics where "up" is bad (for Highlights vs Lowlights classification)
_METRICS_WHERE_UP_IS_BAD = frozenset({
    "churn_rate",
    "burn",
    "escalation_rate",
    "incident_count",
    "security_issues",
    "p95_latency_ms",
})


def _classify_anomaly(metric_name: str, direction: str) -> str:
    """Return 'positive' or 'negative' given the metric name + direction."""
    if direction == "none":
        return "neutral"
    if metric_name in _METRICS_WHERE_UP_IS_BAD:
        return "negative" if direction == "up" else "positive"
    return "positive" if direction == "up" else "negative"


async def render_bullets(
    metrics: dict,
    hypotheses: dict,
    gaps: list[str],
) -> str:
    """Produce structured Markdown bullets from metrics + anomaly hypotheses.

    Sections (in order):
      ## Highlights
      ## Lowlights
      ## Numbers
      ## Anomalies needing founder explanation
      ## Suggested asks

    Parameters
    ----------
    metrics:
        Dict of ``{metric_name: {"current": float, "history": [float, ...]}}``.
    hypotheses:
        Dict of ``{metric_name: "one-sentence hypothesis"}``.
    gaps:
        List of needs_external_help strings from mission_lessons (can be empty).
    """
    # Run anomaly detection on all metrics
    anomalies: list[dict] = []
    for name, data in metrics.items():
        current = data.get("current", 0.0)
        history = data.get("history", [])
        result = _detect_anomaly(name, current, history)
        if result["is_anomaly"]:
            sentiment = _classify_anomaly(name, result["direction"])
            anomalies.append({
                "name": name,
                "current": current,
                "direction": result["direction"],
                "sigma": result["sigma"],
                "sentiment": sentiment,
                "hypothesis": hypotheses.get(name, ""),
            })

    # Sort: positives by sigma desc, negatives by |sigma| desc
    positives = sorted(
        [a for a in anomalies if a["sentiment"] == "positive"],
        key=lambda a: abs(a["sigma"]),
        reverse=True,
    )
    negatives = sorted(
        [a for a in anomalies if a["sentiment"] == "negative"],
        key=lambda a: abs(a["sigma"]),
        reverse=True,
    )

    lines: list[str] = []

    # ── Highlights ──────────────────────────────────────────────────────────
    lines.append("## Highlights")
    if positives:
        for a in positives[:3]:
            direction_str = "above" if a["direction"] == "up" else "below"
            lines.append(
                f"- **{a['name']}** {a['current']} "
                f"({a['sigma']:+.1f}σ {direction_str} 3-month median)"
            )
    else:
        lines.append("- No positive outliers this month.")
    lines.append("")

    # ── Lowlights ───────────────────────────────────────────────────────────
    lines.append("## Lowlights")
    if negatives:
        for a in negatives[:3]:
            direction_str = "above" if a["direction"] == "up" else "below"
            lines.append(
                f"- **{a['name']}** {a['current']} "
                f"({a['sigma']:+.1f}σ {direction_str} 3-month median)"
            )
    else:
        lines.append("- No negative outliers this month.")
    lines.append("")

    # ── Numbers ─────────────────────────────────────────────────────────────
    lines.append("## Numbers")
    lines.append("")
    if metrics:
        lines.append("| Metric | Value | 3-mo Median |")
        lines.append("|---|---|---|")
        for name, data in sorted(metrics.items()):
            current = data.get("current", "N/A")
            history = data.get("history", [])
            if len(history) >= 2:
                med = round(statistics.median(history), 4)
            else:
                med = "N/A"
            lines.append(f"| {name} | {current} | {med} |")
    else:
        lines.append("_No metrics available this month._")
    lines.append("")

    # ── Anomalies ────────────────────────────────────────────────────────────
    lines.append("## Anomalies needing founder explanation")
    top_anomalies = (positives + negatives)[:5]
    if top_anomalies:
        for a in top_anomalies:
            hyp = a["hypothesis"] or "_needs founder explanation_"
            lines.append(f"- **{a['name']}** ({a['sigma']:+.1f}σ): {hyp}")
    else:
        lines.append("- No significant anomalies this month.")
    lines.append("")

    # ── Suggested asks ───────────────────────────────────────────────────────
    lines.append("## Suggested asks")
    if gaps:
        for gap in gaps[:5]:
            lines.append(f"- {gap}")
    else:
        lines.append("- _No external-help gaps flagged in mission lessons this month._")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A9.r1 — Segmented template variants
# ---------------------------------------------------------------------------


def emit_segmented_variants(
    bullets_md: str,
    contacts: list[dict],
) -> list[dict]:
    """Emit 3 template variants when any CRM contacts exist.

    Parameters
    ----------
    bullets_md:
        The core Markdown bullets (output of render_bullets).
    contacts:
        List of contact dicts from list_contacts (may include investor/advisor).

    Returns
    -------
    List of dicts: ``[{"template_kind": str, "content_md": str}, ...]``.
    All 3 variants always emitted when contacts list is non-empty.
    Empty list when no contacts exist.
    """
    if not contacts:
        return []

    variants: list[dict] = []
    for template_kind, preamble in _TEMPLATE_PREAMBLES.items():
        content_md = preamble + bullets_md
        variants.append({
            "template_kind": template_kind,
            "content_md": content_md,
        })

    return variants


# ---------------------------------------------------------------------------
# founder_action surface (monkeypatchable)
# ---------------------------------------------------------------------------


async def _create_founder_action(**kwargs) -> Any:
    """Thin wrapper around ``src.founder_actions.create`` (monkeypatchable)."""
    from src.founder_actions import create as _create
    return await _create(**kwargs)


async def _list_contacts(product_id: str) -> list[dict]:
    """Thin wrapper around ``src.app.crm.list_contacts`` (monkeypatchable)."""
    try:
        from src.app.crm import list_contacts
        return await list_contacts(product_id)
    except Exception as exc:
        logger.warning("investor_bullets: list_contacts failed", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# Suggested asks — mission_lessons query
# ---------------------------------------------------------------------------


async def _fetch_gaps(product_id: str) -> list[str]:
    """Return needs_external_help items from mission_lessons (degrade if absent)."""
    from src.infra.db import get_db
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT pattern, fix FROM mission_lessons "
            "WHERE tags_json LIKE '%needs_external_help%' "
            "ORDER BY created_at DESC LIMIT 5",
        )
        rows = await cur.fetchall()
        return [f"{r[0]}: {r[1]}" for r in rows if r[0]]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_investor_bullets(
    product_id: str = "default",
    *,
    mission_id: int = 0,
) -> dict:
    """Monthly investor bullets entry point. Called by mr_roboto for
    ``investor_bullets`` executor.

    Returns ``{"ok": True, "variants": N}`` on success.
    """
    try:
        # 1. Collect metrics
        metrics, missing = await collect_metrics(product_id)
        logger.info(
            "investor_bullets: metrics collected",
            product_id=product_id,
            metric_count=len(metrics),
            missing_sources=missing,
        )

        # 2. Anomaly detection + LLM hypotheses for outliers
        hypotheses: dict[str, str] = {}
        anomaly_items: list[tuple[str, float, list[float]]] = []
        for name, data in metrics.items():
            current = data.get("current", 0.0)
            history = data.get("history", [])
            res = _detect_anomaly(name, current, history)
            if res["is_anomaly"]:
                anomaly_items.append((name, current, history))

        # Call LLM for each anomaly (OVERHEAD lane, sequential — cheap model)
        for name, current, history in anomaly_items[:5]:  # cap at 5 LLM calls
            hyp = await _call_llm_anomaly_hypothesis(name, current, history)
            if hyp:
                hypotheses[name] = hyp

        # 3. Fetch suggested asks from mission_lessons
        gaps = await _fetch_gaps(product_id)

        # 4. Render bullets
        bullets_md = await render_bullets(metrics, hypotheses, gaps)

        # 5. A9.r1 — segmented variants
        contacts = await _list_contacts(product_id)
        variants = emit_segmented_variants(bullets_md, contacts)

        logger.info(
            "investor_bullets: variants emitted",
            product_id=product_id,
            variant_count=len(variants),
        )

        # 6. Surface as founder_action (copy-to-clipboard only — NO send button)
        context_payload = {
            "product_id": product_id,
            "bullets_md": bullets_md,
            "variants": variants,
            "missing_sources": missing,
            "anomaly_count": len(anomaly_items),
            "_investor_bullets": True,
        }

        instructions = [
            "Review the bullet points below for accuracy before use.",
            "Edit any anomaly hypotheses — you know the context better than the system.",
            "Copy the variant that matches your recipient category.",
            "DO NOT auto-send — these bullets are your starting point, not a finished update.",
        ]

        await _create_founder_action(
            mission_id=mission_id,
            kind="generic",
            title="Review monthly investor bullets",
            why=(
                "Monthly investor-bullet digest ready. "
                f"{len(anomaly_items)} metric anomal{'y' if len(anomaly_items) == 1 else 'ies'} "
                f"flagged vs 3-month baseline. "
                f"{len(variants)} segmented template variant{'s' if len(variants) != 1 else ''} "
                f"generated (investor / advisor). "
                "Copy your variant — the prose is yours to write."
            ),
            instructions=instructions,
            expected_output_kind="ack_only",
            notify_telegram=True,
            context_json=json.dumps(context_payload),
        )

        return {"ok": True, "variants": len(variants), "anomalies": len(anomaly_items)}

    except Exception as exc:
        logger.error("investor_bullets: failed", product_id=product_id, error=str(exc))
        return {"ok": False, "reason": str(exc)}
