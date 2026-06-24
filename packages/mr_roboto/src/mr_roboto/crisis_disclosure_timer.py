"""Z7 T3E — B6: crisis/disclosure_timer mr_roboto verb.

Tier 3 only: runs every 6h after a Tier-3 crisis opens.
Jurisdiction-aware via compliance_overlay (defaults to GDPR 72h if unknown).
Emits an escalating founder_action reminder as the 72h window closes.

Payload::

    {
        "product_id":    "prod-abc",  # required
        "event_id":      42,          # required
        "mission_id":    1,           # optional
        "hours_elapsed": 12,          # hours since crisis_events.opened_at
        "jurisdiction":  "GDPR",      # optional; defaults to "GDPR"
        "tier":          3,           # optional; skip if < 3
    }

Returns::

    {
        "status": "ok" | "skipped",
        "hours_elapsed": int,
        "hours_remaining": int,
        "founder_action_id": int | None,
    }
"""
from __future__ import annotations

from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.crisis_disclosure_timer")

# Jurisdiction → notification window (hours)
_JURISDICTION_HOURS: dict[str, int] = {
    "GDPR": 72,
    "UK GDPR": 72,
    "PDPA": 72,        # Turkey
    "LGPD": 72,        # Brazil (approximate)
    "CCPA": 168,       # 7 days (approximate)
    "DEFAULT": 72,
}


async def _create_founder_action(**kwargs) -> Any:
    """Thin wrapper — mockable in tests."""
    from src.founder_actions import create as fa_create
    return await fa_create(**kwargs)


def _urgency_level(hours_elapsed: int, window_hours: int) -> tuple[bool, str]:
    """Return (urgent: bool, urgency_label: str) based on percentage of window used."""
    pct = hours_elapsed / window_hours if window_hours > 0 else 1.0
    if pct >= 0.833:  # >80% used (≥60h of 72h)
        return True, "CRITICAL"
    if pct >= 0.667:  # >66% used (≥48h of 72h)
        return True, "ELEVATED"
    return False, "ROUTINE"


async def run(payload: dict) -> dict:
    """Execute crisis/disclosure_timer.

    Skips when tier < 3. For Tier 3+, emits an escalating founder_action.
    """
    product_id = payload.get("product_id") or ""
    event_id = payload.get("event_id")
    mission_id = int(payload.get("mission_id") or 0)
    hours_elapsed = int(payload.get("hours_elapsed") or 0)
    jurisdiction = str(payload.get("jurisdiction") or "GDPR").upper()
    tier = int(payload.get("tier") or 3)  # default to 3 (timer is Tier-3 only)

    if not event_id:
        return {"status": "error", "error": "event_id is required"}

    # Only fire for Tier 3+
    if tier < 3:
        logger.info(
            "crisis_disclosure_timer: tier=%d < 3 — skipped",
            tier,
            event_id=event_id,
            product_id=product_id,
        )
        return {"status": "skipped", "reason": "tier < 3", "tier": tier}

    window_hours = _JURISDICTION_HOURS.get(jurisdiction, _JURISDICTION_HOURS["DEFAULT"])
    hours_remaining = max(0, window_hours - hours_elapsed)
    urgent, urgency_label = _urgency_level(hours_elapsed, window_hours)

    # Compose title and instructions
    title = (
        f"[{urgency_label}] {jurisdiction} 72h breach disclosure — "
        f"{hours_remaining}h remaining"
    )
    instructions = [
        f"Jurisdiction: {jurisdiction}. Disclosure window: {window_hours}h.",
        f"Hours elapsed: {hours_elapsed}h. Hours remaining: {hours_remaining}h.",
    ]

    if hours_remaining == 0:
        instructions.insert(0, "DEADLINE REACHED — regulator notification must be filed NOW.")
        instructions.append("Contact counsel immediately if not already done.")
        instructions.append(
            "If filing has been submitted, resolve this crisis event: "
            f"`/crisis resolve {product_id}`."
        )
    elif urgent:
        instructions.append(
            f"Regulator notice must be filed within {hours_remaining}h. "
            "Counsel must review the draft before submission."
        )
        instructions.append("See playbooks/crisis_comms_tier3.md for regulator notice template.")
    else:
        instructions.append(
            f"Next check-in in 6h. Ensure counsel is engaged and "
            "breach scope is fully documented."
        )
        instructions.append("See playbooks/crisis_comms_tier3.md for jurisdiction matrix.")

    fa_result = None
    fa_id = None
    try:
        fa_result = await _create_founder_action(
            mission_id=mission_id,
            kind="generic",
            title=title,
            why=(
                f"Tier 3 crisis open for product '{product_id}' (event #{event_id}). "
                f"{jurisdiction} mandates breach notification within {window_hours}h. "
                f"Currently {hours_elapsed}h elapsed."
            ),
            instructions=instructions,
            urgent=urgent,
            notify_telegram=True,
        )
        fa_id = int(getattr(fa_result, "id", 0) or 0)
    except Exception as exc:
        logger.warning(
            "crisis_disclosure_timer: founder_action create failed",
            event_id=event_id,
            error=str(exc),
        )

    logger.info(
        "crisis_disclosure_timer: reminder emitted",
        event_id=event_id,
        product_id=product_id,
        jurisdiction=jurisdiction,
        hours_elapsed=hours_elapsed,
        hours_remaining=hours_remaining,
        urgency=urgency_label,
        founder_action_id=fa_id,
    )

    return {
        "status": "ok",
        "hours_elapsed": hours_elapsed,
        "hours_remaining": hours_remaining,
        "urgency": urgency_label,
        "jurisdiction": jurisdiction,
        "founder_action_id": fa_id,
    }
