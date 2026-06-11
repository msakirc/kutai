"""Z0 minimal slice — mission preflight writer.

The full z0 zone (founder profile, vault, north-star, kill-switch contract,
idle handling, etc.) is a separate multi-feature scope; see
`docs/i2p-evolution/z0-mission-preflight.md`.

This module ships the **minimal contract** that downstream Z1 gates need:

* `ambition_tier` ∈ {prototype, private_beta, public_launch, revenue_product}
  — drives default attention budget + severity-gate strictness.
* `cost_ceiling_usd` — enforced by 06-real-world-bridge vendor_call.
* `attention_budget_minutes` — z0 default by tier when founder skips.
* `jurisdictions`, `user_classes` — coarse hand-off to 0.4a refinement.

Writes `mission_<id>/.preflight/mission_preflight.json` AND sets the
corresponding columns on `missions`. Idempotent: re-running overwrites.
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.z0_preflight")


_DEFAULT_ATTENTION_BY_TIER = {
    "prototype": 120,
    "private_beta": 240,
    "public_launch": 480,
    "revenue_product": None,  # unbounded
}

VALID_TIERS = tuple(_DEFAULT_ATTENTION_BY_TIER.keys())


def default_attention_minutes(tier: str | None) -> int | None:
    """Return the spec'd attention default for a tier (None = unbounded)."""
    if tier in _DEFAULT_ATTENTION_BY_TIER:
        return _DEFAULT_ATTENTION_BY_TIER[tier]
    # Unknown / undeclared tier → use private_beta default (matches
    # attention_check's "no overlay" fallback semantics).
    return _DEFAULT_ATTENTION_BY_TIER["private_beta"]


async def z0_preflight_write(
    *,
    mission_id: int,
    ambition_tier: str | None = None,
    cost_ceiling_usd: float | None = None,
    attention_budget_minutes: int | None = None,
    jurisdictions: list[str] | None = None,
    user_classes: list[str] | None = None,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Write mission preflight JSON + update missions row.

    All fields optional. When ``ambition_tier`` is set and
    ``attention_budget_minutes`` is None, the default for the tier is used.
    """
    if ambition_tier is not None and ambition_tier not in VALID_TIERS:
        return {
            "ok": False,
            "error": (
                f"ambition_tier must be one of {VALID_TIERS}, got "
                f"{ambition_tier!r}"
            ),
        }

    # Defaulting: if tier is set but attention budget isn't, apply tier default.
    effective_attention = attention_budget_minutes
    if effective_attention is None and ambition_tier is not None:
        effective_attention = default_attention_minutes(ambition_tier)

    payload = {
        "_schema_version": "1",
        "mission_id": int(mission_id),
        "ambition_tier": ambition_tier,
        "cost_ceiling_usd": cost_ceiling_usd,
        "attention_budget_minutes": effective_attention,
        "jurisdictions": jurisdictions or [],
        "user_classes": user_classes or [],
    }

    # Workspace path.
    if workspace_path is None:
        from src.tools.workspace import get_mission_workspace
        workspace_path = get_mission_workspace(int(mission_id))
    preflight_dir = os.path.join(workspace_path, ".preflight")
    os.makedirs(preflight_dir, exist_ok=True)
    out_path = os.path.join(preflight_dir, "mission_preflight.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    # Persist DB columns. Best-effort: file write is authoritative; DB
    # update failures log + don't block.
    # Persist DB columns via the beckman write API.
    # z0_preflight previously built a dynamic SET clause; we now pass only
    # the non-None values as keyword arguments — update_mission_fields
    # validates every key against its whitelist (ambition_tier,
    # cost_ceiling_usd, founder_attention_budget_minutes are all whitelisted).
    try:
        from general_beckman import update_mission_fields as _umf
        _fields: dict = {}
        if ambition_tier is not None:
            _fields["ambition_tier"] = ambition_tier
        if cost_ceiling_usd is not None:
            _fields["cost_ceiling_usd"] = float(cost_ceiling_usd)
        if effective_attention is not None:
            _fields["founder_attention_budget_minutes"] = int(effective_attention)
        if _fields:
            await _umf(int(mission_id), **_fields)
    except Exception as exc:
        logger.warning("z0_preflight_write: DB update failed: %s", exc)

    return {"ok": True, "preflight_path": out_path, "payload": payload}
