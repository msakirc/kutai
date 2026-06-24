"""Z8 T5D — cost_pull mechanical executor.

Pulls daily spend from a vendor (stripe / vercel / aws) by delegating
to the ``vendor_call`` executor and aggregating the result. The
aggregated USD total is returned for the caller to feed into
``cost_anomaly.is_anomaly()``.

V1 vendors
----------
- **stripe** — sums ``amount`` from ``GET /v1/balance_transactions``
  (Stripe amounts are in cents).
- **vercel** — reads the top-level ``total`` from
  ``GET /v9/teams/{teamId}/usage``.
- **aws** — Cost Explorer. Skipped when AWS creds are not present
  (the integration's adapter check will refuse).

Payload
-------
```
{
    "action": "cost_pull",
    "vendor": "stripe" | "vercel" | "aws",
    "params": {...},   # vendor-specific (team_id, time_window, ...)
}
```

Returns ``{"ok": bool, "vendor": str, "today_usd": float|None,
"skipped": bool, "reason": str|None}``.
"""
from __future__ import annotations

from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.cost_pull")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    vendor = (payload.get("vendor") or "").lower()
    if vendor not in ("stripe", "vercel", "aws"):
        return {
            "ok": False,
            "vendor": vendor,
            "today_usd": None,
            "skipped": False,
            "reason": f"unsupported vendor: {vendor!r}",
        }

    action_map = {
        "stripe": "list_balance_transactions",
        "vercel": "team_usage",
        "aws": "cost_explorer",
    }
    vendor_action = action_map[vendor]
    params = payload.get("params") or {}

    # Delegate to vendor_call via its mechanical entry point.
    vc_task = {
        "context": {},
        "payload": {
            "service": vendor,
            "action": vendor_action,
            "params": params,
        },
    }
    try:
        from mr_roboto.executors.vendor_call import run as _vc_run
        vc_res = await _vc_run(vc_task)
    except Exception as e:
        return {
            "ok": False,
            "vendor": vendor,
            "today_usd": None,
            "skipped": False,
            "reason": f"vendor_call invocation failed: {e}",
        }

    if not isinstance(vc_res, dict):
        return {
            "ok": False,
            "vendor": vendor,
            "today_usd": None,
            "skipped": False,
            "reason": "vendor_call returned non-dict",
        }
    if not vc_res.get("ok"):
        # Adapter missing / cred missing — treat as skipped, not failure.
        reason = vc_res.get("reason") or "vendor_call failed"
        skipped = "adapter" in reason or "credential" in reason or "cred" in reason
        return {
            "ok": False,
            "vendor": vendor,
            "today_usd": None,
            "skipped": skipped,
            "reason": reason,
        }

    raw = vc_res.get("result") or {}
    try:
        today_usd = _aggregate(vendor, raw)
    except Exception as e:
        return {
            "ok": False,
            "vendor": vendor,
            "today_usd": None,
            "skipped": False,
            "reason": f"aggregation failed: {e}",
        }

    return {
        "ok": True,
        "vendor": vendor,
        "today_usd": today_usd,
        "skipped": False,
        "reason": None,
    }


def _aggregate(vendor: str, raw: dict) -> float:
    """Vendor-specific aggregation to a single USD figure."""
    if vendor == "stripe":
        # Stripe amounts are in cents; sum balance_transactions.
        data = raw.get("data") or raw.get("balance_transactions") or []
        total_cents = 0
        for tx in data:
            if isinstance(tx, dict):
                total_cents += int(tx.get("amount") or 0)
        return round(total_cents / 100.0, 2)
    if vendor == "vercel":
        # Vercel usage payload exposes a top-level total (cents or USD
        # depending on plan — treat as USD here, scale if Vercel later
        # standardises). Fallback to summing line items.
        if "total" in raw:
            return float(raw.get("total") or 0.0)
        items = raw.get("items") or raw.get("usage") or []
        total = 0.0
        for it in items:
            if isinstance(it, dict):
                total += float(it.get("cost") or it.get("price") or 0.0)
        return round(total, 2)
    if vendor == "aws":
        # Cost Explorer: { "ResultsByTime": [ { "Total": {"BlendedCost": {"Amount": "..."}}}] }
        results = raw.get("ResultsByTime") or []
        total = 0.0
        for r in results:
            if not isinstance(r, dict):
                continue
            cost = (r.get("Total") or {}).get("BlendedCost") or {}
            try:
                total += float(cost.get("Amount") or 0.0)
            except (TypeError, ValueError):
                continue
        return round(total, 2)
    return 0.0
