"""Z10 T2A D8 — vendor cost adapter.

Thin re-export of ``src.infra.db.record_vendor_cost`` so vendor / provider
adapters (06-real-world-bridge) call into a stable surface owned by
kuleden_donen_var. The DB implementation owns the schema; this module
owns the public contract.
"""
from __future__ import annotations


async def record_vendor_cost(
    mission_id: int,
    vendor: str,
    usd: float,
    line_item: str,
) -> None:
    """Append vendor cost to ``cost_budgets`` scoped ``vendor:{vendor}``.

    See ``src.infra.db.record_vendor_cost`` for storage semantics. T2A
    keeps this as a re-export so vendor adapters (when 06 lands) hold a
    stable import path independent of db.py internals.
    """
    from dabidabi import record_vendor_cost as _impl
    await _impl(mission_id, vendor, usd, line_item)


__all__ = ["record_vendor_cost"]
