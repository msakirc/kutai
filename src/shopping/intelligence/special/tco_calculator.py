"""Total cost of ownership calculator.

Estimates the true cost of owning a product over its lifetime by
accounting for energy consumption, consumables, and maintenance --
not just the purchase price.
"""

from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.special.tco")

# ─── Turkey Electricity Rate (TL/kWh, 2024 avg residential) ────────────────
DEFAULT_KWH_PRICE = 2.83

# ─── Consumable Cost Estimates (TL/year) ────────────────────────────────────
# Category -> annual consumable cost estimate

_CONSUMABLE_ANNUAL: dict[str, float] = {
    "printer": 1500.0,        # toner / ink
    "vacuum_cleaner": 300.0,  # bags, filters
    "coffee_machine": 800.0,  # pods, descaler
    "air_purifier": 500.0,    # HEPA filters
    "water_purifier": 400.0,  # replacement filters
    "dishwasher": 600.0,      # salt, rinse aid, tablets
    "washing_machine": 500.0, # detergent, softener
    "dryer": 200.0,           # lint filters
    "shaver": 250.0,          # replacement heads
    "toothbrush": 300.0,      # replacement heads
    "robot_vacuum": 400.0,    # brushes, filters, mop pads
}

# ─── Maintenance Cost Estimates (TL/year) ───────────────────────────────────

_MAINTENANCE_ANNUAL: dict[str, float] = {
    "air_conditioner": 500.0,
    "dishwasher": 200.0,
    "washing_machine": 200.0,
    "car": 5000.0,
    "laptop": 300.0,
    "desktop": 200.0,
    "refrigerator": 150.0,
    "tv": 0.0,
    "phone": 0.0,
}


def calculate_tco(product: dict, years: int = 3) -> dict:
    """Calculate total cost of ownership over *years*.

    Parameters
    ----------
    product:
        Product dict with ``price``, and optionally ``watts``,
        ``daily_usage_hours``, ``category``.
    years:
        Ownership period in years.

    Returns
    -------
    Dict with ``purchase_price``, ``energy_cost``, ``consumable_cost``,
    ``maintenance_cost``, ``total_tco``, ``annual_tco``, ``breakdown``.
    """
    price = product.get("price", 0)
    watts = product.get("watts", 0)
    daily_hours = product.get("daily_usage_hours", _default_daily_hours(product))
    category = product.get("category", "").lower().replace(" ", "_")

    energy = estimate_energy_cost(watts, daily_hours, years) if watts > 0 else 0
    consumable = estimate_consumable_cost(category, years)
    maintenance = _MAINTENANCE_ANNUAL.get(category, 0) * years

    total = price + energy + consumable + maintenance

    return {
        "purchase_price": round(price, 2),
        "energy_cost": round(energy, 2),
        "consumable_cost": round(consumable, 2),
        "maintenance_cost": round(maintenance, 2),
        "total_tco": round(total, 2),
        "annual_tco": round(total / years, 2) if years > 0 else 0,
        "years": years,
        "breakdown": {
            "purchase_pct": round(price / total * 100, 1) if total > 0 else 100,
            "energy_pct": round(energy / total * 100, 1) if total > 0 else 0,
            "consumable_pct": round(consumable / total * 100, 1) if total > 0 else 0,
            "maintenance_pct": round(maintenance / total * 100, 1) if total > 0 else 0,
        },
    }


def estimate_energy_cost(
    watts: float,
    daily_hours: float,
    years: int,
    kwh_price: float = DEFAULT_KWH_PRICE,
) -> float:
    """Estimate electricity cost over *years* at Turkey residential rate.

    Parameters
    ----------
    watts:
        Power consumption in watts.
    daily_hours:
        Average hours of use per day.
    years:
        Ownership period.
    kwh_price:
        Electricity price in TL per kWh (default: current Turkish rate).

    Returns
    -------
    Total energy cost in TL.
    """
    daily_kwh = (watts / 1000) * daily_hours
    annual_kwh = daily_kwh * 365
    return annual_kwh * kwh_price * years


def estimate_consumable_cost(product_category: str, years: int) -> float:
    """Estimate consumable costs for a product category over *years*.

    Parameters
    ----------
    product_category:
        Category key (e.g. ``"printer"``, ``"vacuum_cleaner"``).
    years:
        Ownership period.

    Returns
    -------
    Total estimated consumable cost in TL.
    """
    annual = _CONSUMABLE_ANNUAL.get(product_category.lower().replace(" ", "_"), 0)
    return annual * years


def compare_tco(products: list[dict], years: int = 3) -> list[dict]:
    """Side-by-side TCO comparison of multiple products.

    Parameters
    ----------
    products:
        List of product dicts.
    years:
        Ownership period.

    Returns
    -------
    List of dicts, each with the product's TCO breakdown plus
    ``rank`` (1 = cheapest TCO) and ``savings_vs_worst`` fields.
    """
    results = []
    for product in products:
        tco = calculate_tco(product, years)
        tco["product_name"] = product.get("name", "Bilinmeyen urun")
        results.append(tco)

    # Sort by total TCO ascending
    results.sort(key=lambda r: r["total_tco"])

    if results:
        worst_tco = results[-1]["total_tco"]
        for rank, entry in enumerate(results, start=1):
            entry["rank"] = rank
            entry["savings_vs_worst"] = round(worst_tco - entry["total_tco"], 2)

    return results


def _default_daily_hours(product: dict) -> float:
    """Guess daily usage hours from product category."""
    category = product.get("category", "").lower()
    defaults = {
        "refrigerator": 24.0,
        "tv": 5.0,
        "laptop": 6.0,
        "desktop": 8.0,
        "air_conditioner": 8.0,
        "washing_machine": 1.0,
        "dishwasher": 1.5,
        "phone": 4.0,
        "monitor": 8.0,
        "router": 24.0,
        "light": 6.0,
    }
    return defaults.get(category.replace(" ", "_"), 2.0)
