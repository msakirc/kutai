"""Constraint checking module for the shopping intelligence system.

Validates products against user constraints: dimensional fit, compatibility,
electrical requirements, budget limits, and availability.
"""

from __future__ import annotations

import re

from src.infra.logging_config import get_logger
from src.shopping.models import Product, UserConstraint
from src.shopping.text_utils import (
    extract_dimensions,
    extract_weight,
    extract_energy_rating,
)

logger = get_logger("shopping.intelligence.constraints")

from ._llm import _llm_call

# ── Compatibility knowledge base ─────────────────────────────────────────────

_SOCKET_COMPATIBILITY: dict[str, list[str]] = {
    "LGA1700": ["Intel 12th Gen", "Intel 13th Gen", "Intel 14th Gen"],
    "LGA1200": ["Intel 10th Gen", "Intel 11th Gen"],
    "AM5": ["AMD Ryzen 7000", "AMD Ryzen 9000"],
    "AM4": ["AMD Ryzen 3000", "AMD Ryzen 5000"],
}

_RAM_COMPATIBILITY: dict[str, list[str]] = {
    "DDR5": ["LGA1700", "AM5"],
    "DDR4": ["LGA1700", "LGA1200", "AM4", "AM5"],
}

_VOLTAGE_STANDARDS: dict[str, float] = {
    "TR": 220.0,  # Turkey
    "EU": 230.0,
    "US": 120.0,
}


# ── Individual constraint checkers ───────────────────────────────────────────

def _check_dimensional(product: Product, constraint: UserConstraint) -> dict:
    """Check if a product fits dimensional constraints."""
    result = {
        "product_name": product.name,
        "constraint_type": "dimensional",
        "passes": True,
        "notes": [],
    }

    # Try to extract dimensions from specs or product name
    dim_text = product.specs.get("dimensions", "") or product.name
    dims = extract_dimensions(dim_text)

    if not dims:
        result["notes"].append("Boyut bilgisi bulunamadı, kontrol edilemedi")
        result["passes"] = True  # can't verify, don't exclude
        return result

    # Parse the constraint value, e.g. "width<60cm" or "max_width:60"
    value_str = constraint.value.lower()

    for dim_name in ("width", "depth", "height"):
        # Pattern: dimension < number or dimension: number
        pattern = re.compile(
            rf"{dim_name}\s*[<:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE
        )
        m = pattern.search(value_str)
        if m and dim_name in dims:
            limit = float(m.group(1))
            actual = dims[dim_name]
            if actual > limit:
                result["passes"] = False
                dim_tr = {"width": "genişlik", "depth": "derinlik", "height": "yükseklik"}
                result["notes"].append(
                    f"{dim_tr.get(dim_name, dim_name)}: {actual:.1f}cm > limit {limit:.1f}cm"
                )
            else:
                result["notes"].append(
                    f"{dim_name}: {actual:.1f}cm <= {limit:.1f}cm (OK)"
                )

    return result


def _check_budget(product: Product, constraint: UserConstraint) -> dict:
    """Check if product is within budget."""
    result = {
        "product_name": product.name,
        "constraint_type": "budget",
        "passes": True,
        "notes": [],
    }

    price = product.discounted_price or product.original_price
    if price is None:
        result["notes"].append("Fiyat bilgisi yok")
        return result

    try:
        budget = float(
            re.sub(r"[^\d.,]", "", constraint.value).replace(",", ".")
        )
    except (ValueError, TypeError):
        result["notes"].append(f"Bütçe değeri ayrıştırılamadı: {constraint.value}")
        return result

    total = price + (product.shipping_cost or 0.0)

    if total > budget:
        result["passes"] = False
        result["notes"].append(
            f"Toplam fiyat {total:.2f} TL > bütçe {budget:.2f} TL"
        )
        if constraint.hard_or_soft == "soft":
            overshoot = ((total - budget) / budget) * 100
            result["notes"].append(f"Bütçe aşımı: %{overshoot:.0f}")
            if overshoot <= 10:
                result["passes"] = True
                result["notes"].append("Yumuşak kısıt: %10 tolerans dahilinde")
    else:
        result["notes"].append(f"Fiyat {total:.2f} TL <= bütçe {budget:.2f} TL (OK)")

    return result


def _check_electrical(product: Product, constraint: UserConstraint) -> dict:
    """Check electrical compatibility (voltage, wattage)."""
    result = {
        "product_name": product.name,
        "constraint_type": "electrical",
        "passes": True,
        "notes": [],
    }

    value_lower = constraint.value.lower()
    specs = product.specs

    # Voltage check
    if "volt" in value_lower or "gerilim" in value_lower:
        product_voltage = specs.get("voltage") or specs.get("gerilim")
        if product_voltage:
            try:
                pv = float(re.sub(r"[^\d.]", "", str(product_voltage)))
                if abs(pv - _VOLTAGE_STANDARDS["TR"]) > 20:
                    result["passes"] = False
                    result["notes"].append(
                        f"Gerilim {pv}V, Türkiye standardı 220V ile uyumsuz"
                    )
            except ValueError:
                pass

    # Wattage check
    watt_match = re.search(r"(\d+)\s*(?:watt|w)\b", value_lower)
    if watt_match:
        max_watt = float(watt_match.group(1))
        product_watt = specs.get("wattage") or specs.get("güç")
        if product_watt:
            try:
                pw = float(re.sub(r"[^\d.]", "", str(product_watt)))
                if pw > max_watt:
                    result["passes"] = False
                    result["notes"].append(f"Güç {pw}W > limit {max_watt}W")
            except ValueError:
                pass

    return result


def _check_compatibility(product: Product, constraint: UserConstraint) -> dict:
    """Check component compatibility (CPU socket, RAM type, etc.)."""
    result = {
        "product_name": product.name,
        "constraint_type": "compatibility",
        "passes": True,
        "notes": [],
    }

    value_upper = constraint.value.upper()
    specs = product.specs
    name_upper = product.name.upper()

    # Socket compatibility
    for socket, cpus in _SOCKET_COMPATIBILITY.items():
        if socket in value_upper:
            product_socket = (
                str(specs.get("socket", "")).upper()
                or str(specs.get("soket", "")).upper()
            )
            if product_socket and socket not in product_socket:
                result["passes"] = False
                result["notes"].append(
                    f"Soket uyumsuzluğu: ürün {product_socket}, gerekli {socket}"
                )

    # RAM compatibility
    for ram_type, sockets in _RAM_COMPATIBILITY.items():
        if ram_type in value_upper or ram_type in name_upper:
            product_ram = str(specs.get("ram_type", "")).upper()
            if product_ram and ram_type not in product_ram:
                result["passes"] = False
                result["notes"].append(
                    f"RAM uyumsuzluğu: ürün {product_ram}, gerekli {ram_type}"
                )

    return result


def _check_availability(product: Product, constraint: UserConstraint) -> dict:
    """Check product availability."""
    result = {
        "product_name": product.name,
        "constraint_type": "availability",
        "passes": True,
        "notes": [],
    }

    if product.availability == "out_of_stock":
        result["passes"] = False
        result["notes"].append("Stokta yok")
    elif product.availability == "preorder":
        result["notes"].append("Ön sipariş aşamasında")
        if constraint.hard_or_soft == "hard":
            result["passes"] = False
    elif product.availability == "low_stock":
        result["notes"].append("Düşük stok")

    return result


# ── Constraint dispatcher ────────────────────────────────────────────────────

_CHECKERS: dict[str, type] = {
    "dimensional": _check_dimensional,
    "budget": _check_budget,
    "electrical": _check_electrical,
    "compatibility": _check_compatibility,
    "availability": _check_availability,
}


# ── Public API ───────────────────────────────────────────────────────────────

async def check_constraints(
    products: list[Product],
    constraints: list[UserConstraint],
) -> list[dict]:
    """Check a list of products against a list of user constraints.

    Parameters
    ----------
    products:
        Products to evaluate.
    constraints:
        User constraints to check against.  Each has a ``type`` field
        (dimensional, compatibility, electrical, budget, availability)
        and a ``value`` field.

    Returns
    -------
    List of result dicts, one per product, each with:
    - product_name: str
    - passes_all: bool (True if all hard constraints pass)
    - results: list of per-constraint check results
    - failed_hard: list of constraint types that failed as hard constraints
    - failed_soft: list of constraint types that failed as soft constraints
    """
    if not products:
        return []

    if not constraints:
        return [
            {
                "product_name": p.name,
                "passes_all": True,
                "results": [],
                "failed_hard": [],
                "failed_soft": [],
            }
            for p in products
        ]

    output: list[dict] = []

    for product in products:
        product_result = {
            "product_name": product.name,
            "passes_all": True,
            "results": [],
            "failed_hard": [],
            "failed_soft": [],
        }

        for constraint in constraints:
            checker = _CHECKERS.get(constraint.type)
            if checker is None:
                logger.debug(
                    "No checker for constraint type '%s', skipping", constraint.type
                )
                continue

            try:
                check_result = checker(product, constraint)
                product_result["results"].append(check_result)

                if not check_result["passes"]:
                    if constraint.hard_or_soft == "hard":
                        product_result["passes_all"] = False
                        product_result["failed_hard"].append(constraint.type)
                    else:
                        product_result["failed_soft"].append(constraint.type)

            except Exception as exc:
                logger.warning(
                    "Constraint check failed for '%s' (%s): %s",
                    product.name,
                    constraint.type,
                    exc,
                )
                product_result["results"].append({
                    "product_name": product.name,
                    "constraint_type": constraint.type,
                    "passes": True,  # don't exclude on error
                    "notes": [f"Kontrol hatası: {exc}"],
                })

        output.append(product_result)

    passed = sum(1 for r in output if r["passes_all"])
    logger.info(
        "Constraint check: %d/%d products pass all hard constraints",
        passed,
        len(output),
    )
    return output
