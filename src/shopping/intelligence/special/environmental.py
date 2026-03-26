"""Environmental / Efficiency Advisor.

Assesses energy efficiency, water efficiency, repairability, expected
lifespan, and total cost of ownership — framed entirely as money saved,
not abstract environmental benefits.
"""

from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.special.environmental")

# ─── Turkey Rates (2026) ─────────────────────────────────────────────────────
_KWH_PRICE_TRY = 4.5          # TRY per kWh, residential
_WATER_PRICE_TRY_M3 = 35.0    # TRY per m³, Istanbul

# ─── EU Energy Class → Estimated Annual kWh ──────────────────────────────────
_ENERGY_CLASS_KWH: dict[str, float] = {
    "A+++": 150.0,
    "A++":  200.0,
    "A+":   270.0,
    "A":    350.0,
    "B":    450.0,
    "C":    550.0,
    "D":    700.0,
}

# Reference class for savings comparison
_REFERENCE_CLASS = "C"
_REFERENCE_KWH = _ENERGY_CLASS_KWH[_REFERENCE_CLASS]

# ─── Category Expected Lifespans (years) ─────────────────────────────────────
_CATEGORY_LIFESPAN: dict[str, float] = {
    "buzdolabı":        15.0,
    "çamaşır makinesi": 10.0,
    "bulaşık makinesi": 10.0,
    "klima":            12.0,
    "tv":                8.0,
    "laptop":            5.0,
    "telefon":           3.0,
    "tablet":            4.0,
    "yazıcı":            5.0,
    "kahve makinesi":    5.0,
}

# Normalised aliases so callers can pass English or Turkish category names
_CATEGORY_ALIASES: dict[str, str] = {
    "refrigerator":     "buzdolabı",
    "fridge":           "buzdolabı",
    "washing machine":  "çamaşır makinesi",
    "washing_machine":  "çamaşır makinesi",
    "dishwasher":       "bulaşık makinesi",
    "air conditioner":  "klima",
    "air_conditioner":  "klima",
    "television":       "tv",
    "television set":   "tv",
    "notebook":         "laptop",
    "mobile":           "telefon",
    "phone":            "telefon",
    "smartphone":       "telefon",
    "printer":          "yazıcı",
    "coffee machine":   "kahve makinesi",
    "coffee maker":     "kahve makinesi",
}

# ─── Annual water consumption by category (m³/year) ──────────────────────────
# Only categories that consume water
_WATER_M3_PER_YEAR: dict[str, float] = {
    "çamaşır makinesi": 15.0,   # ~200 litres per wash × ~75 washes/year
    "bulaşık makinesi":  6.0,   # ~12 litres per cycle × ~500 cycles/year
}

# ─── Repairability reference data ────────────────────────────────────────────
# (repairability_score 0-10, spare_parts, service_network)
_REPAIRABILITY: dict[str, tuple[float, str, str]] = {
    "buzdolabı":        (7.0, "easy",     "yaygın"),
    "çamaşır makinesi": (7.5, "easy",     "yaygın"),
    "bulaşık makinesi": (6.5, "moderate", "yaygın"),
    "klima":            (7.0, "moderate", "yaygın"),
    "tv":               (5.0, "moderate", "sınırlı"),
    "laptop":           (4.5, "moderate", "sınırlı"),
    "telefon":          (3.5, "difficult", "sınırlı"),
    "tablet":           (3.0, "difficult", "sınırlı"),
    "yazıcı":           (5.5, "moderate", "sınırlı"),
    "kahve makinesi":   (6.0, "moderate", "yaygın"),
}

# ─── Annual running cost by category when energy class is unknown (TRY/year) ──
_DEFAULT_ANNUAL_RUNNING: dict[str, float] = {
    "buzdolabı":        round(_CATEGORY_LIFESPAN["buzdolabı"]        * 0 + 350 * _KWH_PRICE_TRY, 2),
    "çamaşır makinesi": round(200 * _KWH_PRICE_TRY + 15.0 * _WATER_PRICE_TRY_M3, 2),
    "bulaşık makinesi": round(250 * _KWH_PRICE_TRY + 6.0  * _WATER_PRICE_TRY_M3, 2),
    "klima":            round(800 * _KWH_PRICE_TRY, 2),
    "tv":               round(150 * _KWH_PRICE_TRY, 2),
    "laptop":           round(100 * _KWH_PRICE_TRY, 2),
    "telefon":          round(10  * _KWH_PRICE_TRY, 2),
    "tablet":           round(15  * _KWH_PRICE_TRY, 2),
    "yazıcı":           round(50  * _KWH_PRICE_TRY, 2),
    "kahve makinesi":   round(200 * _KWH_PRICE_TRY, 2),
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _normalise_category(raw: str) -> str:
    """Return canonical Turkish category name, or the lowercased original."""
    lowered = raw.strip().lower()
    return _CATEGORY_ALIASES.get(lowered, lowered)


def _parse_energy_class(product: dict) -> str | None:
    """Extract EU energy class string from product dict."""
    for key in ("energy_class", "enerji_sinifi", "energy_label", "energy_rating"):
        val = product.get(key)
        if val and isinstance(val, str):
            cleaned = val.strip().upper()
            if cleaned in _ENERGY_CLASS_KWH:
                return cleaned
    return None


def _efficiency_score_from_class(energy_class: str | None) -> float:
    """Map energy class to a 0-1 efficiency score."""
    order = ["D", "C", "B", "A", "A+", "A++", "A+++"]
    if energy_class is None or energy_class not in order:
        return 0.5  # unknown → neutral
    idx = order.index(energy_class)
    return round(idx / (len(order) - 1), 3)


# ─── Public API ──────────────────────────────────────────────────────────────

def assess_efficiency(product: dict) -> dict:
    """Assess energy and water efficiency of a product.

    Parameters
    ----------
    product:
        Product dict; recognised keys: ``energy_class``, ``category``,
        ``price``.

    Returns
    -------
    Dict with ``energy_class``, ``estimated_annual_energy_cost``,
    ``estimated_annual_water_cost``, ``efficiency_score``,
    ``cost_savings_note``.
    """
    logger.debug("assess_efficiency called for product: %s", product.get("name"))

    energy_class = _parse_energy_class(product)
    category = _normalise_category(product.get("category", ""))

    # Annual energy cost
    annual_energy_cost: float | None = None
    if energy_class is not None:
        annual_kwh = _ENERGY_CLASS_KWH[energy_class]
        annual_energy_cost = round(annual_kwh * _KWH_PRICE_TRY, 2)

    # Annual water cost (only for water-using appliances)
    annual_water_cost: float | None = None
    water_m3 = _WATER_M3_PER_YEAR.get(category)
    if water_m3 is not None:
        annual_water_cost = round(water_m3 * _WATER_PRICE_TRY_M3, 2)

    # Efficiency score
    efficiency_score = _efficiency_score_from_class(energy_class)

    # Cost savings note (Turkish, framed as money saved vs C-class reference)
    if energy_class is not None and energy_class != _REFERENCE_CLASS:
        reference_annual_cost = round(_REFERENCE_KWH * _KWH_PRICE_TRY, 2)
        savings = round(reference_annual_cost - (annual_energy_cost or 0), 2)
        if savings > 0:
            cost_savings_note = (
                f"{energy_class} enerji sınıfı sayesinde C sınıfı bir ürüne kıyasla "
                f"yılda yaklaşık {savings:,.0f} TRY elektrik tasarrufu sağlarsınız."
            )
        elif savings < 0:
            cost_savings_note = (
                f"Bu {energy_class} sınıfı ürün, C sınıfı bir ürüne kıyasla "
                f"yılda yaklaşık {abs(savings):,.0f} TRY daha fazla elektrik tüketir."
            )
        else:
            cost_savings_note = "Bu ürünün enerji tüketimi referans sınıfıyla aynı düzeydedir."
    elif annual_energy_cost is not None:
        cost_savings_note = (
            f"Yıllık tahmini elektrik maliyeti {annual_energy_cost:,.0f} TRY'dir."
        )
    else:
        cost_savings_note = (
            "Enerji sınıfı bilinmiyor; daha verimli bir model seçerek yılda "
            "yüzlerce TRY tasarruf edebilirsiniz."
        )

    return {
        "energy_class":                 energy_class,
        "estimated_annual_energy_cost": annual_energy_cost,
        "estimated_annual_water_cost":  annual_water_cost,
        "efficiency_score":             efficiency_score,
        "cost_savings_note":            cost_savings_note,
    }


def estimate_lifespan(product: dict) -> dict:
    """Estimate expected product lifespan and cost-per-year.

    Parameters
    ----------
    product:
        Product dict; recognised keys: ``category``, ``price``,
        ``brand``, ``warranty_years``.

    Returns
    -------
    Dict with ``expected_years``, ``category_average_years``,
    ``cost_per_year``, ``note``.
    """
    logger.debug("estimate_lifespan called for product: %s", product.get("name"))

    category = _normalise_category(product.get("category", ""))
    price: float | None = product.get("price")
    warranty_years: float = float(product.get("warranty_years", 2))

    category_avg = _CATEGORY_LIFESPAN.get(category, 5.0)

    # Adjust expected lifespan slightly based on warranty length
    # (longer warranty often signals build confidence)
    bonus = max(0.0, (warranty_years - 2) * 0.5)
    expected_years = round(min(category_avg + bonus, category_avg * 1.2), 1)

    cost_per_year: float | None = None
    if price is not None and expected_years > 0:
        cost_per_year = round(price / expected_years, 2)

    if cost_per_year is not None:
        note = (
            f"Bu ürünün ortalama kategorisi için beklenen ömür "
            f"{category_avg:.0f} yıldır. Tahmini ömrünüz {expected_years} yıl "
            f"olduğundan, her yıla {cost_per_year:,.0f} TRY satın alma maliyeti düşer."
        )
    else:
        note = (
            f"Bu ürün kategorisinin ortalama ömrü {category_avg:.0f} yıldır "
            f"(tahmini: {expected_years} yıl). Fiyat bilgisi olmadan yıllık "
            "maliyet hesaplanamadı."
        )

    return {
        "expected_years":       expected_years,
        "category_average_years": category_avg,
        "cost_per_year":        cost_per_year,
        "note":                 note,
    }


def assess_repairability(product: dict) -> dict:
    """Assess how repairable a product is.

    Parameters
    ----------
    product:
        Product dict; recognised keys: ``category``, ``brand``.

    Returns
    -------
    Dict with ``repairability_score``, ``spare_parts_available``,
    ``service_network_tr``, ``note``.
    """
    logger.debug("assess_repairability called for product: %s", product.get("name"))

    category = _normalise_category(product.get("category", ""))

    if category in _REPAIRABILITY:
        score, parts, network = _REPAIRABILITY[category]
    else:
        score, parts, network = 5.0, "unknown", "sınırlı"

    # Build note
    parts_tr = {
        "easy":     "kolayca bulunabiliyor",
        "moderate": "kısmen bulunabiliyor",
        "difficult": "bulmak güç",
        "unknown":  "bilgi yok",
    }.get(parts, "bilgi yok")

    network_note = {
        "yaygın":  "Türkiye genelinde yaygın servis ağı mevcut.",
        "sınırlı": "Türkiye'de servis ağı sınırlı; şehir dışında sorun yaşanabilir.",
        "yok":     "Türkiye'de yetkili servis bulunmuyor; tamir oldukça zor.",
    }.get(network, "")

    note = (
        f"Tamir edilebilirlik puanı 10 üzerinden {score:.1f}. "
        f"Yedek parçalar {parts_tr}. {network_note} "
        "Kolay tamir edilebilen ürünler uzun vadede çok daha az para harcanmasını sağlar."
    )

    return {
        "repairability_score":   score,
        "spare_parts_available": parts,
        "service_network_tr":    network,
        "note":                  note,
    }


def get_lifetime_cost(product: dict) -> dict:
    """Calculate total cost of ownership over the product's expected lifetime.

    Parameters
    ----------
    product:
        Product dict; recognised keys: ``price``, ``category``,
        ``energy_class``, ``warranty_years``.

    Returns
    -------
    Dict with ``purchase_price``, ``annual_running_cost``,
    ``lifetime_years``, ``total_lifetime_cost``, ``cost_per_year``,
    ``summary``.
    """
    logger.debug("get_lifetime_cost called for product: %s", product.get("name"))

    price: float = float(product.get("price") or 0)
    category = _normalise_category(product.get("category", ""))

    lifespan_info = estimate_lifespan(product)
    lifetime_years = lifespan_info["expected_years"]

    efficiency_info = assess_efficiency(product)
    annual_energy = efficiency_info["estimated_annual_energy_cost"]
    annual_water = efficiency_info["estimated_annual_water_cost"] or 0.0

    # Fall back to category default when energy class is absent
    if annual_energy is None:
        annual_energy = _DEFAULT_ANNUAL_RUNNING.get(category, 0.0)

    annual_running_cost = round(annual_energy + annual_water, 2)
    total_lifetime_cost = round(price + annual_running_cost * lifetime_years, 2)
    cost_per_year = round(
        total_lifetime_cost / lifetime_years if lifetime_years > 0 else 0.0, 2
    )

    summary = (
        f"{lifetime_years:.0f} yıllık kullanım boyunca toplam sahip olma maliyeti "
        f"yaklaşık {total_lifetime_cost:,.0f} TRY'dir "
        f"({price:,.0f} TRY satın alma + yılda {annual_running_cost:,.0f} TRY işletme gideri). "
        f"Bu, yıl başına {cost_per_year:,.0f} TRY'ye karşılık gelir."
    )

    return {
        "purchase_price":       round(price, 2),
        "annual_running_cost":  annual_running_cost,
        "lifetime_years":       lifetime_years,
        "total_lifetime_cost":  total_lifetime_cost,
        "cost_per_year":        cost_per_year,
        "summary":              summary,
    }
