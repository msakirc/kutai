"""Warranty analysis for products.

Evaluates warranty coverage, official service availability, and
gray-market risk based on price deviation and seller information.
"""

from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.special.warranty")

# ─── Known Service Networks ─────────────────────────────────────────────────
# Brand -> {warranty_months, has_official_turkey_service, service_center_cities}

_SERVICE_NETWORKS: dict[str, dict] = {
    "apple": {
        "warranty_months": 24,
        "official_service": True,
        "service_centers": ["Istanbul", "Ankara", "Izmir", "Antalya", "Bursa"],
    },
    "samsung": {
        "warranty_months": 24,
        "official_service": True,
        "service_centers": ["Istanbul", "Ankara", "Izmir", "Antalya", "Bursa", "Adana", "Gaziantep"],
    },
    "lg": {
        "warranty_months": 24,
        "official_service": True,
        "service_centers": ["Istanbul", "Ankara", "Izmir", "Bursa"],
    },
    "sony": {
        "warranty_months": 24,
        "official_service": True,
        "service_centers": ["Istanbul", "Ankara", "Izmir"],
    },
    "bosch": {
        "warranty_months": 24,
        "official_service": True,
        "service_centers": ["Istanbul", "Ankara", "Izmir", "Antalya", "Bursa", "Adana"],
    },
    "arcelik": {
        "warranty_months": 36,
        "official_service": True,
        "service_centers": ["Istanbul", "Ankara", "Izmir", "Antalya", "Bursa", "Adana",
                            "Gaziantep", "Konya", "Kayseri", "Samsun", "Trabzon"],
    },
    "vestel": {
        "warranty_months": 36,
        "official_service": True,
        "service_centers": ["Istanbul", "Ankara", "Izmir", "Antalya", "Bursa", "Manisa"],
    },
    "hp": {
        "warranty_months": 24,
        "official_service": True,
        "service_centers": ["Istanbul", "Ankara", "Izmir"],
    },
    "lenovo": {
        "warranty_months": 24,
        "official_service": True,
        "service_centers": ["Istanbul", "Ankara"],
    },
    "dyson": {
        "warranty_months": 24,
        "official_service": True,
        "service_centers": ["Istanbul"],
    },
    "xiaomi": {
        "warranty_months": 24,
        "official_service": True,
        "service_centers": ["Istanbul", "Ankara", "Izmir"],
    },
    "philips": {
        "warranty_months": 24,
        "official_service": True,
        "service_centers": ["Istanbul", "Ankara", "Izmir", "Bursa"],
    },
}

# Default for unknown brands
_DEFAULT_NETWORK = {
    "warranty_months": 24,
    "official_service": False,
    "service_centers": [],
}


def analyze_warranty(product: dict, store: str) -> dict:
    """Analyse warranty coverage for a product at a given store.

    Parameters
    ----------
    product:
        Product dict with at least ``name`` and optionally ``brand``.
    store:
        Store name (e.g. ``"trendyol"``, ``"hepsiburada"``).

    Returns
    -------
    Dict with keys: ``warranty_months``, ``official_service``,
    ``gray_market_risk``, ``service_centers``, ``notes``.
    """
    brand = _extract_brand(product)
    network = get_service_network(brand)

    # Official stores have lower gray market risk
    official_stores = {"trendyol", "hepsiburada", "amazon_tr", "n11", "ciceksepeti"}
    is_official_store = store.lower() in official_stores

    price = product.get("price", 0)
    avg_price = product.get("avg_price", price)
    gray_risk = assess_gray_market_risk(product, price, avg_price)

    notes = []
    if not network["official_service"]:
        notes.append("Bu marka icin Turkiye'de resmi servis agirligi sinirli olabilir")
    if not is_official_store:
        notes.append("Urun resmi bir e-ticaret platformundan alinmiyor; garanti belgesini kontrol edin")
    if gray_risk > 0.5:
        notes.append("Fiyat ortalamanin cok altinda; ithalat/garanti durumunu sorgulayiniz")

    return {
        "warranty_months": network["warranty_months"],
        "official_service": network["official_service"],
        "gray_market_risk": round(gray_risk, 2),
        "service_centers": network["service_centers"],
        "store": store,
        "brand": brand,
        "notes": notes,
    }


def get_service_network(brand: str) -> dict:
    """Look up the service network for *brand*.

    Parameters
    ----------
    brand:
        Brand name (case-insensitive).

    Returns
    -------
    Dict with ``warranty_months``, ``official_service``, ``service_centers``.
    """
    return _SERVICE_NETWORKS.get(brand.lower().strip(), dict(_DEFAULT_NETWORK))


def assess_gray_market_risk(
    product: dict,
    price: float,
    avg_price: float,
) -> float:
    """Estimate gray-market risk on a 0--1 scale.

    A product significantly cheaper than the market average may lack
    official Turkish warranty or be an unofficial import.

    Parameters
    ----------
    product:
        Product dict (may contain ``seller_type`` or ``import`` hints).
    price:
        The listed price.
    avg_price:
        The average market price for this product.

    Returns
    -------
    Risk score between 0.0 (no risk) and 1.0 (very high risk).
    """
    if avg_price <= 0 or price <= 0:
        return 0.0

    risk = 0.0

    # Price significantly below average is suspicious
    price_ratio = price / avg_price
    if price_ratio < 0.70:
        risk += 0.5
    elif price_ratio < 0.80:
        risk += 0.3
    elif price_ratio < 0.90:
        risk += 0.1

    # Seller hints
    seller_type = product.get("seller_type", "").lower()
    if "ithalat" in seller_type or "import" in seller_type:
        risk += 0.3
    if "yurt disi" in seller_type or "abroad" in seller_type:
        risk += 0.2

    # Product description hints
    name = product.get("name", "").lower()
    if "ithalatci" in name or "import" in name:
        risk += 0.1
    if "garantisiz" in name or "no warranty" in name:
        risk += 0.4

    return min(1.0, risk)


def _extract_brand(product: dict) -> str:
    """Extract brand from product dict, falling back to name parsing."""
    brand = product.get("brand", "")
    if brand:
        return brand.strip()

    # Try to extract from product name (first word is often the brand)
    name = product.get("name", "")
    if name:
        first_word = name.split()[0] if name.split() else ""
        if first_word.lower() in _SERVICE_NETWORKS:
            return first_word
    return "unknown"
