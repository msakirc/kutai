"""Counterfeit and fraud detection for product listings.

Identifies fake / counterfeit goods using keyword analysis, category
heuristics, and price anomaly detection.  Provides category-specific
safety warnings for dangerous product classes.
"""

from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.special.fraud_detector")

# ─── Counterfeit keyword registry ────────────────────────────────────────────

# Each entry: (keyword_lower, severity, meaning)
_COUNTERFEIT_KEYWORDS: list[tuple[str, str, str]] = [
    # Critical — explicit counterfeit / replica indicators
    ("a kalite", "critical", "Orijinal olmayan, birinci sınıf görünümlü taklit ürün"),
    ("a+ kalite", "critical", "Orijinal olmayan, taklit ürün"),
    ("1. kalite", "critical", "Taklit ürünlerde sık kullanılan aldatıcı ifade"),
    ("1.kalite", "critical", "Taklit ürünlerde sık kullanılan aldatıcı ifade"),
    ("birinci kalite kopya", "critical", "Açıkça kopya/taklit ürün"),
    ("replika", "critical", "Orijinal ürünün yetkisiz kopyası"),
    ("kopya", "critical", "Orijinal ürünün kopyası"),
    ("taklit", "critical", "Orijinal marka taklidi"),
    # High — strong counterfeit signals
    ("muadil", "high", "Orijinal yerine geçen, yetkisiz eşdeğer parça veya ürün"),
    ("çakma", "high", "Taklit/sahte ürün için kullanılan argo terim"),
    ("baskı", "high", "Markasız baskı ürünü; lisanssız olabilir"),
    ("super copy", "high", "Yüksek kaliteli kopya ürün"),
    ("süper kopya", "high", "Yüksek kaliteli kopya ürün"),
    ("master copy", "high", "Orijinale benzetilmiş kopya ürün"),
    ("clone", "high", "Orijinal ürünün kopyası"),
    # Medium — context-dependent, possibly legitimate
    ("uyumlu", "medium", "Orijinal marka yerine uyumlu üçüncü taraf ürün (meşru olabilir)"),
    ("benzer", "medium", "Orijinale benzer ürün; marka taklidi olabilir"),
    ("alternatif", "medium", "Orijinal ürüne alternatif; kalite garantisi yok"),
    ("tip", "medium", "Orijinal tipe benzer ürün; lisanssız olabilir"),
    ("model uyumlu", "medium", "Modele uyumlu iddia edilen ürün; doğrulama gerekir"),
    # Info — not necessarily fake, but noteworthy
    ("oem", "info", "Orijinal ekipman üreticisi dışında üretilmiş; kalite değişken olabilir"),
    ("bulk ambalaj", "info", "Toplu satış ambalajı; orijinal perakende paketi değil"),
    ("kutusuz", "info", "Orijinal kutusu olmayan ürün; kaynak belirsiz olabilir"),
    ("ambalajsız", "info", "Ambalajsız ürün; orijinallik doğrulaması zor"),
    ("faturasız", "info", "Faturasız satış; resmi garanti ve menşei belirsiz"),
]

# ─── High-risk categories ─────────────────────────────────────────────────────

_HIGH_RISK_CATEGORIES: frozenset[str] = frozenset(
    [
        "elektronik aksesuar",
        "şarj aleti",
        "şarj",
        "powerbank",
        "power bank",
        "kozmetik",
        "parfüm",
        "hafıza kartı",
        "sd kart",
        "usb bellek",
        "flash bellek",
        "bebek ürünleri",
        "bebek",
        "ilaç",
        "gıda takviyesi",
        "takviye",
        "supplement",
    ]
)

# Categories that pose physical safety hazards when counterfeit
_DANGEROUS_CATEGORIES: frozenset[str] = frozenset(
    [
        "şarj aleti",
        "şarj",
        "powerbank",
        "power bank",
        "elektronik aksesuar",
        "kozmetik",
        "parfüm",
        "bebek ürünleri",
        "bebek",
        "ilaç",
        "gıda takviyesi",
        "takviye",
        "supplement",
    ]
)

# ─── Safety warning templates per category ───────────────────────────────────

_SAFETY_WARNINGS: dict[str, str] = {
    "şarj aleti": (
        "UYARI: Sahte şarj aletleri yangın ve elektrik çarpması riskine yol açabilir. "
        "CE/BS sertifikası olmayan şarj aletlerini kullanmayın."
    ),
    "şarj": (
        "UYARI: Sahte şarj aletleri yangın ve elektrik çarpması riskine yol açabilir. "
        "CE/BS sertifikası olmayan şarj aletlerini kullanmayın."
    ),
    "powerbank": (
        "UYARI: Sahte powerbank'lar batarya patlaması ve yangın riski taşır. "
        "Belirtilen kapasite gerçeği yansıtmıyor olabilir."
    ),
    "power bank": (
        "UYARI: Sahte powerbank'lar batarya patlaması ve yangın riski taşır. "
        "Belirtilen kapasite gerçeği yansıtmıyor olabilir."
    ),
    "elektronik aksesuar": (
        "DİKKAT: Sahte elektronik aksesuarlar cihazınıza zarar verebilir veya "
        "güvenlik riski oluşturabilir."
    ),
    "kozmetik": (
        "UYARI: Sahte kozmetik ürünler zararlı kimyasallar içerebilir. "
        "Cilt tahrişi, alerjik reaksiyon veya kalıcı hasar riski taşır."
    ),
    "parfüm": (
        "UYARI: Sahte parfümler toksik maddeler içerebilir. "
        "Deri üzerinde ciddi tahriş veya alerjik reaksiyon oluşturabilir."
    ),
    "bebek ürünleri": (
        "KRİTİK UYARI: Sahte bebek ürünleri bebeğiniz için ciddi sağlık riskleri taşır. "
        "Yalnızca yetkili satıcılardan orijinal ürün satın alın."
    ),
    "bebek": (
        "KRİTİK UYARI: Sahte bebek ürünleri bebeğiniz için ciddi sağlık riskleri taşır. "
        "Yalnızca yetkili satıcılardan orijinal ürün satın alın."
    ),
    "ilaç": (
        "KRİTİK UYARI: Sahte ilaçlar yanlış etken madde içerebilir veya hiç içermeyebilir. "
        "İlaçları yalnızca eczanelerden ve yetkili kanallardan temin edin."
    ),
    "gıda takviyesi": (
        "UYARI: Sahte gıda takviyeleri zararlı maddeler içerebilir veya "
        "etiketindeki içerikleri barındırmayabilir."
    ),
    "takviye": (
        "UYARI: Sahte gıda takviyeleri zararlı maddeler içerebilir veya "
        "etiketindeki içerikleri barındırmayabilir."
    ),
    "supplement": (
        "UYARI: Sahte supplement ürünleri zararlı maddeler içerebilir veya "
        "etiketindeki içerikleri barındırmayabilir."
    ),
    "hafıza kartı": (
        "DİKKAT: Sahte hafıza kartları belirtilen kapasiteden çok daha az depolama alanı sunar "
        "ve veri kaybına neden olabilir."
    ),
    "sd kart": (
        "DİKKAT: Sahte SD kartlar belirtilen kapasiteden çok daha az depolama alanı sunar "
        "ve veri kaybına neden olabilir."
    ),
    "usb bellek": (
        "DİKKAT: Sahte USB bellekler belirtilen kapasiteden çok daha az depolama alanı sunar "
        "ve veri kaybına neden olabilir."
    ),
    "flash bellek": (
        "DİKKAT: Sahte flash bellekler belirtilen kapasiteden çok daha az depolama alanı sunar "
        "ve veri kaybına neden olabilir."
    ),
}

# ─── Known brand approximate market prices (TL) ──────────────────────────────
# Used for price anomaly detection.  Values are rough midpoints as of 2024-2025.
# Prices below 30 % of these thresholds are flagged as suspicious.

_BRAND_PRICE_FLOOR: dict[str, float] = {
    # Chargers / power banks
    "apple": 800.0,
    "samsung": 400.0,
    "anker": 300.0,
    "baseus": 250.0,
    # Memory / storage
    "sandisk": 200.0,
    "samsung evo": 250.0,
    "kingston": 150.0,
    "lexar": 180.0,
    # Cosmetics / fragrance
    "chanel": 3000.0,
    "dior": 2500.0,
    "lancome": 1500.0,
    "mac": 600.0,
    "nars": 800.0,
    "ysl": 2000.0,
    "versace": 1800.0,
    "armani": 2000.0,
    "dolce gabbana": 2000.0,
    "dolce & gabbana": 2000.0,
    "hugo boss": 1500.0,
    # Electronics accessories
    "belkin": 400.0,
    "ugreen": 200.0,
}


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    """Lowercase and strip *text* for matching."""
    return text.lower().strip()


def _category_matches(product_category: str, target_set: frozenset[str]) -> str | None:
    """Return the first matching category token from *target_set*, or ``None``."""
    cat_lower = _normalise(product_category)
    for token in target_set:
        if token in cat_lower:
            return token
    return None


def _scan_text(text: str) -> list[dict]:
    """Return counterfeit keyword hits found in *text*."""
    text_lower = _normalise(text)
    hits: list[dict] = []
    seen: set[str] = set()
    for keyword, severity, meaning in _COUNTERFEIT_KEYWORDS:
        if keyword in text_lower and keyword not in seen:
            seen.add(keyword)
            hits.append({"keyword": keyword, "severity": severity, "meaning": meaning})
    return hits


def _severity_score(severity: str) -> float:
    """Map severity label to a numeric weight."""
    return {"critical": 1.0, "high": 0.7, "medium": 0.4, "info": 0.1}.get(severity, 0.0)


def _price_anomaly_flag(product: dict) -> str | None:
    """Return a Turkish description if the price is suspiciously low for a known brand."""
    price = product.get("price")
    name = _normalise(product.get("name", "") + " " + product.get("brand", ""))
    if not price or price <= 0:
        return None
    for brand, floor in _BRAND_PRICE_FLOOR.items():
        if brand in name:
            threshold = floor * 0.30
            if price < threshold:
                return (
                    f"Fiyat ({price:.0f} TL), {brand.title()} markası için beklenen "
                    f"minimum pazar fiyatının ({floor:.0f} TL) %30'unun altında — "
                    f"sahte ürün ihtimali yüksek"
                )
    return None


# ─── Public API ───────────────────────────────────────────────────────────────


def assess_counterfeit_risk(product: dict) -> dict:
    """Assess the counterfeit / fraud risk of a product listing.

    Parameters
    ----------
    product:
        Product dict with optional keys: ``name``, ``brand``,
        ``description``, ``category``, ``seller``, ``price``.

    Returns
    -------
    Dict with keys:

    - ``risk_level`` -- ``"low"`` / ``"medium"`` / ``"high"`` / ``"critical"``
    - ``risk_score`` -- 0.0 to 1.0
    - ``red_flags`` -- list of Turkish-language red flag descriptions
    - ``safety_warning`` -- Turkish safety warning string or ``None``
    - ``recommendation`` -- Turkish recommendation string
    """
    name = product.get("name", "")
    brand = product.get("brand", "")
    description = product.get("description", "")
    category = product.get("category", "")

    combined_text = " ".join([name, brand, description])

    red_flags: list[str] = []
    risk_accumulator: float = 0.0

    # 1. Keyword scan across all text fields
    keyword_hits = _scan_text(combined_text)
    for hit in keyword_hits:
        weight = _severity_score(hit["severity"])
        risk_accumulator += weight
        red_flags.append(
            f"Şüpheli ifade bulundu: \"{hit['keyword']}\" — {hit['meaning']}"
        )

    # 2. High-risk category boost
    matched_category = _category_matches(category, _HIGH_RISK_CATEGORIES)
    if matched_category:
        risk_accumulator += 0.15
        red_flags.append(
            f"Yüksek risk kategorisi: \"{matched_category}\" — sahte ürün yaygın olduğu bilinen alan"
        )

    # 3. Price anomaly check
    price_flag = _price_anomaly_flag(product)
    if price_flag:
        risk_accumulator += 0.5
        red_flags.append(price_flag)

    # 4. Seller name heuristic — no-name / generic seller for branded products
    seller = _normalise(product.get("seller", ""))
    if brand and seller and len(seller) < 5 and seller not in (
        _normalise(brand), "resmi", "official"
    ):
        risk_accumulator += 0.1
        red_flags.append(
            "Tanınmış bir marka ürünü, ancak satıcı adı çok kısa / belirsiz"
        )

    # 5. Clamp risk score to [0, 1]
    risk_score = min(1.0, risk_accumulator)

    # Determine risk level
    if risk_score >= 0.8:
        risk_level = "critical"
    elif risk_score >= 0.5:
        risk_level = "high"
    elif risk_score >= 0.2:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Safety warning (only for dangerous categories)
    safety_warning = _build_safety_warning(category) if risk_score >= 0.2 else None

    # Recommendation
    recommendation = _build_recommendation(risk_level, category)

    logger.info(
        "Counterfeit risk assessed for '%s': level=%s score=%.2f flags=%d",
        name or "?",
        risk_level,
        risk_score,
        len(red_flags),
    )

    return {
        "risk_level": risk_level,
        "risk_score": round(risk_score, 2),
        "red_flags": red_flags,
        "safety_warning": safety_warning,
        "recommendation": recommendation,
    }


def detect_counterfeit_keywords(text: str) -> list[dict]:
    """Scan *text* for counterfeit / fraud indicator keywords.

    Parameters
    ----------
    text:
        Any free-form text — product title, description, seller notes, etc.

    Returns
    -------
    List of dicts, each with:

    - ``keyword`` -- matched keyword (lower-case)
    - ``severity`` -- ``"critical"`` / ``"high"`` / ``"medium"`` / ``"info"``
    - ``meaning`` -- Turkish explanation of what the keyword signals
    """
    hits = _scan_text(text)

    if hits:
        logger.debug(
            "detect_counterfeit_keywords: %d hit(s) in text (first 80 chars: %r)",
            len(hits),
            text[:80],
        )

    return hits


def get_safety_warnings(product: dict) -> list[str]:
    """Return category-specific safety warnings for potentially dangerous counterfeits.

    Parameters
    ----------
    product:
        Product dict with an optional ``category`` key.

    Returns
    -------
    List of Turkish-language safety warning strings.  Empty list if no
    warnings apply to the product's category.
    """
    category = product.get("category", "")
    warnings: list[str] = []

    cat_lower = _normalise(category)
    seen: set[str] = set()

    for token, warning_text in _SAFETY_WARNINGS.items():
        if token in cat_lower and warning_text not in seen:
            warnings.append(warning_text)
            seen.add(warning_text)

    # Also scan name/description for category tokens when category field is empty
    if not warnings:
        combined = _normalise(
            product.get("name", "") + " " + product.get("description", "")
        )
        for token, warning_text in _SAFETY_WARNINGS.items():
            if token in combined and warning_text not in seen:
                warnings.append(warning_text)
                seen.add(warning_text)

    return warnings


# ─── Private helpers ──────────────────────────────────────────────────────────


def _build_safety_warning(category: str) -> str | None:
    """Return the primary safety warning for *category*, or ``None``."""
    cat_lower = _normalise(category)
    for token, warning_text in _SAFETY_WARNINGS.items():
        if token in cat_lower:
            return warning_text
    return None


def _build_recommendation(risk_level: str, category: str) -> str:
    """Build a Turkish purchase recommendation based on risk level and category."""
    base: dict[str, str] = {
        "critical": (
            "Bu ürünü SATINALMAYIN. Ürün büyük olasılıkla sahte ya da taklit. "
            "Yetkili satıcı veya resmi mağazalardan alışveriş yapın."
        ),
        "high": (
            "Dikkatli olun. Ürünün orijinalliğini satın almadan doğrulayın; "
            "mümkünse yetkili satıcıları tercih edin."
        ),
        "medium": (
            "Ürün bilgilerini dikkatlice inceleyin. Şüpheli ifadeler varsa "
            "satıcıya kaynak ve garanti belgesi sorun."
        ),
        "low": (
            "Belirgin bir sahtecilik göstergesi saptanmadı. "
            "Yine de satıcı yorumlarını okumayı unutmayın."
        ),
    }
    rec = base.get(risk_level, base["low"])

    # Append category-specific addendum for dangerous categories
    matched = _category_matches(category, _DANGEROUS_CATEGORIES)
    if matched and risk_level in ("critical", "high"):
        rec += (
            f" Bu kategori ({matched}) için sahte ürünler fiziksel zarar riski taşıyabilir."
        )

    return rec
