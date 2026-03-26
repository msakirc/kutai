"""Import vs Domestic Advisor.

Classifies product origin (domestic / imported), detects grey-market
parallel imports, warns about BTK registration requirements for phones
and tablets, and provides a consolidated import advisory.
"""

from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.special.import_domestic")

# ─── Turkish Domestic Brand Registry ────────────────────────────────────────
# Maps canonical lowercase brand key -> {"display": str, "country": str,
#   "notes": str | None}

_DOMESTIC_BRANDS: dict[str, dict] = {
    # Vestel group
    "vestel": {
        "display": "Vestel",
        "country": "Türkiye",
        "notes": "Manisa merkezli yerli üretim",
    },
    "regal": {
        "display": "Regal",
        "country": "Türkiye",
        "notes": "Vestel grubu markası",
    },
    "nms": {
        "display": "NMS",
        "country": "Türkiye",
        "notes": "Vestel grubu kurumsal markası",
    },
    # Arçelik group
    "arcelik": {
        "display": "Arçelik",
        "country": "Türkiye",
        "notes": "İstanbul merkezli yerli üretim; küresel marka",
    },
    "arçelik": {
        "display": "Arçelik",
        "country": "Türkiye",
        "notes": "İstanbul merkezli yerli üretim; küresel marka",
    },
    "beko": {
        "display": "Beko",
        "country": "Türkiye",
        "notes": "Arçelik'in uluslararası markası; Türkiye'de üretim",
    },
    "altus": {
        "display": "Altus",
        "country": "Türkiye",
        "notes": "Arçelik grubu beyaz eşya markası",
    },
    "grundig": {
        "display": "Grundig",
        "country": "Türkiye",
        "notes": "Arçelik'e ait; Türkiye'de üretim yapılıyor",
    },
    "blomberg": {
        "display": "Blomberg",
        "country": "Türkiye",
        "notes": "Arçelik grubu markası",
    },
    "leisure": {
        "display": "Leisure",
        "country": "Türkiye",
        "notes": "Arçelik grubu markası",
    },
    "defy": {
        "display": "Defy",
        "country": "Türkiye",
        "notes": "Arçelik grubu markası",
    },
    # BSH Turkey
    "profilo": {
        "display": "Profilo",
        "country": "Türkiye",
        "notes": "BSH Türkiye üretimi; yerli marka",
    },
    # Furniture / home
    "istikbal": {
        "display": "İstikbal",
        "country": "Türkiye",
        "notes": "Kayseri merkezli yerli mobilya",
    },
    "bellona": {
        "display": "Bellona",
        "country": "Türkiye",
        "notes": "İstikbal grubu mobilya markası",
    },
    "mondi": {
        "display": "Mondi",
        "country": "Türkiye",
        "notes": "İstikbal grubu markası",
    },
    # Food / FMCG
    "ulker": {
        "display": "Ülker",
        "country": "Türkiye",
        "notes": "Türk gıda şirketi; İstanbul",
    },
    "ülker": {
        "display": "Ülker",
        "country": "Türkiye",
        "notes": "Türk gıda şirketi; İstanbul",
    },
    "eti": {
        "display": "ETİ",
        "country": "Türkiye",
        "notes": "Eskişehir merkezli yerli gıda markası",
    },
    # Kitchenware
    "karaca": {
        "display": "Karaca",
        "country": "Türkiye",
        "notes": "İstanbul merkezli mutfak eşyası",
    },
    "emsan": {
        "display": "Emsan",
        "country": "Türkiye",
        "notes": "Türk mutfak ve ev eşyası markası",
    },
    "fakir": {
        "display": "Fakir",
        "country": "Türkiye",
        "notes": "Küçük ev aletleri; yerli marka",
    },
    "simfer": {
        "display": "Simfer",
        "country": "Türkiye",
        "notes": "Yerli pişirici ve beyaz eşya markası",
    },
    "silverline": {
        "display": "Silverline",
        "country": "Türkiye",
        "notes": "Yerli mutfak aletleri markası",
    },
    # Technology
    "casper": {
        "display": "Casper",
        "country": "Türkiye",
        "notes": "İstanbul merkezli yerli bilgisayar ve telefon markası",
    },
    "monster": {
        "display": "Monster",
        "country": "Türkiye",
        "notes": "Yerli oyuncu dizüstü bilgisayar markası (Monster Notebook)",
    },
    "reeder": {
        "display": "Reeder",
        "country": "Türkiye",
        "notes": "Yerli tablet ve telefon markası",
    },
    "piranha": {
        "display": "Piranha",
        "country": "Türkiye",
        "notes": "Vestel grubu oyuncu elektroniği",
    },
    # Automotive / industrial
    "anadol": {
        "display": "Anadol",
        "country": "Türkiye",
        "notes": "Tarihi yerli otomobil markası",
    },
    "togg": {
        "display": "TOGG",
        "country": "Türkiye",
        "notes": "Türkiye'nin elektrikli otomobil girişimi",
    },
}

# ─── Known Imported Brands ───────────────────────────────────────────────────
# Maps canonical lowercase key -> {"display": str, "country": str}

_IMPORTED_BRANDS: dict[str, dict] = {
    # USA
    "apple": {"display": "Apple", "country": "ABD"},
    "microsoft": {"display": "Microsoft", "country": "ABD"},
    "hp": {"display": "HP", "country": "ABD"},
    "dell": {"display": "Dell", "country": "ABD"},
    "google": {"display": "Google", "country": "ABD"},
    "amazon": {"display": "Amazon", "country": "ABD"},
    "motorola": {"display": "Motorola", "country": "ABD"},
    "qualcomm": {"display": "Qualcomm", "country": "ABD"},
    # South Korea
    "samsung": {"display": "Samsung", "country": "Güney Kore"},
    "lg": {"display": "LG", "country": "Güney Kore"},
    "sk": {"display": "SK", "country": "Güney Kore"},
    # Japan
    "sony": {"display": "Sony", "country": "Japonya"},
    "panasonic": {"display": "Panasonic", "country": "Japonya"},
    "sharp": {"display": "Sharp", "country": "Japonya"},
    "toshiba": {"display": "Toshiba", "country": "Japonya"},
    "canon": {"display": "Canon", "country": "Japonya"},
    "nikon": {"display": "Nikon", "country": "Japonya"},
    "fujifilm": {"display": "Fujifilm", "country": "Japonya"},
    # China
    "xiaomi": {"display": "Xiaomi", "country": "Çin"},
    "huawei": {"display": "Huawei", "country": "Çin"},
    "oppo": {"display": "OPPO", "country": "Çin"},
    "vivo": {"display": "Vivo", "country": "Çin"},
    "realme": {"display": "Realme", "country": "Çin"},
    "oneplus": {"display": "OnePlus", "country": "Çin"},
    "tcl": {"display": "TCL", "country": "Çin"},
    "lenovo": {"display": "Lenovo", "country": "Çin"},
    "honor": {"display": "Honor", "country": "Çin"},
    "zte": {"display": "ZTE", "country": "Çin"},
    "haier": {"display": "Haier", "country": "Çin"},
    "hisense": {"display": "Hisense", "country": "Çin"},
    "dji": {"display": "DJI", "country": "Çin"},
    "anker": {"display": "Anker", "country": "Çin"},
    # Taiwan
    "asus": {"display": "ASUS", "country": "Tayvan"},
    "acer": {"display": "Acer", "country": "Tayvan"},
    "msi": {"display": "MSI", "country": "Tayvan"},
    "hTC": {"display": "HTC", "country": "Tayvan"},
    # Europe (Germany)
    "bosch": {"display": "Bosch", "country": "Almanya"},
    "siemens": {"display": "Siemens", "country": "Almanya"},
    "braun": {"display": "Braun", "country": "Almanya"},
    "miele": {"display": "Miele", "country": "Almanya"},
    "aeg": {"display": "AEG", "country": "Almanya"},
    "sennheiser": {"display": "Sennheiser", "country": "Almanya"},
    # Europe (Netherlands / Sweden / UK)
    "philips": {"display": "Philips", "country": "Hollanda"},
    "electrolux": {"display": "Electrolux", "country": "İsveç"},
    "dyson": {"display": "Dyson", "country": "İngiltere"},
    # Europe (France / Italy)
    "moulinex": {"display": "Moulinex", "country": "Fransa"},
    "tefal": {"display": "Tefal", "country": "Fransa"},
    "rowenta": {"display": "Rowenta", "country": "Fransa"},
    "de'longhi": {"display": "De'Longhi", "country": "İtalya"},
    "delonghi": {"display": "De'Longhi", "country": "İtalya"},
    # Finland / Sweden
    "nokia": {"display": "Nokia", "country": "Finlandiya"},
    # Gaming
    "nintendo": {"display": "Nintendo", "country": "Japonya"},
    "razer": {"display": "Razer", "country": "ABD/Singapur"},
    "logitech": {"display": "Logitech", "country": "İsviçre"},
    "corsair": {"display": "Corsair", "country": "ABD"},
}

# ─── Grey Market Keyword Indicators ─────────────────────────────────────────

_GREY_MARKET_KEYWORDS: list[tuple[str, float, str]] = [
    # (keyword, weight, human-readable indicator label)
    ("ithalatçı garantili", 0.70, "İthalatçı garantili ibaresi tespit edildi"),
    ("ithalatci garantili", 0.70, "İthalatçı garantili ibaresi tespit edildi"),
    ("distribütör garantisi yok", 0.80, "Resmi distribütör garantisi bulunmuyor"),
    ("distributor garantisi yok", 0.80, "Resmi distribütör garantisi bulunmuyor"),
    ("yurt dışı", 0.50, "Yurt dışı ifadesi mevcut"),
    ("yurt disi", 0.50, "Yurt dışı ifadesi mevcut"),
    ("paralel ithalat", 0.75, "Paralel ithalat ifadesi mevcut"),
    ("resmi distribütör değil", 0.80, "Resmi distribütör değil ifadesi mevcut"),
    ("resmi distributor degil", 0.80, "Resmi distribütör değil ifadesi mevcut"),
    ("a kalite", 0.40, "A kalite ifadesi mevcut (orijinallik belirsiz)"),
    ("a+ kalite", 0.35, "A+ kalite ifadesi mevcut"),
    ("garantisiz", 0.60, "Garantisiz olarak belirtilmiş"),
    ("no warranty", 0.60, "Garanti bulunmuyor"),
    ("uluslararası garanti", 0.45, "Yalnızca uluslararası garanti belirtilmiş"),
    ("uluslararasi garanti", 0.45, "Yalnızca uluslararası garanti belirtilmiş"),
    ("eu spec", 0.35, "AB spesifikasyonu ürünü olabilir"),
    ("us spec", 0.35, "ABD spesifikasyonu ürünü olabilir"),
    ("global version", 0.40, "Global versiyon belirtilmiş"),
    ("çin sürümü", 0.55, "Çin sürümü olarak belirtilmiş"),
    ("cin surumu", 0.55, "Çin sürümü olarak belirtilmiş"),
    ("chinese version", 0.55, "Çin sürümü olarak belirtilmiş"),
]

# ─── BTK-Applicable Product Keywords ────────────────────────────────────────

_BTK_PRODUCT_KEYWORDS: list[str] = [
    "telefon", "akıllı telefon", "akilli telefon", "smartphone",
    "iphone", "android", "tablet", "ipad",
    "samsung galaxy", "xiaomi", "huawei", "oppo", "vivo", "realme",
]

_BTK_REGISTRATION_COST_TRY = 7_500.0  # Approximate 2026 figure


def classify_origin(brand: str, product_name: str = "") -> dict:
    """Classify a brand as domestic, imported, or unknown.

    Parameters
    ----------
    brand:
        Brand name string (case-insensitive, tolerates Turkish characters).
    product_name:
        Optional product name for additional heuristic hints.

    Returns
    -------
    Dict with keys:

    - ``origin``: ``"domestic"`` | ``"imported"`` | ``"unknown"``
    - ``country``: ISO country name in Turkish, or ``None``
    - ``confidence``: float in [0.0, 1.0]
    - ``notes``: list of human-readable observation strings
    """
    key = _normalize(brand)
    notes: list[str] = []

    if key in _DOMESTIC_BRANDS:
        info = _DOMESTIC_BRANDS[key]
        if info.get("notes"):
            notes.append(info["notes"])
        logger.debug("classify_origin: '%s' -> domestic (%s)", brand, info["country"])
        return {
            "origin": "domestic",
            "country": info["country"],
            "confidence": 0.95,
            "notes": notes,
        }

    if key in _IMPORTED_BRANDS:
        info = _IMPORTED_BRANDS[key]
        notes.append(f"Menşei ülke: {info['country']}")
        logger.debug("classify_origin: '%s' -> imported (%s)", brand, info["country"])
        return {
            "origin": "imported",
            "country": info["country"],
            "confidence": 0.90,
            "notes": notes,
        }

    # Heuristic: check if the brand key appears inside any known brand entry
    # (handles partial matches like "arcelik" inside "arcelik-lge")
    for db, label in ((_DOMESTIC_BRANDS, "domestic"), (_IMPORTED_BRANDS, "imported")):
        for known_key, info in db.items():
            if known_key in key or key in known_key:
                notes.append(
                    f"'{brand}' bilinen bir markayla kısmi eşleşme: {info['display']}"
                )
                country = info.get("country")
                logger.debug(
                    "classify_origin: '%s' -> %s (partial match)", brand, label
                )
                return {
                    "origin": label,
                    "country": country,
                    "confidence": 0.55,
                    "notes": notes,
                }

    # Final fallback: scan product name for any known brand mention
    name_key = _normalize(product_name)
    for db, label in ((_DOMESTIC_BRANDS, "domestic"), (_IMPORTED_BRANDS, "imported")):
        for known_key, info in db.items():
            if known_key in name_key:
                notes.append(
                    f"Ürün adında '{info['display']}' markası tespit edildi"
                )
                return {
                    "origin": label,
                    "country": info.get("country"),
                    "confidence": 0.50,
                    "notes": notes,
                }

    notes.append("Marka kökeni belirlenemedi; ek araştırma önerilir")
    logger.debug("classify_origin: '%s' -> unknown", brand)
    return {
        "origin": "unknown",
        "country": None,
        "confidence": 0.0,
        "notes": notes,
    }


def detect_grey_market(product: dict) -> dict:
    """Detect parallel / grey market indicators in a product listing.

    Scans the product ``name``, ``description``, ``seller_type``, and
    ``seller_name`` fields for grey-market keywords.

    Parameters
    ----------
    product:
        Product dict.  Relevant keys: ``name``, ``description``,
        ``seller_type``, ``seller_name``, ``price``, ``avg_price``.

    Returns
    -------
    Dict with keys:

    - ``is_grey_market``: bool
    - ``confidence``: float in [0.0, 1.0]
    - ``indicators``: list of matched indicator labels
    - ``warnings``: list of actionable warning strings
    """
    # Build a single lowercase search corpus from the relevant fields
    corpus_parts = [
        product.get("name", ""),
        product.get("description", ""),
        product.get("seller_type", ""),
        product.get("seller_name", ""),
    ]
    corpus = _normalize(" ".join(filter(None, corpus_parts)))

    indicators: list[str] = []
    raw_score = 0.0

    for keyword, weight, label in _GREY_MARKET_KEYWORDS:
        if keyword in corpus:
            indicators.append(label)
            raw_score += weight
            logger.debug("detect_grey_market: keyword match '%s' (+%.2f)", keyword, weight)

    # Price deviation heuristic
    price = float(product.get("price", 0) or 0)
    avg_price = float(product.get("avg_price", 0) or 0)
    if price > 0 and avg_price > 0:
        ratio = price / avg_price
        if ratio < 0.70:
            raw_score += 0.45
            indicators.append(
                f"Fiyat piyasa ortalamasının {int((1 - ratio) * 100)}% altında"
            )
        elif ratio < 0.80:
            raw_score += 0.25
            indicators.append(
                f"Fiyat piyasa ortalamasının {int((1 - ratio) * 100)}% altında"
            )

    confidence = min(1.0, raw_score)
    is_grey = confidence >= 0.40

    warnings: list[str] = []
    if is_grey:
        warnings.append(
            "Bu ürün gri piyasa (paralel ithalat) ürünü olabilir. "
            "Resmi Türkiye garantisi bulunmayabilir."
        )
        warnings.append(
            "Arıza durumunda resmi yetkili servis reddedebilir; "
            "satıcıdan garanti belgesi talep edin."
        )
        if confidence >= 0.70:
            warnings.append(
                "Yüksek güvenle gri piyasa işareti tespit edildi. "
                "Satın almadan önce dikkatlice değerlendirin."
            )

    logger.info(
        "detect_grey_market: is_grey=%s confidence=%.2f indicators=%d",
        is_grey,
        confidence,
        len(indicators),
    )
    return {
        "is_grey_market": is_grey,
        "confidence": round(confidence, 2),
        "indicators": indicators,
        "warnings": warnings,
    }


def check_btk_requirement(product: dict) -> dict:
    """Check whether a product likely requires BTK IMEI registration.

    Phones and tablets imported unofficially into Turkey must be registered
    with BTK (Bilgi Teknolojileri ve İletişim Kurumu) within 120 days or
    they will be blocked from cellular networks.

    Parameters
    ----------
    product:
        Product dict.  Relevant keys: ``name``, ``category``, ``brand``.

    Returns
    -------
    Dict with keys:

    - ``needs_btk``: bool
    - ``warning``: str — explanation (empty string if ``needs_btk`` is False)
    - ``registration_cost_estimate``: float — estimated fee in TRY
    """
    corpus = _normalize(
        " ".join(
            filter(
                None,
                [
                    product.get("name", ""),
                    product.get("category", ""),
                    product.get("brand", ""),
                ],
            )
        )
    )

    needs_btk = any(kw in corpus for kw in _BTK_PRODUCT_KEYWORDS)

    if not needs_btk:
        logger.debug("check_btk_requirement: not applicable for this product")
        return {
            "needs_btk": False,
            "warning": "",
            "registration_cost_estimate": 0.0,
        }

    warning = (
        "Bu ürün bir telefon veya tablet olarak görünmektedir. "
        "Yurt dışından veya gri piyasadan alınan cihazların 120 gün içinde "
        f"BTK'ya IMEI kaydı yaptırılması zorunludur; aksi takdirde mobil ağlarda "
        f"kullanılamaz hale gelir. "
        f"Tahmini kayıt ücreti: ≈{_BTK_REGISTRATION_COST_TRY:,.0f} TRY."
    )

    logger.info("check_btk_requirement: BTK registration likely required")
    return {
        "needs_btk": True,
        "warning": warning,
        "registration_cost_estimate": _BTK_REGISTRATION_COST_TRY,
    }


def get_import_advisory(product: dict) -> dict:
    """Produce a consolidated import advisory for a product.

    Combines origin classification, grey-market detection, and BTK
    registration check into a single advisory dict.

    Parameters
    ----------
    product:
        Product dict.  Relevant keys: ``name``, ``brand``, ``category``,
        ``description``, ``price``, ``avg_price``, ``seller_type``,
        ``seller_name``.

    Returns
    -------
    Dict with keys:

    - ``origin``: result of :func:`classify_origin`
    - ``grey_market``: result of :func:`detect_grey_market`
    - ``btk``: result of :func:`check_btk_requirement`
    - ``warranty_implications``: list[str]
    - ``service_center_note``: str
    - ``price_context``: str
    - ``overall_risk``: ``"low"`` | ``"medium"`` | ``"high"``
    - ``recommendations``: list[str]
    """
    brand = product.get("brand", "") or _infer_brand_from_name(product.get("name", ""))
    name = product.get("name", "")

    origin_result = classify_origin(brand, name)
    grey_result = detect_grey_market(product)
    btk_result = check_btk_requirement(product)

    # ── Warranty implications ────────────────────────────────────────────────
    warranty_implications: list[str] = []
    if origin_result["origin"] == "domestic":
        warranty_implications.append(
            "Yerli üretim ürünlerde Türkiye garantisi standarttır; "
            "yetkili servisler yaygındır."
        )
    elif grey_result["is_grey_market"]:
        warranty_implications.append(
            "Gri piyasa ürününde resmi Türkiye garantisi geçersiz olabilir."
        )
        warranty_implications.append(
            "Satıcının sunduğu garanti (ithalatçı garantisi) yetersiz kalabilir; "
            "kapsam ve süreyi yazılı olarak alın."
        )
    else:
        warranty_implications.append(
            "Resmi kanaldan ithal üründe Türkiye distribütörü garantisi geçerlidir."
        )

    # ── Service center note ──────────────────────────────────────────────────
    if origin_result["origin"] == "domestic":
        service_note = (
            "Yerli marka için yaygın yetkili servis ağı mevcut olması beklenir."
        )
    elif grey_result["is_grey_market"]:
        service_note = (
            "Gri piyasa ürünleri resmi yetkili servisler tarafından reddedilebilir; "
            "yalnızca bağımsız servislere başvurabilirsiniz."
        )
    else:
        country = origin_result.get("country") or "yurt dışı"
        service_note = (
            f"Bu marka ({country} menşeli) için Türkiye'deki yetkili servis "
            "ağı büyük şehirlerle sınırlı olabilir."
        )

    # ── Price context ────────────────────────────────────────────────────────
    price = float(product.get("price", 0) or 0)
    avg_price = float(product.get("avg_price", 0) or 0)
    price_context = _build_price_context(
        price, avg_price, origin_result["origin"], grey_result["is_grey_market"]
    )

    # ── Overall risk ─────────────────────────────────────────────────────────
    risk_score = 0
    if grey_result["confidence"] >= 0.70:
        risk_score += 2
    elif grey_result["is_grey_market"]:
        risk_score += 1
    if btk_result["needs_btk"] and grey_result["is_grey_market"]:
        risk_score += 1
    if origin_result["origin"] == "domestic":
        risk_score = max(0, risk_score - 1)

    if risk_score >= 3:
        overall_risk = "high"
    elif risk_score >= 1:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    # ── Recommendations ──────────────────────────────────────────────────────
    recommendations: list[str] = []
    if grey_result["is_grey_market"]:
        recommendations.append(
            "Satın almadan önce satıcıdan resmi Türkiye garanti belgesi isteyin."
        )
    if btk_result["needs_btk"] and grey_result["is_grey_market"]:
        recommendations.append(
            f"BTK IMEI kaydı zorunlu olabilir (≈{_BTK_REGISTRATION_COST_TRY:,.0f} TRY ek maliyet)."
        )
    if origin_result["origin"] == "imported" and not grey_result["is_grey_market"]:
        recommendations.append(
            "Resmi distribütörden alınan ithal üründe Türkiye garantisi ve "
            "servis ağı mevcuttur; sorun beklenmez."
        )
    if overall_risk == "low":
        recommendations.append("Ürün düşük riskli görünmektedir; standart satın alma sürecini uygulayabilirsiniz.")

    logger.info(
        "get_import_advisory: brand='%s' origin=%s grey=%s btk=%s risk=%s",
        brand,
        origin_result["origin"],
        grey_result["is_grey_market"],
        btk_result["needs_btk"],
        overall_risk,
    )

    return {
        "origin": origin_result,
        "grey_market": grey_result,
        "btk": btk_result,
        "warranty_implications": warranty_implications,
        "service_center_note": service_note,
        "price_context": price_context,
        "overall_risk": overall_risk,
        "recommendations": recommendations,
    }


# ─── Private Helpers ─────────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """Lowercase and apply basic Turkish character folding."""
    return (
        text.lower()
        .replace("ı", "i")
        .replace("ğ", "g")
        .replace("ş", "s")
        .replace("ç", "c")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("İ", "i")
        .replace("Ğ", "g")
        .replace("Ş", "s")
        .replace("Ç", "c")
        .replace("Ö", "o")
        .replace("Ü", "u")
    )


def _infer_brand_from_name(name: str) -> str:
    """Return the first word of *name* as a rough brand guess."""
    parts = name.strip().split()
    return parts[0] if parts else ""


def _build_price_context(
    price: float,
    avg_price: float,
    origin: str,
    is_grey: bool,
) -> str:
    """Generate a human-readable price-context sentence."""
    if price <= 0 or avg_price <= 0:
        return "Fiyat karşılaştırması için yeterli veri yok."

    ratio = price / avg_price
    diff_pct = abs(1 - ratio) * 100

    if ratio < 0.75:
        base = f"Fiyat piyasa ortalamasının %{diff_pct:.0f} altında."
        if is_grey:
            return base + " Bu düşüklük gri piyasa/paralel ithalat ile açıklanabilir."
        return base + " Fiyat avantajı gerçek olabilir; ancak kaynak araştırılmalıdır."
    if ratio < 0.92:
        return f"Fiyat piyasa ortalamasının hafif altında (%{diff_pct:.0f})."
    if ratio <= 1.08:
        return "Fiyat piyasa ortalamasıyla uyumlu."
    if ratio <= 1.25:
        base = f"Fiyat piyasa ortalamasının %{diff_pct:.0f} üzerinde."
        if origin == "imported":
            return base + " İthal ürünlerde ithalat primi yansımış olabilir."
        return base
    base = f"Fiyat piyasa ortalamasının %{diff_pct:.0f} üzerinde — belirgin prim."
    if origin == "imported":
        return base + " Yüksek ithalat maliyetleri veya kur etkisi söz konusu olabilir."
    return base
