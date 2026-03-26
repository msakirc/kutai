"""Used Market Awareness — advisory module for second-hand buying guidance.

Evaluates whether a product category is suitable for buying used/refurbished,
identifies the right platforms to check, and surfaces Turkish-specific tips.
No actual web scraping is performed; this is purely advisory.
"""

from __future__ import annotations

import json

from src.infra.logging_config import get_logger
from src.shopping.intelligence._llm import _llm_call

logger = get_logger("shopping.intelligence.special.used_market")

# ─── Safety blocklist ────────────────────────────────────────────────────────

# Categories where used products are NEVER recommended for safety or hygiene
# reasons. Matching is done case-insensitively against the normalised category.
_UNSAFE_CATEGORIES: list[str] = [
    "bebek",
    "bebek ürünleri",
    "bebek bezi",
    "medikal",
    "tıbbi",
    "sağlık",
    "güvenlik",
    "gıda",
    "yiyecek",
    "kozmetik",
    "makyaj",
    "ilaç",
    "ilaçlar",
    "araba koltuğu",
    "oto koltuk",
    "çocuk koltuğu",
    "kask",
    "bisiklet kasığı",
    "motosiklet kasığı",
]

# ─── Platform definitions ─────────────────────────────────────────────────────

_PLATFORMS: list[dict] = [
    {
        "name": "sahibinden.com",
        "url_pattern": "https://www.sahibinden.com/arama?query={query}",
        "strengths": ["Geniş kategori yelpazesi", "Araç, elektronik, mobilya", "Yerel satıcılar"],
        "categories": [
            "elektronik",
            "mobilya",
            "araç",
            "spor",
            "müzik",
            "kitap",
            "genel",
            "beyaz eşya",
            "ev aletleri",
        ],
    },
    {
        "name": "dolap.com",
        "url_pattern": "https://dolap.com/arama?q={query}",
        "strengths": ["Kadın/erkek/çocuk giyim", "Marka doğrulama", "Güvenli ödeme"],
        "categories": ["giyim", "moda", "ayakkabı", "aksesuar", "çanta"],
    },
    {
        "name": "letgo",
        "url_pattern": "https://tr.letgo.com/tr/search?q={query}",
        "strengths": ["Kolay kullanım", "Yerel ilanlar", "Hızlı pazarlık"],
        "categories": ["elektronik", "mobilya", "genel", "spor", "oyun"],
    },
    {
        "name": "gardrops.com",
        "url_pattern": "https://www.gardrops.com/search?q={query}",
        "strengths": ["Lüks/marka moda", "Kimlik doğrulaması", "Kalite kontrolü"],
        "categories": ["giyim", "moda", "lüks", "aksesuar", "çanta"],
    },
    {
        "name": "gittigidiyor.com",
        "url_pattern": "https://www.gittigidiyor.com/arama?k={query}&sf=1",
        "strengths": ["Elektronik odaklı", "Açık artırma ve sabit fiyat", "Geniş seçenek"],
        "categories": ["elektronik", "bilgisayar", "telefon", "oyun", "kamera"],
    },
    {
        "name": "apple.com/tr/shop/refurbished",
        "url_pattern": "https://www.apple.com/tr/shop/refurbished/{query}",
        "strengths": ["Apple garantisi", "Yenilenmiş ürün sertifikası", "Tam test edilmiş"],
        "categories": ["apple", "iphone", "ipad", "mac", "macbook"],
    },
]

# ─── Category suitability map ─────────────────────────────────────────────────

# Maps broad category keywords to expected discount range and platform hints.
_CATEGORY_SUITABILITY: dict[str, dict] = {
    "elektronik": {
        "suitable": True,
        "typical_discount_pct": 35.0,
        "platforms": ["sahibinden.com", "letgo", "gittigidiyor.com"],
        "tips": [
            "Açmadan önce ürünün IMEI veya seri numarasını kontrol edin.",
            "Mümkünse yüz yüze teslim alın ve yerinde test edin.",
            "Fatura veya garanti belgesi olup olmadığını sorun.",
            "Şarj döngüsü ve batarya sağlığını kontrol edin (telefonlar için).",
        ],
    },
    "telefon": {
        "suitable": True,
        "typical_discount_pct": 30.0,
        "platforms": ["sahibinden.com", "gittigidiyor.com", "apple.com/tr/shop/refurbished"],
        "tips": [
            "IMEI numarasını sorgulayin (e-devlet veya operatör).",
            "Ekran bütünlüğünü, kamera ve hoparlörü test edin.",
            "iCloud/Google hesabı kilitli olmadığından emin olun.",
            "Batarya sağlık yüzdesini kontrol edin (ideal: %80+).",
        ],
    },
    "bilgisayar": {
        "suitable": True,
        "typical_discount_pct": 40.0,
        "platforms": ["sahibinden.com", "gittigidiyor.com", "letgo"],
        "tips": [
            "RAM, işlemci ve disk bilgilerini doğrulayın.",
            "Fan gürültüsü ve ısınma sorunlarını test edin.",
            "Ekran üzerinde dead pixel veya renk sorunu olup olmadığına bakın.",
            "Mümkünse fabrika sıfırlaması talep edin.",
        ],
    },
    "mobilya": {
        "suitable": True,
        "typical_discount_pct": 50.0,
        "platforms": ["sahibinden.com", "letgo"],
        "tips": [
            "Fotoğraflarda tüm açılardan detay isteyin.",
            "Kargo yerine kendiniz veya nakliyeci ile taşıyın.",
            "Nem hasarı veya böcek belirtilerini gözlemleyin.",
            "Montaj vidaları ve eksik parça olup olmadığını sorun.",
        ],
    },
    "giyim": {
        "suitable": True,
        "typical_discount_pct": 60.0,
        "platforms": ["dolap.com", "gardrops.com", "letgo"],
        "tips": [
            "Beden tablosunu dikkatlice inceleyin.",
            "Satıcının değerlendirme puanına bakın.",
            "Temizleme/bakım talimatlarını sorun.",
            "Kargo öncesi ek fotoğraf isteyin.",
        ],
    },
    "moda": {
        "suitable": True,
        "typical_discount_pct": 55.0,
        "platforms": ["dolap.com", "gardrops.com"],
        "tips": [
            "Marka ürünler için orijinallik belgesi veya etiket isteyin.",
            "Gardrops'ta lüks ürünler için kimlik doğrulama özelliğini kullanın.",
            "İade politikasını satın almadan önce teyit edin.",
        ],
    },
    "kitap": {
        "suitable": True,
        "typical_discount_pct": 60.0,
        "platforms": ["sahibinden.com", "letgo", "gittigidiyor.com"],
        "tips": [
            "Baskı yılını ve sayfa durumunu sorun.",
            "Yazı veya altı çizili notlar olup olmadığını belirtin.",
            "Akademik kitaplar için güncel baskı olup olmadığını kontrol edin.",
        ],
    },
    "müzik": {
        "suitable": True,
        "typical_discount_pct": 35.0,
        "platforms": ["sahibinden.com", "letgo", "gittigidiyor.com"],
        "tips": [
            "Enstrümanı satın almadan önce çalmayı test edin.",
            "Elektronik enstrümanlarda ses çıkışlarını kontrol edin.",
            "Akustik enstrümanlarda çatlak veya yapıştırma izlerine dikkat edin.",
            "Aksesuar (kılıf, kablo) durumunu sorun.",
        ],
    },
    "spor": {
        "suitable": True,
        "typical_discount_pct": 45.0,
        "platforms": ["sahibinden.com", "letgo"],
        "tips": [
            "Darbeye maruz kalan ürünlerde (kayak, bisiklet) hasar kontrolü yapın.",
            "Çerçeve veya yapısal bütünlük sorunlarına dikkat edin.",
            "Elektronik parçalı aletlerde (koşu bandı vb.) test yapın.",
        ],
    },
    "araç": {
        "suitable": True,
        "typical_discount_pct": 25.0,
        "platforms": ["sahibinden.com"],
        "tips": [
            "Ekspertiz raporu alın.",
            "Tramer kaydını e-devlet üzerinden sorgulayın.",
            "Noterde resmi devir işlemi yapın.",
            "Servis bakım geçmişini inceleyin.",
        ],
    },
    "beyaz eşya": {
        "suitable": True,
        "typical_discount_pct": 40.0,
        "platforms": ["sahibinden.com", "letgo"],
        "tips": [
            "Çalışır durumda test edin.",
            "Enerji etiketi sınıfını kontrol edin.",
            "Orijinal kurulum kılavuzu ve parçaların eksiksiz olduğunu doğrulayın.",
            "Nakliye için profesyonel yardım alın.",
        ],
    },
    "oyun": {
        "suitable": True,
        "typical_discount_pct": 30.0,
        "platforms": ["sahibinden.com", "letgo", "gittigidiyor.com"],
        "tips": [
            "Konsol için orijinal güç adaptörü ve kabloların tam olduğunu kontrol edin.",
            "Dijital oyun hesabının bağlı olmadığını doğrulayın.",
            "Disk çiziklerini fotoğrafta netçe isteyin.",
        ],
    },
}

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    """Lowercase and strip a string for comparison."""
    return text.lower().strip()


def _match_suitability(category: str) -> dict | None:
    """Find the best matching suitability entry for a category string."""
    norm = _normalise(category)
    # Exact match first
    if norm in _CATEGORY_SUITABILITY:
        return _CATEGORY_SUITABILITY[norm]
    # Substring match
    for key, data in _CATEGORY_SUITABILITY.items():
        if key in norm or norm in key:
            return data
    return None


# ─── Public API ───────────────────────────────────────────────────────────────


def is_safe_to_buy_used(category: str) -> bool:
    """Return ``True`` if buying used is safe for this category.

    NEVER recommends used for: bebek (baby), medikal (medical), güvenlik
    (safety), gıda (food), kozmetik (cosmetics), ilaç (medicine), araba
    koltuğu (car seat), kask (helmet).

    Parameters
    ----------
    category:
        Product category string (Turkish or English).

    Returns
    -------
    ``False`` if the category is in the safety blocklist, ``True`` otherwise.
    """
    norm = _normalise(category)
    for unsafe in _UNSAFE_CATEGORIES:
        if unsafe in norm or norm in unsafe:
            logger.debug("Category '%s' matched unsafe pattern '%s'", category, unsafe)
            return False
    return True


def get_used_platforms(category: str) -> list[dict]:
    """Return relevant second-hand platforms for a given category.

    Parameters
    ----------
    category:
        Product category string.

    Returns
    -------
    List of platform dicts, each with ``name``, ``url_pattern``, and
    ``strengths``.  Always returns at least sahibinden.com and letgo as
    general fallbacks.
    """
    norm = _normalise(category)
    matched: list[dict] = []
    seen: set[str] = set()

    for platform in _PLATFORMS:
        for cat in platform["categories"]:
            if cat in norm or norm in cat:
                if platform["name"] not in seen:
                    matched.append(
                        {
                            "name": platform["name"],
                            "url_pattern": platform["url_pattern"],
                            "strengths": platform["strengths"],
                        }
                    )
                    seen.add(platform["name"])
                break

    # Always include general fallbacks if nothing was matched
    if not matched:
        for platform in _PLATFORMS:
            if "genel" in platform["categories"]:
                if platform["name"] not in seen:
                    matched.append(
                        {
                            "name": platform["name"],
                            "url_pattern": platform["url_pattern"],
                            "strengths": platform["strengths"],
                        }
                    )
                    seen.add(platform["name"])

    logger.debug("Found %d platforms for category '%s'", len(matched), category)
    return matched


async def assess_used_viability(
    product_name: str,
    category: str | None = None,
) -> dict:
    """Assess whether buying a product used is a good idea.

    Combines rule-based category safety checks with LLM-powered contextual
    advice to produce a comprehensive used-market recommendation.

    Parameters
    ----------
    product_name:
        Name or description of the product (e.g. ``"iPhone 14 Pro"``).
    category:
        Optional category hint (e.g. ``"elektronik"``).  When omitted the
        function infers the category from the product name.

    Returns
    -------
    Dict with:
    - ``used_recommended`` (bool) — overall recommendation
    - ``reason`` (str) — human-readable explanation
    - ``safety_concern`` (bool) — ``True`` if category is blocklisted
    - ``platforms`` (list[str]) — suggested platform names to check
    - ``typical_discount_pct`` (float) — expected discount for used
    - ``tips`` (list[str]) — Turkish buying tips
    """
    effective_category = category or product_name

    # ── Safety check ────────────────────────────────────────────────────
    if not is_safe_to_buy_used(effective_category):
        logger.info(
            "Used not recommended for '%s' (safety concern in category '%s')",
            product_name,
            effective_category,
        )
        return {
            "used_recommended": False,
            "reason": (
                f"'{effective_category}' kategorisindeki ürünler için ikinci el alım "
                "önerilmez. Güvenlik, hijyen veya sağlık riski taşıyabilir."
            ),
            "safety_concern": True,
            "platforms": [],
            "typical_discount_pct": 0.0,
            "tips": [
                "Bu kategoride sadece sıfır, onaylı ürün satın alın.",
                "Yetkili satıcı ve resmi garanti belgesi isteyin.",
            ],
        }

    # ── Rule-based suitability lookup ───────────────────────────────────
    suitability = _match_suitability(effective_category)
    platforms = get_used_platforms(effective_category)
    platform_names = [p["name"] for p in platforms]

    if suitability:
        base_discount = suitability["typical_discount_pct"]
        base_tips = suitability["tips"]
        rule_recommended = suitability["suitable"]
    else:
        base_discount = 20.0
        base_tips = [
            "Satıcı değerlendirmelerini dikkatlice okuyun.",
            "Mümkünse ürünü yerinde görün ve test edin.",
            "Ödeme güvenliği için platform içi ödeme sistemini tercih edin.",
        ]
        rule_recommended = True

    # ── LLM enrichment ───────────────────────────────────────────────────
    system_prompt = (
        "Sen Türkiye'de ikinci el alışveriş konusunda uzman bir danışmansın. "
        "Kısa, pratik ve Türkçe tavsiye veriyorsun."
    )
    user_prompt = (
        f"Ürün: {product_name}\n"
        f"Kategori: {effective_category}\n\n"
        "Bu ürünü ikinci el olarak satın almak mantıklı mı? "
        "Aşağıdaki JSON formatında yanıt ver (başka metin ekleme):\n"
        "{\n"
        '  "recommended": true/false,\n'
        '  "reason": "kısa gerekçe (1-2 cümle)",\n'
        '  "extra_tips": ["ipucu1", "ipucu2"]\n'
        "}"
    )

    llm_recommended = rule_recommended
    llm_reason = ""
    extra_tips: list[str] = []

    try:
        raw = await _llm_call(user_prompt, system=system_prompt, temperature=0.2)
        # Extract JSON even if there is surrounding text
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end])
            llm_recommended = bool(parsed.get("recommended", rule_recommended))
            llm_reason = parsed.get("reason", "")
            extra_tips = parsed.get("extra_tips", [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM call failed for used_market assessment: %s", exc)
        llm_reason = ""

    # ── Merge results ────────────────────────────────────────────────────
    final_recommended = rule_recommended and llm_recommended

    if llm_reason:
        reason = llm_reason
    elif final_recommended:
        reason = (
            f"'{effective_category}' kategorisi ikinci el alım için uygundur. "
            f"Ortalama %{base_discount:.0f} indirim beklentisiyle iyi bir seçenek olabilir."
        )
    else:
        reason = (
            f"'{effective_category}' kategorisinde ikinci el alım dikkat gerektirir. "
            "Ürün koşullarını ve satıcı güvenilirliğini mutlaka doğrulayın."
        )

    all_tips = base_tips + [t for t in extra_tips if t not in base_tips]

    logger.info(
        "Used viability for '%s': recommended=%s, discount=%.0f%%",
        product_name,
        final_recommended,
        base_discount,
    )

    return {
        "used_recommended": final_recommended,
        "reason": reason,
        "safety_concern": False,
        "platforms": platform_names,
        "typical_discount_pct": base_discount,
        "tips": all_tips,
    }
