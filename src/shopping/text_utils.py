"""Turkish text utilities for the shopping system.

Provides functions for parsing Turkish prices, normalizing text with
Turkish-specific characters, generating bilingual search variants,
and extracting product attributes (dimensions, weight, capacity, etc.)
from Turkish product descriptions.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Translation dictionary: Turkish <-> English product terms
# ---------------------------------------------------------------------------
_TR_EN_MAP: dict[str, list[str]] = {
    "bellek": ["RAM"],
    "anakart": ["motherboard"],
    "ekran kartı": ["GPU", "graphics card"],
    "çamaşır makinesi": ["washing machine"],
    "bulaşık makinesi": ["dishwasher"],
    "buzdolabı": ["refrigerator"],
    "fırın": ["oven"],
    "hoparlör": ["speaker"],
    "kulaklık": ["headphone"],
    "klavye": ["keyboard"],
    "fare": ["mouse"],
    "monitör": ["monitor"],
    "yazıcı": ["printer"],
    "telefon": ["phone"],
}

# Build a reverse map: English term -> list of Turkish equivalents
_EN_TR_MAP: dict[str, list[str]] = {}
for _tr, _en_list in _TR_EN_MAP.items():
    for _en in _en_list:
        _EN_TR_MAP.setdefault(_en.lower(), []).append(_tr)


# ---------------------------------------------------------------------------
# Turkish-specific character maps
# ---------------------------------------------------------------------------
_TR_UPPER_TO_LOWER: dict[str, str] = {
    "İ": "i",
    "I": "ı",
    "Ş": "ş",
    "Ç": "ç",
    "Ö": "ö",
    "Ü": "ü",
    "Ğ": "ğ",
}

_TR_LOWER_TO_UPPER: dict[str, str] = {
    "i": "İ",
    "ı": "I",
    "ş": "Ş",
    "ç": "Ç",
    "ö": "Ö",
    "ü": "Ü",
    "ğ": "Ğ",
}

# Characters that signal Turkish text
_TR_SPECIFIC_CHARS = set("şçğıöüİ")

# Filler words/phrases to strip from product names (ordered longest-first to
# avoid partial matches when shorter phrases are substrings of longer ones).
_FILLER_PHRASES: list[str] = sorted(
    [
        "süper fırsat",
        "kampanyalı",
        "orjinal ürün",
        "orijinal ürün",
        "hızlı kargo",
        "aynı gün kargo",
        "ücretsiz kargo",
        "indirimli",
        "fırsat ürünü",
        "stokta",
        "son 5 adet",
        "çok satan",
    ],
    key=len,
    reverse=True,
)

# Material keywords mapped to canonical names
_MATERIALS: list[tuple[str, str]] = [
    ("paslanmaz çelik", "paslanmaz çelik"),
    ("stainless steel", "paslanmaz çelik"),
    ("meşe", "meşe"),
    ("mdf", "mdf"),
    ("cam", "cam"),
    ("ahşap", "ahşap"),
    ("plastik", "plastik"),
    ("seramik", "seramik"),
    ("granit", "granit"),
    ("mermer", "mermer"),
]

# Energy rating pattern (A+++ down to G)
_ENERGY_RE = re.compile(r"\b([A-G])\+{0,3}\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# 1. parse_turkish_price
# ---------------------------------------------------------------------------

def parse_turkish_price(text: str) -> float | None:
    """Parse a Turkish-formatted price string and return a float.

    Supported formats:
        "1.299,99 TL"   -> 1299.99
        "₺1299.99"      -> 1299.99
        "1299,99"        -> 1299.99
        "1.299 TL"       -> 1299.0
        "TL 1.299,99"   -> 1299.99
    """
    if not text or not isinstance(text, str):
        return None

    s = text.strip()

    # Remove currency markers and surrounding whitespace
    s = s.replace("₺", "").replace("TL", "").replace("tl", "").strip()

    if not s:
        return None

    # Determine whether the string uses Turkish formatting (comma as decimal
    # separator, dot as thousands separator) or international formatting.
    #
    # Heuristic:
    #   - If there is a comma followed by 1-2 digits at the end -> Turkish decimal
    #   - If there is a dot followed by exactly 3 digits (and possibly more groups) -> Turkish thousands
    #   - Otherwise fall back to trying float() directly.

    # Turkish format: optional thousands dots, comma as decimal
    # e.g. "1.299,99" or "1.299" (no decimal) or "1299,99"
    tr_match = re.match(
        r"^(\d{1,3}(?:\.\d{3})*)(?:,(\d{1,2}))?$", s
    )
    if tr_match:
        integer_part = tr_match.group(1).replace(".", "")
        decimal_part = tr_match.group(2) or "0"
        try:
            return float(f"{integer_part}.{decimal_part}")
        except ValueError:
            return None

    # Plain number with comma decimal but no thousands dot: "1299,99"
    plain_comma = re.match(r"^(\d+),(\d{1,2})$", s)
    if plain_comma:
        try:
            return float(f"{plain_comma.group(1)}.{plain_comma.group(2)}")
        except ValueError:
            return None

    # International / simple float: "1299.99" or "1299"
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 2. normalize_turkish
# ---------------------------------------------------------------------------

def normalize_turkish(text: str) -> str:
    """Lowercase *text* using Turkish-specific casing rules.

    Turkish İ -> i, I -> ı, Ş -> ş, Ç -> ç, Ö -> ö, Ü -> ü, Ğ -> ğ.
    All other characters are lowercased with the default Python rules.
    """
    if not text:
        return text

    chars: list[str] = []
    for ch in text:
        mapped = _TR_UPPER_TO_LOWER.get(ch)
        if mapped is not None:
            chars.append(mapped)
        else:
            chars.append(ch.lower())
    return "".join(chars)


# ---------------------------------------------------------------------------
# 3. generate_search_variants
# ---------------------------------------------------------------------------

def generate_search_variants(query: str) -> list[str]:
    """Return *query* plus Turkish/English translation variants.

    Matching is case-insensitive.  The original query is always the first
    element in the returned list.  Duplicates are removed while preserving
    order.
    """
    if not query:
        return [query] if query is not None else []

    variants: list[str] = [query]
    lower = normalize_turkish(query)

    # Check Turkish -> English
    for tr_term, en_list in _TR_EN_MAP.items():
        if normalize_turkish(tr_term) == lower or normalize_turkish(tr_term) in lower:
            for en in en_list:
                if en not in variants:
                    variants.append(en)
            # Also add the query with the Turkish term replaced by English
            if normalize_turkish(tr_term) != lower:
                for en in en_list:
                    replaced = re.sub(
                        re.escape(tr_term), en, query, flags=re.IGNORECASE
                    )
                    if replaced not in variants:
                        variants.append(replaced)

    # Check English -> Turkish
    for en_term, tr_list in _EN_TR_MAP.items():
        if en_term == lower or en_term in lower:
            for tr in tr_list:
                if tr not in variants:
                    variants.append(tr)
            if en_term != lower:
                for tr in tr_list:
                    replaced = re.sub(
                        re.escape(en_term), tr, query, flags=re.IGNORECASE
                    )
                    if replaced not in variants:
                        variants.append(replaced)

    return variants


# ---------------------------------------------------------------------------
# 4. extract_dimensions
# ---------------------------------------------------------------------------

_DIM_WDH_RE = re.compile(
    r"[Ww]?\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*"
    r"[Dd]?\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*"
    r"[Hh]?\s*(\d+(?:[.,]\d+)?)\s*"
    r"(cm|mm|m)?\b",
    re.IGNORECASE,
)

_DIM_LABELED_EN_RE = re.compile(
    r"[Ww]\s*(\d+(?:[.,]\d+)?)\s*[xX×]?\s*"
    r"[Dd]\s*(\d+(?:[.,]\d+)?)\s*[xX×]?\s*"
    r"[Hh]\s*(\d+(?:[.,]\d+)?)\s*"
    r"(cm|mm|m)?\b",
    re.IGNORECASE,
)

# Turkish labelled dimensions
_DIM_TR_WIDTH_RE = re.compile(
    r"(?:genişlik|en)\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*(cm|mm|m)?\b",
    re.IGNORECASE,
)
_DIM_TR_DEPTH_RE = re.compile(
    r"(?:derinlik|boy)\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*(cm|mm|m)?\b",
    re.IGNORECASE,
)
_DIM_TR_HEIGHT_RE = re.compile(
    r"(?:yükseklik)\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*(cm|mm|m)?\b",
    re.IGNORECASE,
)


def _parse_num(s: str) -> float:
    """Parse a number string that may use comma as decimal separator."""
    return float(s.replace(",", "."))


def _to_cm(value: float, unit: str | None) -> float:
    """Convert *value* in *unit* to centimetres."""
    if unit is None:
        return value
    unit = unit.lower()
    if unit == "mm":
        return value / 10.0
    if unit == "m":
        return value * 100.0
    return value  # cm


def extract_dimensions(text: str) -> dict:
    """Extract width, depth, height from *text* and return values in cm.

    Supported formats:
        "60x55x45 cm"
        "genişlik: 60cm"
        "W600 x D550 x H450mm"
        "en: 60 boy: 55 yükseklik: 45"
    """
    if not text:
        return {}

    result: dict[str, float] = {}

    # Try WxDxH pattern (with optional W/D/H prefixes)
    m = _DIM_WDH_RE.search(text)
    if m:
        unit = m.group(4)
        result["width"] = _to_cm(_parse_num(m.group(1)), unit)
        result["depth"] = _to_cm(_parse_num(m.group(2)), unit)
        result["height"] = _to_cm(_parse_num(m.group(3)), unit)
        return result

    # Try Turkish labelled dimensions
    mw = _DIM_TR_WIDTH_RE.search(text)
    md = _DIM_TR_DEPTH_RE.search(text)
    mh = _DIM_TR_HEIGHT_RE.search(text)

    if mw:
        result["width"] = _to_cm(_parse_num(mw.group(1)), mw.group(2))
    if md:
        result["depth"] = _to_cm(_parse_num(md.group(1)), md.group(2))
    if mh:
        result["height"] = _to_cm(_parse_num(mh.group(1)), mh.group(2))

    return result


# ---------------------------------------------------------------------------
# 5. extract_weight
# ---------------------------------------------------------------------------

_WEIGHT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(kg|gr|g)\b", re.IGNORECASE
)


def extract_weight(text: str) -> float | None:
    """Parse a weight string and return the value in kilograms.

    Handles "5 kg", "5000 g", "5,2 kg", "250 gr", etc.
    """
    if not text:
        return None

    m = _WEIGHT_RE.search(text)
    if not m:
        return None

    value = _parse_num(m.group(1))
    unit = m.group(2).lower()

    if unit in ("g", "gr"):
        return value / 1000.0
    return value  # kg


# ---------------------------------------------------------------------------
# 6. extract_capacity
# ---------------------------------------------------------------------------

_CAPACITY_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(kg|litre|lt|liter|fincan|bardak|kişilik|porsiyon)\b",
    re.IGNORECASE,
)

_CAPACITY_CONTEXT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(kg)\s+"
    r"(yıkama|kurutma|yıkama/kurutma)?\s*kapasitesi?",
    re.IGNORECASE,
)


def extract_capacity(text: str) -> dict:
    """Extract capacity information from *text*.

    Returns a dict with ``value`` (float) and ``unit`` (str) keys.
    Examples:
        "9 kg yıkama kapasitesi" -> {"value": 9.0, "unit": "kg"}
        "500 litre"              -> {"value": 500.0, "unit": "litre"}
        "12 fincan"              -> {"value": 12.0, "unit": "fincan"}
    """
    if not text:
        return {}

    # Try the more specific context pattern first
    mc = _CAPACITY_CONTEXT_RE.search(text)
    if mc:
        return {"value": _parse_num(mc.group(1)), "unit": mc.group(2).lower()}

    m = _CAPACITY_RE.search(text)
    if m:
        unit = m.group(2).lower()
        # Normalise litre variants
        if unit in ("lt", "liter"):
            unit = "litre"
        return {"value": _parse_num(m.group(1)), "unit": unit}

    return {}


# ---------------------------------------------------------------------------
# 7. extract_energy_rating
# ---------------------------------------------------------------------------

_ENERGY_RATING_RE = re.compile(
    r"\b([A-G])(\+{1,3})?(?=\s|$|[^+\w])"
)


def extract_energy_rating(text: str) -> str | None:
    """Extract a European energy rating label from *text*.

    Returns labels like "A+++", "A++", "A+", "A", "B", ... "G", or ``None``.
    """
    if not text:
        return None

    # Search for the best (highest) rating present
    best: str | None = None
    best_rank: int = -1

    for m in _ENERGY_RATING_RE.finditer(text):
        letter = m.group(1).upper()
        pluses = m.group(2) or ""
        label = f"{letter}{pluses}"
        # Rank: A+++ > A++ > A+ > A > B > ... > G
        rank = (ord("G") - ord(letter)) * 10 + len(pluses)
        if rank > best_rank:
            best = label
            best_rank = rank

    return best


# ---------------------------------------------------------------------------
# 8. extract_material
# ---------------------------------------------------------------------------

def extract_material(text: str) -> str | None:
    """Detect material keywords in *text* and return the canonical Turkish name.

    Recognised materials: paslanmaz celik, mese, mdf, cam, ahsap, plastik,
    seramik, granit, mermer.
    """
    if not text:
        return None

    lower = normalize_turkish(text)

    for keyword, canonical in _MATERIALS:
        if keyword in lower:
            return canonical

    return None


# ---------------------------------------------------------------------------
# 9. extract_volume_weight_for_grocery
# ---------------------------------------------------------------------------

_GROCERY_PACK_RE = re.compile(
    r"(\d+)\s*[''´`]?\s*l[iıİI]\s*paket",
    re.IGNORECASE,
)

_GROCERY_VW_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(kg|gr|g|ml|lt|liter|litre|l)\b",
    re.IGNORECASE,
)


def extract_volume_weight_for_grocery(text: str) -> dict | None:
    """Parse grocery volume/weight info from *text*.

    Returns a dict with ``value``, ``unit``, and optionally
    ``per_kg_or_liter`` (always 1.0 for the base unit) when the value
    can be normalised to kg or litre.

    Examples:
        "1 kg"        -> {"value": 1.0, "unit": "kg", "per_kg_or_liter": 1.0}
        "500 ml"      -> {"value": 500.0, "unit": "ml", "per_kg_or_liter": 0.5}
        "6'lı paket"  -> {"value": 6.0, "unit": "paket"}
        "250 gr"      -> {"value": 250.0, "unit": "gr", "per_kg_or_liter": 0.25}
    """
    if not text:
        return None

    # Check for pack notation first ("6'lı paket")
    mp = _GROCERY_PACK_RE.search(text)
    if mp:
        return {"value": float(mp.group(1)), "unit": "paket"}

    m = _GROCERY_VW_RE.search(text)
    if not m:
        return None

    value = _parse_num(m.group(1))
    raw_unit = m.group(2).lower()

    # Normalise unit names
    unit = raw_unit
    if raw_unit in ("lt", "liter", "litre"):
        unit = "l"
    if raw_unit == "gr":
        unit = "gr"

    result: dict = {"value": value, "unit": unit if unit != raw_unit else raw_unit}

    # Calculate per_kg_or_liter where possible
    if unit == "kg":
        result["per_kg_or_liter"] = value
    elif unit in ("g", "gr"):
        result["per_kg_or_liter"] = value / 1000.0
    elif unit == "l":
        result["per_kg_or_liter"] = value
    elif unit == "ml":
        result["per_kg_or_liter"] = value / 1000.0

    return result


# ---------------------------------------------------------------------------
# 10. detect_language
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Return ``"tr"`` if *text* contains Turkish-specific characters, else ``"en"``."""
    if not text:
        return "en"

    for ch in text:
        if ch in _TR_SPECIFIC_CHARS:
            return "tr"

    return "en"


# ---------------------------------------------------------------------------
# 11. normalize_product_name
# ---------------------------------------------------------------------------

def normalize_product_name(text: str) -> str:
    """Strip marketing filler words and excessive whitespace from *text*.

    Filler phrases like "süper fırsat", "kampanyalı", "ücretsiz kargo", etc.
    are removed.  The result is stripped and collapsed to single spaces.
    """
    if not text:
        return text

    result = text

    # Remove filler phrases (case-insensitive, Turkish-aware)
    for phrase in _FILLER_PHRASES:
        # Build a pattern that matches the phrase surrounded by word boundaries
        # or start/end of string, case-insensitive
        pattern = re.compile(
            r"(?<!\w)" + re.escape(phrase) + r"(?!\w)",
            re.IGNORECASE,
        )
        result = pattern.sub("", result)

    # Strip surrounding punctuation remnants (dashes, pipes, slashes)
    result = re.sub(r"[|\-/\\]+\s*$", "", result)
    result = re.sub(r"^\s*[|\-/\\]+", "", result)

    # Collapse multiple spaces and strip
    result = re.sub(r"\s{2,}", " ", result).strip()

    return result
