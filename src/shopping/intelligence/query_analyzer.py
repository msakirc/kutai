"""Query analysis module for the shopping intelligence system.

Sends a user's raw shopping query to a local LLM for structured analysis.
Falls back to keyword-based heuristics when the LLM is unavailable.
"""

from __future__ import annotations

import json
import re

from src.infra.logging_config import get_logger
from src.shopping.text_utils import (
    detect_language,
    normalize_turkish,
    generate_search_variants,
)

logger = get_logger("shopping.intelligence.query_analyzer")

from ._llm import _llm_call

# ── Keyword-based fallback ───────────────────────────────────────────────────

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "compare": ["karşılaştır", "kıyasla", "fark", "vs", "versus", "compare"],
    "find_cheapest": [
        "ucuz", "uygun fiyat", "en ucuz", "hesaplı", "cheap", "budget",
        "bütçe", "indirim", "kampanya",
    ],
    "find_best": [
        "en iyi", "best", "tavsiye", "öner", "recommend", "top",
    ],
    "specific_product": [],  # determined by brand/model detection
    "explore": [
        "ne almalıyım", "hangisi", "which", "bakıyorum", "arıyorum",
    ],
}

_URGENCY_KEYWORDS: list[str] = [
    "acil", "hemen", "bugün", "yarın", "urgent", "today", "tomorrow",
    "hızlı", "çabuk",
]

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "electronics": [
        "telefon", "laptop", "bilgisayar", "tablet", "kulaklık", "hoparlör",
        "monitor", "ekran kartı", "ram", "ssd", "phone", "computer",
    ],
    "appliances": [
        "buzdolabı", "çamaşır", "bulaşık", "fırın", "mikrodalga",
        "klima", "süpürge", "ütü", "refrigerator", "washing", "dishwasher",
    ],
    "furniture": [
        "koltuk", "masa", "sandalye", "dolap", "yatak", "kanepe",
        "chair", "table", "desk", "sofa",
    ],
    "grocery": [
        "yiyecek", "içecek", "market", "gıda", "süt", "peynir",
        "food", "grocery", "drink",
    ],
    "clothing": [
        "giyim", "ayakkabı", "çanta", "elbise", "gömlek", "pantolon",
        "shoes", "bag", "dress", "shirt",
    ],
}

_BUDGET_RE = re.compile(
    r"(\d[\d.,]*)\s*(tl|lira|₺)",
    re.IGNORECASE,
)

# Extended budget patterns: "30k", "30bin", "$500", "under 500", "budget 500",
# "max 500", "bütçe 30000", "30.000 lira", "30000₺"
_BUDGET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Turkish lira: "5000 TL", "5.000 TL", "5000₺", "5000 lira"
    (re.compile(r"(\d[\d.,]*)\s*(tl|lira|₺)", re.IGNORECASE), "TRY"),
    # Turkish shorthand: "30k", "30bin" (interpret as TRY)
    (re.compile(r"(\d+)\s*(?:k|bin)\b", re.IGNORECASE), "TRY_k"),
    # Turkish budget keyword: "bütçe 30000", "bütçem 5000"
    (re.compile(r"b[üu]t[çc]e\w*\s+(\d[\d.,]*)", re.IGNORECASE), "TRY"),
    # English dollar: "$500", "$ 500"
    (re.compile(r"\$\s*(\d[\d.,]*)", re.IGNORECASE), "USD"),
    # English keywords: "under 500", "budget 500", "max 500"
    (re.compile(r"(?:under|budget|max|maximum)\s+(\d[\d.,]*)", re.IGNORECASE), "AMB"),
]

_CONSTRAINT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("dimensional", re.compile(r"\d+\s*(cm|mm|m|inç|inch)\b", re.IGNORECASE)),
    ("budget", _BUDGET_RE),
    ("electrical", re.compile(r"\d+\s*(watt|volt|amper|hz)\b", re.IGNORECASE)),
    ("compatibility", re.compile(r"uyumlu|compatible|fit", re.IGNORECASE)),
]


def _detect_intent(lower: str) -> str:
    """Return best-guess intent from keyword matching."""
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return intent
    return "explore"


def _detect_category(lower: str) -> str | None:
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return cat
    return None


def _extract_constraints(raw: str) -> list[str]:
    found: list[str] = []
    for ctype, pattern in _CONSTRAINT_PATTERNS:
        if pattern.search(raw):
            found.append(ctype)
    # Also check extended budget patterns if not already detected
    if "budget" not in found:
        for pattern, _ in _BUDGET_PATTERNS:
            if pattern.search(raw):
                found.append("budget")
                break
    return found


def _detect_urgency(lower: str) -> str:
    if any(kw in lower for kw in _URGENCY_KEYWORDS):
        return "high"
    return "normal"


def _extract_budget(raw: str) -> float | None:
    """Extract a numeric budget from the query string.

    Supports Turkish (TL, lira, ₺, k/bin shorthand, bütçe keyword)
    and English ($, under/budget/max prefix) patterns.
    """
    for pattern, currency_hint in _BUDGET_PATTERNS:
        m = pattern.search(raw)
        if m:
            try:
                val = float(m.group(1).replace(".", "").replace(",", "."))
                if currency_hint == "TRY_k":
                    val *= 1000
                return val
            except ValueError:
                continue
    return None


def _is_vague_query(raw_query: str, constraints: list[str], budget: float | None) -> bool:
    """Determine if a query is too vague to search.

    In practice, almost nothing is too vague — scrapers handle broad queries
    and return results the user can refine. Only truly empty queries are vague.
    """
    return not raw_query.strip()


def _fallback_analyze(raw_query: str) -> dict:
    """Keyword-based analysis when the LLM is unavailable."""
    lower = normalize_turkish(raw_query)
    variants = generate_search_variants(raw_query)
    lang = detect_language(raw_query)
    constraints = _extract_constraints(raw_query)
    budget = _extract_budget(raw_query)
    vague = _is_vague_query(raw_query, constraints, budget)

    return {
        "intent": _detect_intent(lower),
        "category": _detect_category(lower),
        "products_mentioned": variants[1:] if len(variants) > 1 else [],
        "constraints": constraints,
        "missing_info": [],
        "follow_up_questions": [],
        "urgency": _detect_urgency(lower),
        "experience_level": "unknown",
        "search_complexity": "simple" if len(raw_query.split()) <= 4 else "moderate",
        "language": lang,
        "budget": budget,
        "needs_clarification": vague,
        "source": "fallback",
    }


# ── System prompt for the LLM ───────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a Turkish shopping-query analyser.  Given a user query, return ONLY
a JSON object (no markdown, no explanation) with these keys:

- intent: one of "compare", "find_cheapest", "find_best", "specific_product", "explore"
- category: product category (electronics, appliances, furniture, grocery, clothing, or null)
- products_mentioned: list of product names / brands explicitly mentioned
- constraints: list of constraint strings (e.g. "budget:5000TL", "width<60cm")
- missing_info: list of information that would help narrow results
- follow_up_questions: list of clarifying questions to ask the user (in Turkish)
- urgency: "high" | "normal" | "low"
- experience_level: "beginner" | "intermediate" | "expert" | "unknown"
- search_complexity: "simple" | "moderate" | "complex"
- budget: numeric budget in TRY or null
"""

# ── Public API ───────────────────────────────────────────────────────────────


async def analyze_query(raw_query: str) -> dict:
    """Analyse a shopping query and return structured information.

    Attempts an LLM call first; falls back to keyword heuristics on failure.

    Parameters
    ----------
    raw_query:
        The user's original shopping query text.

    Returns
    -------
    dict with keys: intent, category, products_mentioned, constraints,
    missing_info, follow_up_questions, urgency, experience_level,
    search_complexity (and possibly budget, language, source).
    """
    if not raw_query or not raw_query.strip():
        logger.warning("analyze_query called with empty query")
        return _fallback_analyze("")

    try:
        llm_response = await _llm_call(
            prompt=raw_query,
            system=_SYSTEM_PROMPT,
            temperature=0.2,
        )

        if llm_response:
            # Strip markdown fences if present
            cleaned = llm_response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)

            result = json.loads(cleaned)

            # Ensure all expected keys exist
            defaults = {
                "intent": "explore",
                "category": None,
                "products_mentioned": [],
                "constraints": [],
                "missing_info": [],
                "follow_up_questions": [],
                "urgency": "normal",
                "experience_level": "unknown",
                "search_complexity": "simple",
            }
            for key, default in defaults.items():
                result.setdefault(key, default)

            # Derive needs_clarification if the LLM didn't set it
            if "needs_clarification" not in result:
                result["needs_clarification"] = _is_vague_query(
                    raw_query,
                    result.get("constraints", []),
                    result.get("budget"),
                )

            result["source"] = "llm"
            logger.info("Query analysed via LLM: intent=%s", result["intent"])
            return result

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("LLM response parsing failed: %s", exc)
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)

    logger.info("Falling back to keyword-based analysis")
    return _fallback_analyze(raw_query)
