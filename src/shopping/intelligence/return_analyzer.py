"""Return Policy Analyzer — evaluates return ease for a product/store
combination using the store_profiles knowledge base."""

from __future__ import annotations

import json
from pathlib import Path

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.return_analyzer")

_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


# ─── LLM helper ─────────────────────────────────────────────────────────────

async def _llm_call(prompt: str, system: str = "", temperature: float = 0.3) -> str:
    try:
        import litellm
        response = await litellm.acompletion(
            model="openai/local",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception:
        return ""


# ─── Knowledge loader ───────────────────────────────────────────────────────

_profiles_cache: str | None = None


def _load_store_profiles() -> str:
    """Load store_profiles.md (cached in-process)."""
    global _profiles_cache
    if _profiles_cache is not None:
        return _profiles_cache
    path = _KNOWLEDGE_DIR / "store_profiles.md"
    try:
        _profiles_cache = path.read_text(encoding="utf-8")
        return _profiles_cache
    except Exception as exc:
        logger.warning("Could not load store_profiles.md", error=str(exc))
        return ""


# ─── Store return policy database (parsed from knowledge) ──────────────────

_STORE_RETURN_POLICIES: dict[str, dict] = {
    "trendyol": {
        "return_window_days": 15,
        "free_return": True,
        "return_method": "Trendyol drop-off points or cargo pickup",
        "electronics_note": "15 days unopened, defect-only after opening",
        "marketplace_caveat": True,
        "physical_stores": False,
    },
    "hepsiburada": {
        "return_window_days": 15,
        "free_return": True,
        "return_method": "HepsiJet pickup",
        "electronics_note": "Installation-day inspection critical for appliances",
        "marketplace_caveat": True,
        "physical_stores": False,
    },
    "amazon_tr": {
        "return_window_days": 30,
        "free_return": True,
        "return_method": "Amazon return label / cargo",
        "electronics_note": "Amazon-direct returns seamless, third-party can be slower",
        "marketplace_caveat": True,
        "physical_stores": False,
    },
    "n11": {
        "return_window_days": 15,
        "free_return": False,
        "return_method": "Seller-dependent, platform mediates disputes",
        "electronics_note": "Return process depends on seller cooperation",
        "marketplace_caveat": True,
        "physical_stores": False,
    },
    "mediamarkt": {
        "return_window_days": 15,
        "free_return": True,
        "return_method": "In-store return or cargo",
        "electronics_note": "Open-box electronics accepted with original packaging",
        "marketplace_caveat": False,
        "physical_stores": True,
    },
    "vatanbilgisayar": {
        "return_window_days": 14,
        "free_return": True,
        "return_method": "In-store return or cargo",
        "electronics_note": "Opened software/games excluded",
        "marketplace_caveat": False,
        "physical_stores": True,
    },
    "koctas": {
        "return_window_days": 15,
        "free_return": True,
        "return_method": "In-store return for most items",
        "electronics_note": "Opened paint/custom-cut items excluded",
        "marketplace_caveat": False,
        "physical_stores": True,
    },
    "ikea_tr": {
        "return_window_days": 365,
        "free_return": True,
        "return_method": "In-store return with receipt",
        "electronics_note": "365 days unopened, 180 days opened",
        "marketplace_caveat": False,
        "physical_stores": True,
    },
    "migros": {
        "return_window_days": 0,
        "free_return": True,
        "return_method": "Refund/replacement on delivery issues",
        "electronics_note": "Perishables replaced on quality issues",
        "marketplace_caveat": False,
        "physical_stores": True,
    },
    "getir": {
        "return_window_days": 0,
        "free_return": True,
        "return_method": "In-app refund/replacement",
        "electronics_note": "Simple quality issue process",
        "marketplace_caveat": False,
        "physical_stores": False,
    },
}

_STORE_ALIASES: dict[str, str] = {
    "trendyol": "trendyol",
    "trendyol.com": "trendyol",
    "hepsiburada": "hepsiburada",
    "hepsiburada.com": "hepsiburada",
    "amazon": "amazon_tr",
    "amazon.com.tr": "amazon_tr",
    "amazon_tr": "amazon_tr",
    "n11": "n11",
    "n11.com": "n11",
    "mediamarkt": "mediamarkt",
    "mediamarkt.com.tr": "mediamarkt",
    "vatanbilgisayar": "vatanbilgisayar",
    "vatanbilgisayar.com": "vatanbilgisayar",
    "vatan": "vatanbilgisayar",
    "koctas": "koctas",
    "koçtaş": "koctas",
    "koctas.com.tr": "koctas",
    "ikea": "ikea_tr",
    "ikea.com.tr": "ikea_tr",
    "ikea_tr": "ikea_tr",
    "getir": "getir",
    "migros": "migros",
    "migros.com.tr": "migros",
}


def _normalize_store(store: str) -> str:
    return _STORE_ALIASES.get(store.lower().strip(), store.lower().strip())


# ─── Category detection ─────────────────────────────────────────────────────

_ELECTRONICS_KEYWORDS = [
    "telefon", "phone", "laptop", "tablet", "bilgisayar", "computer",
    "tv", "televizyon", "monitor", "kulaklık", "headphone", "kamera",
    "camera", "gpu", "cpu", "ram", "ssd", "hdd", "yazıcı", "printer",
    "klima", "kombi", "bulaşık", "çamaşır", "buzdolabı", "fırın",
    "elektrikli", "electronic", "oyun konsolu", "console",
]


def _is_electronics(product: dict) -> bool:
    """Detect if a product falls under electronics/appliances."""
    searchable = " ".join([
        product.get("name", ""),
        product.get("category_path", ""),
        str(product.get("specs", {})),
    ]).lower()
    return any(kw in searchable for kw in _ELECTRONICS_KEYWORDS)


# ─── Scoring ─────────────────────────────────────────────────────────────────

def _score_return_ease(policy: dict, is_electronic: bool) -> tuple[str, float]:
    """Score the return ease and produce a badge.

    Returns: (badge, score 0.0-1.0)
    """
    score = 0.0

    window = policy.get("return_window_days", 0)
    if window >= 30:
        score += 0.35
    elif window >= 15:
        score += 0.25
    elif window >= 14:
        score += 0.2
    elif window > 0:
        score += 0.1

    if policy.get("free_return"):
        score += 0.25

    if policy.get("physical_stores"):
        score += 0.15  # in-store return is easier

    if not policy.get("marketplace_caveat"):
        score += 0.15  # direct retail more reliable

    # Electronics penalty for strict policies
    if is_electronic and "defect-only" in policy.get("electronics_note", "").lower():
        score -= 0.1

    score = max(0.0, min(1.0, score))

    if score >= 0.7:
        badge = "Easy return \u2705"
    elif score >= 0.4:
        badge = "Standard return"
    else:
        badge = "Difficult return \u26a0\ufe0f"

    return badge, round(score, 2)


# ─── Main entry point ───────────────────────────────────────────────────────

async def analyze_return_policy(product: dict, store: str) -> dict:
    """Analyze return policy for a product at a given store.

    Args:
        product: product dict (from models.Product as dict)
        store: store name or domain

    Returns:
        dict with: badge, score, return_window_days, free_return, return_method,
                   conditions, warnings, store_notes
    """
    logger.info("Analyzing return policy", product=product.get("name", "?"), store=store)

    store_key = _normalize_store(store)
    policy = _STORE_RETURN_POLICIES.get(store_key)
    is_electronic = _is_electronics(product)

    if not policy:
        # Try LLM with store profiles knowledge for unknown stores
        profiles_md = _load_store_profiles()
        if profiles_md:
            llm_prompt = (
                f"Store: {store}\nProduct: {product.get('name', '')}\n\n"
                f"Based on the store profiles below, what is the return policy?\n"
                f"Return JSON: {{\"return_window_days\": N, \"free_return\": bool, "
                f"\"return_method\": \"...\", \"conditions\": \"...\"}}\n\n"
                f"--- Store Profiles ---\n{profiles_md[:3000]}"
            )
            llm_resp = await _llm_call(
                llm_prompt,
                system="You are a Turkish e-commerce return policy expert. Return only valid JSON.",
            )
            if llm_resp:
                try:
                    llm_data = json.loads(llm_resp)
                    policy = {
                        "return_window_days": llm_data.get("return_window_days", 15),
                        "free_return": llm_data.get("free_return", False),
                        "return_method": llm_data.get("return_method", "Unknown"),
                        "electronics_note": llm_data.get("conditions", ""),
                        "marketplace_caveat": True,
                        "physical_stores": False,
                    }
                except (json.JSONDecodeError, TypeError):
                    pass

    if not policy:
        logger.info("No return policy data available", store=store_key)
        return {
            "badge": "Unknown return policy \u2753",
            "score": 0.0,
            "return_window_days": None,
            "free_return": None,
            "return_method": "Unknown",
            "conditions": [],
            "warnings": [f"Return policy for '{store}' not found in knowledge base"],
            "store_notes": "",
        }

    badge, score = _score_return_ease(policy, is_electronic)

    # Build conditions list
    conditions: list[str] = []
    if policy.get("electronics_note"):
        conditions.append(policy["electronics_note"])
    if policy.get("marketplace_caveat"):
        conditions.append("Marketplace seller: return experience may vary by seller")

    # Warnings
    warnings: list[str] = []
    if is_electronic and policy.get("return_window_days", 0) <= 15:
        warnings.append("Electronics: test thoroughly within return window, opened items may be defect-only")
    if policy.get("marketplace_caveat"):
        seller_name = product.get("seller_name", "")
        if seller_name and seller_name.lower() not in [store_key, store]:
            warnings.append(f"Third-party seller ({seller_name}): return handling may differ from platform standard")

    # Seller rating context
    seller_rating = product.get("seller_rating")
    if seller_rating is not None and seller_rating < 4.0:
        warnings.append(f"Seller rating is {seller_rating}/5 — may indicate return issues")

    result = {
        "badge": badge,
        "score": score,
        "return_window_days": policy.get("return_window_days"),
        "free_return": policy.get("free_return"),
        "return_method": policy.get("return_method", ""),
        "has_physical_stores": policy.get("physical_stores", False),
        "conditions": conditions,
        "warnings": warnings,
        "store_notes": "",
    }

    logger.info("Return policy analyzed", store=store_key, badge=badge, score=score)
    return result
