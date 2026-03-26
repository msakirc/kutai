"""Installment (Taksit) Calculator — loads store-bank partnership data and
computes monthly payment options for a given price and store."""

from __future__ import annotations

import json
from pathlib import Path

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.installment_calculator")

_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"

from ._llm import _llm_call

# ─── Knowledge loader ───────────────────────────────────────────────────────

_installment_cache: dict | None = None


def _load_installments() -> dict:
    """Load installments.json knowledge base (cached in-process)."""
    global _installment_cache
    if _installment_cache is not None:
        return _installment_cache
    path = _KNOWLEDGE_DIR / "installments.json"
    try:
        with open(path, encoding="utf-8") as f:
            _installment_cache = json.load(f)
        return _installment_cache
    except Exception as exc:
        logger.warning("Could not load installments.json", error=str(exc))
        return {}


# ─── Interest rate estimation ───────────────────────────────────────────────

# Typical monthly interest rates for non-faizsiz taksit tiers in Turkey (approximate)
_DEFAULT_MONTHLY_RATES: dict[int, float] = {
    2: 0.00,    # 2-taksit almost always faizsiz
    3: 0.00,    # often faizsiz
    6: 0.0189,  # ~1.89% monthly
    9: 0.0199,  # ~1.99% monthly
    12: 0.0209, # ~2.09% monthly
}


def _monthly_rate(tier: int, is_faizsiz: bool) -> float:
    """Return monthly interest rate for a given tier."""
    if is_faizsiz:
        return 0.0
    return _DEFAULT_MONTHLY_RATES.get(tier, 0.02)


def _compute_installment(price: float, tier: int, is_faizsiz: bool) -> dict:
    """Compute installment details for a single tier.

    Returns:
        dict with monthly_payment, total_amount, interest_amount, interest_pct, is_faizsiz, tier
    """
    rate = _monthly_rate(tier, is_faizsiz)
    if rate == 0.0:
        monthly = price / tier
        total = price
    else:
        # Standard amortization
        monthly = price * (rate * (1 + rate) ** tier) / (((1 + rate) ** tier) - 1)
        total = monthly * tier

    interest_amount = total - price
    interest_pct = (interest_amount / price) * 100 if price else 0.0

    return {
        "tier": tier,
        "months": tier,
        "monthly_payment": round(monthly, 2),
        "total_amount": round(total, 2),
        "interest_amount": round(interest_amount, 2),
        "interest_pct": round(interest_pct, 1),
        "is_faizsiz": is_faizsiz,
    }


# ─── Store name normalization ───────────────────────────────────────────────

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
}


def _normalize_store(store: str) -> str:
    return _STORE_ALIASES.get(store.lower().strip(), store.lower().strip())


# ─── Main entry point ───────────────────────────────────────────────────────

async def calculate_installments(price: float, store: str) -> list[dict]:
    """Calculate installment options for a price at a given store.

    Args:
        price: product price in TRY
        store: store name or domain

    Returns:
        list of installment option dicts, one per bank-tier combination,
        sorted by monthly_payment ascending.
    """
    logger.info("Calculating installments", price=price, store=store)

    if price <= 0:
        logger.warning("Invalid price for installment calculation", price=price)
        return []

    kb = _load_installments()
    store_key = _normalize_store(store)
    stores_data = kb.get("stores", {})
    bank_cards = kb.get("bank_cards", {})

    store_info = stores_data.get(store_key)
    if not store_info:
        logger.info("Store not found in installments KB, using generic tiers", store=store_key)
        # Fallback: generic tiers without faizsiz info
        results = []
        for tier in [2, 3, 6, 9, 12]:
            entry = _compute_installment(price, tier, is_faizsiz=(tier <= 3))
            entry["bank"] = "generic"
            entry["card_name"] = ""
            entry["notes"] = "Store not in knowledge base — tiers are estimated"
            results.append(entry)
        return sorted(results, key=lambda r: r["monthly_payment"])

    # --- Build options from store-bank partnerships ---
    results: list[dict] = []
    partnerships = store_info.get("bank_partnerships", {})

    for bank_key, bank_info in partnerships.items():
        max_taksit = bank_info.get("max_taksit", 6)
        faizsiz_tiers = set(bank_info.get("faizsiz_tiers", []))
        bank_notes = bank_info.get("notes", "")

        # Card names from bank_cards
        card_info = bank_cards.get(bank_key, {})
        card_names = card_info.get("card_names", [])
        card_name_str = ", ".join(card_names) if card_names else ""

        # Generate an entry for each standard tier up to max_taksit
        for tier in [2, 3, 6, 9, 12]:
            if tier > max_taksit:
                continue

            is_faizsiz = tier in faizsiz_tiers
            entry = _compute_installment(price, tier, is_faizsiz)
            entry["bank"] = bank_key
            entry["card_name"] = card_name_str
            entry["notes"] = bank_notes
            entry["requires_specific_card"] = bool(card_name_str)
            results.append(entry)

    # --- Add peşin (single payment) option ---
    results.insert(0, {
        "tier": 1,
        "months": 1,
        "monthly_payment": round(price, 2),
        "total_amount": round(price, 2),
        "interest_amount": 0.0,
        "interest_pct": 0.0,
        "is_faizsiz": True,
        "bank": "any",
        "card_name": "",
        "notes": "Peşin ödeme (tek çekim)",
        "requires_specific_card": False,
    })

    # --- Sort: faizsiz first within each tier, then by monthly payment ---
    results.sort(key=lambda r: (r["tier"], not r["is_faizsiz"], r["monthly_payment"]))

    # --- LLM recommendation on best option ---
    if results:
        faizsiz_options = [r for r in results if r["is_faizsiz"] and r["tier"] > 1]
        if faizsiz_options:
            best_faizsiz = max(faizsiz_options, key=lambda r: r["tier"])
            # Attach recommendation to results metadata
            for r in results:
                r["recommended"] = (
                    r["bank"] == best_faizsiz["bank"]
                    and r["tier"] == best_faizsiz["tier"]
                    and r["is_faizsiz"]
                )
        else:
            for r in results:
                r["recommended"] = r["tier"] == 1  # recommend peşin if no faizsiz

    store_notes = store_info.get("notes", "")
    if store_notes:
        for r in results:
            if not r.get("notes"):
                r["notes"] = store_notes

    logger.info("Installment options calculated", store=store_key, options_count=len(results))
    return results
