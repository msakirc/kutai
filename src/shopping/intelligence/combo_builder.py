"""Combo Builder — assembles multi-component purchase sets (PC builds, kitchen
setups, etc.) with compatibility checking and tiered budget recommendations."""

from __future__ import annotations

import json
from itertools import product as itertools_product
from pathlib import Path

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.combo_builder")

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


# ─── Knowledge loaders ──────────────────────────────────────────────────────

def _load_compatibility_data() -> dict:
    """Load all compatibility JSON files from the knowledge base."""
    compat_dir = _KNOWLEDGE_DIR / "compatibility"
    data: dict = {}
    if not compat_dir.exists():
        return data
    try:
        for path in compat_dir.glob("*.json"):
            with open(path, encoding="utf-8") as f:
                data[path.stem] = json.load(f)
    except Exception as exc:
        logger.warning("Error loading compatibility data", error=str(exc))
    return data


# ─── Compatibility checking ─────────────────────────────────────────────────

def _check_pair_compatibility(a: dict, b: dict, compat_data: dict) -> dict | None:
    """Check whether two component dicts are compatible.

    Returns a dict with {compatible: bool, note: str} or None if no rule applies.
    """
    a_cat = (a.get("category") or "").lower()
    b_cat = (b.get("category") or "").lower()

    # CPU + Motherboard socket check
    if {"cpu", "motherboard"} <= {a_cat, b_cat} or {"işlemci", "anakart"} <= {a_cat, b_cat}:
        sockets = compat_data.get("cpu_sockets", {})
        a_socket = (a.get("specs") or {}).get("socket", "")
        b_socket = (b.get("specs") or {}).get("socket", "")
        if a_socket and b_socket:
            if a_socket == b_socket:
                return {"compatible": True, "note": f"Socket match: {a_socket}"}
            return {"compatible": False, "note": f"Socket mismatch: {a_socket} vs {b_socket}"}

    # RAM + Motherboard DDR check
    if {"ram", "motherboard"} <= {a_cat, b_cat} or {"bellek", "anakart"} <= {a_cat, b_cat}:
        ram_compat = compat_data.get("ram_compatibility", {})
        a_type = (a.get("specs") or {}).get("ram_type", "")
        b_type = (b.get("specs") or {}).get("ram_type", "")
        if a_type and b_type:
            if a_type == b_type:
                return {"compatible": True, "note": f"RAM type match: {a_type}"}
            return {"compatible": False, "note": f"RAM type mismatch: {a_type} vs {b_type}"}

    return None  # no rule found


def _build_compatibility_graph(candidates_per_slot: list[list[dict]], compat_data: dict) -> list[dict]:
    """Check pairwise compatibility across all component slots.

    Returns a list of compatibility notes (both passes and failures).
    """
    notes: list[dict] = []
    for i, slot_a in enumerate(candidates_per_slot):
        for j, slot_b in enumerate(candidates_per_slot):
            if j <= i:
                continue
            # Check first candidate from each slot as representative
            if slot_a and slot_b:
                result = _check_pair_compatibility(slot_a[0], slot_b[0], compat_data)
                if result:
                    notes.append({
                        "slot_a": i,
                        "slot_b": j,
                        **result,
                    })
    return notes


# ─── Scoring ─────────────────────────────────────────────────────────────────

def _score_combo(combo_products: list[dict]) -> dict:
    """Score a combination of products.

    Returns {total_price, value_score, convenience_score, store_count}.
    """
    total_price = 0.0
    stores: set[str] = set()
    ratings_sum = 0.0
    rated_count = 0

    for p in combo_products:
        price = p.get("discounted_price") or p.get("original_price") or 0.0
        total_price += price
        source = p.get("source", "")
        if source:
            stores.add(source)
        rating = p.get("rating")
        if rating:
            ratings_sum += rating
            rated_count += 1

    avg_rating = (ratings_sum / rated_count) if rated_count else 3.0
    # Fewer stores = more convenient (1 store = 1.0, 5+ stores = 0.2)
    convenience = max(1.0 - (len(stores) - 1) * 0.2, 0.2)
    # Value = average rating normalized to 0-1 * convenience
    value_score = (avg_rating / 5.0) * 0.7 + convenience * 0.3

    return {
        "total_price": round(total_price, 2),
        "value_score": round(value_score, 2),
        "convenience_score": round(convenience, 2),
        "store_count": len(stores),
        "stores": sorted(stores),
    }


def _assign_tier(price: float, budget: float) -> str:
    """Assign a combo to budget/mid/premium tier based on proportion of budget used."""
    ratio = price / budget if budget else 1.0
    if ratio <= 0.7:
        return "budget"
    elif ratio <= 1.0:
        return "mid"
    else:
        return "premium"


# ─── Main entry point ───────────────────────────────────────────────────────

async def build_combos(
    components: list[dict],
    budget: float,
    constraints: list,
) -> list[dict]:
    """Build multi-component combo recommendations.

    Args:
        components: list of dicts, each with:
            - role: str (e.g. "cpu", "gpu", "monitor")
            - candidates: list[dict] — product dicts per slot
        budget: total budget in TRY
        constraints: list of constraint dicts (type, value, hard_or_soft)

    Returns:
        list of combo dicts sorted by value_score, with tier labels.
    """
    logger.info("Building combos", num_components=len(components), budget=budget)

    if not components:
        logger.warning("No components provided")
        return []

    compat_data = _load_compatibility_data()

    # --- Gather candidates per slot ---
    slots: list[list[dict]] = []
    slot_roles: list[str] = []
    for comp in components:
        candidates = comp.get("candidates", [])
        role = comp.get("role", "unknown")
        if not candidates:
            logger.warning("No candidates for slot", role=role)
            continue
        slots.append(candidates)
        slot_roles.append(role)

    if not slots:
        return []

    # --- Compatibility graph ---
    compat_notes = _build_compatibility_graph(slots, compat_data)
    has_incompatibility = any(not n.get("compatible", True) for n in compat_notes)

    if has_incompatibility:
        logger.info("Incompatibilities detected, will filter combos")

    # --- Generate combinations (capped to avoid explosion) ---
    MAX_COMBOS = 200
    # Limit candidates per slot to keep combinatorial space manageable
    capped_slots = [s[:5] for s in slots]

    all_combos: list[dict] = []
    for combo_tuple in itertools_product(*capped_slots):
        combo_products = list(combo_tuple)

        # Check pairwise compatibility for this specific combo
        skip = False
        combo_compat_notes: list[str] = []
        for i in range(len(combo_products)):
            for j in range(i + 1, len(combo_products)):
                result = _check_pair_compatibility(combo_products[i], combo_products[j], compat_data)
                if result:
                    if not result["compatible"]:
                        skip = True
                        break
                    combo_compat_notes.append(result["note"])
            if skip:
                break

        if skip:
            continue

        scores = _score_combo(combo_products)
        tier = _assign_tier(scores["total_price"], budget)

        # Apply hard budget constraint
        hard_budget = any(
            c.get("type") == "budget" and c.get("hard_or_soft") == "hard"
            for c in constraints
        )
        if hard_budget and scores["total_price"] > budget:
            continue

        combo_entry = {
            "products": [
                {
                    "role": slot_roles[i] if i < len(slot_roles) else "unknown",
                    "name": p.get("name", ""),
                    "price": p.get("discounted_price") or p.get("original_price") or 0.0,
                    "source": p.get("source", ""),
                    "url": p.get("url", ""),
                }
                for i, p in enumerate(combo_products)
            ],
            "total_price": scores["total_price"],
            "value_score": scores["value_score"],
            "convenience_score": scores["convenience_score"],
            "store_count": scores["store_count"],
            "stores": scores["stores"],
            "compatibility_notes": combo_compat_notes,
            "tier": tier,
        }
        all_combos.append(combo_entry)

        if len(all_combos) >= MAX_COMBOS:
            break

    # --- Sort by value score descending ---
    all_combos.sort(key=lambda c: c["value_score"], reverse=True)

    # --- Pick best per tier ---
    tiered: dict[str, dict | None] = {"budget": None, "mid": None, "premium": None}
    for combo in all_combos:
        t = combo["tier"]
        if t in tiered and tiered[t] is None:
            tiered[t] = combo

    # --- LLM summary for the top combos ---
    top_combos = [c for c in tiered.values() if c is not None]
    if top_combos:
        summary_prompt = (
            f"Budget: {budget} TRY\n"
            f"Components needed: {', '.join(slot_roles)}\n"
            f"Top combos:\n{json.dumps(top_combos[:3], ensure_ascii=False, indent=2)}\n\n"
            "Write a brief (2-3 sentence) Turkish comparison of these tiers. "
            "Focus on value trade-offs. Return JSON: {\"summary\": \"...\"}"
        )
        llm_resp = await _llm_call(
            summary_prompt,
            system="You are a Turkish shopping advisor specializing in multi-component purchases.",
        )
        if llm_resp:
            try:
                summary_data = json.loads(llm_resp)
                for combo in top_combos:
                    combo["llm_summary"] = summary_data.get("summary", "")
            except (json.JSONDecodeError, TypeError):
                pass

    result = [c for c in [tiered["budget"], tiered["mid"], tiered["premium"]] if c is not None]
    logger.info("Combos built", total_generated=len(all_combos), returned=len(result))
    return result
