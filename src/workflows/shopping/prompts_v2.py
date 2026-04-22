"""Frozen LLM prompt templates for shopping pipeline v2.

Keep prompts versioned in this module. Any change that affects output shape
must bump the version comment and update the parser tests.
"""
from __future__ import annotations

# ─── Grouping ──────────────────────────────────────────────────────────────
# Version: v2.0.0
# Output contract: JSON list of {representative_title, member_indices, is_accessory_or_part}
GROUPING_PROMPT = """You are grouping shopping search results across multiple e-commerce sites.

Input candidates (each has an integer `index`):
{candidates_json}

Task:
1. Cluster candidates that refer to the SAME product (same brand + model + variant).
   Different colours or storage tiers of the same model are the same group.
   Different models from the same product line are DIFFERENT groups (e.g. Siemens EQ.3 vs EQ.6).
2. Flag accessories, replacement parts, filters, covers, or spare components as `is_accessory_or_part: true`.
   A full coffee machine is NOT a part. A brewing unit / demleme ünitesi sold separately IS a part.
3. Pick a clean representative_title for each group (shortest member title is usually best).

Return ONLY valid JSON in this exact shape, no prose, no markdown fences:
{{
  "groups": [
    {{
      "representative_title": "string",
      "member_indices": [int, ...],
      "is_accessory_or_part": bool
    }}
  ]
}}
"""

# ─── Synthesis ─────────────────────────────────────────────────────────────
# Version: v2.0.0
# Output contract: JSON {praise, complaints, red_flags, insufficient_data}
SYNTHESIS_PROMPT = """You are summarising user reviews for a product across multiple sources.

Product: {representative_title}

Review snippets (Turkish and/or English, from multiple sources):
{review_snippets_json}

Task:
- Extract recurring PRAISE points (what users like). Up to 5 bullets, short phrases.
- Extract recurring COMPLAINTS (what users dislike). Up to 5 bullets.
- Extract RED FLAGS (safety, reliability, fraud concerns, complaint-site mentions). Up to 5 bullets.
- If the snippets are too few, too short, or irrelevant to judge this product, set insufficient_data=true and leave lists empty.
- Do NOT fabricate points that aren't supported by the snippets. Better to output insufficient_data=true than to guess.
- Output in the dominant language of the snippets (Turkish if Turkish dominates).

Return ONLY valid JSON, no prose, no markdown fences:
{{
  "praise": ["string", ...],
  "complaints": ["string", ...],
  "red_flags": ["string", ...],
  "insufficient_data": bool
}}
"""

# ─── Label ─────────────────────────────────────────────────────────────────
# Version: v2.0.0
# Output contract: JSON {groups: [{group_id, product_type, base_model, variant, authenticity_confidence, matches_user_intent}]}
LABEL_PROMPT = """You classify product-search result groups for a Turkish shopping bot.

User query: {query}

Groups (each is one or more scraped listings that we already think refer to the same product):
{groups_json}

For EVERY group, return a JSON object:
- group_id: copy the input id verbatim
- product_type: one of "authentic_product", "accessory", "replacement_part", "knockoff", "refurbished", "unknown"
  * authentic_product = a real, new, first-party product that matches the query
  * accessory = a case, charger, cable, holder, screen protector, bag, strap, etc.
  * replacement_part = a screen panel, battery, motherboard, button — a part of a product, not the product
  * knockoff = counterfeit / non-branded clone / suspicious listing
  * refurbished = used / refurbished / grade-B / open-box
  * unknown = cannot tell from title + category
- base_model: the canonical product line, e.g. "Samsung Galaxy S25" (strip variant suffix)
- variant: the variant suffix if any (e.g. "FE", "Plus", "Ultra", "Pro", "Mini", color code, storage); null when base model has no variant
- authenticity_confidence: 0.0–1.0 — how sure you are the listing is the authentic product
- matches_user_intent: true if answering this group tells the user what they asked; false if they'd consider it the wrong thing (accessory for a phone query, part for a product query, knockoff, etc.)

Return only the JSON object:
{{
  "groups": [ {{...}}, {{...}}, ... ]
}}
"""
