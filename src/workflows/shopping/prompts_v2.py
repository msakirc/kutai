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
# Version: v3.0.0 — aspect-based with verbatim quotes + comparative mining
# Output contract: JSON {praise, complaints, red_flags, insufficient_data,
#   aspects, overall_sentiment, notable_quote, comparative_mentions}
SYNTHESIS_PROMPT = """You are an INTELLIGENCE module summarising user reviews for one product across many sources. Mine substance, ignore boilerplate ("teşekkürler", "kargo hızlı"). Output ONLY valid JSON, no prose, no fences. Use the dominant language of the snippets (Turkish if Turkish dominates).

Product: {representative_title}

Review snippets (Turkish and/or English, multi-source):
{review_snippets_json}

Mine these dimensions. For each aspect ACTUALLY discussed in snippets, emit one entry:
- aspect: one of `kamera`, `pil`, `ekran`, `performans`, `yapım_kalitesi`, `yazılım`, `fiyat`, `satıcı`, `kargo`, `ses`, `şarj`, `güncellemeler`, `oyun`, `boyut`, `ergonomi`, `ısınma` (use only those that actually appear; do not invent).
- sentiment: float in [-1.0, 1.0] (avg user feeling on this aspect).
- mention_count: int (how many distinct snippets discussed this aspect).
- summary: single short Turkish line capturing the consensus on this aspect (no fluff).
- quote: ONE verbatim snippet that best represents the aspect (≤140 chars, copied as-is, no ellipses unless the original had them).

Rules:
- Skip generic praise/complaints. "Çok iyi telefon" is NOT an aspect insight.
- Prefer SPECIFIC, SUBSTANTIVE statements ("Kamera 50MP çok net", "Pil 1.5 günde bitiyor", "Şarj 25W yavaş gelmiş").
- aspects must be sorted by mention_count desc.
- Up to 8 aspects.
- Emit `comparative_mentions` (≤3): verbatim snippets that compare this product to a rival ("X'ten daha iyi", "Y'ye göre").
- Emit `notable_quote`: the single most informative review excerpt (≤200 chars, verbatim).
- `praise`, `complaints`, `red_flags` (≤5 each) — short bullets, language as snippets.
- `red_flags`: only safety / reliability / fraud / build-defect concerns. NOT "fiyat yüksek".
- `overall_sentiment`: float [-1.0, 1.0] derived from snippet balance.
- `insufficient_data`: true only if snippets are <3 substantive ones or all boilerplate. When true, leave aspects=[], lists=[], notable_quote="".
- DO NOT fabricate. If a snippet doesn't say it, don't write it.

Return ONLY this JSON:
{{
  "aspects": [{{"aspect": "string", "sentiment": float, "mention_count": int, "summary": "string", "quote": "string"}}],
  "comparative_mentions": ["string", ...],
  "notable_quote": "string",
  "overall_sentiment": float,
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
- line_id: a canonical lowercase ASCII slug uniquely identifying the product LINE this listing belongs to (NOT the SKU, NOT the variant). Use only [a-z0-9-]. Same line ⇒ same slug across listings, regardless of color/storage/seller wording. Examples:
  * "Xiaomi Redmi Note 15 Pro 256 GB" → "xiaomi-redmi-note-15-pro"
  * "Xiaomi REDMı Note 15 Pro 8 GB+256 GB Titanyum Gri" → "xiaomi-redmi-note-15-pro"
  * "Samsung Galaxy S25 256 GB Buz Mavisi" → "samsung-galaxy-s25"
  * "Samsung Galaxy S25 FE" → "samsung-galaxy-s25-fe"
  * "Apple MacBook Pro 14 inch M3" → "apple-macbook-pro-14"
  Two listings of the SAME product line with different color/storage/seller MUST share line_id. Two DIFFERENT lines (S25 vs S25 FE vs S25 Ultra) MUST have different line_ids.
- product_type: one of "authentic_product", "accessory", "replacement_part", "knockoff", "refurbished", "unknown"
  * authentic_product = a real, new, first-party product that matches the query
  * accessory = a case, charger, cable, holder, screen protector, bag, strap, etc.
  * replacement_part = a screen panel, battery, motherboard, button — a part of a product, not the product
  * knockoff = counterfeit / non-branded clone / suspicious listing
  * refurbished = used / refurbished / grade-B / open-box
  * unknown = cannot tell from title + category
- base_model: the canonical product LINE including any line-extension qualifier.
  Line-extension qualifiers (tokens that denote a DIFFERENT product in the same family, sold alongside the base) MUST stay in base_model. Examples: "FE", "Pro", "Max", "Plus", "Ultra", "Lite", "Mini", "SE", "Air", "XL", "Neo", "Edge".
  Correct: "Samsung Galaxy S25", "Samsung Galaxy S25 FE", "Samsung Galaxy S25 Ultra" are THREE distinct base_models.
  Strip only sub-axis details: color, storage size, RAM, regional SKU code, seller tag.
- variant: the sub-axis suffix — color, storage, RAM (e.g. "256 GB Buz Mavisi", "12/512GB Black"); null when listing has no sub-axis info
- authenticity_confidence: 0.0–1.0 — how sure you are the listing is the authentic product
- matches_user_intent: true if answering this group tells the user what they asked; false if they'd consider it the wrong thing (accessory for a phone query, part for a product query, knockoff, etc.)

Return only the JSON object:
{{
  "groups": [
    {{
      "group_id": int,
      "line_id": "lowercase-ascii-slug",
      "product_type": "string",
      "base_model": "string",
      "variant": "string or null",
      "authenticity_confidence": float,
      "matches_user_intent": bool
    }}
  ]
}}
"""
