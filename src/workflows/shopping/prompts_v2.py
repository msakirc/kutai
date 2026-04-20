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
- Extract recurring PRAISE points (what users like). Max 3 bullets, short phrases.
- Extract recurring COMPLAINTS (what users dislike). Max 3 bullets.
- Extract RED FLAGS (safety, reliability, fraud concerns, complaint-site mentions). Max 3 bullets.
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
