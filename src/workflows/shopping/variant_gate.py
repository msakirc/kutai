"""Pure-code filter + gate logic for variant disambiguation."""
from __future__ import annotations

import re

from src.workflows.shopping.pipeline_v2 import ProductGroup

FILTER_AUTHENTICITY_MIN = 0.7
MAX_CLARIFY_OPTIONS = 5

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Storage / capacity / RAM tokens. Generic across electronics — strips "256 GB",
# "1 TB", "12/512", "8GB" leaks from base_model so listings whose LLM-label kept
# storage in the base_model still bucket together with clean labels.
_STORAGE_RE = re.compile(
    r"\b\d+\s*(?:gb|tb|mb|gib|tib|mib|g|t|m)\b|"  # 256 GB, 1TB, 8g
    r"\b\d+\s*/\s*\d+\b|"                          # 12/512
    r"\b\d{2,4}\b",                                # bare 256, 512 (3-4 digit chunks)
    re.IGNORECASE,
)

# Marketplace boilerplate — "Türkiye Garantili", "(Garanti)", seller tags.
# Generic enough across electronics retail to fold without category-specific seed.
_NOISE_PATTERNS = [
    re.compile(r"\([^)]*\)"),                                  # parenthetical content
    re.compile(r"\b(?:türkiye|turkiye)\s+garant[iı]l[iı]\b", re.IGNORECASE),
    re.compile(r"\bgarant[iı]l[iı]\b", re.IGNORECASE),
    re.compile(r"\bram\b", re.IGNORECASE),                     # standalone "Ram" leftover
    re.compile(r"\b(?:resmi|orijinal|distribütör|distributor)\b", re.IGNORECASE),
]

# ASCII fold for Turkish letters so "REDMı" matches "Redmi".
_ASCII_FOLD = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})


def _canonical_base(base_model: str, fallback: str = "", *, max_tokens: int = 5) -> str:
    """Aggressive normalization: ascii-fold, drop storage/parens/noise, keep first N
    tokens (alphanumeric kept so model codes like "s25"/"iphone15" survive; pure-
    digit tokens dropped). Returns "xiaomi redmi note pro" for any of:

      - "Xiaomi Redmi Note 15 Pro 256 GB"
      - "Xiaomi REDMı Note 15 Pro 256 GB 8 GB Ram (Xiaomi Türkiye Garantili) Mavi"
      - "Xiaomi Redmi Note 15 Pro 8 GB+256 GB Titanyum Gri"
    """
    text = (base_model or fallback or "")
    text = text.translate(_ASCII_FOLD).lower()
    for pat in _NOISE_PATTERNS:
        text = pat.sub(" ", text)
    text = _STORAGE_RE.sub(" ", text)
    toks = [t for t in _TOKEN_RE.findall(text) if not t.isdigit()]
    return " ".join(toks[:max_tokens]).strip()


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _query_overlap(query_tokens: set[str], base_model: str) -> tuple[float, int]:
    """(overlap ratio, extra-token count). Higher overlap + fewer extras = closer match."""
    if not query_tokens:
        return 1.0, 0
    btoks = _tokens(base_model)
    if not btoks:
        return 0.0, 0
    shared = query_tokens & btoks
    extras = len(btoks - query_tokens - {"the", "and", "for"})
    return len(shared) / max(1, len(query_tokens)), extras


def step_filter(groups: list[ProductGroup]) -> list[ProductGroup]:
    """Drop groups that aren't authentic, intent-matched products."""
    return [
        g for g in groups
        if g.product_type == "authentic_product"
        and g.matches_user_intent
        and g.authenticity_confidence >= FILTER_AUTHENTICITY_MIN
    ]


def step_variant_gate(
    survivors: list[ProductGroup],
    all_groups: list[ProductGroup],
    query: str = "",
) -> dict:
    """Decide how to proceed after filtering.

    Returns one of:
      {"kind": "chosen", "group": ProductGroup}
      {"kind": "clarify", "options": [{label, group_id, prominence}], "payloads": {gid: group}}
      {"kind": "escalation", "reason": "all_filtered"}
    """
    if not survivors:
        return {"kind": "escalation", "reason": "all_filtered"}

    # Bucket by base_model (product LINE) only — color/storage are sub-details the
    # user doesn't pick at the gate. Line choice: S25 vs S25 FE vs S25 Ultra.
    # Bucketing key — prefer LLM-emitted `line_id` (canonical slug per product
    # line). Fall back to deterministic canonical_base when missing, then a
    # structural prefix-merge (≥2 extra tokens = sub-axis like color+storage).
    raw_keys: list[str] = []
    for g in survivors:
        lid = (getattr(g, "line_id", "") or "").strip().lower()
        raw_keys.append(lid or _canonical_base(g.base_model, fallback=g.representative_title))

    # Structural prefix-merge as fallback for fallback-bucketed entries:
    # if key A is a strict token-prefix of B with ≥2 extra tokens, B → A.
    # No qualifier vocab — purely a "extras count ≥ 2 = sub-axis" rule.
    raw_set = set(raw_keys)
    merged_keys: list[str] = []
    for k in raw_keys:
        toks = k.split()
        chosen = k
        for n in range(len(toks) - 1, 0, -1):
            short = " ".join(toks[:n])
            if short and short in raw_set and short != k and (len(toks) - n) >= 2:
                chosen = short
                break
        merged_keys.append(chosen)

    by_line: dict[str, list[ProductGroup]] = {}
    for g, key in zip(survivors, merged_keys):
        by_line.setdefault(key, []).append(g)

    if len(by_line) == 1:
        only_bucket = next(iter(by_line.values()))
        chosen = max(only_bucket, key=lambda g: g.prominence)
        return {"kind": "chosen", "group": chosen}

    # Rank by query specificity first, then prominence — so exact-query matches
    # surface above line-extensions (e.g. "S25" query ranks base S25 above S25 FE).
    q_tokens = _tokens(query)

    def _bucket_score(bucket: list[ProductGroup]) -> tuple[float, int, float]:
        rep = max(bucket, key=lambda g: g.prominence)
        overlap, extras = _query_overlap(q_tokens, rep.base_model)
        return (overlap, -extras, rep.prominence)

    sorted_buckets = sorted(
        by_line.items(),
        key=lambda kv: _bucket_score(kv[1]),
        reverse=True,
    )

    # Stable group_id = position in all_groups list. Use index-by-identity-or-equality
    # with a fallback per-bucket numbering, so collapsed duplicates get unique ids.
    all_groups_index: dict[int, int] = {id(g): i for i, g in enumerate(all_groups)}
    next_synth_id = len(all_groups)

    options: list[dict] = []
    payloads: dict[int, ProductGroup] = {}
    for _key, bucket in sorted_buckets[:MAX_CLARIFY_OPTIONS]:
        rep = max(bucket, key=lambda g: g.prominence)
        # Merge all groups of the same line into one virtual ProductGroup so that
        # compare-all aggregates prices/reviews across every color/storage listing
        # for that line (instead of showing one color's price + another "fiyat yok").
        merged_indices: list[int] = []
        seen: set[int] = set()
        for g in bucket:
            for idx in g.member_indices:
                if idx not in seen:
                    merged_indices.append(idx)
                    seen.add(idx)
        merged = ProductGroup(
            representative_title=rep.base_model or rep.representative_title,
            member_indices=merged_indices,
            is_accessory_or_part=rep.is_accessory_or_part,
            prominence=sum(g.prominence for g in bucket),
            product_type=rep.product_type,
            base_model=rep.base_model,
            variant=None,
            authenticity_confidence=rep.authenticity_confidence,
            matches_user_intent=rep.matches_user_intent,
            line_id=rep.line_id,
        )
        gid = all_groups_index.get(id(rep))
        if gid is None or gid in payloads:
            gid = next_synth_id
            next_synth_id += 1
        options.append({
            "label": merged.representative_title,
            "group_id": gid,
            "prominence": merged.prominence,
        })
        payloads[gid] = merged

    return {"kind": "clarify", "options": options, "payloads": payloads}
