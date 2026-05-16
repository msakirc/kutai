"""Confidence scoring for matched artifacts.

confidence = score × source_trust × owner_trust × hint_bonus

score arrives on the Artifact from yalayut.query (Phase 1 ranks the
index by vector similarity over the name+name_original+body embedding).
source_trust / owner_trust are looked up from yalayut_sources /
yalayut_owners by flash.py and passed in. hint_bonus rewards artifacts
whose intent keywords overlap the step's recipe_hint.
"""
from __future__ import annotations

# Maximum multiplicative lift from a recipe_hint keyword match. A full
# keyword overlap caps the bonus here; partial overlap scales linearly.
HINT_BONUS_MAX: float = 1.30


def _tokenize(text: str) -> set[str]:
    """Lowercase word split, drop tokens <= 2 chars. Mirrors the
    coarse tokenisation used elsewhere in KutAI for keyword overlap."""
    import re
    out: set[str] = set()
    for w in re.split(r"[\s,;.:/()_\-]+", (text or "").lower()):
        w = w.strip("'\"")
        if len(w) > 2:
            out.add(w)
    return out


def compute_hint_bonus(artifact, recipe_hint: str | None) -> float:
    """Return a multiplicative bonus in [1.0, HINT_BONUS_MAX].

    1.0 when there is no recipe_hint or zero keyword overlap. Scales
    linearly with the fraction of recipe_hint tokens found in the
    artifact's intent keywords or name.
    """
    if not recipe_hint:
        return 1.0
    hint_tokens = _tokenize(recipe_hint)
    if not hint_tokens:
        return 1.0
    art_tokens = set()
    for kw in getattr(artifact, "intent_keywords", None) or []:
        art_tokens |= _tokenize(str(kw))
    art_tokens |= _tokenize(getattr(artifact, "name", "") or "")
    overlap = len(hint_tokens & art_tokens) / len(hint_tokens)
    if overlap <= 0.0:
        return 1.0
    return 1.0 + overlap * (HINT_BONUS_MAX - 1.0)


def score_artifact(
    artifact,
    *,
    source_trust: float,
    owner_trust: float,
    hint_bonus: float = 1.0,
) -> float:
    """Compute final confidence, clamped to [0.0, 1.0].

    Reads artifact.score (addendum rename from vector_sim).
    """
    vector_sim = float(getattr(artifact, "score", 0.0) or 0.0)
    raw = vector_sim * float(source_trust) * float(owner_trust) * float(hint_bonus)
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        return 1.0
    return raw
