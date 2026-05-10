"""Cheap heuristic check against ``non_goals.md`` — Z1 Tier 2 (A2).

Mechanical pre-flag: scans a target artifact text for explicit token-level
overlap with bullets in ``non_goals.md``. NOT a substitute for the LLM
reviewer at 3.11 / 4.16 / 5.10 — surfaces obvious contradictions cheaply
so the reviewer can focus on subtle ones.

Returns a list of ``matches`` (one per non-goal bullet that overlaps the
target text) plus the per-match overlap context.

Pure function. Caller (post-hook wiring) provides:
    - ``non_goals_text`` (or ``non_goals_paths``): the refusal artifact
    - ``target_text`` (or ``target_paths``): the artifact to check
"""
from __future__ import annotations

import re
from typing import Any

# Stopwords removed from non-goal bullets when computing significant tokens.
_STOPWORDS = {
    "no", "not", "do", "don't", "dont", "doesn't", "doesnt",
    "the", "a", "an", "is", "are", "of", "in", "on", "for", "to",
    "and", "or", "but", "with", "without", "this", "that", "we",
    "our", "us", "be", "will", "would", "should", "shall", "may",
    "yet", "iteration", "launch",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.+)$", re.MULTILINE)


def _gather_text(text: str | None, paths: list[str] | None) -> str:
    if text:
        return text
    if not paths:
        return ""
    bufs: list[str] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                bufs.append(fh.read())
        except OSError:
            continue
    return "\n\n".join(bufs)


def _strip_frontmatter(md: str) -> str:
    m = _FRONTMATTER_RE.match(md)
    return md[m.end():] if m else md


def _extract_bullets(md: str) -> list[str]:
    body = _strip_frontmatter(md)
    return [m.group(1).strip() for m in _BULLET_RE.finditer(body)]


def _significant_tokens(s: str) -> set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(s)
        if t.lower() not in _STOPWORDS
    }


def check_against_non_goals(
    *,
    non_goals_text: str | None = None,
    non_goals_paths: list[str] | None = None,
    target_text: str | None = None,
    target_paths: list[str] | None = None,
    min_overlap_tokens: int = 2,
) -> dict[str, Any]:
    """Surface explicit token-overlap between non-goal bullets and target text.

    Parameters
    ----------
    min_overlap_tokens
        How many significant tokens (after stopword filter) of a non-goal
        bullet must appear in the target text to count as an overlap match.
    """
    ng_md = _gather_text(non_goals_text, non_goals_paths)
    target = _gather_text(target_text, target_paths)
    if not ng_md.strip() or not target.strip():
        return {"matches": [], "checked": 0, "non_goals_present": bool(ng_md.strip())}

    bullets = _extract_bullets(ng_md)
    target_tokens = _significant_tokens(target)
    target_lower = target.lower()

    matches: list[dict[str, Any]] = []
    for bullet in bullets:
        bullet_tokens = _significant_tokens(bullet)
        if not bullet_tokens:
            continue
        overlap = sorted(bullet_tokens & target_tokens)
        # Bonus signal: full literal substring match (any 4+ char run).
        literal_hit = False
        norm = " ".join(bullet.lower().split())
        if len(norm) >= 8 and norm in target_lower:
            literal_hit = True
        if literal_hit or len(overlap) >= min_overlap_tokens:
            matches.append({
                "non_goal": bullet,
                "overlap_tokens": overlap,
                "literal_substring": literal_hit,
            })

    return {
        "matches": matches,
        "checked": len(bullets),
        "non_goals_present": True,
    }
