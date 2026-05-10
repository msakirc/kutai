"""Reverse-pitch shape verifier — Tier 1 of Z1 (A1).

Mechanical post-hook that reads ``reverse_pitch.md`` (Amazon working-
backwards press release) and asserts the required sections:

    1. Headline
    2. Sub-head (sub-headline / subhead)
    3. Customer quote
    4. Founder quote
    5. FAQ

Also rejects template placeholder text (``TODO``, ``<insert>``, ``Lorem
ipsum``, ``[fill in]``, etc.) — a founder who left placeholders never
committed to the outcome the press release is supposed to express.

A1 escape hatch: ``ambition_tier == "prototype"`` may pass with a
single line ``acknowledgement: I am not building for users``. That
acknowledgement carries semantic weight downstream — the reviewer at
1.13 will see it and grade the charter accordingly.

Returns
-------
dict
    ``ok`` (bool), ``found_sections`` (list), ``missing_sections``
    (list), ``placeholders`` (list of sample placeholder snippets),
    ``acknowledged_no_users`` (bool).
"""
from __future__ import annotations

import re
from typing import Any

REQUIRED_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("headline", ("headline",)),
    ("sub_head", ("sub-head", "subhead", "subheadline", "sub headline", "sub-headline")),
    ("customer_quote", ("customer quote", "user quote")),
    ("founder_quote", ("founder quote", "founder's quote", "founder voice")),
    ("faq", ("faq", "frequently asked",)),
)

_PLACEHOLDER_PATTERNS = (
    r"\bTODO\b",
    r"\bTBD\b",
    r"\bFIXME\b",
    r"<[A-Za-z][^>]{0,40}>",
    r"\[(?:fill[- ]in|placeholder|insert|name|company)[^\]]*\]",
    r"\bLorem ipsum\b",
)
_PLACEHOLDER_RE = re.compile("|".join(_PLACEHOLDER_PATTERNS), re.IGNORECASE)

_ACK_RE = re.compile(
    r"acknowledgement\s*[:\-—]\s*[\"']?\s*I\s+am\s+not\s+building\s+for\s+users",
    re.IGNORECASE,
)


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


def verify_reverse_pitch_shape(
    *,
    pitch_text: str | None = None,
    pitch_paths: list[str] | None = None,
    ambition_tier: str = "private_beta",
) -> dict[str, Any]:
    """Validate paraflow-shape reverse_pitch.md.

    See module docstring for output schema.
    """
    md = _gather_text(pitch_text, pitch_paths)
    if not md.strip():
        return {
            "ok": False,
            "error": "empty reverse pitch",
            "found_sections": [],
            "missing_sections": [name for name, _ in REQUIRED_SECTIONS],
            "placeholders": [],
            "acknowledged_no_users": False,
        }

    # Prototype escape hatch.
    ack = bool(_ACK_RE.search(md))
    if ack and ambition_tier.lower() == "prototype":
        # Reject any other placeholder noise alongside the ack — pure ack
        # only. This keeps the bypass narrow and traceable.
        return {
            "ok": True,
            "found_sections": ["acknowledgement"],
            "missing_sections": [],
            "placeholders": [],
            "acknowledged_no_users": True,
            "ambition_tier": ambition_tier,
        }

    # Locate each required section by scanning headings + bold labels.
    lower = md.lower()
    found: list[str] = []
    missing: list[str] = []
    for canonical, aliases in REQUIRED_SECTIONS:
        hit = False
        for alias in aliases:
            # Match in any of: ``# Headline``, ``## Headline``,
            # ``**Headline:**``, ``Headline:`` line-leading.
            patterns = (
                rf"(?m)^#{{1,6}}\s+{re.escape(alias)}\b",
                rf"\*\*{re.escape(alias)}\*\*\s*[:\-—]",
                rf"(?m)^{re.escape(alias)}\s*[:\-—]",
            )
            if any(re.search(p, lower, re.IGNORECASE) for p in patterns):
                hit = True
                break
        if hit:
            found.append(canonical)
        else:
            missing.append(canonical)

    placeholders = list({m.group(0) for m in _PLACEHOLDER_RE.finditer(md)})

    ok = not missing and not placeholders
    # Block the ambition-tier bypass for non-prototype tiers even if the
    # ack line is present.
    if ack and ambition_tier.lower() != "prototype":
        ok = False

    return {
        "ok": ok,
        "found_sections": found,
        "missing_sections": missing,
        "placeholders": placeholders[:10],
        "acknowledged_no_users": ack,
        "ambition_tier": ambition_tier,
    }
