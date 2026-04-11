"""Heuristic content quality detectors.

Each function returns (score, breached, reason_tag).
- score: numeric measurement (size in chars, ratio 0.0-1.0, entropy in bits)
- breached: True if threshold exceeded
- reason_tag: short string for ContentQualityResult.reasons, or None if not breached
"""

from __future__ import annotations

import math
import re
from collections import Counter

HARD_CAP = 50_000

MIN_SECTIONS_FOR_HEADER_CHECK = 5
MIN_PARAGRAPHS_FOR_CHECK = 4
MIN_WORDS_FOR_ENTROPY = 20

_HEADER_SUFFIX_RE = re.compile(
    r'\s+(summary|examples?|notes|details)\s*$', re.IGNORECASE,
)


def check_size(
    text: str, max_size: int = 20_000,
) -> tuple[int, bool, str | None]:
    effective_max = min(max_size, HARD_CAP)
    size = len(text)
    if size > effective_max:
        return size, True, "size_exceeded"
    return size, False, None


def check_header_repetition(text: str) -> tuple[float, bool, str | None]:
    sections = text.split("\n## ")
    if len(sections) < MIN_SECTIONS_FOR_HEADER_CHECK + 1:
        return 0.0, False, None

    norm_headers: list[str] = []
    for sec in sections[1:]:
        header = sec.split("\n", 1)[0].strip()
        norm = _HEADER_SUFFIX_RE.sub("", header.lower()).strip()
        norm_headers.append(norm)

    if not norm_headers:
        return 0.0, False, None

    counts = Counter(norm_headers)
    duplicated = sum(c - 1 for c in counts.values() if c > 1)
    ratio = duplicated / len(norm_headers)

    if ratio > 0.4:
        return ratio, True, "header_repetition"
    return ratio, False, None


def check_paragraph_repetition(text: str) -> tuple[float, bool, str | None]:
    """Detect repeated paragraph blocks.

    Splits by double-newlines, normalizes whitespace, hashes blocks,
    counts blocks sharing hash with 2+ others.
    """
    blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]
    if len(blocks) < MIN_PARAGRAPHS_FOR_CHECK:
        return 0.0, False, None

    normalized = [re.sub(r'\s+', ' ', b).lower() for b in blocks]
    counts = Counter(normalized)
    duplicated = sum(c - 1 for c in counts.values() if c > 1)
    ratio = duplicated / len(normalized)

    if ratio > 0.3:
        return ratio, True, "paragraph_repetition"
    return ratio, False, None


def check_token_entropy(text: str) -> tuple[float, bool, str | None]:
    """Measure Shannon entropy of whitespace-split tokens.

    Natural English ~9-10 bits. Repetitive garbage < 3 bits.
    """
    tokens = text.split()
    if len(tokens) < MIN_WORDS_FOR_ENTROPY:
        return 0.0, False, None

    total = len(tokens)
    counts = Counter(tokens)
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
    )

    if entropy < 3.0:
        return entropy, True, "low_entropy"
    return entropy, False, None
