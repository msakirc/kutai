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

HARD_CAP = 60_000

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
    counts blocks that appear 2+ times (duplicated = count - 1 per block).
    """
    blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]
    if len(blocks) < MIN_PARAGRAPHS_FOR_CHECK:
        return 0.0, False, None

    normalized = [re.sub(r'\s+', ' ', b).lower() for b in blocks]
    counts = Counter(normalized)
    duplicated = sum(c - 1 for c in counts.values() if c > 1)
    ratio = duplicated / len(normalized)

    if ratio > 0.4:
        return ratio, True, "paragraph_repetition"
    return ratio, False, None


_CONTROL_TOKEN_RE = re.compile(
    r'<\w+_(?:tool_call|tool_calls|arg_key|arg_value|function_call)>',
    re.IGNORECASE,
)


def check_control_token_leak(text: str) -> tuple[int, bool, str | None]:
    """Detect leaked model function-calling control tokens.

    Some models (notably LongCat-Flash, seen in prod via the cloaked OpenRouter
    alias ``owl-alpha``) emit their native namespaced tool-call special tokens —
    ``<longcat_tool_call>``, ``<longcat_arg_key>``, ``<longcat_arg_value>`` — as
    literal text when driven through the json / text tool path instead of native
    tool calls. The tokens then trip downstream placeholder shape gates and DLQ
    the task after retries on the SAME model (2026-06-20). Flagging them as
    degenerate lets the call-path quality gate reject the response and
    failure-adapt to a clean model — model-agnostic, no hardcoded aliases.

    The namespaced form ``<name_tool_call>`` is specific enough that it never
    appears in legitimate prose, so a single occurrence is enough to flag.
    """
    count = len(_CONTROL_TOKEN_RE.findall(text))
    if count >= 1:
        return count, True, "control_token_leak"
    return count, False, None


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
