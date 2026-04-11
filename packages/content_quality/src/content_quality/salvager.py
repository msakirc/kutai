"""Section deduplication for degenerate markdown output."""

from __future__ import annotations

import re

_HEADER_SUFFIX_RE = re.compile(
    r'\s+(summary|examples?|notes|details)\s*$', re.IGNORECASE,
)


def salvage(text: str) -> str:
    """Deduplicate repeated markdown sections.

    1. Split text by ## headers
    2. Normalize each header (strip trailing suffixes, lowercase)
    3. Keep first occurrence of each normalized header
    4. Drop sections with no content (empty body)
    5. Reassemble

    Returns:
        Cleaned text, or empty string if nothing salvageable survives.
        Non-markdown text (no ## headers) is returned unchanged.
    """
    # Normalize: if text starts with "## ", treat it as if preceded by newline
    normalized = text if not text.startswith("## ") else "\n" + text
    parts = normalized.split("\n## ")

    if len(parts) <= 1:
        return text

    preamble = parts[0]
    sections = parts[1:]

    seen_normalized: set[str] = set()
    kept: list[str] = []

    for sec in sections:
        header, _, body = sec.partition("\n")
        header = header.strip()
        body = body.strip()

        if not body:
            continue

        norm = _HEADER_SUFFIX_RE.sub("", header.lower()).strip()

        if norm in seen_normalized:
            continue

        seen_normalized.add(norm)
        kept.append(f"## {header}\n{body}")

    if not kept:
        return ""

    result_parts = []
    preamble_stripped = preamble.strip()
    if preamble_stripped:
        result_parts.append(preamble_stripped)
    result_parts.extend(kept)

    return "\n\n".join(result_parts)
