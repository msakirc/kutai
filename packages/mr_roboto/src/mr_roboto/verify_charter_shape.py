"""Charter-shape verifier — Tier 1 of Z1 (B1+C1+A9+C6+A14).

Mechanical post-hook that reads ``product_charter.md`` and asserts the
five-section paraflow shape:

    1. Product Positioning           — single paragraph
    2. Brand Keywords                — at least 5 bullets, each ``name — desc``
    3. Core Problem / JTBD           — single paragraph
    4. Goals & Mission               — Mission line + Desired Outcomes bullets
    5. Solutions We Own              — 3-7 solutions, each with sub-shape:
        - What it solves
        - Typical path
        - Outcome for the user
        - Boundaries
        - Guiding principles

The check is pure (no I/O, no LLM). The caller (post-hook wiring or test
fixture) provides the markdown text via ``payload['charter_text']`` or a
list of ``charter_paths`` it has already read.

Returns
-------
dict
    ``ok`` (bool), ``sections_found`` (list), ``missing_sections`` (list),
    ``solution_count`` (int), ``solution_problems``
    (list of ``{name, missing_subfields}``), ``placeholders`` (list of
    sample placeholder snippets — non-empty rejects).
"""
from __future__ import annotations

import re
from typing import Any

REQUIRED_SECTIONS = (
    "Product Positioning",
    "Brand Keywords",
    "Core Problem",  # tolerate "Core Problem", "Core Problem / JTBD", etc.
    "Goals & Mission",
    "Solutions We Own",
)

# Per-solution sub-fields (paraflow + A14 boundaries + guiding-principles).
# We want at least these keys present in every Solution block. "Boundaries"
# is required by A14 even though paraflow has it on only some solutions —
# Z1 mandates it everywhere (per acceptance criterion + reviewer 1.13).
REQUIRED_SOLUTION_FIELDS = (
    "What it solves",
    "Typical path",
    "Outcome for the user",
    "Boundaries",
    "Guiding principles",
)

# Strings that almost always indicate template/placeholder leftovers.
_PLACEHOLDER_PATTERNS = (
    r"(?-i:\bTODO\b)",
    r"(?-i:\bTBD\b)",
    r"(?-i:\bFIXME\b)",
    r"<[A-Za-z][^>]{0,40}>",        # <fill in>, <name>, <one-line>
    r"\[(?:fill[- ]in|placeholder|insert)[^\]]*\]",
    r"\bLorem ipsum\b",
    r"\.\.\.\.\.+",                 # long ellipses runs
)
_PLACEHOLDER_RE = re.compile("|".join(_PLACEHOLDER_PATTERNS), re.IGNORECASE)


def _split_top_sections(md: str) -> dict[str, str]:
    """Split markdown by top-level ``##`` headings.

    Returns ``{heading_text: body}``. Headings are normalized — leading
    ``N) `` / ``N. `` numbering stripped so paraflow's ``## 1) Product
    Positioning`` matches ``## Product Positioning``.
    """
    out: dict[str, str] = {}
    current_head: str | None = None
    buf: list[str] = []
    head_re = re.compile(r"^##\s+(.*?)\s*$")
    num_prefix = re.compile(r"^\s*\d+\s*[)\.\:\-]\s*")
    for line in md.splitlines():
        m = head_re.match(line)
        if m:
            if current_head is not None:
                out[current_head] = "\n".join(buf).strip()
            current_head = num_prefix.sub("", m.group(1)).strip()
            buf = []
        else:
            buf.append(line)
    if current_head is not None:
        out[current_head] = "\n".join(buf).strip()
    return out


def _find_section(sections: dict[str, str], needle: str) -> tuple[str, str] | None:
    """Find a section whose heading contains ``needle`` (case-insensitive)."""
    n = needle.lower()
    for head, body in sections.items():
        if n in head.lower():
            return head, body
    return None


def _parse_solutions(body: str) -> list[tuple[str, str]]:
    """Parse the ``Solutions We Own`` body into ``[(name, block_text)]``.

    Each solution is delimited by a ``###`` heading.
    """
    out: list[tuple[str, str]] = []
    current_name: str | None = None
    buf: list[str] = []
    for line in body.splitlines():
        m = re.match(r"^###\s+(.*?)\s*$", line)
        if m:
            if current_name is not None:
                out.append((current_name, "\n".join(buf).strip()))
            current_name = m.group(1).strip()
            buf = []
        else:
            buf.append(line)
    if current_name is not None:
        out.append((current_name, "\n".join(buf).strip()))
    return out


def _has_field(block: str, field: str) -> bool:
    """Check whether a solution block contains a labeled field.

    Accepts ``- **What it solves:** ...``, ``- What it solves: ...``, or
    a leading ``What it solves —`` line — the three shapes that show up
    in real paraflow output.
    """
    field_re = re.compile(
        r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?"
        + re.escape(field)
        + r"(?:\*\*)?\s*[:\-—]",
        re.IGNORECASE,
    )
    return bool(field_re.search(block))


def _gather_text(charter_text: str | None, charter_paths: list[str] | None) -> str:
    if charter_text:
        return charter_text
    if not charter_paths:
        return ""
    bufs: list[str] = []
    for p in charter_paths:
        try:
            with open(p, encoding="utf-8") as fh:
                bufs.append(fh.read())
        except OSError:
            continue
    return "\n\n".join(bufs)


def verify_charter_shape(
    *,
    charter_text: str | None = None,
    charter_paths: list[str] | None = None,
    min_solutions: int = 3,
    max_solutions: int = 7,
    min_brand_keywords: int = 5,
) -> dict[str, Any]:
    """Validate paraflow-shape ``product_charter.md``.

    See module docstring for output schema.
    """
    md = _gather_text(charter_text, charter_paths)
    if not md.strip():
        return {
            "ok": False,
            "error": "empty charter",
            "sections_found": [],
            "missing_sections": list(REQUIRED_SECTIONS),
            "solution_count": 0,
            "solution_problems": [],
            "placeholders": [],
        }

    sections = _split_top_sections(md)
    sections_found = list(sections.keys())

    missing_sections: list[str] = []
    section_bodies: dict[str, str] = {}
    for needed in REQUIRED_SECTIONS:
        found = _find_section(sections, needed)
        if found is None:
            missing_sections.append(needed)
        else:
            section_bodies[needed] = found[1]

    # Brand keywords — count list items.
    brand_count = 0
    if "Brand Keywords" in section_bodies:
        brand_count = sum(
            1
            for line in section_bodies["Brand Keywords"].splitlines()
            if re.match(r"^\s*[-*]\s+\S", line)
        )

    # Goals & Mission — must include a Mission line + at least one Desired
    # Outcome bullet.
    goals_problems: list[str] = []
    if "Goals & Mission" in section_bodies:
        body = section_bodies["Goals & Mission"]
        if not re.search(r"\bMission\b\s*[:\-—]", body, re.IGNORECASE):
            goals_problems.append("missing 'Mission:' line")
        if not re.search(r"Desired\s+Outcomes?", body, re.IGNORECASE):
            goals_problems.append("missing 'Desired Outcomes' header")

    # Solutions We Own — count + per-solution sub-shape.
    solutions: list[tuple[str, str]] = []
    if "Solutions We Own" in section_bodies:
        solutions = _parse_solutions(section_bodies["Solutions We Own"])
    solution_problems: list[dict[str, Any]] = []
    for name, block in solutions:
        missing_fields = [
            f for f in REQUIRED_SOLUTION_FIELDS if not _has_field(block, f)
        ]
        if missing_fields:
            solution_problems.append(
                {"name": name, "missing_subfields": missing_fields}
            )

    # Placeholder hunt across the whole doc.
    placeholders = list({m.group(0) for m in _PLACEHOLDER_RE.finditer(md)})

    ok = (
        not missing_sections
        and not goals_problems
        and not solution_problems
        and not placeholders
        and min_solutions <= len(solutions) <= max_solutions
        and brand_count >= min_brand_keywords
    )

    return {
        "ok": ok,
        "sections_found": sections_found,
        "missing_sections": missing_sections,
        "goals_problems": goals_problems,
        "brand_keyword_count": brand_count,
        "min_brand_keywords": min_brand_keywords,
        "solution_count": len(solutions),
        "min_solutions": min_solutions,
        "max_solutions": max_solutions,
        "solution_problems": solution_problems,
        "placeholders": placeholders[:10],
    }
