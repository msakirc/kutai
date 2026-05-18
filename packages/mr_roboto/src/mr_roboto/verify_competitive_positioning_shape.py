"""Competitive-positioning shape verifier — Tier 2 of Z1 (C2).

Mechanical post-hook that reads ``competitive_positioning.md`` and asserts
the paraflow PRD §6 shape:

    1. Landscape
    2. Value Thesis
    3. Strengths / Weaknesses
    4. Our Differentiators
    5. Switching Costs & Risks
    6. Notes

Plus a YAML front-matter block that names ``mission_id`` and a
non-empty ``named_competitors`` list, plus ``_schema_version: "1"``.

The check is pure (no I/O, no LLM). Caller provides the markdown text
via ``payload['positioning_text']`` or a list of ``positioning_paths``
the caller has already read.

Returns
-------
dict
    ``ok`` (bool), ``sections_found`` (list), ``missing_sections``
    (list), ``named_competitors`` (list), ``placeholders`` (list),
    ``schema_version`` (str|None), ``mission_id`` (int|None).
"""
from __future__ import annotations

import re
from typing import Any

REQUIRED_SECTIONS = (
    "Landscape",
    "Value Thesis",
    "Strengths",          # tolerate "Strengths / Weaknesses" / "Strengths & Weaknesses"
    "Our Differentiators",
    "Switching Costs",    # tolerate "Switching Costs & Risks"
    "Notes",
)

_PLACEHOLDER_PATTERNS = (
    r"(?-i:\bTODO\b)",
    r"(?-i:\bTBD\b)",
    r"(?-i:\bFIXME\b)",
    r"<[A-Za-z][^>]{0,40}>",
    r"\[(?:fill[- ]in|placeholder|insert)[^\]]*\]",
    r"\bLorem ipsum\b",
    r"\.\.\.\.\.+",
)
_PLACEHOLDER_RE = re.compile("|".join(_PLACEHOLDER_PATTERNS), re.IGNORECASE)

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


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


def _parse_front_matter(md: str) -> dict[str, Any]:
    """Tiny front-matter parser — accepts simple ``key: value`` and
    ``key: [a, b, c]`` lines. Avoids a YAML dep."""
    m = _FRONT_MATTER_RE.match(md)
    if not m:
        return {}
    body = m.group(1)
    out: dict[str, Any] = {}
    for line in body.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            if not inner:
                out[key] = []
            else:
                out[key] = [
                    s.strip().strip('"').strip("'") for s in inner.split(",") if s.strip()
                ]
        else:
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]
            out[key] = val
    return out


def _split_top_sections(md: str) -> dict[str, str]:
    """Split markdown by top-level ``##`` headings."""
    out: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    head_re = re.compile(r"^##\s+(.*?)\s*$")
    for line in md.splitlines():
        m = head_re.match(line)
        if m:
            if current is not None:
                out[current] = "\n".join(buf).strip()
            current = m.group(1).strip()
            buf = []
        else:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf).strip()
    return out


def _has_section(sections: dict[str, str], needle: str) -> tuple[str, str] | None:
    n = needle.lower()
    for head, body in sections.items():
        if n in head.lower():
            return head, body
    return None


def verify_competitive_positioning_shape(
    *,
    positioning_text: str | None = None,
    positioning_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Validate paraflow-PRD-§6-shape ``competitive_positioning.md``."""
    md = _gather_text(positioning_text, positioning_paths)
    if not md.strip():
        return {
            "ok": False,
            "error": "empty competitive_positioning",
            "sections_found": [],
            "missing_sections": list(REQUIRED_SECTIONS),
            "named_competitors": [],
            "placeholders": [],
            "schema_version": None,
            "mission_id": None,
        }

    fm = _parse_front_matter(md)
    body = _FRONT_MATTER_RE.sub("", md, count=1)

    schema_version = fm.get("_schema_version")
    mission_id = fm.get("mission_id")
    named = fm.get("named_competitors") or []
    if isinstance(named, str):
        named = [named] if named.strip() else []

    sections = _split_top_sections(body)
    sections_found = list(sections.keys())

    missing: list[str] = []
    section_bodies: dict[str, str] = {}
    for needed in REQUIRED_SECTIONS:
        hit = _has_section(sections, needed)
        if hit is None:
            missing.append(needed)
        else:
            section_bodies[needed] = hit[1]

    # Flag empty bodies — present-but-blank section is a fail.
    empty_sections = [
        name for name, b in section_bodies.items() if not b.strip()
    ]

    placeholders = list({m.group(0) for m in _PLACEHOLDER_RE.finditer(md)})

    ok = (
        not missing
        and not empty_sections
        and not placeholders
        and bool(named)
        and schema_version == "1"
    )

    return {
        "ok": ok,
        "sections_found": sections_found,
        "missing_sections": missing,
        "empty_sections": empty_sections,
        "named_competitors": list(named),
        "placeholders": placeholders[:10],
        "schema_version": schema_version,
        "mission_id": mission_id,
    }
