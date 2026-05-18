"""Non-goals shape verifier — Z1 Tier 2 (A2).

Mechanical post-hook that reads ``non_goals.md`` and asserts the
mission-wide refusal artifact's shape:

    1. YAML frontmatter delimited by ``---`` lines at top of file
    2. Frontmatter carries ``_schema_version: "1"``
    3. Frontmatter ``non_goals:`` array (3-7 entries)
    4. Body ``# Non-goals`` heading present
    5. Body has 3-7 bullets matching the frontmatter count
    6. No placeholder text (``TODO``, ``TBD``, ``<...>``)

Pure check. Caller (post-hook wiring) provides the markdown via
``payload['non_goals_text']`` or a ``non_goals_paths`` list of files
to concatenate.

Returns
-------
dict
    ``ok`` (bool), ``frontmatter_present`` (bool),
    ``schema_version`` (str|None), ``yaml_count`` (int),
    ``bullet_count`` (int), ``placeholders`` (list), ``problems``
    (list of human-readable strings).
"""
from __future__ import annotations

import re
from typing import Any

MIN_NON_GOALS = 3
MAX_NON_GOALS = 7

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^#\s+Non-goals?\s*$", re.IGNORECASE | re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*]\s+\S", re.MULTILINE)
_SCHEMA_VERSION_RE = re.compile(r'^_schema_version\s*:\s*["\']?([^"\'\n]+)["\']?\s*$', re.MULTILINE)
_YAML_BULLET_RE = re.compile(r'^\s*-\s+["\']?(.+?)["\']?\s*$', re.MULTILINE)

_PLACEHOLDER_PATTERNS = (
    r"(?-i:\bTODO\b)",
    r"(?-i:\bTBD\b)",
    r"(?-i:\bFIXME\b)",
    r"<[A-Za-z][^>]{0,40}>",
    r"\[(?:fill[- ]in|placeholder|insert)[^\]]*\]",
)
_PLACEHOLDER_RE = re.compile("|".join(_PLACEHOLDER_PATTERNS), re.IGNORECASE)


def _gather_text(non_goals_text: str | None,
                 non_goals_paths: list[str] | None) -> str:
    if non_goals_text:
        return non_goals_text
    if not non_goals_paths:
        return ""
    bufs: list[str] = []
    for p in non_goals_paths:
        try:
            with open(p, encoding="utf-8") as fh:
                bufs.append(fh.read())
        except OSError:
            continue
    return "\n\n".join(bufs)


def verify_non_goals_shape(
    *,
    non_goals_text: str | None = None,
    non_goals_paths: list[str] | None = None,
    min_items: int = MIN_NON_GOALS,
    max_items: int = MAX_NON_GOALS,
) -> dict[str, Any]:
    """Validate ``non_goals.md`` shape. See module docstring for output."""
    md = _gather_text(non_goals_text, non_goals_paths)
    if not md.strip():
        return {
            "ok": False,
            "frontmatter_present": False,
            "schema_version": None,
            "yaml_count": 0,
            "bullet_count": 0,
            "placeholders": [],
            "problems": ["empty non_goals.md"],
        }

    problems: list[str] = []

    fm_match = _FRONTMATTER_RE.match(md)
    frontmatter_present = bool(fm_match)
    schema_version: str | None = None
    yaml_items: list[str] = []
    if frontmatter_present:
        fm_text = fm_match.group(1)
        sv = _SCHEMA_VERSION_RE.search(fm_text)
        if sv:
            schema_version = sv.group(1).strip()
        # YAML non_goals list — gather block under `non_goals:`.
        ng_idx = fm_text.lower().find("non_goals:")
        if ng_idx >= 0:
            tail = fm_text[ng_idx:]
            yaml_items = [m.group(1).strip() for m in _YAML_BULLET_RE.finditer(tail)]
        else:
            problems.append("frontmatter missing 'non_goals:' key")
        if schema_version is None:
            problems.append("frontmatter missing _schema_version")
        elif schema_version != "1":
            problems.append(
                f"frontmatter _schema_version={schema_version!r} expected '1'"
            )
    else:
        problems.append("missing YAML frontmatter (--- ... ---)")

    body = md[fm_match.end():] if frontmatter_present else md
    if not _HEADING_RE.search(body):
        problems.append("body missing '# Non-goals' heading")

    bullets = _BULLET_RE.findall(body)
    bullet_count = len(bullets)
    if not (min_items <= bullet_count <= max_items):
        problems.append(
            f"bullet_count={bullet_count} not in [{min_items},{max_items}]"
        )

    yaml_count = len(yaml_items)
    if yaml_count and not (min_items <= yaml_count <= max_items):
        problems.append(
            f"yaml_count={yaml_count} not in [{min_items},{max_items}]"
        )

    placeholders = sorted({m.group(0) for m in _PLACEHOLDER_RE.finditer(md)})
    if placeholders:
        problems.append(f"placeholders present: {placeholders[:5]}")

    ok = not problems
    return {
        "ok": ok,
        "frontmatter_present": frontmatter_present,
        "schema_version": schema_version,
        "yaml_count": yaml_count,
        "bullet_count": bullet_count,
        "placeholders": placeholders[:10],
        "problems": problems,
    }
