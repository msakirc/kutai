"""Per-screen plan shape verifier — Z1 Tier 3 (C3+A10+C14).

Mechanical post-hook that reads a paraflow-shape ``screen_plan.md`` and
asserts the structural contract emitted by step ``5.1
generate_per_screen_plans``:

    1. YAML frontmatter delimited by ``---`` lines at top of file
    2. Frontmatter carries:
        - ``_schema_version: "1"``
        - ``mission_id`` (any value)
        - ``screen_id``  (non-empty string)
        - ``route``      (non-empty string)
        - ``surface``    (non-empty string)
        - ``inherits_shell`` (list — may be empty if explicit override
          comment is present in the body)
    3. Body has a description paragraph (non-blank text) before any ``##``
    4. At least one ``## <Section>`` heading beyond the ``States`` section
    5. ``## States`` H2 with H3 sub-sections for ``Default``, ``Empty``,
       ``Loading``, and ``Error`` (case-insensitive). C14 contract.
    6. No placeholder text (``TODO``, ``TBD``, ``<fill in>``…)

Pure check. Caller (post-hook wiring) provides the markdown via
``payload['plan_text']`` or a ``plan_paths`` list of files to validate
(each file checked independently — return is per-file).

Returns
-------
dict
    ``ok`` (bool), ``per_file`` (list of per-file dicts when paths given),
    or top-level ``frontmatter_*``/``problems`` when ``plan_text`` given.
"""
from __future__ import annotations

import re
from typing import Any

REQUIRED_FRONTMATTER_KEYS = (
    "_schema_version",
    "mission_id",
    "screen_id",
    "route",
    "surface",
    "inherits_shell",
)

REQUIRED_STATE_SUBSECTIONS = ("Default", "Empty", "Loading", "Error")

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_KEY_RE = re.compile(
    r'^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$', re.MULTILINE
)
_H2_RE = re.compile(r"^##\s+(.*?)\s*$", re.MULTILINE)
_H3_RE = re.compile(r"^###\s+(.*?)\s*$", re.MULTILINE)

_PLACEHOLDER_PATTERNS = (
    r"(?-i:\bTODO\b)",
    r"(?-i:\bTBD\b)",
    r"(?-i:\bFIXME\b)",
    r"<[A-Za-z][^>]{0,40}>",
    r"\[(?:fill[- ]in|placeholder|insert)[^\]]*\]",
    r"\bLorem ipsum\b",
)
_PLACEHOLDER_RE = re.compile("|".join(_PLACEHOLDER_PATTERNS), re.IGNORECASE)


def _parse_frontmatter(md: str) -> tuple[dict[str, str] | None, str]:
    """Return ``(frontmatter_dict, body_after_frontmatter)``.

    ``frontmatter_dict`` is ``None`` when no frontmatter block is present.
    The frontmatter parser is a bare-bones key:value scanner — paraflow
    plans don't need full YAML.
    """
    m = _FRONTMATTER_RE.match(md)
    if not m:
        return None, md
    block = m.group(1)
    out: dict[str, str] = {}
    for km in _KEY_RE.finditer(block):
        out[km.group(1)] = km.group(2).strip()
    body = md[m.end():]
    return out, body


def _verify_one(md: str) -> dict[str, Any]:
    problems: list[str] = []

    fm, body = _parse_frontmatter(md)
    if fm is None:
        problems.append("missing YAML frontmatter")
        return {
            "ok": False,
            "frontmatter_present": False,
            "frontmatter_keys": [],
            "missing_frontmatter_keys": list(REQUIRED_FRONTMATTER_KEYS),
            "missing_state_subsections": list(REQUIRED_STATE_SUBSECTIONS),
            "h2_sections": [],
            "placeholders": [],
            "problems": problems,
        }

    missing_fm = [
        k for k in REQUIRED_FRONTMATTER_KEYS
        if not fm.get(k) and k != "inherits_shell"
    ]
    # inherits_shell may be `[]` or even an explicit `[]` — only require
    # presence of the key, not non-empty value.
    if "inherits_shell" not in fm:
        missing_fm.append("inherits_shell")
    if missing_fm:
        problems.append(f"frontmatter missing keys: {missing_fm}")

    schema_version = fm.get("_schema_version", "").strip(' "\'')
    if schema_version and schema_version != "1":
        problems.append(
            f"_schema_version expected '1', got {schema_version!r}"
        )

    # Description paragraph: non-blank text between frontmatter end and
    # first ``##`` heading (after the optional ``# <ScreenName>`` H1).
    pre_h2 = body
    first_h2 = _H2_RE.search(body)
    if first_h2:
        pre_h2 = body[:first_h2.start()]
    # strip any leading H1 line ``# Foo`` to find the description prose.
    pre_lines = [
        ln for ln in pre_h2.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not pre_lines:
        problems.append("missing description paragraph before sections")

    # Top-level H2 headings.
    h2_sections = [m.group(1).strip() for m in _H2_RE.finditer(body)]
    non_state_h2 = [h for h in h2_sections if h.lower() != "states"]
    if not non_state_h2:
        problems.append(
            "no content sections beyond `## States` — at least one is required"
        )

    # `## States` block + H3 sub-sections.
    states_subs: list[str] = []
    states_idx = None
    for i, h in enumerate(h2_sections):
        if h.lower() == "states":
            states_idx = i
            break
    if states_idx is None:
        problems.append("missing `## States` section")
    else:
        # Slice body from the `## States` heading to the next H2 (or EOF).
        h2_positions = [(m.start(), m.group(1).strip())
                        for m in _H2_RE.finditer(body)]
        start_pos = h2_positions[states_idx][0]
        end_pos = (
            h2_positions[states_idx + 1][0]
            if states_idx + 1 < len(h2_positions)
            else len(body)
        )
        states_block = body[start_pos:end_pos]
        states_subs = [
            m.group(1).strip() for m in _H3_RE.finditer(states_block)
        ]

    states_subs_lower = {s.lower() for s in states_subs}
    missing_states = [
        sub for sub in REQUIRED_STATE_SUBSECTIONS
        if sub.lower() not in states_subs_lower
    ]
    if missing_states:
        problems.append(
            f"`## States` missing required H3 sub-sections: {missing_states}"
        )

    placeholders = list({m.group(0) for m in _PLACEHOLDER_RE.finditer(md)})
    if placeholders:
        problems.append(f"placeholder text present: {placeholders[:5]}")

    ok = not problems

    return {
        "ok": ok,
        "frontmatter_present": True,
        "frontmatter_keys": list(fm.keys()),
        "missing_frontmatter_keys": missing_fm,
        "schema_version": schema_version or None,
        "h2_sections": h2_sections,
        "states_subsections": states_subs,
        "missing_state_subsections": missing_states,
        "placeholders": placeholders[:10],
        "problems": problems,
    }


def verify_screen_plan_shape(
    *,
    plan_text: str | None = None,
    plan_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Validate one or more paraflow-shape ``screen_plan.md`` files.

    See module docstring for output schema.
    """
    if plan_text is not None:
        return _verify_one(plan_text)

    if not plan_paths:
        return {
            "ok": False,
            "error": "no plan_text or plan_paths provided",
            "per_file": [],
        }

    # Per-screen plans are authored under a runtime directory
    # (mission_<id>/.screens/) whose individual filenames are unknown at
    # workflow-author time, so the `checks` payload points at the DIRECTORY.
    # Expand any directory entry to its contained .md files (sorted for
    # determinism) — mirrors verify_screen_consistency. Without this the
    # verifier open()s a directory and the gate reports the uninformative
    # `problems=[]` that DLQ'd m90 task 567454.
    import os
    import glob as _glob
    # Production writes `.screens/<slug>/screen_plan.md` (one subdir per screen),
    # so the expansion RECURSES (also matches a flat `.screens/<slug>.md`).
    expanded: list[str] = []
    for p in plan_paths:
        if isinstance(p, str) and os.path.isdir(p):
            expanded.extend(
                sorted(_glob.glob(os.path.join(p, "**", "*.md"), recursive=True))
            )
        else:
            expanded.append(p)
    if not expanded:
        return {
            "ok": False,
            "error": "no per-screen .md plans found under the produces directory",
            "per_file": [],
        }

    per_file: list[dict[str, Any]] = []
    all_ok = True
    for p in expanded:
        try:
            with open(p, encoding="utf-8") as fh:
                md = fh.read()
        except OSError as e:
            per_file.append({"path": p, "ok": False, "error": str(e)})
            all_ok = False
            continue
        res = _verify_one(md)
        res["path"] = p
        per_file.append(res)
        if not res.get("ok"):
            all_ok = False

    return {"ok": all_ok, "per_file": per_file}
