"""Interview-script shape verifier — Tier 2 of Z1 (A4).

Mechanical post-hook that reads ``interview_script.md`` and asserts:

* YAML front matter with ``_schema_version: "1"``, ``mission_id``,
  non-empty ``target_assumptions`` list, and ``question_count`` 5-7.
* A ``Logistics`` section.
* 5-7 ``### Q<N> — <topic>`` blocks, each carrying:
    - ``**Question:**``
    - ``**Probes:**`` (with at least one bullet)
    - ``**Looking for:**``

The check is pure (no I/O, no LLM).

Returns
-------
dict
    ``ok`` (bool), ``question_count`` (int), ``question_problems`` (list),
    ``target_assumptions`` (list), ``placeholders`` (list),
    ``schema_version`` (str|None), ``mission_id`` (int|None).
"""
from __future__ import annotations

import re
from typing import Any

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

_PLACEHOLDER_PATTERNS = (
    r"\bTODO\b",
    r"\bTBD\b",
    r"\bFIXME\b",
    r"<[A-Za-z][^>]{0,40}>",
    r"\[(?:fill[- ]in|placeholder|insert)[^\]]*\]",
    r"\bLorem ipsum\b",
)
_PLACEHOLDER_RE = re.compile("|".join(_PLACEHOLDER_PATTERNS), re.IGNORECASE)

_QUESTION_HEADER_RE = re.compile(
    r"^###\s+Q\d+\b[^\n]*$", re.MULTILINE
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


def _parse_front_matter(md: str) -> dict[str, Any]:
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
            quoted = False
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]
                quoted = True
            if not quoted:
                # Coerce bare integers (e.g. ``question_count: 5``) to int.
                # Quoted values stay strings (``_schema_version: "1"``).
                try:
                    out[key] = int(val)
                    continue
                except ValueError:
                    pass
            out[key] = val
    return out


def _split_questions(body: str) -> list[tuple[str, str]]:
    """Return [(header, block_text), ...] split by ``### Q<N>`` headers."""
    out: list[tuple[str, str]] = []
    current: str | None = None
    buf: list[str] = []
    for line in body.splitlines():
        if _QUESTION_HEADER_RE.match(line):
            if current is not None:
                out.append((current, "\n".join(buf).strip()))
            current = line.strip()
            buf = []
        else:
            buf.append(line)
    if current is not None:
        out.append((current, "\n".join(buf).strip()))
    return out


def _has_field(block: str, field: str) -> bool:
    # Accept any of:
    #   **Field:** ...
    #   **Field**: ...
    #   - **Field:** ...
    #   Field: ...        (line-leading bare label)
    pat = re.compile(
        r"(?:^|\n)[ \t]*(?:-[ \t]+)?(?:\*\*"
        + re.escape(field)
        + r"(?::)?\*\*[ \t]*[:\-—]?|"
        + re.escape(field)
        + r"[ \t]*[:\-—])",
        re.IGNORECASE,
    )
    return bool(pat.search(block))


def _probes_have_bullets(block: str) -> bool:
    """Does the Probes field have at least one bullet beneath it?"""
    pat = re.compile(
        r"\*\*Probes(?::)?\*\*[ \t]*[:\-—]?[ \t]*\n(.*?)(?=\n\s*\*\*[A-Z]|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pat.search(block)
    if not m:
        return False
    tail = m.group(1)
    bullets = [
        ln for ln in tail.splitlines() if re.match(r"^\s*[-*]\s+\S", ln)
    ]
    return len(bullets) >= 1


def verify_interview_script_shape(
    *,
    script_text: str | None = None,
    script_paths: list[str] | None = None,
    min_questions: int = 5,
    max_questions: int = 7,
) -> dict[str, Any]:
    """Validate ``interview_script.md`` shape (A4)."""
    md = _gather_text(script_text, script_paths)
    if not md.strip():
        return {
            "ok": False,
            "error": "empty interview script",
            "question_count": 0,
            "question_problems": [],
            "target_assumptions": [],
            "placeholders": [],
            "schema_version": None,
            "mission_id": None,
            "min_questions": min_questions,
            "max_questions": max_questions,
        }

    fm = _parse_front_matter(md)
    body = _FRONT_MATTER_RE.sub("", md, count=1)

    schema_version = fm.get("_schema_version")
    mission_id = fm.get("mission_id")
    target_assumptions = fm.get("target_assumptions") or []
    if isinstance(target_assumptions, str):
        target_assumptions = (
            [target_assumptions] if target_assumptions.strip() else []
        )

    has_logistics = bool(re.search(r"^##\s+Logistics\b", body, re.MULTILINE))

    questions = _split_questions(body)
    qcount = len(questions)

    question_problems: list[dict[str, Any]] = []
    for header, block in questions:
        missing_fields: list[str] = []
        for field in ("Question", "Probes", "Looking for"):
            if not _has_field(block, field):
                missing_fields.append(field)
        extra: list[str] = []
        if "Probes" not in missing_fields and not _probes_have_bullets(block):
            extra.append("probes_no_bullets")
        if missing_fields or extra:
            question_problems.append({
                "header": header,
                "missing_fields": missing_fields,
                "issues": extra,
            })

    placeholders = list({m.group(0) for m in _PLACEHOLDER_RE.finditer(md)})

    ok = (
        schema_version == "1"
        and bool(target_assumptions)
        and has_logistics
        and min_questions <= qcount <= max_questions
        and not question_problems
        and not placeholders
    )

    return {
        "ok": ok,
        "question_count": qcount,
        "question_problems": question_problems,
        "target_assumptions": list(target_assumptions),
        "placeholders": placeholders[:10],
        "schema_version": schema_version,
        "mission_id": mission_id,
        "has_logistics": has_logistics,
        "min_questions": min_questions,
        "max_questions": max_questions,
    }
