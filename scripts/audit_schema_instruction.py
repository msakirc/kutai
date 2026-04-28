"""Audit i2p_v3 schema vs instruction field-name mismatches.

Heuristic only — instructions are prose, no perfect parser. Flags steps
where the instruction text mentions field names NOT in the artifact_schema
required_fields/item_fields lists. Human reviews each flag and decides
whether to tighten schema or accept the gap.

Pattern detection:
  - snake_case identifiers preceded by "For each:" or "with"
  - "`field_name`" backtick-wrapped
  - "field_name:" colon-suffixed at start of clause
  - comma-separated lists after a colon

Usage: python scripts/audit_schema_instruction.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WF_PATH = _PROJECT_ROOT / "src" / "workflows" / "i2p" / "i2p_v3.json"

# A "field name" candidate: snake_case (lowercase letters, digits,
# underscores), at least 2 chars, no leading underscore, no leading digit.
_FIELD_TOKEN_RE = re.compile(r"\b([a-z][a-z0-9_]{2,})\b")

# Common English words that match snake_case pattern but aren't fields.
# Aggressive list — we'd rather miss a field than spam the audit with
# false positives.
_STOPWORDS = frozenset({
    "and", "or", "the", "for", "each", "with", "must", "should", "include",
    "use", "via", "ensure", "every", "this", "that", "these", "those",
    "across", "between", "into", "from", "where", "when", "what", "which",
    "before", "after", "during", "while", "until", "since", "than", "then",
    "first", "next", "last", "previous", "any", "all", "some", "none",
    "true", "false", "yes", "no", "minimum", "maximum", "exactly", "only",
    "also", "still", "even", "just", "perhaps", "maybe", "above", "below",
    "here", "there", "now", "today", "tomorrow", "yesterday",
    "review", "validate", "produce", "create", "extract", "build", "generate",
    "based", "input", "output", "step", "task", "item", "items", "list",
    "array", "object", "string", "number", "boolean", "field", "fields",
    "value", "values", "key", "keys", "type", "types", "format", "schema",
    "json", "markdown", "text", "content", "document", "section", "sections",
    "your", "their", "its", "his", "her", "our",
    "data", "result", "results", "response", "answer", "final", "raw",
    "source", "target", "primary", "secondary", "main", "core", "central",
    "high", "medium", "low", "critical", "blocker", "minor", "major",
    "wcag", "level", "criterion", "phase", "phases", "milestone", "milestones",
    "user", "users", "users", "developer", "developers",
    "id", "ids", "name", "names", "title", "titles", "description",
    "descriptions", "summary", "summaries",
    "status", "verdict", "issue", "issues", "fix", "fixes", "error", "errors",
    "implementation", "guidance", "testing", "method",
    "specific", "explicit", "concrete", "structured", "comprehensive",
    "complete", "completed", "incomplete", "partial", "full", "empty",
    "needs", "wants", "requires", "demand", "expect", "expected",
    "real", "concrete", "abstract", "logical", "physical",
    "platform", "platforms", "browser", "browsers", "mobile",
    "language", "languages", "country", "countries", "region", "regions",
})


def _extract_candidate_fields(text: str) -> set[str]:
    """Extract snake_case identifiers from prose, filter common words."""
    if not text:
        return set()
    candidates = set(_FIELD_TOKEN_RE.findall(text.lower()))
    return {c for c in candidates if c not in _STOPWORDS and "_" in c}


# Comma-list field enumeration: "For each: a, b, c." / "must include: x, y."
# / "fields: alpha_one, beta, gamma_three." / "with pricing, setup, lock_in".
# Captures every token between the trigger and the next sentence break,
# regardless of underscore presence. 3.5 hits the colon variant; 4.8 hits
# the comma-after-"with" variant ("recommend a vendor with pricing, setup
# complexity, SDK quality, lock-in risk"). Live grader DLQs called these
# out by name (2026-04-28).
_LIST_TRIGGER_RE = re.compile(
    r"(?:for each|must include|include|fields|with|consist(?:ing)? of|"
    r"following|these)\s*[:,]\s*([^.\n]{8,400}?)(?:\.|\n|$)",
    re.IGNORECASE,
)
_LIST_ITEM_RE = re.compile(r"\b([a-z][a-z0-9_]{2,})\b")


def _extract_listed_fields(text: str) -> set[str]:
    """Extract tokens from comma-list enumerations following list-trigger words.

    Looser than ``_extract_candidate_fields``: drops the underscore-only
    requirement because instructions often mix snake_case (rate_limits) with
    plain words (pricing, alternatives).
    """
    if not text:
        return set()
    found: set[str] = set()
    for m in _LIST_TRIGGER_RE.finditer(text):
        body = m.group(1)
        for tok in _LIST_ITEM_RE.findall(body.lower()):
            if tok in _STOPWORDS:
                continue
            found.add(tok)
    return found


def _required_fields_from_schema(schema: dict) -> set[str]:
    """Collect every field name declared as required in artifact_schema.

    Walks rules recursively. Handles both legacy form (``required_fields``
    / ``item_fields`` flat lists) and canonical E1 form (``fields`` map /
    ``items`` rule, recursively).
    """
    out: set[str] = set()
    if not isinstance(schema, dict):
        return out
    for art_name, rules in schema.items():
        if not isinstance(rules, dict):
            continue
        out.add(art_name)
        _collect_field_names(rules, out)
    return out


def _collect_field_names(rule: dict, out: set[str]) -> None:
    """Recursively pull every field name from a dialect rule."""
    if not isinstance(rule, dict):
        return
    # Canonical: object.fields map
    fields_map = rule.get("fields")
    if isinstance(fields_map, dict):
        for fname, frule in fields_map.items():
            if isinstance(fname, str):
                out.add(fname)
            if isinstance(frule, dict):
                _collect_field_names(frule, out)
    # Canonical: array.items rule (single rule, not list)
    items_rule = rule.get("items")
    if isinstance(items_rule, dict):
        _collect_field_names(items_rule, out)
    # Legacy: required_fields / item_fields flat lists
    for key in ("required_fields", "item_fields"):
        for f in rule.get(key) or []:
            if isinstance(f, str):
                out.add(f)


def audit() -> int:
    with open(_WF_PATH, encoding="utf-8") as f:
        wf = json.load(f)

    flagged = []
    for step in wf.get("steps", []):
        schema = step.get("artifact_schema") or {}
        if not schema:
            continue
        instr = step.get("instruction") or ""
        if not instr:
            continue

        schema_fields = _required_fields_from_schema(schema)
        instr_candidates = _extract_candidate_fields(instr)
        listed = _extract_listed_fields(instr)
        instr_candidates |= listed

        # Subtract upstream artifact names — they're inputs to THIS step,
        # not output fields. Reduces false positives when the instruction
        # refers to a prior phase's artifact.
        input_arts = set(step.get("input_artifacts") or [])
        # Also drop the literal ``final_answer`` (the envelope JSON tag,
        # never a field).
        ignored = input_arts | {"final_answer"}
        instr_candidates -= ignored

        # Mismatch: instruction mentions snake_case identifier NOT in schema.
        gap = instr_candidates - schema_fields
        # Filter further: the candidate has to look like a field that
        # would plausibly belong in THIS schema. Heuristic: it appears
        # in a field-list-like context within the instruction. Crude
        # check: the instruction contains "<candidate>:" or "<candidate>(",
        # or the word appears 2+ times.
        likely = []
        instr_lower = instr.lower()
        for cand in gap:
            count = instr_lower.count(cand)
            if count >= 2:
                likely.append((cand, count))
            elif (f"{cand}:" in instr_lower
                  or f"{cand}(" in instr_lower
                  or f"`{cand}`" in instr_lower):
                likely.append((cand, count))
            elif cand in listed:
                # Appeared explicitly in a comma-list field enumeration —
                # high-confidence regardless of overall mention count.
                likely.append((cand, count))

        if likely:
            flagged.append({
                "step_id": step.get("id"),
                "name": step.get("name"),
                "agent": step.get("agent"),
                "schema_fields": sorted(schema_fields),
                "missing_in_schema": sorted([(n, c) for n, c in likely],
                                            key=lambda x: -x[1]),
            })

    print(f"flagged {len(flagged)} of {len(wf.get('steps', []))} steps")
    print()
    for f in flagged:
        print(f"=== {f['step_id']} ({f['agent']}) — {f['name']} ===")
        print(f"  schema declares: {f['schema_fields']}")
        print(f"  instruction mentions but NOT in schema (candidate, count):")
        for name, count in f["missing_in_schema"]:
            print(f"    {name} (x{count})")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(audit())
