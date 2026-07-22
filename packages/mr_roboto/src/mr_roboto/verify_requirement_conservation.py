"""Requirement-conservation verifier — assembly-fidelity gate.

Mechanical post-hook asserting that an ASSEMBLED artifact (traceability
matrix, requirements-spec part 1, final requirements spec) carries EVERY
requirement id present in its upstream source artifact(s). The `easy`/
`medium`-tier assembly steps are LLM writers told to "compile / map /
format"; handed a long requirement list they silently compress it (m90:
15 functional requirements in, 11 out — the 4 newest dropped). This gate
catches the drop deterministically at the producer and re-pends it with
feedback naming the dropped ids, so the mission self-heals rather than
halting at the downstream reviewer.

Pure function — no I/O, no LLM. The caller (mr_roboto dispatch) reads the
produced + source artifacts off the mission workspace and passes their text.

Conservation is a subset check: source_ids ⊆ produced_ids. Extra ids in the
produced artifact are fine (assembly may legitimately add). A source that
carries no matching ids is a vacuous pass (``empty`` flagged) — the safe
direction, since false-blocking a producer is worse than a missed drop that
the LLM reviewer still backstops.
"""
from __future__ import annotations

import re
from typing import Any


def _ids(pattern: str, text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    try:
        return re.findall(pattern, text)
    except re.error:
        return []


def _sort_ids(ids) -> list[str]:
    """Natural-ish sort so FR-2 < FR-10 and the missing list reads in order."""
    def key(x: str):
        m = re.search(r"(\d+)\s*$", x)
        return (re.sub(r"\d+\s*$", "", x), int(m.group(1)) if m else 0)
    return sorted(ids, key=key)


def verify_requirement_conservation(
    *,
    produced_text: str,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assert every id in each source appears in ``produced_text``.

    Parameters
    ----------
    produced_text
        Full text of the assembled artifact.
    sources
        List of ``{"label": str, "source_text": str, "id_pattern": str}``.
        Each rule extracts ids from ``source_text`` via ``id_pattern`` and
        requires all of them to be present in ``produced_text``.
    """
    sources = sources or []
    missing: list[dict[str, Any]] = []
    checked = 0
    saw_any = False

    for rule in sources:
        if not isinstance(rule, dict):
            continue
        pattern = rule.get("id_pattern") or ""
        label = str(rule.get("label") or pattern or "source")
        source_ids = set(_ids(pattern, rule.get("source_text") or ""))
        if not source_ids:
            continue
        saw_any = True
        checked += len(source_ids)
        produced_ids = set(_ids(pattern, produced_text or ""))
        dropped = source_ids - produced_ids
        if dropped:
            missing.append({
                "label": label,
                "id_pattern": pattern,
                "missing_ids": _sort_ids(dropped),
            })

    empty = not saw_any
    ok = not missing
    return {
        "ok": ok,
        "checked": checked,
        "missing": missing[:25],
        "empty": empty,
    }
