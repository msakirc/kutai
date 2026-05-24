"""Falsification-triple verifier — Z1 Tier 2 (P4).

Mechanical post-hook that reads a phase-3 requirement-bundle artifact
(functional_requirements, non_functional_requirements, business_rules,
system_quality_attributes, persona lists, ADR option lists) and asserts
that EVERY item carries the falsification triple introduced by Z1 Tier 2:

    - ``risk_if_wrong``: one of "low" | "medium" | "high" | "critical"
    - ``validation_method``: non-empty string (how we'd know it's wrong)
    - ``falsification_signal``: non-empty string (specific observable)

When ``risk_if_wrong == "critical"`` the validation_method must name a
measurable / specific check (heuristic: at least one digit OR one of the
keywords {"interview", "monitor", "test", "fact-check", "audit",
"benchmark", "track", "measure", "scan"} present).

Pure function — no I/O, no LLM. Caller (post-hook wiring) provides the
artifact via ``payload['artifacts']`` as either a list of items, a dict
with an ``items`` / ``personas`` / ``options_considered`` array, or a
mapping of ``artifact_name -> artifact_value``.

Returns
-------
dict
    ``ok`` (bool), ``checked`` (int total items), ``missing`` (list of
    ``{artifact, item_id, missing_fields}``), ``critical_underspecified``
    (list of ``{artifact, item_id, validation_method}``), ``empty`` (bool
    when no items found at all — signals a wiring bug to the caller).
"""
from __future__ import annotations

from typing import Any

ALLOWED_RISK = ("low", "medium", "high", "critical")

REQUIRED_TRIPLE = (
    "risk_if_wrong",
    "validation_method",
    "falsification_signal",
)

# When risk_if_wrong=critical, validation_method must reference a
# concrete check. Heuristic — full LLM rigor lives at reviewer 3.11.
_CRITICAL_METHOD_KEYWORDS = (
    "interview",
    "monitor",
    "fact-check",
    "fact check",
    "audit",
    "benchmark",
    "track",
    "measure",
    "scan",
    "survey",
    "regulation",
    "policy",
    "pen-test",
    "pentest",
    "p95",
    "p99",
    "latency",
    "compliance",
    "regression",
    "a/b",
)

# Phrases that look specific but are NOT — they tend to appear in vague
# "users will tell us" / "we'll see" rationalizations.
_CRITICAL_METHOD_BLOCKLIST = (
    "users will tell us",
    "we'll see",
    "we will see",
    "feels off",
    "tell us",
)


def _has_text(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _is_specific_validation(text: str) -> bool:
    """Heuristic: validation_method names a measurable / specific check."""
    if not _has_text(text):
        return False
    s = text.lower()
    if any(b in s for b in _CRITICAL_METHOD_BLOCKLIST):
        return False
    if any(c.isdigit() for c in s):
        return True
    return any(k in s for k in _CRITICAL_METHOD_KEYWORDS)


def _extract_items(value: Any) -> list[Any]:
    """Walk the common shapes phase-3 artifacts use to expose item lists.

    Accepts:
      - bare list (functional_requirements is `[{...}, {...}]`)
      - dict with `items` / `personas` / `options_considered` array
      - dict-of-dicts (rare; keyed by req_id)
    """
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        for k in ("items", "personas", "options_considered", "requirements",
                  "rules", "attributes"):
            inner = value.get(k)
            if isinstance(inner, list):
                return list(inner)
        # dict-of-dicts fallback (rare).
        if value and all(isinstance(v, dict) for v in value.values()):
            return list(value.values())
    return []


def _item_id(item: dict) -> str:
    for k in ("req_id", "id", "feature_id", "rule_id", "persona_id",
              "option_id", "title", "name"):
        v = item.get(k)
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v)
    return "<unnamed>"


def verify_falsification_present(
    *,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the falsification-triple check across one or more artifacts.

    Parameters
    ----------
    artifacts
        Mapping of ``artifact_name -> artifact_value``. Each value is
        walked for items via :func:`_extract_items`.
    """
    artifacts = artifacts or {}

    missing: list[dict[str, Any]] = []
    critical_underspecified: list[dict[str, Any]] = []
    checked = 0
    saw_any_items = False

    for name, value in artifacts.items():
        items = _extract_items(value)
        if items:
            saw_any_items = True
        for item in items:
            if not isinstance(item, dict):
                continue
            checked += 1
            iid = _item_id(item)
            missing_fields: list[str] = []
            for f in REQUIRED_TRIPLE:
                if not _has_text(item.get(f)):
                    missing_fields.append(f)
            risk = (item.get("risk_if_wrong") or "").strip().lower()
            if risk and risk not in ALLOWED_RISK:
                missing_fields.append(
                    f"risk_if_wrong={risk!r} not in {list(ALLOWED_RISK)}"
                )
            if missing_fields:
                missing.append(
                    {"artifact": name, "item_id": iid,
                     "missing_fields": missing_fields}
                )
                continue
            if risk == "critical":
                if not _is_specific_validation(item.get("validation_method", "")):
                    critical_underspecified.append(
                        {"artifact": name, "item_id": iid,
                         "validation_method": item.get("validation_method", "")}
                    )

    empty = not saw_any_items
    ok = (not missing) and (not critical_underspecified) and not empty
    return {
        "ok": ok,
        "checked": checked,
        "missing": missing[:25],
        "critical_underspecified": critical_underspecified[:10],
        "empty": empty,
    }
