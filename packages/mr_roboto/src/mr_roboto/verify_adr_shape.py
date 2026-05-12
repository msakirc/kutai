"""ADR-shape verifier — Tier 2 of Z1 (P3 + C7 + A8) + Z3 T4A schema v2.

Mechanical post-hook that reads an Architecture Decision Record (ADR) JSON
artifact and asserts the universal Nygard-extended shape:

    adr_id, title, status, context, decision, consequences,
    options_considered (list), chosen_option_id, falsification_signal,
    reversal_cost, supersedes_adr_id, _schema_version

The check is pure (no I/O, no LLM). The caller (post-hook wiring or test
fixture) provides either:
  - ``payload['adr_text']``: a JSON string OR markdown with a fenced
    ```json``` block carrying the ADR object,
  - ``payload['adr_obj']``: a dict already parsed,
  - ``payload['adr_paths']``: a list of filesystem paths the verifier reads.

Validations
-----------
1. Required top-level fields present.
2. ``status`` ∈ {proposed, accepted, deprecated, superseded}.
3. ``reversal_cost`` ∈ {low, medium, high}.
4. ``options_considered`` is a non-empty list and every option has
   ``id``/``name``/``rationale_for``/``rationale_against``.
5. ``chosen_option_id`` resolves to an option in ``options_considered``.
6. ``supersedes_adr_id`` is null OR a string matching the ADR-ID pattern.
7. ``_schema_version`` matches the expected version (default ``"1"``).
8. ``falsification_signal`` validated per schema version (see below).

falsification_signal rules (Z3 T4A)
-------------------------------------
* ``null`` — always valid: signals the decision is inherently judgment-only
  (e.g. REST vs GraphQL). T4B will route these to the LLM-only drift path.
* ``_schema_version == "2"`` — REQUIRES v2 object form:
  ``{"forbidden_imports": [...], "forbidden_patterns": [...],
     "required_test_coverage": true|false}``
  At least one key must be present and non-empty/non-None.
  Key types: ``forbidden_imports``/``forbidden_patterns`` = list[str];
  ``required_test_coverage`` = bool.
* ``_schema_version == "1"`` (or missing/unknown) — lenient: accepts either
  a non-empty string (original form) OR a valid v2 object (forward-compat).

Back-compat window: ``_schema_version "1"`` is accepted for one mission
cycle after the v2 rollout. The next planned mission run should drop v1
acceptance and require v2 from all new ADRs.

When ``require_cost_curve=True`` (used by the cost-curve verifier as a
sanity backstop), every option must include ``monthly_cost_curve`` with
``at_mvp``/``at_1k_users``/``at_100k_users`` fields. The dedicated
``verify_cost_curve_present`` action is the primary cost-curve guard.

Returns
-------
dict
    ``ok`` (bool), ``adr_id`` (str|None), ``missing_fields`` (list),
    ``status_invalid`` (bool), ``reversal_cost_invalid`` (bool),
    ``options_count`` (int), ``options_problems`` (list of dicts),
    ``orphan_chosen_option_id`` (bool), ``supersedes_invalid`` (bool),
    ``schema_version_mismatch`` (dict|None),
    ``falsification_missing`` (bool),
    ``falsification_invalid`` (bool).
"""
from __future__ import annotations

import json
import re
from typing import Any

REQUIRED_FIELDS = (
    "adr_id",
    "title",
    "status",
    "context",
    "decision",
    "consequences",
    "options_considered",
    "chosen_option_id",
    "falsification_signal",
    "reversal_cost",
    "supersedes_adr_id",
    "_schema_version",
)

ALLOWED_STATUS = {"proposed", "accepted", "deprecated", "superseded"}
ALLOWED_REVERSAL = {"low", "medium", "high"}

REQUIRED_OPTION_FIELDS = (
    "id",
    "name",
    "rationale_for",
    "rationale_against",
)

# ADR-2026-05-10-001-style id (also tolerates short slugs for tests).
_ADR_ID_RE = re.compile(r"^ADR-\d{4}-\d{2}-\d{2}-\d{2,4}(?:-[A-Za-z0-9_-]+)?$")
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _coerce_to_obj(value: Any) -> dict | None:
    """Best-effort: return a dict from value, or None if no parse possible."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass
        for block in _FENCE_RE.findall(s):
            try:
                obj = json.loads(block.strip())
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _gather_adr(
    adr_text: str | None,
    adr_obj: dict | None,
    adr_paths: list[str] | None,
) -> dict | None:
    if adr_obj is not None:
        return adr_obj if isinstance(adr_obj, dict) else None
    if adr_text:
        return _coerce_to_obj(adr_text)
    if not adr_paths:
        return None
    for p in adr_paths:
        try:
            with open(p, encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            continue
        obj = _coerce_to_obj(text)
        if obj is not None:
            return obj
    return None


def verify_adr_shape(
    *,
    adr_text: str | None = None,
    adr_obj: dict | None = None,
    adr_paths: list[str] | None = None,
    expected_schema_version: str = "1",
    require_cost_curve: bool = False,
) -> dict[str, Any]:
    """Validate universal-shape ADR.

    See module docstring for output schema.
    """
    adr = _gather_adr(adr_text, adr_obj, adr_paths)
    if adr is None:
        return {
            "ok": False,
            "error": "no ADR provided or parseable",
            "adr_id": None,
            "missing_fields": list(REQUIRED_FIELDS),
            "status_invalid": False,
            "reversal_cost_invalid": False,
            "options_count": 0,
            "options_problems": [],
            "orphan_chosen_option_id": False,
            "supersedes_invalid": False,
            "schema_version_mismatch": None,
            "falsification_missing": False,
        }

    missing = [f for f in REQUIRED_FIELDS if f not in adr]
    adr_id = adr.get("adr_id")

    status = adr.get("status")
    status_invalid = status not in ALLOWED_STATUS

    reversal = adr.get("reversal_cost")
    reversal_invalid = reversal not in ALLOWED_REVERSAL

    options = adr.get("options_considered")
    options_count = len(options) if isinstance(options, list) else 0
    options_problems: list[dict[str, Any]] = []
    option_ids: list[str] = []
    if isinstance(options, list):
        for idx, opt in enumerate(options):
            if not isinstance(opt, dict):
                options_problems.append(
                    {"index": idx, "missing_fields": list(REQUIRED_OPTION_FIELDS)}
                )
                continue
            opt_id = opt.get("id")
            if opt_id is not None:
                option_ids.append(str(opt_id))
            mf = [f for f in REQUIRED_OPTION_FIELDS if f not in opt]
            if require_cost_curve:
                curve = opt.get("monthly_cost_curve")
                if not isinstance(curve, dict) or not all(
                    k in curve for k in ("at_mvp", "at_1k_users", "at_100k_users")
                ):
                    mf.append("monthly_cost_curve")
            if mf:
                options_problems.append(
                    {"index": idx, "id": opt_id, "missing_fields": mf}
                )
    elif options is not None:
        # Non-list under options_considered is itself a problem.
        options_problems.append({"index": None, "missing_fields": ["options_considered:not-a-list"]})

    chosen = adr.get("chosen_option_id")
    orphan_chosen = (
        chosen is not None
        and option_ids
        and str(chosen) not in option_ids
    )

    supersedes = adr.get("supersedes_adr_id")
    supersedes_invalid = (
        supersedes is not None
        and not (isinstance(supersedes, str) and _ADR_ID_RE.match(supersedes))
    )

    sv = adr.get("_schema_version")
    schema_mismatch = None
    if sv is None:
        # Already counted in missing.
        pass
    elif str(sv) != str(expected_schema_version):
        schema_mismatch = {"found": str(sv), "expected": str(expected_schema_version)}

    # ── falsification_signal validation (Z3 T4A) ────────────────────────────
    # Determine effective schema version for falsification rules.
    # When expected_schema_version is "2" AND the ADR claims "2", enforce v2.
    # Otherwise use lenient (v1-compat) mode.
    falsification = adr.get("falsification_signal")
    _effective_sv = str(sv) if sv is not None else "1"
    _strict_v2 = (str(expected_schema_version) == "2" and _effective_sv == "2")

    falsification_missing = False
    falsification_invalid = False

    if falsification is None:
        # null is always valid — judgment-only ADR; T4B handles drift path.
        pass
    elif _strict_v2:
        # v2 requires object form.
        if not isinstance(falsification, dict):
            falsification_invalid = True
        else:
            _valid_keys = {"forbidden_imports", "forbidden_patterns", "required_test_coverage"}
            _bad_keys = set(falsification) - _valid_keys
            _has_nonempty = False
            _type_ok = True
            for k, v in falsification.items():
                if k in ("forbidden_imports", "forbidden_patterns"):
                    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                        _type_ok = False
                    elif v:  # non-empty list counts as present
                        _has_nonempty = True
                elif k == "required_test_coverage":
                    if not isinstance(v, bool):
                        _type_ok = False
                    else:
                        _has_nonempty = True
            if not _type_ok or not _has_nonempty:
                falsification_invalid = True
    else:
        # Lenient v1 mode: accept non-empty string OR valid v2 object.
        is_str_ok = isinstance(falsification, str) and bool(falsification.strip())
        is_obj_ok = isinstance(falsification, dict) and bool(falsification)
        if not is_str_ok and not is_obj_ok:
            falsification_missing = True

    ok = (
        not missing
        and not status_invalid
        and not reversal_invalid
        and options_count >= 2
        and not options_problems
        and not orphan_chosen
        and not supersedes_invalid
        and schema_mismatch is None
        and not falsification_missing
        and not falsification_invalid
    )

    return {
        "ok": ok,
        "adr_id": adr_id,
        "missing_fields": missing,
        "status_invalid": status_invalid,
        "reversal_cost_invalid": reversal_invalid,
        "options_count": options_count,
        "options_problems": options_problems,
        "orphan_chosen_option_id": bool(orphan_chosen),
        "supersedes_invalid": supersedes_invalid,
        "schema_version_mismatch": schema_mismatch,
        "falsification_missing": falsification_missing,
        "falsification_invalid": falsification_invalid,
    }
