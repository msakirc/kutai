"""Cost-curve presence verifier — Tier 2 of Z1 (A8).

Mechanical post-hook for stack-related ADRs (4.2, 4.4, 4.6, 4.8, 4.9, 4.10):
asserts every ``options_considered[*]`` carries a populated
``monthly_cost_curve`` (with ``at_mvp``, ``at_1k_users``, ``at_100k_users``
keys) plus a top-level ``cost_at_target_users_usd`` and ``cost_mitigation_plan``
field on the ADR. Reviewer at 4.16 then enforces:
``cost_at_target_users_usd > cost_ceiling_monthly_usd`` ⇒ fail unless
``cost_mitigation_plan`` is non-null.

Pure (no I/O when ``adr_obj`` provided; reads files when ``adr_paths``
given). No LLM.

Returns
-------
dict
    ``ok`` (bool), ``adr_id`` (str|None), ``options_missing_curve``
    (list of {index, id}), ``cost_at_target_missing`` (bool),
    ``cost_mitigation_field_missing`` (bool — distinct from ``null`` value
    which is allowed when the curve fits the ceiling).
"""
from __future__ import annotations

import json
import re
from typing import Any

REQUIRED_CURVE_KEYS = ("at_mvp", "at_1k_users", "at_100k_users")
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _coerce_to_obj(value: Any) -> dict | None:
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


def _gather(
    adr_text: str | None, adr_obj: dict | None, adr_paths: list[str] | None
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
                obj = _coerce_to_obj(fh.read())
            if obj is not None:
                return obj
        except OSError:
            continue
    return None


def _curve_populated(curve: Any) -> bool:
    if not isinstance(curve, dict):
        return False
    for k in REQUIRED_CURVE_KEYS:
        v = curve.get(k)
        if v is None:
            return False
        if isinstance(v, str) and not v.strip():
            return False
    return True


def verify_cost_curve_present(
    *,
    adr_text: str | None = None,
    adr_obj: dict | None = None,
    adr_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Validate cost-curve presence on a stack-related ADR."""
    adr = _gather(adr_text, adr_obj, adr_paths)
    if adr is None:
        return {
            "ok": False,
            "error": "no ADR provided or parseable",
            "adr_id": None,
            "options_missing_curve": [],
            "cost_at_target_missing": True,
            "cost_mitigation_field_missing": True,
        }

    adr_id = adr.get("adr_id")
    options = adr.get("options_considered") or []
    options_missing: list[dict[str, Any]] = []
    if isinstance(options, list):
        for idx, opt in enumerate(options):
            if not isinstance(opt, dict):
                options_missing.append({"index": idx, "id": None})
                continue
            if not _curve_populated(opt.get("monthly_cost_curve")):
                options_missing.append({"index": idx, "id": opt.get("id")})
    else:
        options_missing.append({"index": None, "id": None})

    cost_at_target_missing = "cost_at_target_users_usd" not in adr or adr.get(
        "cost_at_target_users_usd"
    ) is None
    # cost_mitigation_plan must exist as a key (value may be null when
    # cost_at_target_users_usd is below the z0 ceiling).
    cost_mitigation_field_missing = "cost_mitigation_plan" not in adr

    ok = (
        not options_missing
        and not cost_at_target_missing
        and not cost_mitigation_field_missing
        and isinstance(options, list)
        and len(options) >= 2
    )

    return {
        "ok": ok,
        "adr_id": adr_id,
        "options_missing_curve": options_missing,
        "cost_at_target_missing": cost_at_target_missing,
        "cost_mitigation_field_missing": cost_mitigation_field_missing,
    }
