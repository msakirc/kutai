"""Premortem shape verifier — Z1 Tier 5 (T5B / A6).

Mechanical post-hook on step ``6.5z failure_premortem``. Asserts the
premortem.md emitted by the analyst carries a JSON envelope with the
contract:

    {
      "_schema_version": "1",
      "scenarios": [
        {
          "kind": "technical" | "market" | "founder",
          "obituary": "<paragraph>",
          "plausibility": 1-5,
          "cause": "<one sentence>",
          "mapped_monitoring_rule": "<rule_id> | null"
        },
        ...
      ]
    }

Contract:
  - At least 3 scenarios.
  - Every scenario kind in {technical, market, founder} appears at least once.
  - plausibility ∈ [1, 5] (int).
  - obituary + cause + kind required, non-empty.
  - mapped_monitoring_rule may be null (premortem run BEFORE 6.6 reviewer
    decides; reviewer's rule is what closes the loop, not the premortem).

Pure check. Caller (workflow post-hook) provides either an envelope dict
via ``payload['premortem']`` or a path via ``payload['premortem_path']``.
The on-disk file may be either a ``.json`` envelope or a ``.md`` with a
fenced ``json`` block — the parser tries both.
"""
from __future__ import annotations

import json
import re
from typing import Any

REQUIRED_KINDS = ("technical", "market", "founder")
MIN_SCENARIOS = 3

_FENCE_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def _extract_envelope(text: str) -> Any:
    """Return the first JSON object in `text`, parsed.

    Tries: (1) raw json.loads, (2) the first ```json fenced block, (3) the
    first ``{...}`` balanced span. Returns the parsed object or raises
    ``ValueError``.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty premortem text")
    # Try raw JSON first.
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))
    # Last resort: find first balanced {...}.
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    raise ValueError("no JSON envelope found in premortem")


def _verify_envelope(env: Any) -> dict[str, Any]:
    problems: list[str] = []

    if not isinstance(env, dict):
        return {
            "ok": False,
            "problems": ["envelope_not_object"],
            "scenarios": [],
            "kinds_seen": [],
            "missing_kinds": list(REQUIRED_KINDS),
        }

    schema = str(env.get("_schema_version") or "")
    if schema != "1":
        problems.append(f"schema_version_mismatch:{schema!r}")

    scenarios = env.get("scenarios")
    if not isinstance(scenarios, list):
        return {
            "ok": False,
            "problems": problems + ["scenarios_not_list"],
            "scenarios": [],
            "kinds_seen": [],
            "missing_kinds": list(REQUIRED_KINDS),
        }

    if len(scenarios) < MIN_SCENARIOS:
        problems.append(f"too_few_scenarios:{len(scenarios)}<{MIN_SCENARIOS}")

    kinds_seen: list[str] = []
    per_item_problems: list[dict[str, Any]] = []
    for i, sc in enumerate(scenarios):
        item_problems: list[str] = []
        if not isinstance(sc, dict):
            per_item_problems.append({"index": i, "problems": ["not_object"]})
            continue
        kind = sc.get("kind")
        if kind not in REQUIRED_KINDS:
            item_problems.append(f"bad_kind:{kind!r}")
        else:
            kinds_seen.append(kind)
        obituary = sc.get("obituary")
        if not isinstance(obituary, str) or not obituary.strip():
            item_problems.append("obituary_missing_or_empty")
        cause = sc.get("cause")
        if not isinstance(cause, str) or not cause.strip():
            item_problems.append("cause_missing_or_empty")
        plaus = sc.get("plausibility")
        if not isinstance(plaus, int) or isinstance(plaus, bool):
            item_problems.append(f"plausibility_not_int:{type(plaus).__name__}")
        elif plaus < 1 or plaus > 5:
            item_problems.append(f"plausibility_out_of_range:{plaus}")
        # mapped_monitoring_rule may be null or a string; presence not required.
        mmr = sc.get("mapped_monitoring_rule", None)
        if mmr is not None and not isinstance(mmr, str):
            item_problems.append(
                f"mapped_monitoring_rule_bad_type:{type(mmr).__name__}"
            )
        if item_problems:
            per_item_problems.append({"index": i, "problems": item_problems})

    missing_kinds = [k for k in REQUIRED_KINDS if k not in set(kinds_seen)]
    if missing_kinds:
        problems.append(f"missing_kinds:{missing_kinds}")

    if per_item_problems:
        problems.append("scenario_item_problems")

    ok = (
        not problems
        and not per_item_problems
        and len(scenarios) >= MIN_SCENARIOS
        and not missing_kinds
    )
    return {
        "ok": ok,
        "problems": problems,
        "scenarios": scenarios,
        "kinds_seen": kinds_seen,
        "missing_kinds": missing_kinds,
        "per_item_problems": per_item_problems,
        "schema_version": schema or None,
    }


def verify_premortem_shape(
    *,
    premortem: Any | None = None,
    premortem_text: str | None = None,
    premortem_path: str | None = None,
) -> dict[str, Any]:
    """Validate a premortem envelope. See module docstring for contract.

    Caller provides exactly one of:
      - ``premortem``: parsed envelope object
      - ``premortem_text``: raw text (json or fenced markdown)
      - ``premortem_path``: file path
    """
    env: Any = None
    if premortem is not None:
        env = premortem
    elif premortem_text is not None:
        try:
            env = _extract_envelope(premortem_text)
        except Exception as e:
            return {
                "ok": False,
                "problems": [f"parse_error:{e}"],
                "scenarios": [],
                "kinds_seen": [],
                "missing_kinds": list(REQUIRED_KINDS),
            }
    elif premortem_path:
        try:
            with open(premortem_path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as e:
            return {
                "ok": False,
                "problems": [f"read_error:{e}"],
                "scenarios": [],
                "kinds_seen": [],
                "missing_kinds": list(REQUIRED_KINDS),
            }
        try:
            env = _extract_envelope(text)
        except Exception as e:
            return {
                "ok": False,
                "problems": [f"parse_error:{e}"],
                "scenarios": [],
                "kinds_seen": [],
                "missing_kinds": list(REQUIRED_KINDS),
            }
    else:
        return {
            "ok": False,
            "problems": ["no_input"],
            "scenarios": [],
            "kinds_seen": [],
            "missing_kinds": list(REQUIRED_KINDS),
        }
    return _verify_envelope(env)
