"""Schema-version verifier — Layer 1 of P7.

Mechanical post-hook that reads a step's emitted artifacts and asserts that
each carries a ``_schema_version`` field matching the declared expectation.
The pair ``(artifact_name, _schema_version)`` is what reviewer prompts cite
(see N4 in docs/i2p-evolution/01-pre-code-plan-v3.md): downstream reviewers
must know which schema rev they are grading so a future bump to ``"2"`` does
not silently regress them against ``"1"`` fixtures.

Inputs (via ``payload``):
    artifacts:
        Either a dict ``{name: artifact_value}`` or a list of
        ``{"name": str, "value": Any}`` items. ``value`` may be a JSON object
        (preferred — version is read directly), a JSON string (parsed first),
        or markdown text containing a fenced ``json`` block whose top-level
        object carries ``_schema_version``.
    expected_versions:
        Dict mapping ``artifact_name -> expected_version_string``. Sourced
        from the workflow step's ``artifact_schema`` block by the expander.

Returns
-------
dict
    ``ok`` (bool), ``checked`` (count), ``missing`` (list of names without
    a version field), ``mismatched`` (list of ``{name, found, expected}``).
"""
from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_version(value: Any) -> str | None:
    """Best-effort extraction of ``_schema_version`` from a single artifact.

    Returns the version string when found, otherwise ``None``.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        v = value.get("_schema_version")
        return str(v) if v is not None else None
    if isinstance(value, list):
        # Lists are not versioned at the top level; the workflow always
        # wraps versioned arrays in an outer object. Return None so callers
        # can flag as missing.
        return None
    if isinstance(value, str):
        s = value.strip()
        # Try direct JSON parse.
        try:
            obj = json.loads(s)
            return _extract_version(obj)
        except (json.JSONDecodeError, ValueError):
            pass
        # Markdown with embedded ```json``` fence.
        for block in _FENCE_RE.findall(s):
            try:
                obj = json.loads(block.strip())
                v = _extract_version(obj)
                if v is not None:
                    return v
            except (json.JSONDecodeError, ValueError):
                continue
        # Last-ditch: look for `"_schema_version": "X"` literal.
        m = re.search(r'"_schema_version"\s*:\s*"([^"]+)"', s)
        if m:
            return m.group(1)
        return None
    return None


def _normalize_artifacts(raw: Any) -> dict[str, Any]:
    """Coerce ``payload['artifacts']`` into a name→value dict."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):
        out: dict[str, Any] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("artifact_name")
            if not name:
                continue
            out[str(name)] = item.get("value")
        return out
    return {}


def verify_schema_version(
    *,
    artifacts: Any,
    expected_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the schema-version check. Pure function — no I/O.

    See module docstring for parameter semantics.
    """
    expected = {k: str(v) for k, v in (expected_versions or {}).items()}
    arts = _normalize_artifacts(artifacts)

    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    checked = 0
    for name, exp in expected.items():
        if name not in arts:
            # Not produced by this step — caller (post-hook wiring) should
            # only pass artifacts the step actually emits, so a missing entry
            # is treated as a soft skip rather than failure.
            continue
        checked += 1
        found = _extract_version(arts[name])
        if found is None:
            missing.append(name)
            continue
        if found != exp:
            mismatched.append({"name": name, "found": found, "expected": exp})

    ok = not missing and not mismatched
    return {
        "ok": ok,
        "checked": checked,
        "missing": missing,
        "mismatched": mismatched,
    }
