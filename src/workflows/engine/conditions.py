"""Evaluate v2 conditional groups based on artifact state."""

from __future__ import annotations

import json
import re
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("workflows.engine.conditions")


def _parse_artifact(artifact_str: str) -> Any:
    """JSON-parse *artifact_str*, falling back to the raw string on failure."""
    try:
        return json.loads(artifact_str)
    except (json.JSONDecodeError, TypeError):
        return artifact_str


def evaluate_condition(condition_check: str, artifact_value: str) -> bool:
    """Evaluate a condition DSL expression against an artifact value.

    Supported patterns:
        length(field) >= N
        any(item.field == 'value')
        field != 'value'
        field == true | false
        field == 'value'
        platforms_include('value')
        expr OR expr

    Returns False on any parse or evaluation error (safe default: skip
    optional conditional steps).
    """
    try:
        # --- OR ---
        if " OR " in condition_check:
            parts = condition_check.split(" OR ")
            return any(evaluate_condition(p.strip(), artifact_value) for p in parts)

        data = _parse_artifact(artifact_value)

        # --- length(field) >= N ---
        m = re.match(r"length\((\w+)\)\s*>=\s*(\d+)", condition_check)
        if m:
            field, threshold = m.group(1), int(m.group(2))
            if not isinstance(data, dict):
                return False
            container = data.get(field)
            if not isinstance(container, (list, dict)):
                return False
            return len(container) >= threshold

        # --- any(item.field == 'value') ---
        m = re.match(r"any\(\w+\.(\w+)\s*==\s*'([^']+)'\)", condition_check)
        if m:
            field, value = m.group(1), m.group(2)
            items = data if isinstance(data, list) else data.values() if isinstance(data, dict) else []
            return any(
                item.get(field) == value
                for item in items
                if isinstance(item, dict)
            )

        # --- platforms_include('value') ---
        m = re.match(r"platforms_include\('([^']+)'\)", condition_check)
        if m:
            value = m.group(1)
            platforms = data.get("platforms", []) if isinstance(data, dict) else []
            return value in platforms

        # --- field != 'value' ---
        m = re.match(r"(\w+)\s*!=\s*'([^']+)'", condition_check)
        if m:
            field, value = m.group(1), m.group(2)
            if isinstance(data, dict):
                return data.get(field) != value
            return False

        # --- field == true / false ---
        m = re.match(r"(\w+)\s*==\s*(true|false)", condition_check)
        if m:
            field, bool_str = m.group(1), m.group(2)
            expected = bool_str == "true"
            if isinstance(data, dict):
                return data.get(field) is expected
            return False

        # --- field == 'value' ---
        m = re.match(r"(\w+)\s*==\s*'([^']+)'", condition_check)
        if m:
            field, value = m.group(1), m.group(2)
            if isinstance(data, dict):
                return data.get(field) == value
            return False

        # Unrecognised expression
        return False

    except Exception:
        return False


def resolve_group(
    group: dict, artifact_value: str
) -> tuple[list[str], list[str]]:
    """Resolve a conditional group into included and excluded step IDs.

    Parameters
    ----------
    group:
        A conditional-group dict from the workflow definition.
    artifact_value:
        The JSON string of the condition artifact.

    Returns
    -------
    (included_step_ids, excluded_step_ids)
        If condition is True:  (if_true, if_false)
        If condition is False: (if_false + fallback_ids, if_true)
    """
    result = evaluate_condition(group["condition_check"], artifact_value)

    if_true: list[str] = group.get("if_true", [])
    if_false: list[str] = group.get("if_false", [])
    fallback_ids: list[str] = [
        fb["id"] for fb in group.get("fallback_steps", [])
    ]

    if result:
        return (list(if_true), list(if_false))

    # Merge if_false with fallback_ids, deduplicating while preserving order
    included = list(if_false)
    for fid in fallback_ids:
        if fid not in included:
            included.append(fid)
    return (included, list(if_true))
