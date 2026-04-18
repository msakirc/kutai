"""Centralized parse/serialize for the `context` field of a task row.

Task context is stored as a JSON string in SQLite. Before this module, each
call site had slightly different fallback handling; this centralizes the
parse-with-fallback and the serialize-back pattern.
"""

import json


def parse_context(task: dict) -> dict:
    """Return task context as a dict. Empty dict on any parse failure or type mismatch."""
    raw = task.get("context")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def set_context(task: dict, ctx: dict) -> dict:
    """Return a shallow copy of task with `context` field set to serialized ctx."""
    updated = dict(task)
    updated["context"] = json.dumps(ctx)
    return updated
