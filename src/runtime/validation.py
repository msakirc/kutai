"""Output validation and Doğru mu Samet quality wrappers.

Pure functions — no agent state, no async.

Public API
----------
validate_final_answer(result, task) -> str | None
    Refusal / length / empty checks. Returns an error message or None if valid.

is_degenerate(text) -> tuple[bool, str | None]
    (is_degenerate, summary). Wraps dogru_mu_samet.assess.

salvage_or_drop(text) -> str | None
    Try to salvage degenerate text; None if unsalvageable. Wraps dogru_mu_samet.salvage.
"""
from __future__ import annotations


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


def validate_final_answer(result, task: dict) -> str | None:
    """Validate a final_answer result.

    Returns an error string if the response is invalid, or None if it passes.
    Identical logic to BaseAgent._validate_response minus ``self``.
    """
    if isinstance(result, dict):
        result = result.get("result", "") or str(result)
    if not result or not str(result).strip():
        return "Your response was empty. Please provide a substantive answer."

    stripped = str(result).strip()

    # For non-trivial tasks, require > 20 chars
    title = task.get("title", "").lower()
    trivial_keywords = ["list", "ls", "status", "count", "version", "ping"]
    is_trivial = any(kw in title for kw in trivial_keywords)

    if not is_trivial and len(stripped) < 20:
        return (
            "Your response seems too short for this task. "
            "Please provide a more complete answer."
        )

    # Check for refusal / error-only patterns
    refusal_patterns = [
        "i cannot", "i can't", "i'm unable", "as an ai",
        "i don't have access", "i am not able",
    ]
    lower = stripped.lower()
    if any(p in lower for p in refusal_patterns) and len(stripped) < 100:
        return (
            "Your response appears to be a refusal. "
            "Try a different approach or use the available tools."
        )

    return None  # validation passed


def is_degenerate(text: str) -> tuple[bool, str | None]:
    """Return ``(is_degenerate, summary)`` by delegating to dogru_mu_samet.assess.

    ``summary`` is a short description of the degeneracy when detected, or None
    when the text is clean.
    """
    from dogru_mu_samet import assess
    r = assess(text)
    return (r.is_degenerate, r.summary)


def salvage_or_drop(text: str) -> str | None:
    """Try to salvage degenerate text.

    Returns the cleaned text when salvage succeeds, or None when the text is
    unsalvageable. Delegates to dogru_mu_samet.salvage which already returns
    None on failure.
    """
    from dogru_mu_samet import salvage
    return salvage(text)
