"""Mechanical artifact-schema gate (fix #1).

Promotes the existing ``src.workflows.engine.hooks.validate_artifact_schema``
(backed by the schema_dialect, which normalizes legacy ``required_fields`` /
``item_fields`` at entry) from a produces-gated file-persistence tie-breaker
into a *gating* mechanical post-hook that runs BEFORE the LLM grader.

On FAIL the validator's own precise message (e.g.
``'user_stories' has ~0 list items, need >= 5`` or
``monetization_strategy.revenue_projections: missing required field``) becomes
the producer's retry feedback — deterministic and actionable, unlike the
prose-reading grader's bare ``COMPLETE: NO``.

Result mirrors the verify_artifacts / check_grounding shape so the
apply-verdict path can reuse the retry-with-feedback mechanics.

#289735 (single object vs required array) / #289737 (field completeness).
"""
from __future__ import annotations

from typing import Any


def schema_gate(*, output_value: str, schema: dict) -> dict[str, Any]:
    """Return ``{passed, error}``.

    Parameters
    ----------
    output_value:
        The producer's artifact, as the raw result string. A ``final_answer``
        envelope is unwrapped by the underlying validator.
    schema:
        The step's ``artifact_schema`` dict, keyed by artifact name.

    Returns
    -------
    dict
        ``passed``: True when the artifact satisfies the schema (or no schema
        is declared — a vacuous pass).
        ``error``: empty on pass; the validator's precise failure reason
        otherwise.
    """
    if not isinstance(schema, dict) or not schema:
        return {"passed": True, "error": ""}

    from src.workflows.engine.hooks import validate_artifact_schema

    ok, error = validate_artifact_schema(output_value, schema)
    return {"passed": bool(ok), "error": "" if ok else (error or "schema validation failed")}
