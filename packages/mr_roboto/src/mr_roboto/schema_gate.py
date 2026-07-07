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


def schema_gate(
    *, output_value: str, schema: dict, inputs: dict | None = None,
    produces_markdown: bool = False,
) -> dict[str, Any]:
    """Return ``{passed, error}``.

    Parameters
    ----------
    output_value:
        The producer's artifact, as the raw result string. A ``final_answer``
        envelope is unwrapped by the underlying validator.
    schema:
        The step's ``artifact_schema`` dict, keyed by artifact name.
    inputs:
        Optional ``{artifact_name: parsed_value}`` map of the step's upstream
        input artifacts. Anchors the dialect's ``empty_ok_when_input_empty``
        per-field exemption (an empty required field is valid only when the
        named upstream scope is itself empty). Loaded by the caller from the
        produced files — NEVER from the producer's own output — so a lazy
        model cannot fake an empty-scope exemption.
    produces_markdown:
        True when the validated artifact is a markdown file (the step's produces
        is ``*.md``). For an object/array schema whose structured value cannot be
        extracted, the underlying validator otherwise degenerates into a literal
        substring search for the field NAMES — meaningless on prose (both false
        pass AND false reject; e.g. ``mermaid_per_surface`` matched against a
        markdown flow doc). Forwarding this makes the grade-path gate DEFER to the
        step's ``verify_*_shape`` check, exactly as the producer gate already does
        (``hooks.py`` ``produces_markdown=all(.md)``). Closes the grade/producer
        asymmetry that false-rejected clean markdown (m90 5.0c user_flow).

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

    ok, error = validate_artifact_schema(
        output_value, schema, inputs=inputs, produces_markdown=produces_markdown,
    )
    return {"passed": bool(ok), "error": "" if ok else (error or "schema validation failed")}
