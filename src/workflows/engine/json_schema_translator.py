"""Translate KutAI's ``artifact_schema`` dialect into a strict JSON Schema
the OpenAI / llama.cpp / Ollama ``response_format: json_schema`` mode can
constrain the decoder against.

The dialect itself is defined in ``schema_dialect`` (single source of
truth — validator, translator, example generator, and per-artifact
checklist all consume the same helpers). This module is a thin
multi-artifact wrapper around ``schema_dialect.translate_rule``.

Multi-artifact wrapping:

    When the dialect declares two or more artifacts in the same step, the
    translated schema wraps them as an outer ``object`` whose required
    properties are each artifact name. This matches the validator's
    ``data.get(artifact_name)`` lookup pattern and lets the agent emit
    ``{"primary": {...}, "secondary": [...]}`` in a single call.

Usage::

    schema = artifact_schema_to_json_schema(step["artifact_schema"])
    if schema is not None:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "artifact", "schema": schema, "strict": True},
        }
"""
from __future__ import annotations

from typing import Optional

from src.workflows.engine.schema_dialect import translate_rule


def artifact_schema_to_json_schema(artifact_schema: dict) -> Optional[dict]:
    """Translate a KutAI artifact_schema dict to a JSON Schema dict.

    Returns ``None`` when no constrainable artifacts exist (all-markdown
    steps, missing dialect, or empty input). Callers should treat ``None``
    as "do not enable constrained decoding for this step".
    """
    if not isinstance(artifact_schema, dict) or not artifact_schema:
        return None

    artifact_schemas: dict[str, dict] = {}
    for name, rules in artifact_schema.items():
        if not isinstance(rules, dict):
            # Skip configuration entries like max_output_chars.
            continue
        # Top-level scalar artifacts (string/number/boolean/markdown) gain
        # nothing from constrained decoding — the validator already covers
        # presence/length. Skip so the caller doesn't enable strict mode
        # for a no-op case.
        if rules.get("type") in ("string", "number", "boolean", "markdown"):
            continue
        translated = translate_rule(rules)
        if translated is not None:
            artifact_schemas[name] = translated

    if not artifact_schemas:
        return None

    # Single artifact: emit its schema directly. Validator's
    # ``data.get(artifact_name) or any list value`` lookup tolerates both
    # bare-shape and wrapped-shape outputs, but bare is cheaper for the
    # decoder (one less object level).
    if len(artifact_schemas) == 1:
        return next(iter(artifact_schemas.values()))

    # Multi-artifact: wrap. Every artifact required at the top level so
    # the model can't omit one.
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(artifact_schemas.keys()),
        "properties": artifact_schemas,
    }


def build_response_format(artifact_schema: dict, name: str = "artifact") -> Optional[dict]:
    """Build the OpenAI / llama.cpp ``response_format: json_schema`` payload.

    ``name`` is required by OpenAI's structured-output API; defaults to a
    generic value since the schema itself carries the artifact identity.
    Returns ``None`` when ``artifact_schema`` is unconstrainable — callers
    must skip the response_format kwarg in that case (passing None to
    litellm raises).
    """
    schema = artifact_schema_to_json_schema(artifact_schema)
    if schema is None:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            "strict": True,
        },
    }
