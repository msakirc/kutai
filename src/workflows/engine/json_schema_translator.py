"""Translate KutAI's ``artifact_schema`` dialect into a strict JSON Schema
the OpenAI / llama.cpp / Ollama ``response_format: json_schema`` mode can
constrain the decoder against.

The KutAI dialect is intentionally narrow — see ``hooks.validate_artifact_schema``
for the validator side. Each top-level key in an ``artifact_schema`` dict is
either:

  * A configuration entry (``max_output_chars: int``) — skipped.
  * An artifact rules dict ``{type, ...}`` — translated.

Supported artifact types:

  * ``object`` with ``required_fields: [str]`` -> JSON Schema object with
    every required field listed in both ``required`` and ``properties``,
    ``additionalProperties: false`` so OpenAI strict mode accepts it.
  * ``array`` with ``min_items: int`` and optional ``item_fields: [str]`` ->
    JSON Schema array with ``minItems`` and (when ``item_fields`` present)
    items typed as objects whose required fields are ``item_fields``.
  * ``string`` / ``markdown`` -> not constrained (constrained decoding on a
    single string field gains nothing the validator doesn't already check).

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

from typing import Any, Optional


# Permissive value type for unconstrained-by-dialect fields. Using bare
# ``{}`` as the property schema (the obvious choice) was a trap: OpenAI
# strict + llama.cpp interpret it as "object accepting no further
# constraint" and the decoder emits empty ``{}`` for every property.
# Mission 46 task 2964 (8.0 implementation_backlog_initialization)
# observed: draft was 4549 chars of substantive plan content; constrained
# emit returned 169 chars of ``[{feature_id: {}, feature_name: {}, ...}]``
# — required fields PRESENT (validator passes) but every value an empty
# object (downstream can't use them).
#
# Listing all permitted JSON types via ``"type": [...]`` keeps the dialect
# permissive (any value goes) while telling the decoder this is a leaf
# value, not another object level. The decoder then samples real
# content from the draft instead of zero-filling a nested shape.
_LEAF_VALUE_SCHEMA = {
    "type": ["string", "number", "boolean", "array", "object", "null"],
}


def _translate_object(rules: dict) -> dict:
    """Convert ``{type:"object", required_fields:[...]}`` to JSON Schema."""
    required = list(rules.get("required_fields") or [])
    # OpenAI strict mode demands additionalProperties:false AND every field
    # listed in required must appear in properties. We constrain only the
    # SHAPE (which fields exist), not the value type within each field.
    # See _LEAF_VALUE_SCHEMA above for why ``{}`` was wrong.
    properties: dict[str, dict] = {f: dict(_LEAF_VALUE_SCHEMA) for f in required}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _translate_array(rules: dict) -> dict:
    """Convert ``{type:"array", min_items:N, item_fields:[...]}`` to JSON Schema."""
    schema: dict[str, Any] = {"type": "array"}
    min_items = rules.get("min_items")
    if isinstance(min_items, int) and min_items > 0:
        schema["minItems"] = min_items
    item_fields = list(rules.get("item_fields") or [])
    if item_fields:
        schema["items"] = {
            "type": "object",
            "additionalProperties": False,
            "required": item_fields,
            "properties": {f: dict(_LEAF_VALUE_SCHEMA) for f in item_fields},
        }
    return schema


def _translate_one(rules: dict) -> Optional[dict]:
    """Translate a single artifact rules dict. Returns None for unconstrainable types."""
    schema_type = rules.get("type", "string")
    if schema_type == "object":
        return _translate_object(rules)
    if schema_type == "array":
        return _translate_array(rules)
    # markdown / string / unknown — not constrained.
    return None


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
        translated = _translate_one(rules)
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
