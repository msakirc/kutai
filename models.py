# models.py
"""
Pydantic response models for structured agent output.

These models define the canonical action types an agent can produce.
They are used for:
  - Validating parsed JSON from model responses
  - Generating JSON schemas for models that support response_format
  - Typed tool argument validation
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ─── Agent Action Models ─────────────────────────────────────────────────────

class ToolCallAction(BaseModel):
    """Agent wants to execute a tool."""
    action: Literal["tool_call"] = "tool_call"
    tool: str = Field(..., description="Name of the tool to invoke")
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool",
    )
    reasoning: Optional[str] = Field(
        None, description="Agent's reasoning for this action"
    )


class FinalAnswerAction(BaseModel):
    """Agent is providing its final response."""
    action: Literal["final_answer"] = "final_answer"
    result: str = Field(..., description="The complete answer / result")
    memories: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value pairs to remember for future tasks",
    )
    reasoning: Optional[str] = Field(
        None, description="Agent's reasoning"
    )


class ClarifyAction(BaseModel):
    """Agent needs more information from the user."""
    action: Literal["clarify"] = "clarify"
    question: str = Field(..., description="Clarification question to ask")
    reasoning: Optional[str] = Field(
        None, description="Agent's reasoning"
    )


class DecomposeAction(BaseModel):
    """Agent wants to break the task into subtasks."""
    action: Literal["decompose"] = "decompose"
    subtasks: list[dict[str, Any]] = Field(
        ..., description="List of subtask definitions"
    )
    plan_summary: Optional[str] = Field(
        None, description="Summary of the decomposition plan"
    )
    reasoning: Optional[str] = Field(
        None, description="Agent's reasoning"
    )


# ─── Validation helpers ──────────────────────────────────────────────────────

# Map action type strings to their Pydantic model for quick lookup.
ACTION_MODELS: dict[str, type[BaseModel]] = {
    "tool_call":    ToolCallAction,
    "final_answer": FinalAnswerAction,
    "clarify":      ClarifyAction,
    "decompose":    DecomposeAction,
}


def validate_action(parsed: dict) -> dict:
    """Validate a parsed action dict against its Pydantic model.

    Returns the validated (and potentially coerced) dict on success.
    Raises ``ValueError`` with a clear message on failure.
    """
    action = parsed.get("action")
    model_cls = ACTION_MODELS.get(action)
    if model_cls is None:
        # Unknown action — pass through without validation
        return parsed
    try:
        validated = model_cls.model_validate(parsed)
        return validated.model_dump(exclude_none=True)
    except Exception as exc:
        raise ValueError(
            f"Invalid {action} response: {exc}"
        ) from exc


def get_action_json_schema() -> dict:
    """Build a JSON schema that accepts any of the defined action types.

    This can be passed to models that support ``response_format``
    with ``type: "json_schema"`` (OpenAI, Gemini).
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_action",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": list(ACTION_MODELS.keys()),
                        "description": "The type of action to take",
                    },
                    "tool": {
                        "type": "string",
                        "description": "Tool name (for tool_call)",
                    },
                    "args": {
                        "type": "object",
                        "description": "Tool arguments (for tool_call)",
                    },
                    "result": {
                        "type": "string",
                        "description": "Final answer text (for final_answer)",
                    },
                    "question": {
                        "type": "string",
                        "description": "Clarification question (for clarify)",
                    },
                    "subtasks": {
                        "type": "array",
                        "description": "Subtask list (for decompose)",
                    },
                    "plan_summary": {
                        "type": "string",
                        "description": "Plan summary (for decompose)",
                    },
                    "memories": {
                        "type": "object",
                        "description": "Key-value memories to store",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Optional reasoning / chain of thought",
                    },
                },
                "required": ["action"],
            },
        },
    }


# ─── Tool Argument Validation ────────────────────────────────────────────────

# JSON Schema type → Python type coercion map
_JSON_SCHEMA_COERCE: dict[str, type] = {
    "integer": int,
    "number": float,
    "boolean": bool,
    "string": str,
}


def validate_tool_args(
    tool_name: str,
    args: dict[str, Any],
    schema: dict,
) -> tuple[dict[str, Any], list[str]]:
    """Validate and coerce tool arguments against a JSON schema.

    Args:
        tool_name: Name of the tool (for error messages).
        args: The arguments dict from the LLM.
        schema: The tool's parameter schema from TOOL_SCHEMAS
                (the ``parameters`` sub-dict).

    Returns:
        (coerced_args, errors) — coerced_args has type-cast values;
        errors is a list of human-readable error strings (empty on success).
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    errors: list[str] = []
    coerced: dict[str, Any] = {}

    # Check required params
    for param in required:
        if param not in args:
            errors.append(
                f"Missing required argument '{param}' for tool '{tool_name}'"
            )

    # Validate and coerce each provided arg
    for key, value in args.items():
        if key not in properties:
            # Extra arg — silently keep (will be filtered by execute_tool)
            coerced[key] = value
            continue

        expected_type = properties[key].get("type", "string")
        coerce_to = _JSON_SCHEMA_COERCE.get(expected_type)

        if coerce_to is not None and not isinstance(value, coerce_to):
            try:
                coerced[key] = coerce_to(value)
            except (ValueError, TypeError):
                errors.append(
                    f"Argument '{key}' for tool '{tool_name}' should be "
                    f"{expected_type}, got {type(value).__name__}: {value!r}"
                )
                coerced[key] = value  # keep original so tool can still try
        else:
            coerced[key] = value

    return coerced, errors
