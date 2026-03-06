# tests/test_phase2_structured.py
"""
Tests for Phase 2: Structured Output & Parsing

  2.1 Pydantic response models (validate_action)
  2.2 JSON schema for response_format (get_action_json_schema)
  2.3 Tool argument validation & coercion (validate_tool_args)
  2.4 Format retry logic (tested via _parse_agent_response inline)
"""
import hashlib
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    validate_action,
    validate_tool_args,
    get_action_json_schema,
    ToolCallAction,
    FinalAnswerAction,
    ClarifyAction,
    DecomposeAction,
    ACTION_MODELS,
)


# ─── 2.1 Pydantic Response Models ──────────────────────────────────────────

class TestValidateAction(unittest.TestCase):

    def test_valid_tool_call(self):
        parsed = {
            "action": "tool_call",
            "tool": "shell",
            "args": {"command": "ls -la"},
        }
        result = validate_action(parsed)
        self.assertEqual(result["action"], "tool_call")
        self.assertEqual(result["tool"], "shell")
        self.assertEqual(result["args"]["command"], "ls -la")

    def test_valid_final_answer(self):
        parsed = {
            "action": "final_answer",
            "result": "Done!",
            "memories": {"key": "value"},
        }
        result = validate_action(parsed)
        self.assertEqual(result["action"], "final_answer")
        self.assertEqual(result["result"], "Done!")
        self.assertEqual(result["memories"]["key"], "value")

    def test_valid_clarify(self):
        parsed = {
            "action": "clarify",
            "question": "What file?",
        }
        result = validate_action(parsed)
        self.assertEqual(result["action"], "clarify")
        self.assertEqual(result["question"], "What file?")

    def test_valid_decompose(self):
        parsed = {
            "action": "decompose",
            "subtasks": [{"title": "Step 1"}],
            "plan_summary": "Two steps",
        }
        result = validate_action(parsed)
        self.assertEqual(result["action"], "decompose")
        self.assertEqual(len(result["subtasks"]), 1)

    def test_missing_required_field_raises(self):
        # tool_call missing 'tool'
        parsed = {"action": "tool_call", "args": {"command": "ls"}}
        with self.assertRaises(ValueError):
            validate_action(parsed)

    def test_final_answer_missing_result_raises(self):
        parsed = {"action": "final_answer"}
        with self.assertRaises(ValueError):
            validate_action(parsed)

    def test_unknown_action_passes_through(self):
        parsed = {"action": "unknown_action", "data": "test"}
        result = validate_action(parsed)
        self.assertEqual(result, parsed)

    def test_extra_fields_stripped(self):
        parsed = {
            "action": "final_answer",
            "result": "Done!",
            "reasoning": "Because...",
        }
        result = validate_action(parsed)
        self.assertIn("reasoning", result)
        self.assertEqual(result["reasoning"], "Because...")

    def test_none_reasoning_excluded(self):
        parsed = {
            "action": "tool_call",
            "tool": "shell",
            "args": {},
        }
        result = validate_action(parsed)
        # reasoning=None should be excluded
        self.assertNotIn("reasoning", result)

    def test_tool_call_default_args(self):
        parsed = {"action": "tool_call", "tool": "shell"}
        result = validate_action(parsed)
        self.assertEqual(result["args"], {})


# ─── 2.2 JSON Schema ───────────────────────────────────────────────────────

class TestGetActionJsonSchema(unittest.TestCase):

    def test_schema_structure(self):
        schema = get_action_json_schema()
        self.assertEqual(schema["type"], "json_schema")
        self.assertIn("json_schema", schema)
        inner = schema["json_schema"]
        self.assertEqual(inner["name"], "agent_action")
        self.assertIn("schema", inner)

    def test_schema_has_action_enum(self):
        schema = get_action_json_schema()
        props = schema["json_schema"]["schema"]["properties"]
        self.assertIn("action", props)
        self.assertEqual(
            set(props["action"]["enum"]),
            {"tool_call", "final_answer", "clarify", "decompose"},
        )

    def test_schema_required_action(self):
        schema = get_action_json_schema()
        required = schema["json_schema"]["schema"]["required"]
        self.assertIn("action", required)

    def test_schema_has_all_fields(self):
        schema = get_action_json_schema()
        props = schema["json_schema"]["schema"]["properties"]
        expected_keys = {
            "action", "tool", "args", "result", "question",
            "subtasks", "plan_summary", "memories", "reasoning",
        }
        self.assertTrue(expected_keys.issubset(set(props.keys())))


# ─── 2.3 Tool Argument Validation ──────────────────────────────────────────

class TestValidateToolArgs(unittest.TestCase):

    def _make_schema(self, properties, required=None):
        return {
            "type": "object",
            "properties": properties,
            "required": required or [],
        }

    def test_valid_args_pass(self):
        schema = self._make_schema(
            {"command": {"type": "string"}},
            required=["command"],
        )
        coerced, errors = validate_tool_args("shell", {"command": "ls"}, schema)
        self.assertEqual(errors, [])
        self.assertEqual(coerced["command"], "ls")

    def test_missing_required_arg(self):
        schema = self._make_schema(
            {"command": {"type": "string"}},
            required=["command"],
        )
        coerced, errors = validate_tool_args("shell", {}, schema)
        self.assertEqual(len(errors), 1)
        self.assertIn("Missing required", errors[0])

    def test_string_to_int_coercion(self):
        schema = self._make_schema(
            {"start_line": {"type": "integer"}},
            required=["start_line"],
        )
        coerced, errors = validate_tool_args(
            "edit_file", {"start_line": "42"}, schema
        )
        self.assertEqual(errors, [])
        self.assertEqual(coerced["start_line"], 42)
        self.assertIsInstance(coerced["start_line"], int)

    def test_string_to_float_coercion(self):
        schema = self._make_schema(
            {"temperature": {"type": "number"}},
        )
        coerced, errors = validate_tool_args(
            "config", {"temperature": "0.7"}, schema
        )
        self.assertEqual(errors, [])
        self.assertAlmostEqual(coerced["temperature"], 0.7)

    def test_string_to_bool_coercion(self):
        schema = self._make_schema(
            {"verbose": {"type": "boolean"}},
        )
        coerced, errors = validate_tool_args(
            "test", {"verbose": 1}, schema
        )
        self.assertEqual(errors, [])
        self.assertTrue(coerced["verbose"])

    def test_invalid_coercion_produces_error(self):
        schema = self._make_schema(
            {"start_line": {"type": "integer"}},
            required=["start_line"],
        )
        coerced, errors = validate_tool_args(
            "edit_file", {"start_line": "not_a_number"}, schema
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("should be integer", errors[0])
        # Original value kept for best-effort
        self.assertEqual(coerced["start_line"], "not_a_number")

    def test_extra_args_kept(self):
        schema = self._make_schema(
            {"command": {"type": "string"}},
            required=["command"],
        )
        coerced, errors = validate_tool_args(
            "shell", {"command": "ls", "extra": "ignored"}, schema
        )
        self.assertEqual(errors, [])
        self.assertIn("extra", coerced)

    def test_correct_type_no_coercion(self):
        schema = self._make_schema(
            {"count": {"type": "integer"}},
        )
        coerced, errors = validate_tool_args(
            "test", {"count": 5}, schema
        )
        self.assertEqual(errors, [])
        self.assertEqual(coerced["count"], 5)


# ─── 2.4 Format Retry Logic (inline parse test) ────────────────────────────
# Tests the _parse_agent_response function to verify format retry conditions.
# We replicate the function here to avoid importing BaseAgent (heavy deps).

def _try_parse_json(text: str) -> dict | None:
    try:
        stripped = text
        if stripped.startswith("```"):
            stripped = (
                stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
            )
            stripped = stripped.rsplit("```", 1)[0]
        obj = json.loads(stripped.strip())
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, IndexError):
        return None


def _parse_agent_response_minimal(content: str) -> dict:
    """Minimal replication of BaseAgent._parse_agent_response for testing
    the JSON fallback path."""
    cleaned = content.strip()

    # Try direct parse
    parsed = _try_parse_json(cleaned)
    if parsed is not None and "action" in parsed:
        return parsed

    # Try code fences
    json_blocks = re.findall(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL
    )
    for block in json_blocks:
        parsed = _try_parse_json(block.strip())
        if parsed is not None and "action" in parsed:
            return parsed

    # Fallback
    return {"action": "final_answer", "result": content}


class TestFormatRetryCondition(unittest.TestCase):
    """Test conditions that trigger format retry."""

    def test_clean_json_no_retry(self):
        content = '{"action": "tool_call", "tool": "shell", "args": {"command": "ls"}}'
        parsed = _parse_agent_response_minimal(content)
        self.assertEqual(parsed["action"], "tool_call")
        # Not a fallback, so no retry needed
        self.assertNotEqual(parsed.get("result"), content)

    def test_garbled_json_triggers_fallback(self):
        content = 'Sure, here is the result: {"action": "tool_call", "tool": "shell" bad json'
        parsed = _parse_agent_response_minimal(content)
        # Falls through to final_answer fallback
        self.assertEqual(parsed["action"], "final_answer")
        self.assertEqual(parsed["result"], content)

    def test_plain_text_no_braces_no_retry(self):
        content = "The answer is 42"
        parsed = _parse_agent_response_minimal(content)
        self.assertEqual(parsed["action"], "final_answer")
        # No braces, so format retry should NOT trigger
        self.assertNotIn("{", content)

    def test_fallback_with_braces_would_retry(self):
        """Simulate the condition: fallback path + braces in content."""
        content = 'Here is my JSON: {"action": "tool_call" BROKEN'
        parsed = _parse_agent_response_minimal(content)
        # Detect the retry condition
        should_retry = (
            parsed.get("action") == "final_answer"
            and parsed.get("result") == content
            and "{" in content
        )
        self.assertTrue(should_retry)


# ─── Action Models registry ────────────────────────────────────────────────

class TestActionModels(unittest.TestCase):

    def test_all_action_types_registered(self):
        self.assertIn("tool_call", ACTION_MODELS)
        self.assertIn("final_answer", ACTION_MODELS)
        self.assertIn("clarify", ACTION_MODELS)
        self.assertIn("decompose", ACTION_MODELS)

    def test_model_classes_match(self):
        self.assertIs(ACTION_MODELS["tool_call"], ToolCallAction)
        self.assertIs(ACTION_MODELS["final_answer"], FinalAnswerAction)
        self.assertIs(ACTION_MODELS["clarify"], ClarifyAction)
        self.assertIs(ACTION_MODELS["decompose"], DecomposeAction)


if __name__ == "__main__":
    unittest.main()
