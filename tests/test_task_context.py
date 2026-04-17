import json
import pytest
from src.core.task_context import parse_context, set_context


def test_parse_context_dict_passthrough():
    task = {"context": {"classification": {"agent_type": "executor"}}}
    ctx = parse_context(task)
    assert ctx == {"classification": {"agent_type": "executor"}}


def test_parse_context_json_string():
    task = {"context": json.dumps({"chat_id": 1001})}
    ctx = parse_context(task)
    assert ctx == {"chat_id": 1001}


def test_parse_context_missing_returns_empty_dict():
    task = {"id": 1}
    ctx = parse_context(task)
    assert ctx == {}


def test_parse_context_empty_string_returns_empty_dict():
    task = {"context": ""}
    ctx = parse_context(task)
    assert ctx == {}


def test_parse_context_malformed_json_returns_empty_dict():
    task = {"context": "{not valid json"}
    ctx = parse_context(task)
    assert ctx == {}


def test_parse_context_non_dict_json_returns_empty_dict():
    task = {"context": json.dumps(["list", "not", "dict"])}
    ctx = parse_context(task)
    assert ctx == {}


def test_set_context_serializes_to_string():
    task = {"id": 1, "context": "{}"}
    updated = set_context(task, {"chat_id": 42})
    parsed = json.loads(updated["context"])
    assert parsed == {"chat_id": 42}


def test_set_context_does_not_mutate_input():
    task = {"id": 1, "context": "{}"}
    set_context(task, {"chat_id": 42})
    # Original task unchanged
    assert task["context"] == "{}"
