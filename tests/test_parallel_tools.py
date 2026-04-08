"""Tests for parallel tool execution (multi_tool_call)."""
from src.agents.base import BaseAgent, _partition_tool_calls


class TestParseMultiToolCall:
    def test_single_unchanged(self):
        r = BaseAgent._parse_function_call_response([
            {"name": "read_file", "arguments": {"filepath": "a.py"}}
        ])
        assert r["action"] == "tool_call"
        assert r["tool"] == "read_file"

    def test_multiple_returns_multi(self):
        r = BaseAgent._parse_function_call_response([
            {"name": "read_file", "arguments": {"filepath": "a.py"}},
            {"name": "read_file", "arguments": {"filepath": "b.py"}},
        ])
        assert r["action"] == "multi_tool_call"
        assert len(r["tools"]) == 2

    def test_final_answer_still_works(self):
        r = BaseAgent._parse_function_call_response([
            {"name": "final_answer", "arguments": {"result": "done"}}
        ])
        assert r["action"] == "final_answer"

    def test_multi_with_final_answer_first(self):
        r = BaseAgent._parse_function_call_response([
            {"name": "final_answer", "arguments": {"result": "done"}},
            {"name": "read_file", "arguments": {"filepath": "a.py"}},
        ])
        assert r["action"] == "final_answer"

    def test_empty_returns_none(self):
        assert BaseAgent._parse_function_call_response([]) is None

    def test_multi_filters_pseudo_tools(self):
        r = BaseAgent._parse_function_call_response([
            {"name": "read_file", "arguments": {"filepath": "a.py"}},
            {"name": "clarify", "arguments": {"question": "what?"}},
            {"name": "shell", "arguments": {"command": "ls"}},
        ])
        assert r["action"] == "multi_tool_call"
        assert len(r["tools"]) == 2  # clarify filtered out

    def test_multi_collapses_to_single(self):
        """If after filtering only 1 tool remains, return tool_call."""
        r = BaseAgent._parse_function_call_response([
            {"name": "read_file", "arguments": {"filepath": "a.py"}},
            {"name": "final_answer", "arguments": {"result": "x"}},
        ])
        assert r["action"] == "tool_call"
        assert r["tool"] == "read_file"


class TestNormalizeMultiTool:
    def test_passthrough(self):
        parsed = {"action": "multi_tool_call", "tools": [{"tool": "x", "args": {}}]}
        r = BaseAgent._normalize_action(parsed)
        assert r is not None
        assert r["action"] == "multi_tool_call"

    def test_passthrough_preserves_tools(self):
        tools = [
            {"tool": "read_file", "args": {"filepath": "a.py"}},
            {"tool": "file_tree", "args": {"path": "/tmp"}},
        ]
        parsed = {"action": "multi_tool_call", "tools": tools}
        r = BaseAgent._normalize_action(parsed)
        assert r["tools"] == tools


class TestPartitionTools:
    def test_all_read_only(self):
        p, s = _partition_tool_calls([
            {"tool": "read_file", "args": {}}, {"tool": "file_tree", "args": {}},
        ])
        assert len(p) == 2 and len(s) == 0

    def test_all_side_effect(self):
        p, s = _partition_tool_calls([
            {"tool": "write_file", "args": {}}, {"tool": "shell", "args": {}},
        ])
        assert len(p) == 0 and len(s) == 2

    def test_mixed(self):
        p, s = _partition_tool_calls([
            {"tool": "read_file", "args": {}}, {"tool": "write_file", "args": {}},
        ])
        assert len(p) == 1 and len(s) == 1

    def test_unknown_is_side_effect(self):
        p, s = _partition_tool_calls([{"tool": "unknown", "args": {}}])
        assert len(p) == 0 and len(s) == 1

    def test_empty(self):
        p, s = _partition_tool_calls([])
        assert len(p) == 0 and len(s) == 0
