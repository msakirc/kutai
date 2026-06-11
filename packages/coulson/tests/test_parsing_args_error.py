"""parse_function_call must carry the truncated-arguments marker through.

response.py attaches `arguments_error` to a tool call whose function-call
arguments were truncated mid-stream. The react loop needs that signal to emit
a 'truncated, resend' nudge instead of running the tool arg-less. The parser
must not drop it.
"""
from __future__ import annotations

from coulson.parsing import parse_function_call


def test_single_tool_call_carries_args_error():
    tcs = [{"id": "1", "name": "write_file", "arguments": {},
            "arguments_error": "arguments were not valid JSON (9000 chars received)"}]
    action = parse_function_call(tcs)
    assert action["action"] == "tool_call"
    assert action["tool"] == "write_file"
    assert action["args_error"] == "arguments were not valid JSON (9000 chars received)"


def test_multi_tool_call_carries_args_error_per_tool():
    tcs = [
        {"id": "1", "name": "write_file", "arguments": {},
         "arguments_error": "truncated mid-stream"},
        {"id": "2", "name": "read_file", "arguments": {"filepath": "a.txt"}},
    ]
    action = parse_function_call(tcs)
    assert action["action"] == "multi_tool_call"
    assert action["tools"][0]["args_error"] == "truncated mid-stream"
    assert "args_error" not in action["tools"][1]


def test_clean_single_tool_call_has_no_args_error():
    tcs = [{"id": "1", "name": "search", "arguments": {"query": "x"}}]
    action = parse_function_call(tcs)
    assert action["action"] == "tool_call"
    assert "args_error" not in action
