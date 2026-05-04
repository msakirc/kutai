"""Tests for trim_for_escalation context reset on model escalation.

Phase A.7: moved from BaseAgent to src.runtime.escalation. Plain function,
no agent instance needed.
"""

from src.runtime.escalation import trim_for_escalation


def test_trim_keeps_system_and_task():
    msgs = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "## Task\nDo something"},
        {"role": "assistant", "content": "reasoning..."},
        {"role": "user", "content": "## Tool Result (`shell`):\n```\nok\n```"},
        {"role": "assistant", "content": "more reasoning"},
    ]
    t = trim_for_escalation(msgs, iteration=4, max_iterations=8)
    assert t[0]["role"] == "system"
    assert "## Task" in t[1]["content"]
    assert any("ok" in m["content"] for m in t)
    assert not any("more reasoning" in m.get("content", "") for m in t)
    assert "previous attempt" in t[-1]["content"].lower()


def test_trim_keeps_last_error():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Task"},
        {"role": "user", "content": "❌ timeout"},
    ]
    t = trim_for_escalation(msgs, iteration=3, max_iterations=8)
    assert any("timeout" in m.get("content", "") for m in t)


def test_trim_strips_assistant_messages():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Task"},
        {"role": "assistant", "content": "I think..."},
        {"role": "assistant", "content": "Let me try..."},
    ]
    t = trim_for_escalation(msgs, iteration=3, max_iterations=8)
    assert not any(m["role"] == "assistant" for m in t)


def test_trim_remaining_iterations():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "Task"}]
    t = trim_for_escalation(msgs, iteration=5, max_iterations=8)
    assert "Iterations remaining: 2" in t[-1]["content"]


def test_trim_keeps_multiple_tool_results():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Task"},
        {"role": "user", "content": "## Tool Result (`read_file`):\n```\ncontents\n```"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "## Tool Result (`shell`):\n```\nsuccess\n```"},
    ]
    t = trim_for_escalation(msgs, iteration=4, max_iterations=8)
    tool_results = [m for m in t if "## Tool Result" in m.get("content", "")]
    assert len(tool_results) == 2


def test_trim_excludes_error_tool_results():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Task"},
        {"role": "user", "content": "## Tool Result (`shell`):\n```\nok\n```"},
        {"role": "user", "content": "❌ Tool error: command not found"},
    ]
    t = trim_for_escalation(msgs, iteration=4, max_iterations=8)
    # Successful tool result kept
    assert any("ok" in m.get("content", "") for m in t)
    # Error kept as last_error
    assert any("command not found" in m.get("content", "") for m in t)
