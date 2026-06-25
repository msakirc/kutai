"""Smoke + contract tests for kara_kutu.

These are DB-free: they verify the public surface survived the move and that the
pure markdown formatters behave. Behavioural DB write/read paths are covered by
the caller suites (coulson react, hallederiz, mr_roboto) which own temp_db
fixtures — we deliberately do NOT touch get_db here (it would hit the real DB).
"""
from __future__ import annotations


def test_public_api_exported():
    import kara_kutu

    expected = [
        "audit", "get_audit_log", "format_audit_log",
        "append_trace", "get_trace", "format_trace",
        "ACTOR_AGENT", "ACTOR_SYSTEM", "ACTOR_HUMAN",
        "ACTION_TOOL_EXEC", "ACTION_MODEL_CALL", "ACTION_STATE_CHANGE",
        "ACTION_FILE_MODIFY", "ACTION_HUMAN_APPROVE", "ACTION_MISSION_CREATE",
        "ACTION_MISSION_COMPLETE", "ACTION_TASK_CREATE", "ACTION_TASK_COMPLETE",
    ]
    for name in expected:
        assert hasattr(kara_kutu, name), f"kara_kutu missing public name: {name}"


def test_action_actor_constant_values():
    import kara_kutu
    assert kara_kutu.ACTOR_AGENT == "agent"
    assert kara_kutu.ACTION_TOOL_EXEC == "tool_exec"
    assert kara_kutu.ACTION_MODEL_CALL == "model_call"


def test_format_audit_log_empty():
    from kara_kutu import format_audit_log
    assert format_audit_log([]) == "_No audit entries found._"


def test_format_audit_log_renders_entry():
    from kara_kutu import format_audit_log
    out = format_audit_log([{
        "timestamp": "2026-01-01T00:00:00",
        "actor": "agent:coder",
        "action": "tool_exec",
        "target": "write_file",
        "details": "x.py",
    }])
    assert "tool_exec" in out
    assert "agent:coder" in out
    assert "write_file" in out


def test_format_trace_empty():
    from kara_kutu import format_trace
    assert format_trace([]) == "_No trace entries._"


def test_format_trace_renders_total_cost():
    from kara_kutu import format_trace
    out = format_trace([
        {"type": "tool", "timestamp": "t", "input": "i", "output": "o",
         "cost": 0.01, "duration_ms": 5.0},
        {"type": "model", "timestamp": "t", "input": "i", "output": "o",
         "cost": 0.02, "duration_ms": 10.0},
    ])
    assert "tool" in out
    assert "Total cost" in out
    assert "0.0300" in out  # 0.01 + 0.02
