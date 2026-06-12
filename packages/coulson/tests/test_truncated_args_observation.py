"""The runtime must turn a truncated-arguments marker into a clear, actionable
observation — NOT run the tool arg-less (which reads as 'tool unavailable' to
weak models and DLQs). See mission 81 ADR 4.1.
"""
from __future__ import annotations

from coulson.react import truncated_args_observation


def test_observation_names_tool_and_detail():
    msg = truncated_args_observation(
        "write_file", "arguments were not valid JSON (9000 chars received)")
    assert "write_file" in msg
    assert "9000" in msg


def test_observation_flagged_as_failure():
    """Must start with the ❌ failure marker so the loop counts it as a tool
    failure (drives mid-task model escalation) and the materializer/grounding
    do not treat it as a successful write."""
    msg = truncated_args_observation("write_file", "truncated mid-stream")
    assert msg.startswith("❌")


def test_observation_tells_agent_to_resend_smaller():
    msg = truncated_args_observation("write_file", "truncated mid-stream")
    low = msg.lower()
    assert "resend" in low or "smaller" in low or "shorter" in low
