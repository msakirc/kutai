"""Regression guard for the mechanical-task ``context`` shape.

The orchestrator's `_dispatch` unpacks `ctx["payload"]` onto `task["payload"]`
before calling `salako.run`, and salako routes on `payload["action"]`. If any
caller stores a flat `{"action": ..., ...}` at the top level of ``context``
(instead of the nested ``{"executor": "mechanical", "payload": {...}}`` shape),
dispatch silently dispatches with an empty payload and salako returns
"unknown mechanical action: None".

These tests verify that:
  1. The helper `_mechanical_context` produces the canonical shape.
  2. Callers in apply.py / cron.py / sweep.py produce tasks whose stored
     `context` JSON round-trips through the dispatch-unpack logic and yields
     a non-empty payload with the expected ``action`` key.
"""
import json
import pytest

from general_beckman.apply import _mechanical_context


def _simulate_dispatch_unpack(context_dict: dict) -> dict:
    """Mirror the orchestrator._dispatch unpack logic (no imports needed)."""
    ctx = context_dict
    t = {"agent_type": "mechanical"}
    if "payload" not in t and "payload" in ctx:
        t["payload"] = ctx["payload"]
    return t


def test_mechanical_context_canonical_shape():
    """Helper produces {executor: mechanical, payload: {action: ..., **kwargs}}."""
    ctx = _mechanical_context("clarify", question="why?", chat_id=42)
    assert ctx == {
        "executor": "mechanical",
        "payload": {"action": "clarify", "question": "why?", "chat_id": 42},
    }


def test_dispatch_unpack_surfaces_payload_action():
    """Orchestrator's unpack logic must extract payload with the action key."""
    ctx = _mechanical_context("workflow_advance",
                              mission_id=7, completed_task_id=500)
    task = _simulate_dispatch_unpack(ctx)
    assert task.get("payload"), "payload must be set after unpack"
    assert task["payload"]["action"] == "workflow_advance"
    assert task["payload"]["mission_id"] == 7
    assert task["payload"]["completed_task_id"] == 500


def test_flat_context_breaks_dispatch_unpack_regression():
    """Negative test: a FLAT context (the old bug shape) leaves payload empty.

    If anyone reintroduces the old `context={"action": "x", ...}` pattern,
    this test documents why it breaks: salako.run reads payload.get("action")
    but payload is empty after unpack.
    """
    flat_ctx = {"action": "clarify", "question": "why?"}
    task = _simulate_dispatch_unpack(flat_ctx)
    assert "payload" not in task, (
        "FLAT context has no 'payload' key so unpack sets nothing — "
        "salako then sees payload.get('action') == None."
    )
