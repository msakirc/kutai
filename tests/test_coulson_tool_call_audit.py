"""Coverage: per-execution tool-call audit log captured by coulson runtime.

The grounding guard (Layer 1 + Layer 2 of G) consumes a richer audit
than ``tools_used_names`` (set of names): it needs args (specifically
``path``/``filepath``) and per-call success status to verify the agent
actually wrote to declared ``produces`` paths instead of narrating.
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

from coulson.react import _record_tool_call, _truncate_args, _tool_output_ok


# ── _tool_output_ok ──────────────────────────────────────────────────────

@pytest.mark.parametrize("output,expected", [
    ("Wrote backend/x.py (123 bytes)", True),
    ("OK", True),
    ("", True),  # empty success allowed; no failure marker
    ("❌ Tool 'shell' timed out", False),
    ("\U0001f6ab Tool 'web_search' not permitted", False),
    ("bash: command not found: foo", False),
    ("No such file or directory: /nope", False),
    ("ran with exit code 1", False),
    ("ran with exit code 0\nstdout: hi", True),
])
def test_tool_output_ok(output, expected):
    assert _tool_output_ok(output) is expected


def test_tool_output_ok_non_string():
    assert _tool_output_ok(None) is False
    assert _tool_output_ok(42) is False


# ── _truncate_args ───────────────────────────────────────────────────────

def test_truncate_short_args_unchanged():
    args = {"path": "foo.py", "size": 42}
    assert _truncate_args(args) == args


def test_truncate_long_string_value_capped():
    long = "x" * 5000
    args = {"path": "foo.py", "content": long}
    out = _truncate_args(args, max_chars=200)
    assert out["path"] == "foo.py"
    assert len(out["content"]) < len(long)
    assert "5000 chars" in out["content"]


def test_truncate_non_string_values_kept():
    """Booleans, numbers, lists, dicts pass through as-is — only strings
    get the length cap."""
    args = {"flag": True, "count": 99, "items": [1, 2, 3], "nested": {"k": "v"}}
    assert _truncate_args(args) == args


def test_truncate_non_dict_returns_empty():
    assert _truncate_args("not a dict") == {}  # type: ignore[arg-type]
    assert _truncate_args(None) == {}  # type: ignore[arg-type]


# ── _record_tool_call ────────────────────────────────────────────────────

def test_record_appends_with_ok_true_on_success():
    calls: list[dict] = []
    _record_tool_call(calls, name="write_file", args={"path": "x.py"}, output="Wrote x.py")
    assert len(calls) == 1
    entry = calls[0]
    assert entry["name"] == "write_file"
    assert entry["args"] == {"path": "x.py"}
    assert entry["ok"] is True


def test_record_appends_with_ok_false_on_failure():
    calls: list[dict] = []
    _record_tool_call(calls, name="shell", args={"command": "fail"}, output="❌ bad")
    assert calls[0]["ok"] is False


def test_record_skips_blank_tool_name():
    calls: list[dict] = []
    _record_tool_call(calls, name="", args={"x": 1}, output="ok")
    assert calls == []


def test_record_truncates_long_args():
    calls: list[dict] = []
    _record_tool_call(
        calls,
        name="write_file",
        args={"path": "x.py", "content": "x" * 1000},
        output="Wrote x.py",
    )
    assert "1000 chars" in calls[0]["args"]["content"]
    assert calls[0]["args"]["path"] == "x.py"


def test_record_preserves_call_order():
    """Grounding scan walks tool_calls in order; assert append-only behaviour."""
    calls: list[dict] = []
    _record_tool_call(calls, name="read_file", args={"path": "a.py"}, output="ok")
    _record_tool_call(calls, name="write_file", args={"path": "b.py"}, output="Wrote b.py")
    _record_tool_call(calls, name="shell", args={"command": "ls"}, output="ok")
    assert [c["name"] for c in calls] == ["read_file", "write_file", "shell"]


# ── checkpoint round-trip ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_checkpoint_persists_tool_calls(tmp_path, monkeypatch):
    """save_checkpoint() must include tool_calls in the persisted state
    so the grounding post-hook can read it after task completes (or
    after a checkpoint resume picks up an in-flight task)."""
    from coulson import checkpoint as ckpt_mod

    captured_state: dict = {}

    async def _fake_save(task_id, state):
        captured_state["task_id"] = task_id
        captured_state["state"] = state

    monkeypatch.setattr(ckpt_mod, "save_task_checkpoint", _fake_save)

    @dataclasses.dataclass
    class _Reqs:
        difficulty: str = "easy"

    tool_calls = [
        {"name": "write_file", "args": {"path": "x.py"}, "ok": True},
        {"name": "shell",      "args": {"command": "pytest"}, "ok": False},
    ]

    await ckpt_mod.save_checkpoint(
        task_id=42,
        next_iteration=3,
        messages=[{"role": "user", "content": "go"}],
        total_cost=0.01,
        used_model="local",
        reqs=_Reqs(),
        tools_used=True,
        validation_retried=False,
        completed_tool_ops={},
        format_corrections=0,
        tools_used_names={"write_file", "shell"},
        tool_calls=tool_calls,
    )

    state = captured_state["state"]
    assert state["tool_calls"] == tool_calls
    assert state["tools_used_names"] == ["shell", "write_file"] or sorted(state["tools_used_names"]) == ["shell", "write_file"]


@pytest.mark.asyncio
async def test_save_checkpoint_default_tool_calls_empty(tmp_path, monkeypatch):
    from coulson import checkpoint as ckpt_mod

    captured: dict = {}

    async def _fake_save(task_id, state):
        captured["state"] = state

    monkeypatch.setattr(ckpt_mod, "save_task_checkpoint", _fake_save)

    @dataclasses.dataclass
    class _Reqs:
        difficulty: str = "easy"

    await ckpt_mod.save_checkpoint(
        task_id=1,
        next_iteration=1,
        messages=[],
        total_cost=0.0,
        used_model="x",
        reqs=_Reqs(),
        tools_used=False,
        validation_retried=False,
    )
    assert captured["state"]["tool_calls"] == []
