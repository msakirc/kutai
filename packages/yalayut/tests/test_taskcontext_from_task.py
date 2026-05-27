"""TaskContext.from_task must parse a JSON-string context (P6 regression).

Production bug (mission 77, 4023 hits): DB-sourced task dicts carry `context`
as a JSON STRING. `from_task` did `ctx = task.get("context") or {}` then
`ctx.get("recipe_hint")` — `.get()` on a str raises `'str' object has no
attribute 'get'`. `yalayut.query()` therefore raised on EVERY db task, so
`intersect.flash` hit its outer except, degraded, and attached skills=[] every
time — skills never reached any agent.
"""
from __future__ import annotations

import json

from yalayut.contracts import TaskContext


def test_from_task_parses_json_string_context():
    task = {
        "id": 7, "title": "T", "description": "D", "agent_type": "coder",
        "context": json.dumps({"recipe_hint": "rh", "payload": {"k": "v"}}),
    }
    tc = TaskContext.from_task(task)
    assert tc.recipe_hint == "rh"
    assert tc.payload == {"k": "v"}
    assert tc.title == "T"


def test_from_task_dict_context_still_works():
    task = {"id": 7, "context": {"recipe_hint": "x", "payload": {"a": 1}}}
    tc = TaskContext.from_task(task)
    assert tc.recipe_hint == "x"
    assert tc.payload == {"a": 1}


def test_from_task_handles_none_and_malformed_context():
    assert TaskContext.from_task({"id": 1}).payload == {}
    assert TaskContext.from_task({"id": 1}).recipe_hint is None
    # malformed JSON string → treated as empty, not a crash
    tc = TaskContext.from_task({"id": 1, "context": "{not valid json"})
    assert tc.payload == {}
    assert tc.recipe_hint is None
    # explicit None context
    assert TaskContext.from_task({"id": 1, "context": None}).payload == {}


def test_from_task_non_dict_json_context_is_safe():
    # context that parses to a non-dict (e.g. a JSON list) must not crash
    tc = TaskContext.from_task({"id": 1, "context": json.dumps([1, 2, 3])})
    assert tc.payload == {}
    assert tc.recipe_hint is None
