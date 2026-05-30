"""Regenerate button must resolve its step id durably from the task context.

mission_79 (2026-05-30): the ♻️ Regenerate button on an artifact-confirm card
fired "Regenerating step `?`" and re-emitted the SAME artifact instantly — no
real regeneration. Root cause: the step id was read only from the ephemeral
``_pending_action[chat_id]`` slot, which any other Telegram action between
sending the card and the button press clobbers. When empty, the writer-step
reset matched no rows (only the confirm task reset) and the unchanged on-disk
artifact re-fired immediately.

Fix: ``_regen_step_from_task_context`` reads the durable
``context.payload.regenerate_step_id`` persisted on the confirm task. These
tests pin that resolver against the real confirm-task context shape
(task #225582: payload.regenerate_step_id == "0.6a.draft").
"""
from __future__ import annotations

import json

from src.app.telegram_bot import TelegramInterface as TI


_f = TI._regen_step_from_task_context


def test_reads_payload_regenerate_step_id_from_dict():
    ctx = {"payload": {"action": "clarify", "regenerate_step_id": "0.6a.draft"}}
    assert _f(ctx) == "0.6a.draft"


def test_reads_from_json_string_context():
    ctx = json.dumps({"payload": {"regenerate_step_id": "0.6a.draft"}})
    assert _f(ctx) == "0.6a.draft"


def test_top_level_fallback():
    ctx = {"regenerate_step_id": "1.10"}
    assert _f(ctx) == "1.10"


def test_payload_wins_over_top_level():
    ctx = {"regenerate_step_id": "wrong", "payload": {"regenerate_step_id": "right"}}
    assert _f(ctx) == "right"


def test_missing_returns_empty():
    assert _f({"payload": {"action": "clarify"}}) == ""
    assert _f({}) == ""
    assert _f(None) == ""
    assert _f("not json") == ""
    assert _f("") == ""


def test_real_mission79_confirm_context_shape():
    """The exact persisted shape of confirm task #225582."""
    ctx = {
        "executor": "mechanical",
        "workflow_step_id": "0.6a",
        "step_name": "non_goals_confirm",
        "payload": {
            "action": "clarify",
            "kind": "non_goals_confirm",
            "question": "Mission-wide non-goals draft below...",
            "attach_file_paths": ["mission_79/.charter/non_goals.md"],
            "regenerate_step_id": "0.6a.draft",
        },
    }
    assert _f(ctx) == "0.6a.draft"
    assert _f(json.dumps(ctx)) == "0.6a.draft"
