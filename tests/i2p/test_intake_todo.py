"""Z1 Tier 1 — generate_intake_todo contract tests.

Verifies the deterministic builder path emits the canonical clarify-shape
that general_beckman.result_router treats as needs_clarification + keeps
waiting_human (per its 148-180 branch).
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mr_roboto.generate_intake_todo import generate_intake_todo
from mr_roboto import run as mr_roboto_run


@contextmanager
def _patch_telegram_available(chat_id: int = 12345):
    """Make the inline artifact-confirm keyboard path succeed offline.

    The executor resolves chat_id from the blackboard then sends via
    get_telegram().send_artifact_confirm_keyboard (added in f4013b78). In
    a unit env neither exists, so keyboard_sent would be False. Mock both
    so the real send path runs and keyboard_sent flips True."""
    tg = MagicMock()
    tg.send_artifact_confirm_keyboard = AsyncMock(return_value=None)
    with patch(
        "src.app.telegram_bot.get_telegram", return_value=tg,
    ), patch(
        "src.collaboration.blackboard.read_blackboard",
        new=AsyncMock(return_value={"chat_id": chat_id}),
    ):
        yield tg


def _make_task(tmpdir: str, mission_id: int = 42, **payload_extra) -> dict:
    payload = {
        "action": "generate_intake_todo",
        "workspace_path": tmpdir,
        "inputs": {
            "founder_pitch": "TruthRate — review anything, separate facts from opinions.",
            "reverse_pitch": "# Headline\nTruthRate launches.\n",
            "z0_outputs": "ambition_tier: private_beta",
        },
    }
    payload.update(payload_extra)
    return {"id": 1, "mission_id": mission_id, "payload": payload}


def test_intake_todo_writes_file_and_returns_clarify_shape():
    with tempfile.TemporaryDirectory() as tmp:
        task = _make_task(tmp, mission_id=42)
        with _patch_telegram_available():
            result = asyncio.run(generate_intake_todo(task))

        assert result["status"] == "needs_clarification"
        assert result["keyboard_sent"] is True
        assert result["kind"] == "intake_todo"
        # Path is workspace-relative per v3 N3.
        assert result["todo_path"] == "mission_42/.intake/intake_todo.md"
        assert os.path.exists(result["todo_path_abs"])

        body = Path(result["todo_path_abs"]).read_text(encoding="utf-8")
        assert "# Intake Todo" in body
        assert "## Items" in body
        # Canonical builder emits at least 10 items.
        assert result["item_count"] >= 10
        assert result["item_count"] <= 20


def test_intake_todo_regen_overwrites_file():
    """Founder edits → next run regenerates; the file is rewritten."""
    with tempfile.TemporaryDirectory() as tmp:
        task1 = _make_task(tmp, mission_id=99)
        r1 = asyncio.run(generate_intake_todo(task1))
        # Simulate founder edit by mangling the file in place.
        with open(r1["todo_path_abs"], "w", encoding="utf-8") as fh:
            fh.write("EDITED BY FOUNDER")

        task2 = _make_task(tmp, mission_id=99)
        r2 = asyncio.run(generate_intake_todo(task2))
        body = Path(r2["todo_path_abs"]).read_text(encoding="utf-8")
        assert "EDITED BY FOUNDER" not in body
        assert "# Intake Todo" in body


def test_intake_todo_missing_mission_id_fails():
    task = {"id": 1, "payload": {"workspace_path": "/tmp/x"}}
    result = asyncio.run(generate_intake_todo(task))
    assert result["status"] == "failed"
    assert "mission_id" in result["error"]


def test_dispatch_via_mr_roboto_run_returns_needs_clarification():
    """The wrapper must return Action(status='needs_clarification') so
    general_beckman.result_router keeps the row as waiting_human (the same
    contract that variant_choice clarify uses)."""
    with tempfile.TemporaryDirectory() as tmp:
        task = _make_task(tmp, mission_id=7)
        with _patch_telegram_available():
            result = asyncio.run(mr_roboto_run(task))
        assert result.status == "needs_clarification", result
        assert result.result["keyboard_sent"] is True
        assert result.result["todo_path"] == "mission_7/.intake/intake_todo.md"
