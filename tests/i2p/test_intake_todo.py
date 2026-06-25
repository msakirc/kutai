"""Z1 Tier 1 — generate_intake_todo contract tests.

Verifies the deterministic builder path emits the canonical clarify-shape
(needs_clarification + keyboard_sent) AND self-parks its row as
waiting_human — the orchestrator skips result_router for mechanical
needs_clarification, so the executor must park its own row.
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


def _seed_draft(tmpdir: str, mission_id: int, items: list[dict]) -> str:
    """Write an analyst draft (0.0a.draft output) to disk and return its path."""
    import json
    d = os.path.join(tmpdir, f"mission_{mission_id}", ".intake")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "intake_todo_draft.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"_schema_version": "1", "items": items}, fh)
    return path


def test_analyst_draft_specialises_matching_slots():
    """A valid draft slot whose category matches is used verbatim; the
    canonical wording is replaced."""
    with tempfile.TemporaryDirectory() as tmp:
        _seed_draft(tmp, 42, [
            {"n": 1, "category": "Audience", "question": "Who is the core HabitLoop user?"},
            {"n": 3, "category": "Problem", "question": "What HabitLoop pain bites daily?"},
        ])
        task = _make_task(tmp, mission_id=42)
        with _patch_telegram_available():
            r = asyncio.run(generate_intake_todo(task))
        body = Path(r["todo_path_abs"]).read_text(encoding="utf-8")
        assert "Who is the core HabitLoop user?" in body
        assert "What HabitLoop pain bites daily?" in body
        assert "Specialised 2/15" in body
        # Always the full canonical coverage — never fewer than 14 question slots.
        assert r["item_count"] >= 14


def test_draft_category_mismatch_rejected_falls_back():
    """Coverage contract: a slot whose declared category != canonical category
    is dropped, so that slot keeps the generic canonical question."""
    with tempfile.TemporaryDirectory() as tmp:
        _seed_draft(tmp, 42, [
            {"n": 1, "category": "Problem", "question": "WRONG CATEGORY SHOULD BE DROPPED"},
        ])
        task = _make_task(tmp, mission_id=42)
        with _patch_telegram_available():
            r = asyncio.run(generate_intake_todo(task))
        body = Path(r["todo_path_abs"]).read_text(encoding="utf-8")
        assert "WRONG CATEGORY SHOULD BE DROPPED" not in body
        # Canonical slot-1 wording survives.
        assert "Who is the primary user" in body


def test_draft_invalid_json_falls_back_to_canonical():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "mission_42", ".intake")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "intake_todo_draft.json"), "w", encoding="utf-8") as fh:
            fh.write("{not valid json")
        task = _make_task(tmp, mission_id=42)
        with _patch_telegram_available():
            r = asyncio.run(generate_intake_todo(task))
        body = Path(r["todo_path_abs"]).read_text(encoding="utf-8")
        assert "# Intake Todo" in body
        assert "Specialised" not in body  # no draft applied
        assert r["item_count"] >= 14


def test_no_draft_still_full_canonical():
    """No analyst draft on disk → pure canonical, no LLM, full coverage."""
    with tempfile.TemporaryDirectory() as tmp:
        task = _make_task(tmp, mission_id=77)
        with _patch_telegram_available():
            r = asyncio.run(generate_intake_todo(task))
        body = Path(r["todo_path_abs"]).read_text(encoding="utf-8")
        assert "Specialised" not in body
        assert r["item_count"] >= 14


def test_intake_todo_self_parks_row_waiting_human_when_keyboard_sent():
    """Regression (task 567372 / mission 90): a successfully-sent intake-todo
    keyboard MUST flip its own task row to ``waiting_human``.

    The orchestrator special-cases mechanical ``needs_clarification`` and
    skips ``on_task_finished`` (so ``result_router`` never runs — the
    docstring's "result_router keeps it waiting_human" contract is stale).
    If the executor doesn't park its own row, it stays ``processing`` and the
    queue sweep (Section 1, status='processing' >5min) flips it back to
    pending → re-admit → the keyboard is re-sent every backoff interval,
    spamming the founder with 6-7 full-text reminders/hour. Mirror
    request_interview_data, which self-parks."""
    with tempfile.TemporaryDirectory() as tmp:
        task = _make_task(tmp, mission_id=90)
        task["id"] = 567372
        upd = AsyncMock()
        with _patch_telegram_available(), patch(
            "general_beckman.update_task", new=upd
        ):
            result = asyncio.run(generate_intake_todo(task))
        assert result["keyboard_sent"] is True
        upd.assert_awaited_once_with(567372, status="waiting_human")


def test_intake_todo_fails_closed_when_keyboard_unsent():
    """No keyboard sent (no chat_id / Telegram down) → fail-closed, NOT park.

    This is a founder-confirmation gate with no automated fallback. Returning
    needs_clarification with keyboard_sent=False makes the dispatch wrapper
    COMPLETE the row (advancing the mission with no founder approval of the
    intake scope). Mirror clarify.py artifact_confirm / surface_choice: return
    status='failed' so the mission halts at the gate (DLQ → founder fixes the
    chat wiring and retries), and never self-park (no keyboard to act on)."""
    with tempfile.TemporaryDirectory() as tmp:
        task = _make_task(tmp, mission_id=91)
        task["id"] = 999
        upd = AsyncMock()
        # chat_id resolves, but Telegram is down → keyboard send fails.
        with patch(
            "src.collaboration.blackboard.read_blackboard",
            new=AsyncMock(return_value={"chat_id": 555}),
        ), patch(
            "src.app.telegram_bot.get_telegram", return_value=None,
        ), patch(
            "general_beckman.update_task", new=upd
        ):
            result = asyncio.run(generate_intake_todo(task))
        assert result["status"] == "failed"
        assert result["keyboard_sent"] is False
        upd.assert_not_awaited()
        # File was still written before the (failed) send — keep the path so a
        # post-fix retry/inspection can find the draft.
        assert os.path.exists(result["todo_path_abs"])


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
