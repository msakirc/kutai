"""resend_clarification — re-send a pending clarification's ORIGINAL
interactive message (question + content + buttons) so escalation reminders
are self-contained.

Sweep escalation reminders (4h/24h/48h) were bare pointers ("Task #N needs
your input") — the founder had to scroll history for the actual question +
options. This executor re-runs the clarify executor against the source task
so every gate kind (artifact_confirm / variant_choice / surface_choice /
plain question) re-sends exactly as first sent.
"""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, patch


def _src_row(**over):
    row = {
        "id": 525000,
        "title": "[0.6a] non_goals_confirm",
        "status": "waiting_human",
        "mission_id": 89,
        "parent_task_id": None,
        "context": json.dumps({
            "escalation_count": 2,
            "executor": "mechanical",
            "payload": {
                "action": "clarify",
                "kind": "non_goals_confirm",
                "question": "Mission-wide non-goals draft below.",
                "attach_file_paths": [
                    "mission_89/.charter/non_goals.md",
                ],
                "regenerate_step_id": "0.6a.draft",
            },
        }),
    }
    row.update(over)
    return row


@pytest.mark.asyncio
async def test_resend_reruns_clarify_for_mechanical_gate():
    """A waiting_human mechanical clarify gate re-sends its ORIGINAL
    interactive message by re-running the clarify executor against the
    source task — same payload, source id, with the attention budget gate
    bypassed (we are escalating an already-asked question)."""
    fake_clarify = AsyncMock(return_value={
        "status": "needs_clarification", "keyboard_sent": True,
    })
    with patch("mr_roboto.resend_clarification.get_task",
               AsyncMock(return_value=_src_row())), \
         patch("mr_roboto.resend_clarification.clarify", fake_clarify):
        from mr_roboto import run
        action = await run({
            "id": 525050,
            "payload": {"action": "resend_clarification",
                        "source_task_id": 525000},
        })

    assert action.status == "completed"
    fake_clarify.assert_awaited_once()
    passed = fake_clarify.await_args.args[0]
    # Re-sent against the SOURCE task (so buttons carry the right task_id).
    assert passed["id"] == 525000
    assert passed["mission_id"] == 89
    # Original clarify payload preserved (kind/question/attach paths).
    assert passed["payload"]["kind"] == "non_goals_confirm"
    assert passed["payload"]["attach_file_paths"] == [
        "mission_89/.charter/non_goals.md",
    ]
    # Escalation re-send must always reach Telegram — skip the attention gate.
    assert passed["payload"]["attention_skip"] is True


@pytest.mark.asyncio
async def test_resend_skips_when_source_no_longer_waiting():
    """If the founder already answered (status moved off waiting_human), a
    re-send would be a spurious duplicate — skip it."""
    fake_clarify = AsyncMock()
    with patch("mr_roboto.resend_clarification.get_task",
               AsyncMock(return_value=_src_row(status="pending"))), \
         patch("mr_roboto.resend_clarification.clarify", fake_clarify):
        from mr_roboto import run
        action = await run({
            "id": 525050,
            "payload": {"action": "resend_clarification",
                        "source_task_id": 525000},
        })

    assert action.status == "completed"
    fake_clarify.assert_not_awaited()
    assert action.result.get("resent") is False


@pytest.mark.asyncio
async def test_resend_plain_question_path():
    """LLM-agent clarifications carry no clarify payload — just a stored
    ``_clarification_question``. Re-send via the same request_clarification
    path used on the original send + restart-restore."""
    row = _src_row(context=json.dumps({
        "escalation_count": 1,
        "_clarification_question": "Which currency should prices use?",
    }))
    fake_tg = AsyncMock()
    fake_clarify = AsyncMock()
    with patch("mr_roboto.resend_clarification.get_task",
               AsyncMock(return_value=row)), \
         patch("mr_roboto.resend_clarification.clarify", fake_clarify), \
         patch("mr_roboto.resend_clarification.get_telegram",
               return_value=fake_tg):
        from mr_roboto import run
        action = await run({
            "id": 525050,
            "payload": {"action": "resend_clarification",
                        "source_task_id": 525000},
        })

    assert action.status == "completed"
    fake_clarify.assert_not_awaited()
    fake_tg.request_clarification.assert_awaited_once_with(
        525000, "[0.6a] non_goals_confirm",
        "Which currency should prices use?",
    )


@pytest.mark.asyncio
async def test_resend_missing_source_task_id_fails():
    from mr_roboto import run
    action = await run({
        "id": 525050,
        "payload": {"action": "resend_clarification"},
    })
    assert action.status == "failed"
    assert "source_task_id" in (action.error or "")
