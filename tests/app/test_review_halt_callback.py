import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_update(data: str):
    update = MagicMock()
    update.effective_chat.id = 42
    update.callback_query.data = data
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    return update


@pytest.mark.asyncio
async def test_review_halt_regen_repends_producer_and_reviewer():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    update = _make_update("rr:regen:7:55:3.4")

    with patch("general_beckman.review_routing._repend_producer",
               new=AsyncMock(return_value=True)) as rp, \
         patch("src.infra.db.update_task", new=AsyncMock()) as upd:
        await iface._handle_review_halt(update, MagicMock())

    update.callback_query.answer.assert_awaited_once()
    # Producer re-pended with the founder feedback + right step_id.
    rp.assert_awaited_once()
    assert rp.call_args.kwargs["mission_id"] == 7
    assert rp.call_args.kwargs["step_id"] == "3.4"
    assert "regenerate" in rp.call_args.kwargs["feedback"].lower()
    # Reviewer re-pended back to pending so it re-reviews the fix.
    upd.assert_awaited_once_with(55, status="pending")


@pytest.mark.asyncio
async def test_review_halt_accept_completes_reviewer_and_audits():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    update = _make_update("rr:accept:7:55")

    with patch("general_beckman.review_routing._repend_producer",
               new=AsyncMock()) as rp, \
         patch("src.infra.db.update_task", new=AsyncMock()) as upd, \
         patch("src.infra.db.record_action_event",
               new=AsyncMock(return_value=1)) as rec, \
         patch("general_beckman.enqueue",
               new=AsyncMock(return_value=99)) as enq:
        await iface._handle_review_halt(update, MagicMock())

    update.callback_query.answer.assert_awaited_once()
    rp.assert_not_awaited()
    # Override: reviewer marked completed so the mission proceeds.
    upd.assert_awaited_once_with(55, status="completed")
    # Audit event recorded.
    rec.assert_awaited_once()
    assert rec.call_args.kwargs["verb"] == "review_override"
    assert rec.call_args.kwargs["mission_id"] == 7
    assert rec.call_args.kwargs["task_id"] == 55
    # L3: a workflow_advance must be spawned so a reviewer with no downstream
    # dependents still advances the mission (otherwise it stalls).
    enq.assert_awaited_once()
    spawned = enq.call_args.args[0]
    assert spawned["agent_type"] == "mechanical"
    assert spawned["mission_id"] == 7
    payload = spawned["context"]["payload"]
    assert payload["action"] == "workflow_advance"
    assert payload["mission_id"] == 7


@pytest.mark.asyncio
async def test_review_halt_malformed_callback_noop():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    update = _make_update("rr:nope")

    with patch("general_beckman.review_routing._repend_producer",
               new=AsyncMock()) as rp, \
         patch("src.infra.db.update_task", new=AsyncMock()) as upd:
        await iface._handle_review_halt(update, MagicMock())

    # Answered, but no routing / db writes for malformed arity.
    update.callback_query.answer.assert_awaited_once()
    rp.assert_not_awaited()
    upd.assert_not_awaited()


@pytest.mark.asyncio
async def test_review_halt_regen_missing_step_noop():
    """rr:regen with no producer step (4 parts) must not re-pend a producer."""
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    update = _make_update("rr:regen:7:55")

    with patch("general_beckman.review_routing._repend_producer",
               new=AsyncMock()) as rp, \
         patch("src.infra.db.update_task", new=AsyncMock()) as upd:
        await iface._handle_review_halt(update, MagicMock())

    update.callback_query.answer.assert_awaited_once()
    rp.assert_not_awaited()
    upd.assert_not_awaited()


# ---------------------------------------------------------------------------
# Re-surfacing a parked review halt (restart-restore + sweep-nudge use this).
# ---------------------------------------------------------------------------


def test_build_review_halt_args_prefers_persisted_payload():
    from src.app.telegram_bot import TelegramInterface
    task = {
        "id": 525019, "mission_id": 89,
        "context": json.dumps({
            "step_name": "research_quality_review",
            "_review_halt": {
                "reviewer_name": "1.13",
                "issues": [{"severity": "blocker", "problem": "naming drift"}],
                "producers": ["0.0z", "1.0a"],
            },
        }),
    }
    name, issues, producers = TelegramInterface._build_review_halt_args(task)
    assert name == "1.13"
    assert issues[0]["problem"] == "naming drift"
    assert producers == ["0.0z", "1.0a"]


def test_build_review_halt_args_falls_back_to_result_json():
    """Legacy reviewer parked BEFORE the persist fix has no _review_halt —
    reconstruct issues from the reviewer's fenced-JSON result; producers
    unknown (empty) but the Accept-anyway card still renders."""
    from src.app.telegram_bot import TelegramInterface
    result = (
        "Here is my verdict:\n```json\n"
        + json.dumps({"verdict": "fail",
                      "issues": [{"severity": "blocker", "problem": "no prior art"}]})
        + "\n```\n"
    )
    task = {
        "id": 525019, "mission_id": 89,
        "context": json.dumps({"workflow_step_id": "1.13"}),
        "result": result,
    }
    name, issues, producers = TelegramInterface._build_review_halt_args(task)
    assert name == "1.13"
    assert issues[0]["problem"] == "no prior art"
    assert producers == []


@pytest.mark.asyncio
async def test_resurface_review_halt_renders_card():
    import src.app.telegram_bot as tb
    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    iface.send_review_halt_keyboard = AsyncMock()
    task = {
        "id": 525019, "mission_id": 89,
        "context": json.dumps({
            "_review_halt": {
                "reviewer_name": "1.13",
                "issues": [{"severity": "blocker", "problem": "p"}],
                "producers": ["0.0z"],
            },
        }),
    }
    with patch.object(tb, "TELEGRAM_ADMIN_CHAT_ID", "42"):
        ok = await iface.resurface_review_halt(task)

    assert ok is True
    iface.send_review_halt_keyboard.assert_awaited_once()
    kw = iface.send_review_halt_keyboard.await_args.kwargs
    assert kw["chat_id"] == 42
    assert kw["mission_id"] == 89
    assert kw["reviewer_task_id"] == 525019
    assert kw["reviewer_name"] == "1.13"
    assert kw["producers"] == ["0.0z"]
