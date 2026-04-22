import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_variant_choice_variant_resumes_mission():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    iface._pending_action = {}
    iface._resume_mission_at_step = AsyncMock()
    iface._reply_text = AsyncMock()
    chat_id = 42
    iface._pending_action[chat_id] = {
        "kind": "variant_choice",
        "mission_id": 7,
        "task_id": 2,
        "options": [{"label": "Galaxy S25", "group_id": 0}],
        "base_label": "Samsung Galaxy S25",
    }
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.callback_query.data = "variant_choice:0"
    update.callback_query.answer = AsyncMock()
    context = MagicMock()
    await iface._handle_variant_choice(update, context)
    iface._resume_mission_at_step.assert_awaited_once()
    assert chat_id not in iface._pending_action


@pytest.mark.asyncio
async def test_variant_choice_compare_all_runs_format_compare():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    iface._pending_action = {}
    iface._run_compare_all_and_reply = AsyncMock()
    chat_id = 42
    iface._pending_action[chat_id] = {
        "kind": "variant_choice", "mission_id": 7, "task_id": 2,
        "options": [{"label": "A", "group_id": 0}], "base_label": "A",
    }
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.callback_query.data = "variant_choice:compare_all"
    update.callback_query.answer = AsyncMock()
    context = MagicMock()
    await iface._handle_variant_choice(update, context)
    iface._run_compare_all_and_reply.assert_awaited_once()
    assert chat_id not in iface._pending_action


@pytest.mark.asyncio
async def test_variant_choice_stale_callback_noop():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    iface._pending_action = {}   # no pending entry
    iface._resume_mission_at_step = AsyncMock()
    iface._run_compare_all_and_reply = AsyncMock()
    chat_id = 42
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.callback_query.data = "variant_choice:0"
    update.callback_query.answer = AsyncMock()
    await iface._handle_variant_choice(update, context=MagicMock())
    iface._resume_mission_at_step.assert_not_awaited()
    iface._run_compare_all_and_reply.assert_not_awaited()
    update.callback_query.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_mission_persists_clarify_choice(monkeypatch):
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    stores: list[dict] = []

    async def fake_store(_self, mission_id, name, value):
        stores.append({
            "mission_id": mission_id,
            "name": name,
            "value": value,
        })

    async def fake_update_task(*args, **kwargs):
        pass

    monkeypatch.setattr(
        "src.workflows.engine.artifacts.ArtifactStore.store",
        fake_store,
        raising=True,
    )
    monkeypatch.setattr(
        "src.infra.db.update_task",
        fake_update_task,
        raising=True,
    )

    await iface._resume_mission_at_step(
        mission_id=7,
        after_task_id=2,
        clarify_choice={"kind": "variant", "group_id": 3},
    )

    assert len(stores) == 1, f"expected 1 store call, got {len(stores)}"
    call = stores[0]
    assert call["mission_id"] == 7
    assert call["name"] == "clarify_choice"
    import json as _json
    stored = _json.loads(call["value"])
    assert stored == {"kind": "variant", "group_id": 3}
