import pytest
from unittest.mock import AsyncMock, MagicMock
import src.app.telegram_bot as tb


def test_button_actions_have_new_load_labels():
    a = tb._BUTTON_ACTIONS
    assert a["🖥 Yerel Serbest"] == ("special", "load_full")
    assert a["⚖️ Dengeli"] == ("special", "load_balanced")
    assert a["☁️ Sadece Bulut"] == ("special", "load_minimal")
    assert "🔋 Heavy" not in a
    assert "⚖️ Shared" not in a


def test_otomatik_still_maps_to_workflow_auto():
    assert tb._BUTTON_ACTIONS["🤖 Otomatik"] == ("special", "wf_auto")


@pytest.mark.asyncio
async def test_load_balanced_button_sets_balanced_mode(monkeypatch):
    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    iface._reply = AsyncMock()
    called = {}

    async def _fake_set(mode, source="user"):
        called["mode"] = mode
        return f"set {mode}"
    monkeypatch.setattr("src.infra.load_manager.set_load_mode", _fake_set)

    update = MagicMock()
    update.effective_chat.id = 1
    context = MagicMock()
    context.args = None  # cmd_load reads context.args; the load_ branch sets it
    await iface._handle_special_button(update, context, "load_balanced")
    assert called["mode"] == "balanced"
