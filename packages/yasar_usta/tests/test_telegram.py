"""Tests for yasar_usta.telegram."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from yasar_usta.telegram import TelegramAPI


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestTelegramAPI:
    def test_disabled_when_no_token(self):
        api = TelegramAPI(token="", chat_id="123")
        assert api.enabled is False

    def test_disabled_when_no_chat_id(self):
        api = TelegramAPI(token="tok", chat_id="")
        assert api.enabled is False

    def test_enabled_with_both(self):
        api = TelegramAPI(token="tok", chat_id="123")
        assert api.enabled is True

    def test_send_returns_none_when_disabled(self):
        api = TelegramAPI(token="", chat_id="")
        result = run_async(api.send("test"))
        assert result is None

    def test_get_updates_returns_empty_when_disabled(self):
        api = TelegramAPI(token="", chat_id="")
        result = run_async(api.get_updates())
        assert result == []

    def test_edit_returns_none_when_disabled(self):
        api = TelegramAPI(token="", chat_id="")
        result = run_async(api.edit(message_id=1, text="test"))
        assert result is None

    def test_answer_callback_noop_when_disabled(self):
        api = TelegramAPI(token="", chat_id="")
        # Should not raise
        run_async(api.answer_callback("cb123"))

    def test_flush_updates_noop_when_disabled(self):
        api = TelegramAPI(token="", chat_id="")
        # Should not raise
        run_async(api.flush_updates())

    def test_base_url_construction(self):
        api = TelegramAPI(token="mytoken", chat_id="42")
        assert api._base_url == "https://api.telegram.org/botmytoken"


class TestSidecar:
    """Basic import test for sidecar module."""
    def test_import(self):
        from yasar_usta.sidecar import SidecarManager
        mgr = SidecarManager(name="test", command=["echo", "hi"])
        assert mgr.name == "test"


class TestStatus:
    def test_build_status_text(self):
        from yasar_usta.status import build_status_text
        import time
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time() - 3600,
            app_running=True,
            heartbeat_age=5.0,
            heartbeat_healthy_seconds=90,
            total_crashes=2,
        )
        assert "Guard" in text
        assert "MyApp" in text
        assert "healthy" in text
        assert "Crashes: 2" in text

    def test_build_status_app_down(self):
        from yasar_usta.status import build_status_text
        import time
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time(),
            app_running=False,
            heartbeat_age=None,
            heartbeat_healthy_seconds=90,
            total_crashes=0,
        )
        assert "not running" in text
