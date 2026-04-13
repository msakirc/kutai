"""Tests for yasar_usta.guard — ProcessGuard integration."""

import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from yasar_usta import ProcessGuard, GuardConfig, Messages
from yasar_usta.config import SidecarConfig


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestProcessGuardConstruction:
    def test_creates_with_minimal_config(self):
        cfg = GuardConfig(
            command=[sys.executable, "-c", "pass"],
            log_dir=tempfile.mkdtemp(),
        )
        guard = ProcessGuard(cfg)
        assert guard.cfg.app_name == "App"
        assert guard.telegram.enabled is False
        assert guard.sidecars == {}

    def test_creates_with_full_config(self):
        cfg = GuardConfig(
            name="Yaşar Usta",
            app_name="Kutay",
            command=["python", "run.py"],
            telegram_token="tok",
            telegram_chat_id="123",
            sidecars=[
                SidecarConfig(
                    name="yazbunu",
                    command=["python", "-m", "yazbunu.server"],
                    health_url="http://127.0.0.1:9880/",
                    pid_file="/tmp/yazbunu.pid",
                ),
                SidecarConfig(
                    name="nerd_herd",
                    command=["python", "-m", "nerd_herd"],
                    health_url="http://127.0.0.1:9881/health",
                    pid_file="/tmp/nerd_herd.pid",
                ),
            ],
        )
        guard = ProcessGuard(cfg)
        assert guard.telegram.enabled is True
        assert len(guard.sidecars) == 2
        assert guard.sidecars["yazbunu"].name == "yazbunu"
        assert guard.sidecars["nerd_herd"].name == "nerd_herd"

    def test_custom_messages(self):
        cfg = GuardConfig(
            command=["echo"],
            log_dir=tempfile.mkdtemp(),
            messages=Messages(
                started="✅ *{app_name} Başladı*",
                btn_start="▶️ Başlat",
            ),
        )
        guard = ProcessGuard(cfg)
        assert "Başladı" in guard.msgs.started
        assert guard.msgs.btn_start == "▶️ Başlat"


class TestProcessGuardNotifications:
    def test_notify_started(self):
        cfg = GuardConfig(
            command=["echo"],
            log_dir=tempfile.mkdtemp(),
            app_name="TestApp",
        )
        guard = ProcessGuard(cfg)
        guard.telegram.send = AsyncMock()
        run_async(guard._notify_started())
        guard.telegram.send.assert_called_once()
        call_text = guard.telegram.send.call_args[0][0]
        assert "TestApp" in call_text
        assert "Started" in call_text

    def test_notify_crash(self):
        cfg = GuardConfig(
            command=["echo"],
            log_dir=tempfile.mkdtemp(),
            app_name="TestApp",
        )
        guard = ProcessGuard(cfg)
        guard.telegram.send = AsyncMock()
        guard.subprocess.stderr_tail.append("some error")
        guard.backoff.record_crash()
        run_async(guard._notify_crash(1))
        call_text = guard.telegram.send.call_args[0][0]
        assert "Crashed" in call_text
        assert "some error" in call_text
