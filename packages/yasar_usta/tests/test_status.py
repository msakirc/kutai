"""Tests for yasar_usta.status."""

import sys
import time
from unittest.mock import patch, MagicMock

from yasar_usta.status import build_status_text, _check_process_running


class TestBuildStatusText:
    def test_basic_healthy(self):
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time() - 3600,
            app_running=True,
            heartbeat_age=10.0,
            heartbeat_healthy_seconds=90,
            total_crashes=0,
        )
        assert "Guard" in text
        assert "MyApp" in text
        assert "healthy" in text
        assert "Crashes: 0" in text

    def test_app_down(self):
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time(),
            app_running=False,
            heartbeat_age=None,
            heartbeat_healthy_seconds=90,
            total_crashes=3,
        )
        assert "not running" in text
        assert "Crashes: 3" in text

    def test_unresponsive(self):
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time() - 100,
            app_running=True,
            heartbeat_age=200.0,
            heartbeat_healthy_seconds=90,
            total_crashes=1,
        )
        assert "UNRESPONSIVE" in text

    def test_no_heartbeat_file(self):
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time(),
            app_running=True,
            heartbeat_age=None,
            heartbeat_healthy_seconds=90,
            total_crashes=0,
        )
        assert "no heartbeat file" in text

    def test_uptime_displayed(self):
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time() - 3661,  # 1h 1m
            app_running=False,
            heartbeat_age=None,
            heartbeat_healthy_seconds=90,
            total_crashes=0,
        )
        assert "1h 1m" in text

    def test_crash_count(self):
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time(),
            app_running=False,
            heartbeat_age=None,
            heartbeat_healthy_seconds=90,
            total_crashes=5,
        )
        assert "Crashes: 5" in text

    def test_last_update_timestamp(self):
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time(),
            app_running=False,
            heartbeat_age=None,
            heartbeat_healthy_seconds=90,
            total_crashes=0,
        )
        assert "Last update" in text

    def test_sidecar_http_alive(self):
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time(),
            app_running=True,
            heartbeat_age=5.0,
            heartbeat_healthy_seconds=90,
            total_crashes=0,
            sidecar_name="LogViewer",
            sidecar_http_alive=True,
            sidecar_pid=12345,
            sidecar_health_url="http://localhost:9880",
        )
        assert "LogViewer" in text
        assert "running" in text

    def test_sidecar_not_running(self):
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time(),
            app_running=True,
            heartbeat_age=5.0,
            heartbeat_healthy_seconds=90,
            total_crashes=0,
            sidecar_name="LogViewer",
            sidecar_alive=False,
        )
        assert "LogViewer" in text
        assert "not running" in text

    def test_extra_processes_running(self):
        with patch("yasar_usta.status._check_process_running", return_value=True):
            text = build_status_text(
                name="Guard",
                app_name="MyApp",
                guard_start_time=time.time(),
                app_running=True,
                heartbeat_age=5.0,
                heartbeat_healthy_seconds=90,
                total_crashes=0,
                extra_processes=[{"exe": "llama-server.exe", "label": "LLM Server"}],
            )
        assert "LLM Server" in text
        assert "running" in text

    def test_extra_processes_not_running(self):
        with patch("yasar_usta.status._check_process_running", return_value=False):
            text = build_status_text(
                name="Guard",
                app_name="MyApp",
                guard_start_time=time.time(),
                app_running=True,
                heartbeat_age=5.0,
                heartbeat_healthy_seconds=90,
                total_crashes=0,
                extra_processes=[{"exe": "llama-server.exe", "label": "LLM Server"}],
            )
        assert "LLM Server" in text
        assert "not running" in text


class TestCheckProcessRunning:
    def test_empty_exe_name(self):
        assert _check_process_running("") is False

    def test_windows_process_found(self):
        if sys.platform != "win32":
            return
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="python.exe", returncode=0)
            result = _check_process_running("python.exe")
            assert result is True

    def test_windows_process_not_found(self):
        if sys.platform != "win32":
            return
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="No tasks running", returncode=0)
            result = _check_process_running("nonexistent.exe")
            assert result is False

    def test_unix_process_found(self):
        if sys.platform == "win32":
            return
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = _check_process_running("python")
            assert result is True

    def test_unix_process_not_found(self):
        if sys.platform == "win32":
            return
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = _check_process_running("nonexistent_process")
            assert result is False
