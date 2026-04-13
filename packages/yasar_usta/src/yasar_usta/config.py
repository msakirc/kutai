"""Configuration and i18n for yasar-usta."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Messages:
    """All user-facing strings. Override for i18n."""

    # Notifications
    announce: str = "🔧 *{name} Process Manager*\n\nStarting {app_name}..."
    started: str = "✅ *{app_name} Started*"
    stopped: str = "⏹ *{app_name} Stopped*\nSend /start to restart."
    crash: str = (
        "🔴 *{app_name} Crashed*\n"
        "Exit code: `{exit_code}`\n"
        "Crash #{crash_count}\n"
        "Restarting in {backoff}s\n\n"
        "```\n{stderr}\n```"
    )
    hung: str = "🔴 {app_name} not responding — restarting in {delay}s"
    restarting: str = "♻️ *{app_name} restarting...*"
    self_restarting: str = "🔄 *{name} restarting...*"

    # Down-state prompt
    down_prompt: str = "⚠️ {app_name} is down. Press the button to start."
    down_with_reason: str = "{reason}\n⚠️ {app_name} is down. Press the button to start."
    down_reply: str = "⏸ {app_name} is currently stopped."
    starting: str = "🚀 Starting {app_name}..."

    # Keyboard
    btn_start: str = "▶️ Start {app_name}"
    btn_status: str = "🔧 Status"
    btn_system: str = "⚙️ System"
    btn_restart: str = "🔄 Restart {app_name}"
    btn_stop: str = "⏹ Stop {app_name}"
    btn_logs: str = "📋 Logs"
    btn_remote: str = "🖥️ Claude Code"
    btn_refresh: str = "🔄 Refresh"
    btn_restart_guard: str = "♻️ Restart {name}"
    btn_restart_sidecar: str = "📊 Restart {sidecar_name}"

    # Status panel
    status_title: str = "🔧 *{name}*\n"
    status_guard: str = "🔵 {name}: running ({uptime})"
    status_app_healthy: str = "💚 {app_name}: healthy (heartbeat {age}s ago)"
    status_app_unresponsive: str = "🔴 {app_name}: UNRESPONSIVE ({age}s silent)"
    status_app_down: str = "💀 {app_name}: not running"
    status_app_no_heartbeat: str = "⚪ {app_name}: no heartbeat file"
    status_crashes: str = "\nCrashes: {count}"
    status_updated: str = "\n\n_Last update: {time}_"

    # Logs
    no_log_file: str = "📋 No log file found."
    no_log_entries: str = "📋 No log entries found."
    log_error: str = "❌ Error reading logs: {error}"

    # Remote
    remote_starting: str = "🖥️ Starting Claude Code session..."
    remote_started: str = "🖥️ *Claude Code Remote Control*\n\n🔗 [Connect]({url})\n\nPID: `{pid}`"
    remote_started_no_url: str = "🖥️ *Claude Code Remote Control started*\nPID: `{pid}`"
    remote_not_found: str = "❌ `claude` command not found. Is Claude Code installed?"
    remote_failed: str = "❌ Failed to start Claude Code: `{error}`"

    # Errors
    process_list_error: str = "⚠️ Could not get process list: {error}"
    wrapper_error: str = "⚠️ *Guard Error*\n`{error}`\n\nGuard is still alive. Send /start to retry."


@dataclass
class SidecarConfig:
    """Configuration for a sidecar subprocess (e.g., log viewer)."""

    name: str = "sidecar"
    command: list[str] = field(default_factory=list)
    health_url: str | None = None
    health_timeout: float = 3.0
    pid_file: str | None = None
    detached: bool = True
    auto_start: bool = True
    auto_restart: bool = True


@dataclass
class GuardConfig:
    """All configuration for ProcessGuard."""

    # What to manage
    name: str = "Yaşar Usta"
    app_name: str = "App"
    command: list[str] = field(default_factory=list)
    cwd: str | None = None

    # Telegram
    telegram_token: str = ""
    telegram_chat_id: str = ""

    # Backoff
    backoff_steps: list[int] = field(default_factory=lambda: [5, 15, 60, 300])
    backoff_reset_after: int = 600

    # Heartbeat
    heartbeat_file: str | None = None
    heartbeat_stale_seconds: int = 120
    heartbeat_healthy_seconds: int = 90

    # Exit codes
    restart_exit_code: int = 42

    # Directories
    log_dir: str = "logs"
    log_file: str | None = None

    # Process management
    auto_restart: bool = True
    stop_timeout: int = 30

    # Claude Code remote
    claude_enabled: bool = True
    claude_cmd: str | None = None
    claude_name: str | None = None
    claude_signal_file: str | None = None

    # Sidecars (log viewer, observability, etc.)
    sidecars: list[SidecarConfig] = field(default_factory=list)

    # Hooks
    on_exit: None = None  # callable(exit_code: int) -> None, called after process exits

    # i18n
    messages: Messages = field(default_factory=Messages)

    # Extra commands
    extra_commands: dict = field(default_factory=dict)

    # Extra process names to check in status panel
    extra_processes: list[dict] = field(default_factory=list)
