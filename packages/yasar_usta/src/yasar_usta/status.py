"""Status panel builder."""

from __future__ import annotations

import logging
import subprocess as _sp
import sys
import time

logger = logging.getLogger("yasar_usta.status")


def build_status_text(
    *,
    name: str,
    app_name: str,
    guard_start_time: float,
    app_running: bool,
    heartbeat_age: float | None,
    heartbeat_healthy_seconds: int,
    total_crashes: int,
    sidecar_name: str | None = None,
    sidecar_alive: bool = False,
    sidecar_pid: int | None = None,
    sidecar_health_url: str | None = None,
    sidecar_http_alive: bool = False,
    extra_processes: list[dict] | None = None,
    messages=None,
) -> str:
    """Build the status panel text.

    Args:
        extra_processes: List of dicts with keys: name, exe (process name to check),
            label (display label).
    """
    uptime_w = int(time.time() - guard_start_time)
    uptime_str = f"{uptime_w // 3600}h {(uptime_w % 3600) // 60}m"

    lines = [f"🔧 *{name}*\n"]
    lines.append(f"🔵 {name}: running ({uptime_str})")

    # App health
    if not app_running:
        lines.append(f"💀 {app_name}: not running")
    elif heartbeat_age is not None and heartbeat_age < heartbeat_healthy_seconds:
        lines.append(f"💚 {app_name}: healthy (heartbeat {int(heartbeat_age)}s ago)")
    elif heartbeat_age is not None:
        lines.append(f"🔴 {app_name}: UNRESPONSIVE ({int(heartbeat_age)}s silent)")
    else:
        lines.append(f"⚪ {app_name}: no heartbeat file")

    # Extra processes (e.g., llama-server)
    for proc_info in (extra_processes or []):
        exe = proc_info.get("exe", "")
        label = proc_info.get("label", exe)
        found = _check_process_running(exe)
        if found:
            lines.append(f"🟡 {label}: running")
        else:
            lines.append(f"⚫ {label}: not running")

    # Sidecar
    if sidecar_name:
        if sidecar_http_alive:
            pid_str = f", PID {sidecar_pid}" if sidecar_pid else ""
            lines.append(f"📊 {sidecar_name}: running ({sidecar_health_url}{pid_str})")
        elif sidecar_pid:
            lines.append(f"🟠 {sidecar_name}: process alive but not responding (PID {sidecar_pid})")
        elif sidecar_alive:
            lines.append(f"🟢 {sidecar_name}: running")
        else:
            lines.append(f"⚫ {sidecar_name}: not running")

    lines.append(f"\nCrashes: {total_crashes}")
    ts = time.strftime("%H:%M:%S")
    lines.append(f"\n_Last update: {ts}_")
    return "\n".join(lines)


def _check_process_running(exe_name: str) -> bool:
    """Check if a process with given executable name is running (Windows or Unix)."""
    if not exe_name:
        return False
    if sys.platform == "win32":
        try:
            result = _sp.run(
                ["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return exe_name.lower() in result.stdout.lower()
        except Exception:
            return False
    else:
        try:
            result = _sp.run(
                ["pgrep", "-f", exe_name],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False
