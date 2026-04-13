"""Keyboard builders and log formatting helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Messages

logger = logging.getLogger("yasar_usta.commands")


def build_start_keyboard(messages: Messages, app_name: str = "App") -> dict:
    """Build the persistent reply keyboard for the wrapper bot."""
    return {
        "keyboard": [
            [{"text": messages.btn_start.format(app_name=app_name)},
             {"text": messages.btn_status}],
            [{"text": messages.btn_restart.format(app_name=app_name)},
             {"text": messages.btn_stop.format(app_name=app_name)}],
            [{"text": messages.btn_logs}, {"text": messages.btn_remote}],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False,
        "is_persistent": True,
    }


def build_status_inline_keyboard(messages: Messages, name: str,
                                  sidecar_names: list[str] | None = None,
                                  sidecar_name: str | None = None) -> dict:
    """Build inline keyboard for the status panel."""
    buttons = [
        [{"text": messages.btn_refresh, "callback_data": "guard_refresh"}],
        [{"text": messages.btn_restart_guard.format(name=name), "callback_data": "restart_guard"}],
    ]
    # Multi-sidecar buttons
    names = sidecar_names or ([sidecar_name] if sidecar_name else [])
    if names:
        sidecar_row = []
        for sc_name in names:
            sidecar_row.append({
                "text": f"📊 Restart {sc_name}",
                "callback_data": f"restart_sidecar:{sc_name}",
            })
        buttons.append(sidecar_row)
    return {"inline_keyboard": buttons}


def _escape_md(text: str) -> str:
    """Escape Markdown special characters in log messages."""
    for ch in ("*", "_", "`", "[", "]"):
        text = text.replace(ch, f"\\{ch}")
    return text


def format_log_entries(log_path: str | Path, n: int = 15) -> str | None:
    """Read and format the last N non-DEBUG lines of a JSONL log file.

    Returns formatted text or None if no entries found.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return None

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            chunk = f.read()
            lines = chunk.strip().split("\n")

        # Walk backwards, skip DEBUG, collect up to n entries
        formatted = []
        for line in reversed(lines):
            if len(formatted) >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                level = entry.get("level", "?")[:4]
                if level == "DEBU":
                    continue
                ts = entry.get("ts", "?")
                if "T" in ts:
                    ts = ts.split("T")[1][:8]
                elif " " in ts:
                    ts = ts.split(" ")[1][:8]
                comp = entry.get("src", "?").split(".")[-1]
                msg = _escape_md(entry.get("msg", "")[:120])
                icon = {
                    "ERRO": "🔴", "CRIT": "🔴", "WARN": "🟡",
                    "INFO": "⚪",
                }.get(level, "⚪")
                formatted.append(f"{icon} `{ts}` *{comp}*: {msg}")
            except (ValueError, KeyError):
                continue

        if not formatted:
            return None

        formatted.reverse()  # back to chronological order
        msg = "\n".join(formatted)
        if len(msg) > 4000:
            msg = msg[-4000:]
            msg = "...(truncated)\n" + msg[msg.index("\n") + 1:]
        return msg
    except Exception:
        return None
