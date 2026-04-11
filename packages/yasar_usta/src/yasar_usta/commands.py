"""Keyboard builders and log formatting helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Messages

logger = logging.getLogger("yasar_usta.commands")


def build_start_keyboard(messages: Messages) -> dict:
    """Build the reply keyboard shown when the app is down."""
    return {
        "keyboard": [
            [{"text": messages.btn_start}, {"text": messages.btn_status}],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False,
        "is_persistent": True,
    }


def build_status_inline_keyboard(messages: Messages, name: str, sidecar_name: str | None = None) -> dict:
    """Build inline keyboard for the status panel."""
    buttons = [
        [{"text": messages.btn_refresh, "callback_data": "guard_refresh"}],
        [{"text": messages.btn_restart_guard.format(name=name), "callback_data": "restart_guard"}],
    ]
    if sidecar_name:
        buttons[1].append({
            "text": messages.btn_restart_sidecar.format(sidecar_name=sidecar_name),
            "callback_data": "restart_sidecar",
        })
    return {"inline_keyboard": buttons}


def format_log_entries(log_path: str | Path, n: int = 20) -> str | None:
    """Read and format the last N lines of a JSONL log file.

    Returns formatted text or None if no entries found.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return None

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 100_000))
            chunk = f.read()
            lines = chunk.strip().split("\n")

        last_n = lines[-n:]
        formatted = []
        for line in last_n:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = entry.get("ts", "?")
                if "T" in ts:
                    ts = ts.split("T")[1][:8]
                elif " " in ts:
                    ts = ts.split(" ")[1][:8]
                level = entry.get("level", "?")[:4]
                comp = entry.get("src", "?").split(".")[-1]
                msg = entry.get("msg", "")[:120]
                icon = {
                    "ERRO": "🔴", "CRIT": "🔴", "WARN": "🟡",
                    "INFO": "⚪", "DEBU": "⚫",
                }.get(level, "⚪")
                formatted.append(f"{icon} `{ts}` *{comp}*: {msg}")
            except (ValueError, KeyError):
                formatted.append(f"⚫ {line[:120]}")

        if not formatted:
            return None

        msg = "\n".join(formatted)
        if len(msg) > 4000:
            msg = msg[-4000:]
            msg = "...(truncated)\n" + msg[msg.index("\n") + 1:]
        return msg
    except Exception:
        return None
