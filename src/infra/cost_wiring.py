"""Shim — cost_wiring relocated to ``kuleden_donen_var`` (cloud-cost domain).

Kept as a thin re-export because ``src`` core still imports it
(``src/app/telegram_bot.py`` → ``format_mission_cost``). New code should
import from ``kuleden_donen_var`` directly. The ``src``→``src`` cleanup of
this shim is out of scope (tied to the telegram_bot split).
"""
from __future__ import annotations

from kuleden_donen_var.cost_wiring import (  # noqa: F401
    DEFAULT_COST_DECISION_THRESHOLD_USD,
    QUALITY_MODE_PROFILES,
    await_cost_decision_verdict,
    format_mission_cost,
    open_cost_decision_confirmation,
    quality_mode_profile,
)

__all__ = [
    "DEFAULT_COST_DECISION_THRESHOLD_USD",
    "QUALITY_MODE_PROFILES",
    "await_cost_decision_verdict",
    "format_mission_cost",
    "open_cost_decision_confirmation",
    "quality_mode_profile",
]
