# infra/alerting.py
"""
Phase 9.3 — Alerting

Rule-based alerting checked every orchestrator cycle. Uses Telegram
for notifications. Rules: task failure rate, daily cost, model success rate,
queue depth.
"""
from __future__ import annotations
import time
from typing import Optional

from .logging_config import get_logger
from .runtime_state import runtime_state

logger = get_logger("infra.alerting")

# ── Default alert thresholds ─────────────────────────────────────────────────

ALERT_TASK_FAILURES_IN_60MIN: int = 3
ALERT_DAILY_COST_USD: float = 5.0
ALERT_MODEL_SUCCESS_RATE_MIN: float = 0.50   # 50%
ALERT_QUEUE_DEPTH_MAX: int = 20

# ── Alert state (prevent spam) ───────────────────────────────────────────────

_last_alert: dict[str, float] = {}
_COOLDOWN_SECS = 900  # 15 minutes per alert type


def _should_fire(alert_id: str) -> bool:
    """Return True if alert hasn't fired recently."""
    last = _last_alert.get(alert_id, 0.0)
    if time.monotonic() - last > _COOLDOWN_SECS:
        _last_alert[alert_id] = time.monotonic()
        return True
    return False


async def check_alerts() -> None:
    """Run all alert rules. Called every orchestrator cycle."""
    try:
        from .metrics import get_counter
        from ..app.config import COST_BUDGET_DAILY
    except Exception:
        return

    # Rule 1: Too many task failures in recent period
    try:
        failures = get_counter("tasks_failed")
        completed = get_counter("tasks_completed")
        total = failures + completed
        if total > 0 and failures >= ALERT_TASK_FAILURES_IN_60MIN:
            if _should_fire("task_failures"):
                await _send_alert(
                    "⚠️ High Task Failure Rate",
                    f"{int(failures)} tasks failed recently "
                    f"({failures/(total)*100:.0f}% failure rate)",
                    priority=4,
                )
    except Exception:
        pass

    # Rule 2: Daily cost exceeded
    try:
        daily_cost = get_counter("cost_total")
        budget = COST_BUDGET_DAILY
        if daily_cost > budget:
            if _should_fire("daily_cost"):
                await _send_alert(
                    "💸 Daily Cost Budget Exceeded",
                    f"Spent ${daily_cost:.2f} today (budget: ${budget:.2f})",
                    priority=5,
                )
    except Exception:
        pass

    # Rule 3: Queue depth too high
    try:
        queue_depth = get_counter("queue_depth")
        if queue_depth > ALERT_QUEUE_DEPTH_MAX:
            if _should_fire("queue_depth"):
                await _send_alert(
                    "📥 Queue Backed Up",
                    f"Task queue depth: {int(queue_depth)} (max: {ALERT_QUEUE_DEPTH_MAX})",
                    priority=3,
                )
    except Exception:
        pass

    # Rule 4 (Z8 T5D): vendor cost-slope anomaly.
    # Reads optional ``_vendor_cost_history`` runtime_state slot — populated
    # by ``cost_pull`` executor when it persists today's USD figure. The
    # detector flags >2.5σ jumps over the trailing 14d.
    try:
        history_state = runtime_state.get("vendor_cost_history") or {}
        if isinstance(history_state, dict) and history_state:
            from src.ops.cost_anomaly import is_anomaly
            for vendor_id, vstate in history_state.items():
                if not isinstance(vstate, dict):
                    continue
                today = float(vstate.get("today_usd") or 0.0)
                history = vstate.get("history_14d") or []
                if not isinstance(history, list):
                    continue
                if await is_anomaly(vendor_id, today, [float(x) for x in history]):
                    if _should_fire(f"cost_slope:{vendor_id}"):
                        await _send_alert(
                            "📈 Vendor Cost Spike",
                            f"{vendor_id}: today ${today:.2f} is >2.5σ above "
                            f"the trailing {len(history)}d mean",
                            priority=4,
                        )
    except Exception:
        pass


async def _send_alert(title: str, message: str, priority: int = 3) -> None:
    """Send alert via Telegram."""
    logger.warning(f"ALERT: {title} — {message}")
    try:
        from ..infra.runtime_state import runtime_state
        if runtime_state.get("telegram_available"):
            from ..app.config import TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID
            import aiohttp
            if TELEGRAM_BOT_TOKEN and TELEGRAM_ADMIN_CHAT_ID:
                icons = {5: "\U0001f534", 4: "\U0001f7e0", 3: "\U0001f7e1", 2: "\U0001f535", 1: "\u26aa"}
                icon = icons.get(priority, "\U0001f7e1")
                text = f"{icon} *{title}*\n\n{message}"
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                async with aiohttp.ClientSession() as s:
                    await s.post(url, json={
                        "chat_id": TELEGRAM_ADMIN_CHAT_ID,
                        "text": text,
                        "parse_mode": "Markdown",
                    }, timeout=aiohttp.ClientTimeout(total=5))
    except Exception:
        pass
