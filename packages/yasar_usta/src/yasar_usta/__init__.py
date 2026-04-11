"""Yaşar Usta — Telegram-controlled process manager.

Manages any subprocess with:
- Heartbeat-based hung detection
- Escalating backoff with stability reset
- Telegram as control plane when process is down
- Non-destructive offset polling (no message loss)
- Claude Code remote trigger
- Sidecar process management (log viewer etc.)
- Configurable i18n for all user-facing strings
"""

from .config import GuardConfig, Messages, SidecarConfig
from .guard import ProcessGuard

__all__ = [
    "ProcessGuard",
    "GuardConfig",
    "Messages",
    "SidecarConfig",
]
