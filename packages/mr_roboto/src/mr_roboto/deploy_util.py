"""Bounded poll-until-ready helper for deploy orchestration."""
from __future__ import annotations
import asyncio, time
from typing import Any, Awaitable, Callable

async def poll_until(
    fetch: Callable[[], Awaitable[dict]],
    ready: Callable[[dict], bool],
    fail: Callable[[dict], bool],
    *, max_wait_s: float = 600, base_delay_s: float = 5, cap_delay_s: float = 30,
) -> dict[str, Any]:
    """Poll ``fetch`` until ``ready`` (→ ok), ``fail`` (→ terminal_fail), or timeout.

    Uses monotonic-clock deadline + exponential backoff capped at ``cap_delay_s``.
    Returns {ok, result?|reason}.
    """
    deadline = time.monotonic() + max_wait_s
    delay = base_delay_s
    last = None
    while time.monotonic() < deadline:
        last = await fetch()
        if fail(last):
            return {"ok": False, "reason": "terminal_fail", "result": last}
        if ready(last):
            return {"ok": True, "result": last}
        await asyncio.sleep(delay)
        delay = min(delay * 2, cap_delay_s) if delay else base_delay_s
    return {"ok": False, "reason": "timeout", "result": last}
