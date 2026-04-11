"""Streaming abort callback factory."""

from __future__ import annotations

from typing import Callable

from .detectors import HARD_CAP


def make_stream_callback(
    max_size: int = 20_000,
    check_interval: int = 4096,
) -> Callable[[str], bool]:
    """Create a stateful callback for streaming quality checks.

    Returns callback(accumulated_text) -> should_abort.
    - Size check runs on every call (cheap len()).
    - Full quality assessment runs every check_interval chars.
    - Returns True (abort) when content is degenerate.
    """
    effective_max = min(max_size, HARD_CAP)
    last_checked_len = 0

    def callback(accumulated: str) -> bool:
        nonlocal last_checked_len

        if len(accumulated) > effective_max:
            return True

        if len(accumulated) - last_checked_len >= check_interval:
            last_checked_len = len(accumulated)
            from .assessor import assess
            result = assess(accumulated, max_size=effective_max)
            return result.is_degenerate

        return False

    return callback
