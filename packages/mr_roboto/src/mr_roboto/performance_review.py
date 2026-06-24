"""Composite performance_review verb.

Routes to the appropriate backend based on the ``mode`` field:

- ``mode="web"``  → run_lighthouse(preview_url=..., thresholds=...)
- ``mode="api"``  → run_k6(script_path=..., thresholds=...)
- other           → soft-skip with reason

This is the verb dispatched by the ``performance_review`` post-hook kind.
"""
from __future__ import annotations

from typing import Any

from yazbunu import get_logger
from mr_roboto.run_lighthouse import run_lighthouse
from mr_roboto.run_k6 import run_k6

logger = get_logger("mr_roboto.performance_review")


async def performance_review(
    mode: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a performance audit; backend selected by *mode*.

    Parameters
    ----------
    mode:
        ``"web"`` to run Lighthouse, ``"api"`` to run k6.
    **kwargs:
        Forwarded to the underlying verb unchanged.
        - web: ``preview_url``, ``thresholds``, ``timeout_s``
        - api: ``script_path``, ``thresholds``, ``timeout_s``

    Returns
    -------
    Same shape as ``run_lighthouse`` / ``run_k6``:
    ``{verdict, findings, tools_used, skipped, reason}``.
    """
    if mode == "web":
        return await run_lighthouse(**kwargs)  # type: ignore[arg-type]

    if mode == "api":
        return await run_k6(**kwargs)  # type: ignore[arg-type]

    # Unknown mode → soft-skip.
    reason = f"performance_review: unknown mode {mode!r}; expected 'web' or 'api'"
    logger.warning(reason)
    return {
        "verdict": "pass",
        "findings": [],
        "tools_used": [],
        "skipped": True,
        "reason": reason,
    }
