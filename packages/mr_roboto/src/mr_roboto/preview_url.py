"""Shared pending-URL filter for preview consumers.

Z3 residual R2 — consolidates the ``pending:`` placeholder detection that
``run_axe`` introduced and extends it to ``run_lighthouse`` /
``run_schemathesis``.  A URL is *real* when it parses as ``http://`` or
``https://`` and is not the ``pending:<reason>`` sentinel produced by the
preview emitter when the tunnel could not be brought up yet.
"""
from __future__ import annotations


def is_real_url(url: str | None) -> bool:
    """Return True when *url* is a real http(s) URL (not pending / blank)."""
    if not url:
        return False
    stripped = str(url).strip()
    if not stripped:
        return False
    if stripped.startswith("pending:"):
        return False
    return stripped.startswith("http://") or stripped.startswith("https://")
