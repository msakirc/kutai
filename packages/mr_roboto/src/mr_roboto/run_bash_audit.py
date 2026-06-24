"""Mechanical wrapper for sade_kalsin's quarterly bash-audit harness.

Z1 Tier 7A (B12) — invoked by the `bash_audit` internal cadence (cron
`0 9 1 jan,apr,jul,oct *`). Wraps `sade_kalsin.run_audit`, posts the
report path to the founder via Telegram (best-effort), and returns a
small JSON-serialisable summary.

Boundary: this module legitimately bridges sade_kalsin (standalone audit
package, zero KutAI deps) with the broader system. sade_kalsin itself
does NOT import mr_roboto, db, or telegram — those wirings live here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.run_bash_audit")


async def run(task: dict) -> dict[str, Any]:
    """Mr. Roboto mechanical executor entry point.

    Payload knobs (all optional):

    - ``quarter``: e.g. ``"2026-Q2"`` (default: current calendar quarter)
    - ``layer``: single-layer filter (default: all layers)
    - ``root``: repo root override (default: ``Path.cwd()``)
    - ``out_dir``: report output dir (default: ``<root>/docs/audits``)
    - ``notify``: post the report path to founder via Telegram
      (default: ``True`` when chat id resolvable; otherwise no-op)
    """
    payload = task.get("payload") or task.get("context") or {}

    import sade_kalsin

    root = Path(payload.get("root") or Path.cwd())
    out_dir = Path(payload["out_dir"]) if payload.get("out_dir") else None

    result = sade_kalsin.run_audit(
        root=root,
        quarter=payload.get("quarter"),
        out_dir=out_dir,
        layer_filter=payload.get("layer"),
    )

    notify = payload.get("notify", True)
    if notify:
        try:
            await _notify_founder(result["report_path"], result["quarter"])
            result["notified"] = True
        except Exception as e:  # noqa: BLE001
            logger.warning("bash_audit notify failed: %s", e)
            result["notified"] = False

    logger.info(
        "bash_audit complete quarter=%s layers=%d total_loc=%d report=%s",
        result["quarter"],
        result["layer_count"],
        result["total_loc"],
        result["report_path"],
    )
    # Slim the response — `layers` is internal-only, not for the task row.
    return {
        "ok": True,
        "quarter": result["quarter"],
        "report_path": result["report_path"],
        "layer_count": result["layer_count"],
        "total_loc": result["total_loc"],
        "notified": result.get("notified", False),
    }


async def _notify_founder(report_path: str, quarter: str) -> None:
    """Best-effort telegram notify. Silently no-ops when bot/chat unset."""
    try:
        from src.app.telegram_bot import telegram_send  # type: ignore
    except Exception:
        # Older code path / shim absent — try the orchestrator-level helper.
        try:
            from src.app.notify import telegram_send  # type: ignore
        except Exception:
            return
    msg = (
        f"Bash audit ready: {quarter}\n"
        f"Report: {report_path}\n\n"
        "Mini-SWE-agent showed 65% SWE-bench in 100 LOC + bash. "
        "Walk the hot-spots and decide what stays."
    )
    try:
        await telegram_send(msg)
    except TypeError:
        # Some helpers expect (chat_id, text); skip gracefully.
        return
