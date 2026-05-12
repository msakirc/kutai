"""Z8 T5D — cost slope anomaly detector.

A vendor's daily spend is flagged when today's amount exceeds 2.5σ above
the trailing 14-day mean. Returns False when history is too short
(<7 days) so the detector doesn't fire on a cold start.

The detector is intentionally tiny — it owns one statistical question.
The cost-slope alerting rule in ``src.infra.alerting`` orchestrates the
data fetch + cooldown.
"""
from __future__ import annotations

import statistics


async def is_anomaly(
    integration_id: str,
    today_usd: float,
    history_14d: list[float],
) -> bool:
    """Return True when ``today_usd`` is a 2.5σ outlier vs ``history_14d``.

    Parameters
    ----------
    integration_id:
        Vendor key (e.g. ``"stripe"``); reserved for future per-vendor
        tuning. Currently unused but kept in the signature so callers
        don't need to refactor when v2 lands.
    today_usd:
        Today's spend in USD.
    history_14d:
        Daily USD spend over the last 14 days (or fewer; <7 → False).
    """
    if not history_14d or len(history_14d) < 7:
        return False
    mean = statistics.mean(history_14d)
    try:
        stdev = statistics.stdev(history_14d)
    except statistics.StatisticsError:
        stdev = 0.0
    if stdev <= 0.0:
        stdev = 0.01
    z = (float(today_usd) - mean) / stdev
    return z > 2.5
