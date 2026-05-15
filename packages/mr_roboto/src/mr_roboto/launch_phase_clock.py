"""Z7 T3A (A2) — Launch phase clock.

Given a ``scheduled_publish_at`` datetime (the T-0 anchor), resolves
the absolute UTC timestamp for each of the 8 required phase offsets:

  T-72h  — Asset prep (draft all channels)
  T-24h  — Founder approval of all drafts
  T-0    — Synchronized publish (all channels at once)
  T+1h   — Early response monitor start
  T+4h   — Response monitor mid-check
  T+24h  — Day-1 engagement digest
  T+72h  — Day-3 engagement digest
  T+168h — T+7d lessons writeback

``resolve_phase_times(publish_at)`` returns ``dict[int, datetime]``
mapping offset_hours → absolute UTC datetime.

``relative_to`` field in launch_playbook.json declares ``"scheduled_publish_at"``
so the expander knows the phase anchor.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

# All 8 required phase offsets (hours relative to T-0 scheduled_publish_at)
PHASE_OFFSETS: tuple[int, ...] = (-72, -24, 0, 1, 4, 24, 72, 168)


def resolve_phase_times(
    publish_at: datetime,
) -> dict[int, datetime]:
    """Resolve each phase offset to an absolute UTC datetime.

    Parameters
    ----------
    publish_at:
        The T-0 scheduled publish time.  If naive (no tzinfo), UTC is assumed.

    Returns
    -------
    dict[int, datetime]
        Mapping offset_hours → absolute UTC datetime with timezone.utc attached.
    """
    if publish_at.tzinfo is None:
        publish_at = publish_at.replace(tzinfo=timezone.utc)

    return {
        offset_h: publish_at + timedelta(hours=offset_h)
        for offset_h in PHASE_OFFSETS
    }


def phase_label(offset_h: int) -> str:
    """Human-readable label for each phase offset."""
    labels = {
        -72: "T-72h: Asset prep — draft all channels",
        -24: "T-24h: Founder approval of channel drafts",
        0: "T-0: Synchronized publish",
        1: "T+1h: Response monitor start",
        4: "T+4h: Response monitor mid-check",
        24: "T+24h: Day-1 engagement digest",
        72: "T+72h: Day-3 engagement digest",
        168: "T+7d: Launch lessons writeback",
    }
    return labels.get(offset_h, f"T{'+' if offset_h >= 0 else ''}{offset_h}h")
