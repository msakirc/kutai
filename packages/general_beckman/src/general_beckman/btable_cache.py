"""Module-level B-table cache. Refreshed by btable_rollup cron (Task 26)."""
from __future__ import annotations

_BTABLE: dict[tuple[str, str, str], dict] = {}


def get_btable() -> dict:
    return _BTABLE


def set_btable(rows: dict) -> None:
    global _BTABLE
    _BTABLE = rows
