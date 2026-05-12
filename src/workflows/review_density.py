"""Mission review-density dials — T1C.

Exposes ``get_dials(mission_id)`` + ``to_mission_dial_context()`` for the
expander to decide whether to enable multi-file expansion and other
quality-gate switches.

``MissionDialContext`` is a lightweight dataclass so the expander can read
dial values without importing DB or async machinery at module import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# MissionDialContext
# ---------------------------------------------------------------------------


@dataclass
class MissionDialContext:
    """Resolved dial values for one mission.

    Attributes
    ----------
    mission_id:
        The mission this context was built for (int or str).
    multi_file_expansion:
        When ``True``, the expander may decompose multi-file steps into
        per-file sub-task steps via ``src.workflows.engine.multifile``.
        Default ``False`` (opt-in only).
    template_id:
        Optional template ID to drive multi-file expansion.  When ``None``
        the expander looks for ``template_id`` on the step itself.
    stack_slug:
        Stack slug for expansion rule lookup (e.g. ``"fastapi"``).
        When ``None`` the expander reads ``tech_stack_detected`` from
        mission artifacts.
    extra:
        Arbitrary per-mission dial overrides preserved for future use.
    """

    mission_id: Any = None
    multi_file_expansion: bool = False
    template_id: str | None = None
    stack_slug: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dial resolution
# ---------------------------------------------------------------------------

# In-process cache so callers in the same mission batch share one DB round-trip.
_dial_cache: dict[Any, dict[str, Any]] = {}


def _default_dials() -> dict[str, Any]:
    return {
        "multi_file_expansion": False,
        "template_id": None,
        "stack_slug": None,
    }


async def get_dials(mission_id: Any) -> dict[str, Any]:
    """Return a dials dict for *mission_id*.

    Reads the ``mission_dial_overrides`` JSON field from the ``missions`` row
    when available; otherwise returns defaults.

    Never raises — returns defaults on any DB error so the expander path is
    always safe to call.

    Parameters
    ----------
    mission_id:
        The mission identifier. ``None`` returns defaults immediately.

    Returns
    -------
    dict with keys ``multi_file_expansion``, ``template_id``, ``stack_slug``,
    plus any extra overrides.
    """
    if mission_id is None:
        return _default_dials()

    if mission_id in _dial_cache:
        return dict(_dial_cache[mission_id])

    defaults = _default_dials()
    try:
        from src.infra.db import get_mission
        import json as _json

        mission_row = await get_mission(mission_id)
        if mission_row is None:
            _dial_cache[mission_id] = defaults
            return dict(defaults)

        # The missions table may carry a JSON blob of per-mission dial overrides.
        overrides_raw = (mission_row or {}).get("dial_overrides") or "{}"
        if isinstance(overrides_raw, str):
            try:
                overrides = _json.loads(overrides_raw)
            except Exception:
                overrides = {}
        elif isinstance(overrides_raw, dict):
            overrides = overrides_raw
        else:
            overrides = {}

        dials = {**defaults, **overrides}
        _dial_cache[mission_id] = dials
        return dict(dials)

    except Exception:
        # Fail open: if DB unavailable, return defaults.
        _dial_cache[mission_id] = defaults
        return dict(defaults)


def to_mission_dial_context(
    mission_id: Any,
    dials: dict[str, Any],
) -> MissionDialContext:
    """Convert a raw dials dict into a ``MissionDialContext``.

    Parameters
    ----------
    mission_id:
        The mission this context is for.
    dials:
        Raw dials dict as returned by ``get_dials()``.

    Returns
    -------
    MissionDialContext
    """
    known = {"multi_file_expansion", "template_id", "stack_slug"}
    extra = {k: v for k, v in dials.items() if k not in known}
    return MissionDialContext(
        mission_id=mission_id,
        multi_file_expansion=bool(dials.get("multi_file_expansion", False)),
        template_id=dials.get("template_id") or None,
        stack_slug=dials.get("stack_slug") or None,
        extra=extra,
    )


def invalidate_cache(mission_id: Any = None) -> None:
    """Clear the in-process dial cache.

    Useful in tests and when a mission's dial overrides are updated mid-run.
    When *mission_id* is ``None``, clears the entire cache.
    """
    if mission_id is None:
        _dial_cache.clear()
    else:
        _dial_cache.pop(mission_id, None)
