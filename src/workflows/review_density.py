"""Z3 T1C — review_density resolver.

Founder-controlled dials that adjust post-hook gate severity, QA modality
coverage, and multi-file expansion behaviour for a given mission.

Schema lives in ``missions.review_density_json`` (TEXT NULL, added by the
2026-05-12-missions-review-density migration).  NULL rows return conservative
defaults so existing missions are unaffected.

Wire-up note
------------
T1C only ships the resolver + Telegram command.  Actual expander wire-up (so
``MissionDialContext`` flows into post-hook evaluation) is deferred to T2+.
The helper ``to_mission_dial_context`` is provided so T2+ can call it without
touching this file.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from src.infra.logging_config import get_logger

logger = get_logger("workflows.review_density")

# ---------------------------------------------------------------------------
# Allowed value sets
# ---------------------------------------------------------------------------

_ALLOWED: dict[str, set] = {
    "qa_dial": {"quick", "standard", "strict"},
    "accessibility_dial": {"on", "off"},
    "multi_file_expansion": {True, False},
    "integration_replay": {"quick", "standard", "strict"},
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReviewDensityDials:
    """Founder-controlled dials for one mission.

    All fields default to the *conservative* values — the same behaviour that
    existed before Z3 T1C so that existing missions are unchanged.
    """
    qa_dial: str = "standard"
    accessibility_dial: str = "off"
    multi_file_expansion: bool = False
    integration_replay: str = "standard"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_dials(mission_id: int) -> ReviewDensityDials:
    """Return resolved dials for *mission_id*.

    Reads ``missions.review_density_json``; fills missing keys with defaults
    so partial JSON blobs are safe.  Returns all defaults when the column is
    NULL or the mission does not exist.
    """
    from src.infra.db import get_db  # lazy to avoid circular imports

    db = await get_db()
    cursor = await db.execute(
        "SELECT review_density_json FROM missions WHERE id = ?", (mission_id,)
    )
    row = await cursor.fetchone()
    if row is None or row[0] is None:
        return ReviewDensityDials()

    try:
        raw: dict = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "review_density_json for mission %s is not valid JSON; "
            "using defaults.", mission_id,
        )
        return ReviewDensityDials()

    dials = ReviewDensityDials()
    for key in ("qa_dial", "accessibility_dial", "integration_replay"):
        if key in raw:
            dials = _replace(dials, key, raw[key])
    if "multi_file_expansion" in raw:
        val = raw["multi_file_expansion"]
        # Accept both bool and string "true"/"false" from JSON
        if isinstance(val, bool):
            dials = _replace(dials, "multi_file_expansion", val)
        elif isinstance(val, str):
            dials = _replace(dials, "multi_file_expansion", val.lower() == "true")

    return dials


async def set_dial(mission_id: int, key: str, value) -> ReviewDensityDials:
    """Set a single dial for *mission_id* and return the updated dials.

    Parameters
    ----------
    mission_id:
        Target mission (must exist).
    key:
        One of ``qa_dial``, ``accessibility_dial``, ``multi_file_expansion``,
        ``integration_replay``.
    value:
        A string or bool.  For ``multi_file_expansion`` the strings ``"true"``
        and ``"false"`` are accepted in addition to actual booleans.

    Raises
    ------
    ValueError
        On unknown key or value not in the allowed set.
    """
    from src.infra.db import get_db  # lazy to avoid circular imports

    if key not in _ALLOWED:
        allowed_keys = sorted(_ALLOWED)
        raise ValueError(
            f"Unknown dial key {key!r}. "
            f"Allowed keys: {allowed_keys}"
        )

    # Coerce string booleans for multi_file_expansion
    coerced = value
    if key == "multi_file_expansion":
        if isinstance(value, str):
            if value.lower() == "true":
                coerced = True
            elif value.lower() == "false":
                coerced = False
            else:
                raise ValueError(
                    f"multi_file_expansion accepts True/False (or 'true'/'false'),"
                    f" got {value!r}"
                )
        elif not isinstance(value, bool):
            raise ValueError(
                f"multi_file_expansion accepts bool, got {type(value).__name__}"
            )

    if coerced not in _ALLOWED[key]:
        raise ValueError(
            f"Invalid value {coerced!r} for dial {key!r}. "
            f"Allowed: {sorted(str(v) for v in _ALLOWED[key])}"
        )

    # Read current dials, apply change, write back
    current = await get_dials(mission_id)
    updated = _replace(current, key, coerced)

    from general_beckman import update_mission_fields
    await update_mission_fields(mission_id, review_density_json=json.dumps(asdict(updated)))
    logger.info(
        "Mission %s: set dial %s=%r", mission_id, key, coerced,
    )
    return updated


# ---------------------------------------------------------------------------
# MissionDialContext bridge
# ---------------------------------------------------------------------------

def to_mission_dial_context(dials: ReviewDensityDials):
    """Convert *dials* to a MissionDialContext suitable for post-hook expanders.

    Lives here so T2+ wire-up only imports from this module.  The actual
    ``MissionDialContext`` dataclass is defined in
    ``packages/general_beckman/src/general_beckman/posthooks.py``.
    """
    from general_beckman.posthooks import MissionDialContext  # type: ignore[import]

    return MissionDialContext(
        qa_dial=dials.qa_dial,
        accessibility_dial=dials.accessibility_dial,
        multi_file_expansion=dials.multi_file_expansion,
        integration_replay=dials.integration_replay,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _replace(dials: ReviewDensityDials, key: str, value) -> ReviewDensityDials:
    """Return a new ReviewDensityDials with *key* set to *value*."""
    d = asdict(dials)
    d[key] = value
    return ReviewDensityDials(**d)
