"""Reversibility tag resolution.

Static tag in workflow JSON is the floor. Runtime override (from executor)
may escalate stricter; downgrade is rejected. `locked: true` removes
override entirely (workflow author wins absolutely).
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Reversibility(Enum):
    FULL = ("full", 0)
    PARTIAL = ("partial", 1)
    NONE = ("none", 2)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def rank(self) -> int:
        return self.value[1]

    @classmethod
    def from_str(cls, s: str) -> "Reversibility":
        for r in cls:
            if r.label == s:
                return r
        logger.warning(
            "unknown reversibility value %r; defaulting to FULL", s
        )
        return cls.FULL


def resolve(step: dict, runtime_override: Optional[Reversibility]) -> Reversibility:
    """Return the effective reversibility tag for a step.

    Priority:
      1. If `step.locked` is true: static tag wins, override ignored.
      2. Else, runtime override accepted only if rank >= static rank.
      3. Downgrade attempts logged + rejected.
    """
    static = Reversibility.from_str(step.get("reversibility", "full"))
    locked = bool(step.get("locked", False))
    if locked or runtime_override is None:
        return static
    if runtime_override.rank < static.rank:
        logger.warning(
            "downgrade rejected: runtime=%s static=%s; using static",
            runtime_override.label, static.label,
        )
        return static
    return runtime_override
