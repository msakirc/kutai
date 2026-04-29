"""Diagnostic struct returned alongside the pressure scalar."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PressureBreakdown:
    """Per-(model, task) signal-by-signal contribution. Logged to model_pick_log."""
    scalar: float
    signals: dict[str, float] = field(default_factory=dict)
    # modifiers may include scalars (M1, M3_difficulty) or nested dicts (weights)
    modifiers: dict[str, Any] = field(default_factory=dict)
    bucket_totals: dict[str, float] = field(default_factory=dict)
    positive_total: float = 0.0
    negative_total: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
