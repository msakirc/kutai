"""Action result type for mr_roboto.run()."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    """Outcome of a mr_roboto.run() invocation.

    Attributes
    ----------
    status:
        ``"completed"`` on success, ``"failed"`` on error, ``"skipped"`` when
        the action is a no-op. The orchestrator maps these to task states.
    result:
        Arbitrary JSON-serializable payload produced by the executor.
    error:
        Human-readable error string; populated only when ``status == "failed"``.
    """

    status: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
