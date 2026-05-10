"""Action result type for mr_roboto.run()."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mr_roboto.reversibility import Reversibility


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
    reversibility:
        Z10-T1B reversibility tag for this action's verb. Populated by
        the dispatcher from :data:`mr_roboto.VERB_REVERSIBILITY`, with
        per-invocation override via ``payload["reversibility_override"]``
        (intended for ``run_cmd`` where the caller knows whether their
        specific command is destructive). Defaults to ``"partial"`` —
        the conservative default for unknown verbs and for failure
        paths that bail before the dispatcher can compute the tag.
    """

    status: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    reversibility: Reversibility = "partial"
