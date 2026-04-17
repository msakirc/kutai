"""Decision types emitted by task master to orchestrator.

Phase 1: introduced but minimally wired. Phase 2b wires task master to emit these.

Rule: a Decision exists only if orchestrator must call a different package/subsystem.
Internal state changes (spawning subtasks, marking complete, suspending) are not decisions.
"""

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(frozen=True)
class Dispatch:
    """Run a task. Orchestrator routes to executor based on `executor` tag."""
    task_id: int
    executor: str  # "llm" | "mechanical"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NotifyUser:
    """Send a message to a user. Orchestrator routes to Telegram."""
    chat_id: int
    text: str


# Gate decisions — used by task_gates module (a later Task 4).
@dataclass(frozen=True)
class Allow:
    """Gate passed; task may proceed."""


@dataclass(frozen=True)
class Block:
    """Task is suspended (awaiting approval, clarification, etc.)."""
    reason: str


@dataclass(frozen=True)
class Cancel:
    """Task is cancelled (rejected at gate). No retry."""
    reason: str


GateDecision = Union[Allow, Block, Cancel]
