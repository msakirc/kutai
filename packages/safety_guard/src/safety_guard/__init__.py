"""Pre-action safety guard: reversibility tag resolution + collision guards."""
from safety_guard.tags import Reversibility, resolve
from safety_guard.executor_hook import (
    pre_action,
    Allow,
    WaitForFounder,
    Block,
    Decision,
)

__all__ = [
    "Reversibility",
    "resolve",
    "pre_action",
    "Allow",
    "WaitForFounder",
    "Block",
    "Decision",
]
