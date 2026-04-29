"""Token estimates per task.

Lookup chain: B-table (learned, step_token_stats) → A (STEP_TOKEN_OVERRIDES)
→ AGENT_REQUIREMENTS default. See `estimate_for(task)` in this module
(added in a later task).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Estimates:
    in_tokens: int
    out_tokens: int
    iterations: int

    @property
    def per_call_tokens(self) -> int:
        return self.in_tokens + self.out_tokens

    @property
    def total_tokens(self) -> int:
        return self.per_call_tokens * self.iterations
