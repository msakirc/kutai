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


from typing import Any

MIN_SAMPLES = 5
THINKING_OUT_SCALE = 2.0  # cloud thinking models: char-derived telemetry under-reports


def _btable_lookup(btable: dict, agent_type: str, step_id: str, phase: str) -> dict | None:
    return btable.get((agent_type, step_id or "", phase or ""))


def estimate_for(task: Any, *, btable: dict, model_is_thinking: bool = False) -> Estimates:
    """Token estimate for a task via the lookup chain.

    Order:
      1. step_token_stats (B-table) when samples_n >= MIN_SAMPLES
      2. STEP_TOKEN_OVERRIDES (curated static A-table)
      3. AGENT_REQUIREMENTS default + AVG_ITERATIONS_BY_AGENT

    Args:
        task: object with .agent_type and .context (dict-like with
              workflow_step_id / workflow_phase keys, optional).
        btable: dict keyed by (agent_type, step_id, phase) returning a row
                with samples_n, in_p90, out_p90, iters_p90.
        model_is_thinking: when True, scale out_tokens up to compensate for
                under-reporting from char-based telemetry.
    """
    from fatih_hoca.requirements import AGENT_REQUIREMENTS, AVG_ITERATIONS_BY_AGENT
    from fatih_hoca.step_overrides import STEP_TOKEN_OVERRIDES

    agent_type = getattr(task, "agent_type", "") or "assistant"
    ctx = getattr(task, "context", None) or {}
    step_id = ctx.get("workflow_step_id") if isinstance(ctx, dict) else None
    phase = ctx.get("workflow_phase") if isinstance(ctx, dict) else None

    # Level 1 — learned
    row = _btable_lookup(btable, agent_type, step_id, phase) if step_id else None
    if row and (row.get("samples_n") or 0) >= MIN_SAMPLES:
        out_p90 = int(row.get("out_p90", 0) or 0)
        if model_is_thinking:
            out_p90 = int(out_p90 * THINKING_OUT_SCALE)
        return Estimates(
            in_tokens=int(row.get("in_p90", 0) or 0),
            out_tokens=out_p90,
            iterations=int(round(float(row.get("iters_p90", 1) or 1))),
        )

    # Level 2 — static overrides
    if step_id and step_id in STEP_TOKEN_OVERRIDES:
        return STEP_TOKEN_OVERRIDES[step_id]

    # Level 3 — AGENT_REQUIREMENTS + AVG_ITERATIONS
    reqs = AGENT_REQUIREMENTS.get(agent_type) or AGENT_REQUIREMENTS["assistant"]
    return Estimates(
        in_tokens=reqs.estimated_input_tokens,
        out_tokens=reqs.estimated_output_tokens,
        iterations=AVG_ITERATIONS_BY_AGENT.get(agent_type, 6),
    )
