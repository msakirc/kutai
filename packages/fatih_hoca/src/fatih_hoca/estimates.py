"""Token estimates per task.

Lookup chain: B-table (learned, step_token_stats) → A (STEP_TOKEN_OVERRIDES)
→ AGENT_REQUIREMENTS default. See `estimate_for(task)` in this module
(added in a later task).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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

# Ceiling for the LEARNED (B-table) input estimate. A runaway conversation or
# poisoned step_token_stats row can push in_p90 to ~170k+; unclamped it flows
# into effective_context_needed (selector.py:593), per-request (:643) and TPM
# (:662) gates and collapses the candidate pool to gemini-only — the
# "No model candidates available" DLQ (2026-06-20). Clamping here (the single
# shared estimate seen by admission + worker + ranking) keeps them aligned and
# bounds all three gates at once. 64000 keeps effective_context_needed
# (≈(64k+out)*1.3+512) under cerebras's 128k window, so the ctx gate can never
# zero the cloud pool regardless of B-table hygiene. Defense-in-depth atop the
# btable_rollup prompt_tokens<=64k filter. ONLY the learned path is clamped —
# curated STEP_TOKEN_OVERRIDES and AGENT_REQUIREMENTS are trusted. Output is
# NOT clamped: heavy steps (e.g. 4.5b openapi_spec) legitimately emit >100k and
# output can't run away (bounded by the model's max generation).
def _max_est_in_tokens() -> int:
    """Ceiling read at call time so it's tunable without a restart."""
    try:
        return int(os.environ.get("KUTAI_MAX_EST_IN_TOKENS", "64000"))
    except ValueError:
        return 64000


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
        in_p90 = int(row.get("in_p90", 0) or 0)
        ceiling = _max_est_in_tokens()
        if in_p90 > ceiling:
            logger.debug(
                "estimate_for: clamping learned in_p90 %d->%d "
                "(agent=%s step=%s phase=%s)",
                in_p90, ceiling, agent_type, step_id, phase,
            )
            in_p90 = ceiling
        return Estimates(
            in_tokens=in_p90,
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
