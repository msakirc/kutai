"""Checkpoint persistence, idempotency keys, and safe conversation logging.

Pure async functions. No class state.

  save_checkpoint      — serialise execution state, call save_task_checkpoint.
  clear_checkpoint_safe — call clear_task_checkpoint, swallow errors.
  tool_idempotency_key  — sha256(tool|json(args))[:16] — stable, deterministic.
  safe_log_conversation — fire-and-forget wrapper around log_conversation.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

from dabidabi import log_conversation
from yazbunu import get_logger

logger = get_logger("runtime.checkpoint")


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


async def save_checkpoint(
    task_id,
    next_iteration: int,
    messages: list[dict],
    total_cost: float,
    used_model: str,
    reqs,
    tools_used: bool,
    validation_retried: bool,
    completed_tool_ops: dict[str, str] | None = None,
    format_corrections: int = 0,
    tools_used_names: set[str] | None = None,
    tool_calls: list[dict] | None = None,
    worker_attempts: int = 0,
) -> None:
    """Persist agent loop state so execution can resume after a crash.

    ``tool_calls`` is a per-execution audit list (richer than
    ``tools_used_names`` which is a name-only set). Each entry is
    ``{name, args, ok}``. Used by the grounding guard to verify the
    agent actually called write_file (or other declared tools) for the
    paths it claimed to produce.

    ``worker_attempts`` is the CURRENT dispatch's attempt counter; it is
    serialized as ``saved_attempts`` so ``run()`` can tell a crash-resume
    of the SAME attempt (restore the conversation) from a checkpoint left
    behind by a COMPLETED prior dispatch (a quality re-dispatch bumped the
    counter → rebuild fresh, do NOT restore the bloated messages array).
    A quality re-dispatch increments worker_attempts (apply.py:530/615);
    a crash / heartbeat-timeout resume does NOT (sweep.py flips
    processing→pending without touching the column). ``saved_attempts``
    is thus the load-bearing dispatch-boundary discriminator (spec M1/C4).
    """
    if task_id == "?":
        return
    try:
        state = {
            "iteration": next_iteration,
            "messages": messages,
            "total_cost": total_cost,
            "used_model": used_model,
            "reqs": dataclasses.asdict(reqs),
            "tools_used": tools_used,
            "tools_used_names": list(tools_used_names or []),
            "tool_calls": list(tool_calls or []),
            "validation_retried": validation_retried,
            "format_corrections": format_corrections,
            "completed_tool_ops": completed_tool_ops or {},
            "saved_attempts": int(worker_attempts or 0),
        }
        from general_beckman import save_task_checkpoint as _save_ckpt
        await _save_ckpt(task_id, state)
        logger.debug(
            f"[Task #{task_id}] Checkpoint saved at iteration "
            f"{next_iteration}"
        )
    except Exception as exc:
        logger.warning(
            f"[Task #{task_id}] Checkpoint save failed: {exc}"
        )


async def clear_checkpoint_safe(task_id) -> None:
    """Clear checkpoint on successful completion — never raises."""
    if task_id == "?":
        return
    try:
        from general_beckman import clear_task_checkpoint as _clear_ckpt
        await _clear_ckpt(task_id)
    except Exception as exc:
        logger.warning(
            f"[Task #{task_id}] Checkpoint clear failed: {exc}"
        )


def tool_idempotency_key(tool_name: str, tool_args: dict) -> str:
    """Compute a short hash key for a tool call's identity.

    Used to skip re-execution of side-effect tools (write_file, shell,
    git_commit, etc.) when resuming from a checkpoint.
    """
    # Stable serialisation: sorted keys, no whitespace variance
    raw = f"{tool_name}|{json.dumps(tool_args, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


async def safe_log_conversation(
    task_id,
    role: str,
    content: str,
    model: str | None,
    cost: float,
    agent_name: str,
) -> None:
    """Fire-and-forget conversation log — never breaks the loop.

    Caller passes agent_name (self.name) to avoid importing BaseAgent.
    """
    try:
        await log_conversation(
            task_id, role, content, model, agent_name, cost
        )
    except Exception as exc:
        logger.warning(f"[Task #{task_id}] log_conversation failed: {exc}")
