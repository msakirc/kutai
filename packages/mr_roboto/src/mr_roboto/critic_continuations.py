"""CPS SP6 — surface-B critic gate resume handlers (git_commit / notify_user).

Two-pass self-park: pass-1 enqueues an admitted critic child + parks the gated
task; these handlers stamp the verdict back into the gated task and re-pend it
(verdict_done) or fail it closed (verdict_err). The gated executor's pass-2
reads context['critic_verdict'] and calls the LLM-free confirm_gate.
"""
from __future__ import annotations

import json

from general_beckman import update_task
from src.infra.db import get_task
from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.critic_continuations")


def _extract_content(result: dict) -> str:
    result = result or {}
    inner = result.get("result")
    if isinstance(inner, dict):
        content = inner.get("content", "")
    elif inner is not None:
        content = inner
    else:
        content = result.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content or "")


def _parse_ctx(task: dict) -> dict:
    raw = (task or {}).get("context") or "{}"
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


async def _persist_critic_log(state: dict, verdict: str, reasons: list) -> None:
    from mr_roboto.critic_gate import _persist
    await _persist(
        state.get("mission_id"),
        str(state.get("action_name") or "unknown"),
        verdict,
        list(reasons or []),
        str(state.get("payload_hash") or ""),
    )


async def _verdict_done(child_task_id: int, result: dict, state: dict) -> None:
    """Critic child completed → stamp verdict into the gated task + re-pend.
    Uses parse_verdict_strict (fail-closed on garbage)."""
    from mr_roboto.critic_gate import parse_verdict_strict
    gated_id = state.get("gated_task_id")
    parsed = parse_verdict_strict(_extract_content(result))
    await _persist_critic_log(state, parsed["verdict"], parsed.get("reasons") or [])
    gated = await get_task(int(gated_id)) if gated_id is not None else None
    if gated is None:
        logger.warning("critic verdict_done: gated task missing", gated_id=gated_id)
        return
    ctx = _parse_ctx(gated)
    ctx["critic_verdict"] = {
        "verdict": parsed["verdict"],
        "reasons": parsed.get("reasons") or [],
        "payload_hash": str(state.get("payload_hash") or ""),
    }
    await update_task(int(gated_id), status="pending", context=json.dumps(ctx))


async def _verdict_err(child_task_id: int, result: dict, state: dict) -> None:
    """Critic child failed terminally → FAIL the gated task CLOSED (blocked)."""
    gated_id = state.get("gated_task_id")
    action = str(state.get("action_name") or "action")
    err = (result or {}).get("error", "unknown")
    await _persist_critic_log(state, "veto", [f"producer error: {str(err)[:120]}"])
    if gated_id is None:
        return
    await update_task(
        int(gated_id), status="failed",
        error=f"critic verdict unavailable ({str(err)[:80]}) — {action} blocked (fail-closed)",
    )


def register_continuations() -> None:
    try:
        from general_beckman.continuations import register_resume
        register_resume("mr_roboto.critic.verdict_done", _verdict_done)
        register_resume("mr_roboto.critic.verdict_err", _verdict_err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("critic continuation registration deferred", error=str(exc))


register_continuations()
