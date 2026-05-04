"""Context-window management — pure functions.

Estimates token counts, resolves context-window sizes, and trims or prunes
message lists to keep LLM calls within the model's context budget.

Public API
----------
count_tokens(messages, model) -> int
context_window_for(model, tier_or_reqs=None) -> int
trim_if_needed(messages, model, tier_or_reqs=None) -> list[dict]
prune_tool_results(messages, ctx_window, estimated_output_tokens, task_id="?") -> list[dict]

trim_if_needed internally calls count_tokens and context_window_for directly.
prune_tool_results uses a cheap char/3 estimate (no tokeniser overhead).
"""
from __future__ import annotations

import litellm as _litellm

from ..models.model_registry import get_registry
from fatih_hoca.requirements import ModelRequirements
from ..infra.logging_config import get_logger

logger = get_logger("runtime.window")


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


def count_tokens(messages: list[dict], model: str) -> int:
    """Estimate token count for a message list."""
    try:
        return _litellm.token_counter(model=model, messages=messages)
    except Exception:
        # Fallback: ~4 chars per token
        return sum(len(m.get("content", "")) for m in messages) // 4


def context_window_for(model: str, tier_or_reqs=None) -> int:
    """Return the context window size for a model."""
    try:
        info = _litellm.get_model_info(model=model)
        if info:
            ctx = info.get("max_input_tokens") or info.get("max_tokens")
            if ctx and ctx > 0:
                return ctx
    except Exception:
        pass

    # Try registry
    try:
        registry = get_registry()
        model_info = registry.find_by_litellm_name(model)
        if model_info:
            return model_info.context_length
    except Exception:
        pass

    # Difficulty-based fallback
    if isinstance(tier_or_reqs, ModelRequirements):
        diff = tier_or_reqs.difficulty
    elif isinstance(tier_or_reqs, str):
        diff = {"routing": 1, "cheap": 3, "code": 5,
                "medium": 6, "expensive": 8}.get(tier_or_reqs, 5)
    else:
        diff = 5

    if diff <= 2:
        return 4096
    elif diff <= 4:
        return 8192
    elif diff <= 6:
        return 16384
    else:
        return 32768


def trim_if_needed(
    messages: list[dict], model: str, tier_or_reqs=None,
) -> list[dict]:
    """
    If the conversation exceeds 80% of context, compress older exchanges.
    Accepts tier string or ModelRequirements for compat.
    """
    ctx_window = context_window_for(model, tier_or_reqs)
    threshold = int(ctx_window * 0.80)

    current = count_tokens(messages, model)
    if current <= threshold:
        return messages

    logger.warning(
        f"Context at {current}/{ctx_window} tokens "
        f"({current * 100 // ctx_window}%), compressing…"
    )

    if len(messages) <= 4:
        return messages

    head = messages[:2]
    tail = messages[-2:]
    middle = list(messages[2:-2])

    if not middle:
        return messages

    # Phase 1: truncate long content
    for i, msg in enumerate(middle):
        content = msg.get("content", "")
        if len(content) > 300:
            middle[i] = {
                "role": msg["role"],
                "content": content[:150] + "\n\n… [compressed] …\n\n" + content[-100:],
            }

    result = head + middle + tail
    if count_tokens(result, model) <= threshold:
        final = count_tokens(result, model)
        logger.info(f"Context compressed (truncate): {current} → {final} tokens")
        return result

    # Phase 2: drop oldest pairs
    while len(middle) >= 2:
        if count_tokens(head + middle + tail, model) <= threshold:
            break
        middle = middle[2:]

    summary = {
        "role": "user",
        "content": (
            "[Earlier tool interactions were removed to fit the context "
            "window. Focus on the latest results and the original task.]"
        ),
    }
    result = head + [summary] + middle + tail
    final = count_tokens(result, model)
    logger.info(f"Context compressed (drop): {current} → {final} tokens")

    # Inject context budget warning so the agent knows to wrap up
    remaining_pct = max(0, 100 - int(final * 100 / ctx_window))
    result.append({
        "role": "user",
        "content": (
            f"[System: Context {remaining_pct}% remaining. "
            f"Earlier messages were compressed. "
            f"Focus on completing the task efficiently.]"
        ),
    })

    return result


def prune_tool_results(
    messages: list[dict],
    ctx_window: int,
    estimated_output_tokens: int,
    task_id: int | str = "?",
) -> list[dict]:
    """
    Cheap char/3 prompt-size guard that runs BEFORE the heavier
    trim_if_needed compression.  If the current prompt estimate exceeds
    ``ctx_window - estimated_output_tokens``, drop the oldest *tool-result*
    exchanges (assistant+user pair where the user message is a tool result)
    until we fit.  System prompt (index 0), initial user context (index 1),
    and the most recent exchange (last 2 messages) are always preserved.

    Logs ``[Task #X] Pruned N oldest tool results to fit context`` once
    per call when pruning happened.
    """
    if len(messages) <= 4:
        return messages

    budget = ctx_window - max(0, estimated_output_tokens)
    if budget <= 0:
        return messages

    def _estimate(msgs: list[dict]) -> int:
        total = 0
        for m in msgs:
            c = m.get("content")
            if isinstance(c, str):
                total += len(c)
            elif c is not None:
                total += len(str(c))
        return total // 3

    if _estimate(messages) <= budget:
        return messages

    # Identify tool-result user messages: role=user at index >= 2 whose
    # preceding message is role=assistant.  We drop them paired with
    # that preceding assistant turn (oldest first).  Preserve the last
    # two messages (most recent exchange).
    preserve_tail_from = len(messages) - 2

    pruned = list(messages)
    dropped = 0
    # Walk from oldest to newest, skipping head (0,1) and tail.
    i = 2
    while _estimate(pruned) > budget and i < len(pruned) - 2:
        prev = pruned[i - 1]
        cur = pruned[i]
        if (
            cur.get("role") == "user"
            and prev.get("role") == "assistant"
            and (i - 1) >= 2                       # don't touch head
            and i < preserve_tail_from              # don't touch tail
        ):
            # Drop the assistant+tool-result pair
            del pruned[i - 1:i + 1]
            preserve_tail_from -= 2
            dropped += 1
            # i now points to what was i+1; re-check from same spot
            i = max(2, i - 1)
            continue
        i += 1

    if dropped:
        logger.warning(
            f"[Task #{task_id}] Pruned {dropped} oldest tool result"
            f"{'s' if dropped != 1 else ''} to fit context "
            f"(budget {budget} tokens, estimate before="
            f"{_estimate(messages)}, after={_estimate(pruned)})"
        )
    return pruned
