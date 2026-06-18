"""Response parsing — extract content, tool_calls, thinking from litellm responses."""
from __future__ import annotations
import json
import re


def _get_logger():
    try:
        from src.infra.logging_config import get_logger
        return get_logger("hallederiz_kadir.response")
    except Exception:
        import logging
        return logging.getLogger("hallederiz_kadir.response")


def parse_response(response, model_name: str, is_local: bool, is_thinking: bool) -> dict:
    """Parse a litellm ModelResponse into a flat dict.

    Args:
        response: litellm.ModelResponse object
        model_name: human-readable model name (for logging)
        is_local: whether this was a local model call
        is_thinking: whether thinking was requested for this call

    Returns:
        dict with keys: content, tool_calls, thinking, usage, cost
    """
    msg = response.choices[0].message

    # ── Thinking extraction ──
    thinking = _extract_thinking(msg) if is_thinking else None

    # ── Rescue reasoning_content when content is empty ──
    if not is_thinking and not (msg.content or "").strip():
        rc = getattr(msg, "reasoning_content", None) or ""
        if rc.strip():
            msg.content = re.sub(r"</?think>", "", rc).strip()

    # ── Strip think tags when thinking wasn't requested ──
    if not is_thinking and msg.content and "<think>" in msg.content:
        original = msg.content
        msg.content = re.sub(r"<think>.*?</think>", "", msg.content, flags=re.DOTALL)
        msg.content = re.sub(r"<think>.*", "", msg.content, flags=re.DOTALL)
        msg.content = re.sub(r"</?think>", "", msg.content)
        msg.content = msg.content.strip()
        if not msg.content:
            msg.content = re.sub(r"</?think>", "", original).strip()

    # ── Tool calls ──
    tool_calls = None
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_calls = []
        for tc in msg.tool_calls:
            fn = tc.function
            raw_args = fn.arguments
            args = {}
            args_error = None
            if raw_args:
                try:
                    args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    # NON-empty but unparseable arguments = the function-call
                    # stream was cut mid-JSON (provider truncation on a large
                    # payload, e.g. a big write_file `content`). Do NOT silently
                    # drop to {} and run the tool arg-less — that surfaces as a
                    # misleading "argument error / tool unavailable" loop and
                    # DLQs (mission 81 ADR 4.1). Keep {} for execution safety
                    # but record the truncation so the runtime can re-prompt.
                    _n = len(raw_args)
                    args_error = (
                        f"arguments were not valid JSON ({_n} chars received) "
                        f"— the tool call was truncated mid-stream, likely "
                        f"because the payload was too large for one response"
                    )
                    _get_logger().warning(
                        "[%s] tool_call '%s' arguments unparseable (%d chars) "
                        "— truncated mid-stream; surfacing as retry nudge: %.200s",
                        model_name, getattr(fn, "name", "?"), _n, raw_args,
                    )
            entry = {"id": tc.id, "name": fn.name, "arguments": args}
            if args_error:
                entry["arguments_error"] = args_error
            tool_calls.append(entry)

    # ── Usage ──
    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens or 0,
            "completion_tokens": response.usage.completion_tokens or 0,
        }

    # ── Cost ──
    cost = 0.0
    if not is_local:
        try:
            import litellm
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

    return {
        "content": msg.content or "",
        "tool_calls": tool_calls,
        "thinking": thinking,
        "usage": usage,
        "cost": cost,
        "finish_reason": getattr(response.choices[0], "finish_reason", None),
    }


def _extract_thinking(msg) -> str | None:
    """Extract thinking content from a model response message."""
    if hasattr(msg, "thinking") and msg.thinking:
        return msg.thinking
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        return msg.reasoning_content
    content = msg.content or ""
    match = re.search(
        r"<(?:thinking|think)>(.*?)</(?:thinking|think)>",
        content, re.DOTALL,
    )
    return match.group(1).strip() if match else None
