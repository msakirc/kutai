"""Response parsing — extract content, tool_calls, thinking from litellm responses."""
from __future__ import annotations
import json
import re


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
            try:
                args = json.loads(fn.arguments) if fn.arguments else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append({"id": tc.id, "name": fn.name, "arguments": args})

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
