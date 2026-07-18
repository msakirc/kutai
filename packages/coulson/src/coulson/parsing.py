"""ReAct response parsing — DSL vocabulary decoder.

Pure functions. Decode an LLM's text response into a canonical action dict:
  {"action": "tool_call" | "multi_tool_call" | "final_answer" | "clarify"
            | "decompose" | "ask_agent",
   ...}

Three parse strategies tried in order:
  1. Direct JSON parse (fastest)
  2. Markdown ```json``` fence extraction
  3. Brace-depth scan for first top-level object

Plus alias normalization (tool/use_tool/execute → tool_call,
answer/respond/done → final_answer, etc.) and legacy format mapping.

Function-call (litellm tool_calls) responses convert to the same canonical shape.
"""
from __future__ import annotations

import json
import re

from src.tools import TOOL_REGISTRY


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


def unwrap_final_answer(content: str) -> str:
    """Extract the 'result' value from a final_answer JSON envelope.

    LLMs often wrap their response in ``{"action": "final_answer", "result": "..."}``
    inside a markdown code block. When the result string contains unescaped
    quotes, ``json.loads`` fails. This helper uses a regex to pull out the
    result value so the downstream artifact pipeline gets clean text instead of
    the raw envelope.

    Returns the extracted result text, or the original content unchanged.
    """
    if '"final_answer"' not in content and '"result"' not in content:
        return content

    # Strip markdown code fences so we work on the JSON body.
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        stripped = stripped.rsplit("```", 1)[0].strip()

    # Try clean JSON parse first — fastest path.
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "result" in obj:
            return obj["result"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Regex fallback: grab everything between "result": " and the last "
    # before the closing brace. The result field is always the longest
    # string value so we use a greedy match anchored to the key.
    m = re.search(
        r'"result"\s*:\s*"(.*)",?\s*(?:"memories"|"subtasks"|\})',
        stripped,
        re.DOTALL,
    )
    if m:
        raw = m.group(1)
        # Un-escape JSON string escapes that survived the regex.
        return raw.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')

    return content


def parse_action(content: str) -> dict | None:
    """Extract an action dict from the model's text.

    Pipeline:
      1. try json.loads (clean JSON, also handles fence-stripped)
      2. try every ```json``` fence block
      3. one brace-depth scan (JSON buried in prose)
      4. explicit failure → return None

    Strips ``<think>`` blocks (Qwen3/DeepSeek thinking models). Handles
    legacy action names via alias map and ``{"status": "complete", ...}``
    legacy format.

    Returns None when parsing fails — the caller is responsible for format
    retries or explicit failure handling.
    """
    cleaned = content.strip()

    # Strip <think>…</think> blocks (Qwen3/DeepSeek thinking models).
    # Also handle unclosed <think> (token limit hit mid-think) and
    # orphaned tags from models that ignore enable_thinking=false.
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"</?think>", "", cleaned).strip()

    # Try 1 — direct parse (strips leading fences too)
    parsed = _try_parse_json(cleaned)
    if parsed is not None:
        norm = _normalize_action(parsed)
        if norm is not None:
            return norm

    # Try 2 — every ```json … ``` block
    json_blocks = re.findall(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL
    )
    for block in json_blocks:
        parsed = _try_parse_json(block.strip())
        if parsed is not None:
            norm = _normalize_action(parsed)
            if norm is not None:
                return norm

    # Try 3 — brace-depth scan for first top-level object
    if "{" in cleaned:
        start = cleaned.index("{")
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(cleaned[start: i + 1])
                        if isinstance(parsed, dict):
                            norm = _normalize_action(parsed)
                            if norm is not None:
                                return norm
                    except json.JSONDecodeError:
                        pass
                    break

    # Explicit failure — no silent fallback to final_answer.
    # The caller must handle None (format retry or explicit fail).
    return None


def parse_function_call(tool_calls: list[dict]) -> dict | None:
    """Convert LiteLLM tool_calls into the canonical action dict.

    Returns a single tool_call for one tool, multi_tool_call for multiple
    concurrent tools, or a pseudo-action (final_answer / clarify).
    Returns None when nothing could be parsed.
    """
    if not tool_calls:
        return None

    first = tool_calls[0]
    first_name = first.get("name", "")
    first_args = first.get("arguments", {})

    # Pseudo-tools always take priority (checked on first call only)
    if first_name == "final_answer":
        return {
            "action": "final_answer",
            "result": first_args.get("result", ""),
            "memories": first_args.get("memories", {}),
        }
    if first_name == "clarify":
        return {
            "action": "clarify",
            "question": first_args.get("question", ""),
        }

    # Single tool call — backwards compatible
    if len(tool_calls) == 1:
        action = {
            "action": "tool_call",
            "tool": first_name,
            "args": first_args,
        }
        # Carry the truncated-arguments marker so the runtime surfaces a
        # 'resend, smaller' nudge instead of running the tool arg-less.
        _err = first.get("arguments_error")
        if _err:
            action["args_error"] = _err
        return action

    # Multiple → multi_tool_call (filter out pseudo-tools)
    tools = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})
        if name in ("final_answer", "clarify"):
            continue
        entry = {"tool": name, "args": args}
        _err = tc.get("arguments_error")
        if _err:
            entry["args_error"] = _err
        tools.append(entry)

    if len(tools) == 1:
        single = {"action": "tool_call", "tool": tools[0]["tool"], "args": tools[0]["args"]}
        if tools[0].get("args_error"):
            single["args_error"] = tools[0]["args_error"]
        return single
    if not tools:
        return None
    return {"action": "multi_tool_call", "tools": tools}


# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────


def _try_parse_json(text: str) -> dict | None:
    """Return parsed dict or None."""
    try:
        stripped = text
        if stripped.startswith("```"):
            stripped = (
                stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
            )
            stripped = stripped.rsplit("```", 1)[0]
        obj = json.loads(stripped.strip())
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, IndexError):
        return None


def _normalize_action(parsed: dict) -> dict | None:
    """Map any recognised format to the canonical action dict.

    Returns None when the dict doesn't look like a valid action so the caller
    can fall through to the next parsing strategy.
    """
    action = parsed.get("action")

    # ── multi_tool_call passthrough ──
    if action == "multi_tool_call" and "tools" in parsed:
        return parsed

    # ── alias mapping for action field ──
    _aliases = {
        # → tool_call
        "tool":         "tool_call",
        "use_tool":     "tool_call",
        "execute":      "tool_call",
        "call":         "tool_call",
        "run":          "tool_call",
        "invoke":       "tool_call",
        # → final_answer
        "answer":       "final_answer",
        "respond":      "final_answer",
        "response":     "final_answer",
        "reply":        "final_answer",
        "complete":     "final_answer",
        "done":         "final_answer",
        "output":       "final_answer",
        "finish":       "final_answer",
        "result":       "final_answer",
        "final":        "final_answer",
        "summary":      "final_answer",
        # → clarify
        "ask":          "clarify",
        "question":     "clarify",
        "clarification": "clarify",
        # → ask_agent
        "delegate":     "ask_agent",
        "consult":      "ask_agent",
        "query_agent":  "ask_agent",
        # → decompose
        "plan":         "decompose",
        "decompose":    "decompose",
        "break_down":   "decompose",
    }
    if action in _aliases:
        action = _aliases[action]
        parsed["action"] = action

    # ── Tool name used as action, OR wrong action but "tool" key present ──
    if action and action not in (
        "tool_call", "multi_tool_call", "final_answer", "clarify", "decompose",
        "ask_agent",
        "think", "thinking", "reasoning", "analyze",
        "observation", "reflect", "consider",
    ):
        # Case 1: action IS a registered tool name
        if action in TOOL_REGISTRY:
            parsed["tool"] = action
            parsed["action"] = "tool_call"
            action = "tool_call"
            if "args" not in parsed:
                parsed["args"] = {
                    k: v for k, v in parsed.items()
                    if k not in ("action", "tool", "reasoning")
                }
        # Case 2: action is wrong, but "tool" key exists → clearly a tool call
        elif "tool" in parsed:
            parsed["action"] = "tool_call"
            action = "tool_call"
            if "args" not in parsed:
                parsed["args"] = {
                    k: v for k, v in parsed.items()
                    if k not in ("action", "tool", "reasoning")
                }

    # ── Treat thinking/reasoning as non-actions → return None
    # so parser falls through to final_answer fallback ──
    if action in ("think", "thinking", "reasoning", "analyze",
                  "observation", "reflect", "consider"):
        return None

    # ── infer action when key is missing ──
    if not action:
        if "tool" in parsed:
            parsed["action"] = "tool_call"
        elif any(k in parsed for k in (
            "result", "answer", "response", "text",
            "message", "output", "content", "reply",
        )):
            parsed["action"] = "final_answer"
            # Normalize the result key
            for key in ("answer", "response", "text", "message",
                        "output", "content", "reply"):
                if key in parsed and "result" not in parsed:
                    parsed["result"] = parsed.pop(key)
                    break
        elif "status" in parsed:
            # Legacy orchestrator format
            return {
                "action":              "final_answer",
                # Serialize the whole dict as JSON (not str(parsed) → Python
                # repr with single quotes), so downstream json.loads consumers
                # — e.g. verify_review_verdict — can parse a `{status, issues}`
                # reviewer verdict. ensure_ascii=False keeps Turkish intact.
                "result":              parsed.get("result")
                                       or json.dumps(parsed, ensure_ascii=False),
                "subtasks":            parsed.get("subtasks"),
                "plan_summary":        parsed.get("plan_summary"),
                "needs_clarification": parsed.get("clarification"),
                "memories":            parsed.get("memories", {}),
            }
        else:
            return None          # nothing recognisable

    # ── normalise flat tool args → nested "args" ──
    if parsed.get("action") == "tool_call" and "args" not in parsed:
        parsed["args"] = {
            k: v for k, v in parsed.items()
            if k not in ("action", "tool", "reasoning")
        }

    return parsed
