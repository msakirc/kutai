# agents/base.py
"""
Base agent with iterative ReAct loop:
  Think → Act (tool or respond) → Observe → Think again
"""
from __future__ import annotations

import hashlib
import json
import logging
import re

from router import call_model, classify_task, MODEL_TIERS, grade_response, check_cost_budget
from db import (
    log_conversation,
    store_memory,
    recall_memory,
    get_completed_dependency_results,
    save_task_checkpoint,
    load_task_checkpoint,
    clear_task_checkpoint,
    record_model_call,
    update_task,
    record_cost,
)
from tools import TOOL_REGISTRY, TOOL_SCHEMAS, get_tool_descriptions, execute_tool
from config import MAX_AGENT_ITERATIONS, AGENT_TIER_MAP, MAX_TOOL_OUTPUT_LENGTH
from models import validate_action, validate_tool_args
import litellm as _litellm

logger = logging.getLogger(__name__)


# Tools whose execution has side effects and should not be re-run on retry.
# Read-only tools (file_tree, read_file, git_log, etc.) are always re-executed.
SIDE_EFFECT_TOOLS: frozenset[str] = frozenset({
    "shell", "shell_stdin", "shell_sequential",
    "write_file", "edit_file", "lint",
    "verify_deps", "run_code",
    "git_init", "git_commit", "git_branch", "git_rollback",
})

# Max JSON format-correction retries before falling through to final_answer.
MAX_FORMAT_RETRIES: int = 2

# Mid-task escalation: after this many iterations with tool failures,
# escalate to the next tier up.
ESCALATION_THRESHOLD: int = 3

# Tier escalation order (low → high)
TIER_ESCALATION_ORDER: list[str] = ["cheap", "code", "medium", "expensive"]

# Pre-build tool schema lookup by name for O(1) access during arg validation.
_TOOL_SCHEMAS_BY_NAME: dict[str, dict] = {}
for _ts in TOOL_SCHEMAS:
    _fn = _ts.get("function", {})
    _ts_name = _fn.get("name")
    if _ts_name:
        _TOOL_SCHEMAS_BY_NAME[_ts_name] = _fn.get("parameters", {})
del _ts, _fn, _ts_name


class BaseAgent:
    """
    Base agent implementing a multi-turn ReAct loop.

    Subclasses override ``get_system_prompt`` and optionally set
    ``allowed_tools``, ``max_iterations``, ``min_tier``, etc.
    """

    name: str = "base"
    description: str = "General-purpose agent"
    default_tier: str = "cheap"
    min_tier: str = "cheap"

    # None → all tools allowed;  [] → no tools;  ["x","y"] → only those
    allowed_tools: list[str] | None = None

    # 1 = one-shot, >1 = iterative.  Capped by MAX_AGENT_ITERATIONS.
    max_iterations: int = MAX_AGENT_ITERATIONS

    can_create_subtasks: bool = False

    # ------------------------------------------------------------------ #
    #  System prompt — override in subclasses                             #
    # ------------------------------------------------------------------ #
    def get_system_prompt(self, task: dict) -> str:
        """
        Return the base system prompt.  Override in every concrete agent.
        """
        return (
            f"You are a helpful AI assistant named '{self.name}'.\n"
            f"Complete the given task thoroughly and accurately."
        )

    # ------------------------------------------------------------------ #
    #  Tool-description block                                             #
    # ------------------------------------------------------------------ #
    def _get_available_tools_prompt(self) -> str:
        """Build the tools section appended to the system prompt."""
        if self.allowed_tools is not None and not self.allowed_tools:
            return ""                       # explicitly empty → no tools

        # Build {name: description} from TOOL_REGISTRY directly,
        # since get_tool_descriptions() returns a formatted string, not a dict.
        if self.allowed_tools is not None:
            descs = {
                name: info["description"]
                for name, info in TOOL_REGISTRY.items()
                if name in self.allowed_tools
            }
        else:
            descs = {
                name: info["description"]
                for name, info in TOOL_REGISTRY.items()
            }

        if not descs:
            return ""

        lines = [
            "## Available Tools",
            "",
            "To use a tool, respond with ONLY a JSON block:",
            "```json",
            "{",
            '  "action": "tool_call",',
            '  "tool": "<tool_name>",',
            '  "args": { ... }',
            "}",
            "```",
            "",
            "When you have your FINAL answer, respond with:",
            "```json",
            "{",
            '  "action": "final_answer",',
            '  "result": "<your complete answer here>",',
            '  "memories": {"key": "value"}  // optional',
            "}",
            "```",
            "",
            "### Tools:",
        ]

        for tool_name, tool_desc in descs.items():
            lines.append(f"  • **{tool_name}**: {tool_desc}")
            info = TOOL_REGISTRY.get(tool_name)
            if info and "example" in info:
                lines.append(f"    Example: {info['example']}")

        lines += [
            "",
            "### IMPORTANT RULES:",
            "- Use ONE action per response.",
            "- After using a tool you will see the result and can act again.",
            "- Always inspect the workspace (file_tree) before writing code.",
            "- After writing code, ALWAYS run it to verify it works.",
            "- If you hit an error, read it carefully and fix the code.",
            f"- You have up to {self.max_iterations} iterations — don't waste them.",
            "- When done you MUST respond with the `final_answer` action.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Full system prompt assembly                                        #
    # ------------------------------------------------------------------ #
    def _build_full_system_prompt(self, task: dict) -> str:
        parts = [self.get_system_prompt(task)]

        tools_block = self._get_available_tools_prompt()
        if tools_block:
            parts.append(tools_block)

        if self.max_iterations > 1:
            parts.append(
                f"You have up to {self.max_iterations} iterations. "
                f"Use tools to build, test, and fix. "
                f"Only provide your final_answer when you're truly done."
            )

        return "\n\n".join(parts)

    def _is_action_task(self, task: dict) -> bool:
        """
        Heuristically detect whether a task requires real execution (tools)
        vs. pure text generation (answering questions, writing prose).

        Used by the hallucination guard to catch models that claim to have
        performed actions without actually calling any tools.
        """
        text = (
            f"{task.get('title', '')} {task.get('description', '')}"
        ).lower().strip()

        # ── Questions are almost never action tasks ──
        question_starts = [
            "what ", "who ", "why ", "when ", "where ",
            "how does ", "how is ", "how do ",
            "explain ", "describe ", "summarize ",
            "what's ", "what is ", "do you ", "can you tell",
            "is there ", "are there ", "which ",
        ]
        if any(text.startswith(q) for q in question_starts):
            return False

        # ── Strong action verbs: almost always need tools ──
        strong_verbs = [
            "fetch", "download", "install", "deploy", "execute",
            "run ", "run:", "clone", "pull ", "push ", "start ",
            "stop ", "restart", "compile", "test ", "debug",
            "setup", "set up", "configure", "scan", "scrape",
            "crawl", "ping", "ssh ", "curl ", "grep ", "find ",
            "launch", "migrate", "import ", "export ",
        ]
        if any(v in text for v in strong_verbs):
            return True

        # ── Contextual verbs that need tools ONLY with technical targets ──
        context_verbs = [
            "list", "create", "build", "write", "read",
            "check", "update", "delete", "remove", "add ",
            "modify", "edit", "open", "search", "look up",
            "analyze", "monitor", "show",
        ]
        tech_targets = [
            "file", "folder", "directory", "repo", "repos",
            "repository", "repositories", "server", "database",
            "api", "endpoint", "package", "container", "docker",
            "service", "script", "code", "project", "workspace",
            "branch", "commit", "log ", "logs", "port", "process",
            "module", "dependency", "dependencies", "config",
        ]

        has_verb = any(v in text for v in context_verbs)
        has_target = any(t in text for t in tech_targets)

        return has_verb and has_target

    # ------------------------------------------------------------------ #
    #  Tier helpers                                                       #
    # ------------------------------------------------------------------ #
    def _enforce_min_tier(self, requested_tier: str) -> str:
        tier_order = {"cheap": 0, "medium": 1, "expensive": 2}
        if tier_order.get(requested_tier, 0) < tier_order.get(self.min_tier, 0):
            logger.info(
                f"Agent '{self.name}' requires min tier '{self.min_tier}', "
                f"upgrading from '{requested_tier}'"
            )
            return self.min_tier
        return requested_tier

    @staticmethod
    def _escalate_tier(current_tier: str) -> str | None:
        """Return the next tier up, or None if already at highest."""
        try:
            idx = TIER_ESCALATION_ORDER.index(current_tier)
        except ValueError:
            return None
        if idx < len(TIER_ESCALATION_ORDER) - 1:
            return TIER_ESCALATION_ORDER[idx + 1]
        return None

    # ------------------------------------------------------------------ #
    #  Context builder (DB + inline fallback)                             #
    # ------------------------------------------------------------------ #
    async def _build_context(self, task: dict) -> str:
        """
        Assemble the user message with task info, dependency results,
        inline prior-step data, and recalled memories.
        """
        parts: list[str] = []

        # ── Task description ──
        parts.append(
            f"## Task\n**{task.get('title', 'Untitled')}**\n"
            f"{task.get('description', '')}"
        )

        # ── Parse task.context (may be str or dict) ──
        task_context = task.get("context")
        if isinstance(task_context, str):
            try:
                task_context = json.loads(task_context)
            except (json.JSONDecodeError, TypeError):
                task_context = {}
        if not isinstance(task_context, dict):
            task_context = {}

        # Specific known keys
        if "workspace_snapshot" in task_context:
            parts.append(
                f"## Current Workspace State\n{task_context['workspace_snapshot']}"
            )
        if "tool_result" in task_context:
            parts.append(
                f"## Prior Tool Result\n{task_context['tool_result']}"
            )

        # Remaining context (exclude internal / already-handled keys)
        _skip = {"workspace_snapshot", "tool_result", "prior_steps", "tool_depth", "recent_conversation"}
        extra = {k: v for k, v in task_context.items() if k not in _skip}
        if extra:
            parts.append(
                f"## Additional Context\n{json.dumps(extra, indent=2)}"
            )

        # ── Dependency results from DB ──
        depends_on = task.get("depends_on")
        if isinstance(depends_on, str):
            try:
                depends_on = json.loads(depends_on)
            except (json.JSONDecodeError, TypeError):
                depends_on = []
        if depends_on:
            try:
                dep_results = await get_completed_dependency_results(depends_on)
            except Exception as exc:
                logger.warning(f"Failed to fetch dependency results: {exc}")
                dep_results = {}
            if dep_results:
                parts.append("## Results from Previous Steps")
                for dep_id, dep in dep_results.items():
                    text = dep.get("result") or "(no result)"
                    if len(text) > 4000:
                        text = text[:4000] + "\n... (truncated)"
                    parts.append(
                        f"### Step #{dep_id}: "
                        f"{dep.get('title', 'Unknown')}\n{text}"
                    )

        # ── Inline prior_steps (orchestrator-injected fallback) ──
        if "prior_steps" in task_context:
            parts.append("## Results from Prior Steps (Inline)")
            for step in task_context["prior_steps"]:
                result = step.get("result", "")
                if len(result) > 2000:
                    result = result[:2000] + "\n... [truncated]"
                parts.append(
                    f"### Step: {step.get('title', 'Unknown')} "
                    f"(Status: {step.get('status', '?')})\n{result}"
                )

        # ── Recalled memories ──
        goal_id = task.get("goal_id")

        # ── Recent conversation (for follow-up understanding) ──
        if "recent_conversation" in task_context:
            parts.append("## Recent Conversation (for context)")
            for entry in task_context["recent_conversation"]:
                user_q = entry.get("user_asked", "?")
                result = entry.get("result", "")
                if len(result) > 600:
                    result = result[:600] + "... [truncated]"
                parts.append(
                    f"**User asked:** {user_q}\n**Result:** {result}\n"
                )
            parts.append(
                "_Use this context to understand follow-up references "
                "like 'list them', 'the names', 'do it again', etc._"
            )
        try:
            memories = await recall_memory(goal_id=goal_id, limit=15)
        except Exception as exc:
            logger.warning(f"Failed to recall memory: {exc}")
            memories = []
        if memories:
            parts.append("## Project Memory")
            for mem in memories:
                mem_value = mem.get('value', '')
                # Ensure mem_value is a string before slicing
                if not isinstance(mem_value, str):
                    mem_value = str(mem_value)
                parts.append(f"- **{mem.get('key', 'unknown')}**: {mem_value[:300]}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #  JSON parsing & normalisation                                       #
    # ------------------------------------------------------------------ #
    def _parse_agent_response(self, content: str) -> dict:
        """
        Extract an action dict from the model's text.

        Handles:
        - Clean JSON
        - JSON inside ```json``` fences (multiple blocks)
        - JSON buried in prose (brace-depth + regex fallback)
        - Legacy action names (``tool`` → ``tool_call``, etc.)
        - Legacy ``{"status": "complete", ...}`` format
        - Plain-text fallback → ``final_answer``
        """
        cleaned = content.strip()

        # Try 1 — direct parse (strips leading fences too)
        parsed = self._try_parse_json(cleaned)
        if parsed is not None:
            norm = self._normalize_action(parsed)
            if norm is not None:
                return norm

        # Try 2 — every ```json … ``` block
        json_blocks = re.findall(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL
        )
        for block in json_blocks:
            parsed = self._try_parse_json(block.strip())
            if parsed is not None:
                norm = self._normalize_action(parsed)
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
                            parsed = json.loads(cleaned[start : i + 1])
                            if isinstance(parsed, dict):
                                norm = self._normalize_action(parsed)
                                if norm is not None:
                                    return norm
                        except json.JSONDecodeError:
                            pass
                        break

        # Try 4 — regex for possibly-nested one-liner objects
        brace_match = re.search(
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned
        )
        if brace_match:
            try:
                parsed = json.loads(brace_match.group())
                if isinstance(parsed, dict):
                    norm = self._normalize_action(parsed)
                    if norm is not None:
                        return norm
            except json.JSONDecodeError:
                pass

        # Fallback — entire response is the answer
        return {"action": "final_answer", "result": content}

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        """Return parsed dict or ``None``."""
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

    @staticmethod
    def _normalize_action(parsed: dict) -> dict | None:
        """
        Map any recognised format to the canonical action dict.

        Returns ``None`` when the dict doesn't look like a valid action so
        the caller can fall through to the next parsing strategy.
        """
        action = parsed.get("action")

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
            "clarification":"clarify",
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
            "tool_call", "final_answer", "clarify", "decompose",
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
                    "action":               "final_answer",
                    "result":               parsed.get("result", str(parsed)),
                    "subtasks":             parsed.get("subtasks"),
                    "plan_summary":         parsed.get("plan_summary"),
                    "needs_clarification":  parsed.get("clarification"),
                    "memories":             parsed.get("memories", {}),
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

    # ------------------------------------------------------------------ #
    #  Context window management                                          #
    # ------------------------------------------------------------------ #
    def _count_tokens(self, messages: list[dict], model: str) -> int:
        """Estimate token count for a message list."""
        try:
            return _litellm.token_counter(model=model, messages=messages)
        except Exception:
            # Fallback: ~4 chars per token
            return sum(len(m.get("content", "")) for m in messages) // 4

    def _get_context_window(self, model: str, tier: str) -> int:
        """Return the context window size (input tokens) for a model."""
        try:
            info = _litellm.get_model_info(model=model)
            if info:
                ctx = info.get("max_input_tokens") or info.get("max_tokens")
                if ctx and ctx > 0:
                    return ctx
        except Exception:
            pass
        # Conservative fallbacks per tier
        return {
            "routing": 4096,
            "cheap": 8192,
            "code": 16384,
            "medium": 16384,
            "expensive": 32768,
        }.get(tier, 8192)

    def _trim_messages_if_needed(
        self, messages: list[dict], model: str, tier: str,
    ) -> list[dict]:
        """
        If the conversation exceeds 80 % of the model's context window,
        progressively compress older tool exchanges:
          Phase 1 — truncate long middle messages
          Phase 2 — drop oldest assistant/user pairs entirely
        """
        ctx_window = self._get_context_window(model, tier)
        threshold = int(ctx_window * 0.80)

        current = self._count_tokens(messages, model)
        if current <= threshold:
            return messages

        logger.warning(
            f"Context at {current}/{ctx_window} tokens "
            f"({current * 100 // ctx_window}%%), compressing…"
        )

        # Need at minimum: system + initial user + latest 2
        if len(messages) <= 4:
            return messages

        head = messages[:2]          # system prompt + task context
        tail = messages[-2:]         # latest exchange
        middle = list(messages[2:-2])

        if not middle:
            return messages

        # ── Phase 1: truncate long content in middle messages ──
        for i, msg in enumerate(middle):
            content = msg.get("content", "")
            if len(content) > 300:
                middle[i] = {
                    "role": msg["role"],
                    "content": (
                        content[:150]
                        + "\n\n… [compressed] …\n\n"
                        + content[-100:]
                    ),
                }

        result = head + middle + tail
        if self._count_tokens(result, model) <= threshold:
            final = self._count_tokens(result, model)
            logger.info(f"Context compressed (truncate): {current} → {final} tokens")
            return result

        # ── Phase 2: drop oldest pairs from middle ──
        while len(middle) >= 2:
            if self._count_tokens(head + middle + tail, model) <= threshold:
                break
            middle = middle[2:]      # drop one assistant + user pair

        summary = {
            "role": "user",
            "content": (
                "[Earlier tool interactions were removed to fit the context "
                "window. Focus on the latest results and the original task.]"
            ),
        }
        result = head + [summary] + middle + tail

        final = self._count_tokens(result, model)
        logger.info(f"Context compressed (drop): {current} → {final} tokens")
        return result

    # ------------------------------------------------------------------ #
    #  Function calling support                                            #
    # ------------------------------------------------------------------ #
    def _build_litellm_tools(self) -> list[dict] | None:
        """Build filtered tool schemas for LiteLLM function calling."""
        if self.allowed_tools is not None and not self.allowed_tools:
            return None  # explicitly no tools

        if self.allowed_tools is not None:
            allowed = set(self.allowed_tools) | {"final_answer", "clarify"}
            return [
                s for s in TOOL_SCHEMAS
                if s["function"]["name"] in allowed
            ]
        return list(TOOL_SCHEMAS)

    @staticmethod
    def _parse_function_call_response(tool_calls: list[dict]) -> dict | None:
        """
        Convert LiteLLM tool_calls into the canonical action dict.

        Returns the first valid action found, or None if none could
        be parsed.
        """
        if not tool_calls:
            return None

        tc = tool_calls[0]  # use the first tool call
        name = tc.get("name", "")
        args = tc.get("arguments", {})

        # Pseudo-tool: final_answer
        if name == "final_answer":
            return {
                "action": "final_answer",
                "result": args.get("result", ""),
                "memories": args.get("memories", {}),
            }

        # Pseudo-tool: clarify
        if name == "clarify":
            return {
                "action": "clarify",
                "question": args.get("question", ""),
            }

        # Real tool call
        return {
            "action": "tool_call",
            "tool": name,
            "args": args,
        }

    # ------------------------------------------------------------------ #
    #  Output validation                                                   #
    # ------------------------------------------------------------------ #
    def _validate_response(self, result: str, task: dict) -> str | None:
        """
        Validate a final_answer result.  Returns an error string if
        the response is invalid, or None if it passes.
        """
        if not result or not result.strip():
            return "Your response was empty. Please provide a substantive answer."

        stripped = result.strip()

        # For non-trivial tasks, require > 20 chars
        title = task.get("title", "").lower()
        trivial_keywords = ["list", "ls", "status", "count", "version", "ping"]
        is_trivial = any(kw in title for kw in trivial_keywords)

        if not is_trivial and len(stripped) < 20:
            return (
                "Your response seems too short for this task. "
                "Please provide a more complete answer."
            )

        # Check for refusal / error-only patterns
        refusal_patterns = [
            "i cannot", "i can't", "i'm unable", "as an ai",
            "i don't have access", "i am not able",
        ]
        lower = stripped.lower()
        if any(p in lower for p in refusal_patterns) and len(stripped) < 100:
            return (
                "Your response appears to be a refusal. "
                "Try a different approach or use the available tools."
            )

        return None  # validation passed

    # ------------------------------------------------------------------ #
    #  Main execution loop                                                #
    # ------------------------------------------------------------------ #
    async def execute(self, task: dict) -> dict:
        """
        ReAct loop:  Think → Act → Observe → repeat → final_answer.
        """
        task_id = task.get("id", "?")
        goal_id = task.get("goal_id")

        # ── resolve tier ──
        tier = task.get("tier", self.default_tier)
        if tier == "auto":
            try:
                classification = await classify_task(
                    task.get("title", ""), task.get("description", "")
                )
                tier = classification.get("tier", self.default_tier)
            except Exception:
                tier = AGENT_TIER_MAP.get(self.name, self.default_tier)
        tier = self._enforce_min_tier(tier)

        # ── attempt checkpoint recovery ──
        start_iteration = 0
        checkpoint = None
        if task_id != "?":
            try:
                checkpoint = await load_task_checkpoint(task_id)
            except Exception as exc:
                logger.warning(
                    f"[Task #{task_id}] Checkpoint load failed: {exc}"
                )

        if checkpoint:
            # Restore state from checkpoint
            messages = checkpoint.get("messages", [])
            start_iteration = checkpoint.get("iteration", 0)
            total_cost = checkpoint.get("total_cost", 0.0)
            used_model = checkpoint.get("used_model", "unknown")
            tools_used = checkpoint.get("tools_used", False)
            validation_retried = checkpoint.get("validation_retried", False)
            format_retries = checkpoint.get("format_retries", 0)
            tier = checkpoint.get("tier", tier)
            completed_tool_ops: dict[str, str] = checkpoint.get(
                "completed_tool_ops", {}
            )
            logger.info(
                f"[Task #{task_id}] Resuming from checkpoint "
                f"(iteration {start_iteration}, "
                f"{len(messages)} messages, ${total_cost:.4f} spent, "
                f"{len(completed_tool_ops)} cached tool ops)"
            )
        else:
            # ── build messages from scratch ──
            system_prompt = self._build_full_system_prompt(task)
            context = await self._build_context(task)

            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": context},
            ]

            total_cost = 0.0
            used_model = "unknown"
            tools_used = False
            validation_retried = False
            format_retries = 0
            completed_tool_ops: dict[str, str] = {}

        # Track tool failures for mid-task escalation
        consecutive_tool_failures = 0
        escalated = False

        # ── iterative loop ──
        for iteration in range(start_iteration, self.max_iterations):
            logger.info(
                f"[Task #{task_id}] Agent '{self.name}' iteration "
                f"{iteration + 1}/{self.max_iterations}"
            )

            # Call LLM
            # ── Trim context if approaching window limit ──
            estimation_model = (
                used_model if used_model != "unknown"
                else MODEL_TIERS.get(tier, {}).get("model", "gpt-4o-mini")
            )
            messages = self._trim_messages_if_needed(
                messages, estimation_model, tier,
            )

            # ── Cost budget check (Phase 4) ──
            try:
                budget_status = await check_cost_budget()
                if not budget_status.get("ok", True):
                    logger.warning(
                        f"[Task #{task_id}] Budget exceeded: "
                        f"{budget_status.get('reason')}"
                    )
                    await self._clear_checkpoint_safe(task_id)
                    return {
                        "status":     "completed",
                        "result":     f"Task paused: {budget_status['reason']}",
                        "model":      used_model,
                        "cost":       total_cost,
                        "tier":       tier,
                        "iterations": iteration,
                    }
            except Exception:
                pass  # budget check failure shouldn't block work

            # Call LLM
            try:
                litellm_tools = self._build_litellm_tools()
                response = await call_model(
                    tier, messages, tools=litellm_tools,
                    agent_type=self.name,
                )
            except Exception as exc:
                logger.error(f"[Task #{task_id}] Model call failed: {exc}")
                return {
                    "status":     "completed",
                    "result":     f"Agent failed after {iteration} iteration(s): {exc}",
                    "model":      used_model,
                    "cost":       total_cost,
                    "tier":       tier,
                    "iterations": iteration,
                }

            content    = response.get("content", "")
            used_model = response.get("model", used_model)
            step_cost  = response.get("cost", 0)
            step_latency = response.get("latency", 0)
            total_cost += step_cost

            # ── Record model call stats (Phase 4) ──
            try:
                await record_model_call(
                    model=used_model,
                    agent_type=self.name,
                    success=True,
                    cost=step_cost,
                    latency=step_latency,
                )
            except Exception:
                pass  # never break the loop for stats

            # ── Record cost for budget tracking (Phase 4) ──
            if step_cost > 0:
                try:
                    await record_cost(step_cost)
                except Exception:
                    pass

            logger.debug(f"[Task #{task_id}] Raw response: {content[:200]}...")

            # Log assistant turn
            await self._safe_log(
                task_id, "assistant", content, used_model, step_cost
            )

            # Parse — try function calling first, then regex fallback
            fc_tool_calls = response.get("tool_calls")
            parsed = None
            if fc_tool_calls:
                parsed = self._parse_function_call_response(fc_tool_calls)
                if parsed:
                    logger.debug(
                        f"[Task #{task_id}] Parsed via function calling: "
                        f"{parsed.get('action')}"
                    )
            if parsed is None:
                parsed = self._parse_agent_response(content)

            # ── FORMAT RETRY ─────────────────────────────────────
            # If the model tried to produce JSON but it couldn't be
            # parsed (fell through to final_answer fallback) AND the
            # raw response contains braces (suggesting JSON intent),
            # send a one-shot "fix your JSON" prompt instead of
            # accepting garbled text as the answer.
            if (
                parsed.get("action") == "final_answer"
                and parsed.get("result") == content  # fallback path
                and "{" in content
                and format_retries < MAX_FORMAT_RETRIES
            ):
                format_retries += 1
                logger.warning(
                    f"[Task #{task_id}] JSON parse failed on response "
                    f"with braces — requesting format correction "
                    f"(retry {format_retries}/{MAX_FORMAT_RETRIES})"
                )
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response could not be parsed as valid JSON. "
                        "Please fix the formatting and try again.\n\n"
                        "You MUST respond with ONLY a valid JSON block:\n"
                        "```json\n"
                        '{"action": "tool_call", "tool": "...", '
                        '"args": {...}}\n'
                        "```\n"
                        "or:\n"
                        "```json\n"
                        '{"action": "final_answer", "result": "..."}\n'
                        "```\n\n"
                        "No text before or after the JSON block."
                    ),
                })
                await self._save_checkpoint(
                    task_id, iteration + 1, messages, total_cost,
                    used_model, tier, tools_used, validation_retried,
                    completed_tool_ops, format_retries,
                )
                continue
            # ── END FORMAT RETRY ─────────────────────────────────

            # ── PYDANTIC VALIDATION ──────────────────────────────
            # Validate the parsed action against its Pydantic model.
            # On failure, log the error but keep the original parsed
            # dict to avoid breaking the loop.
            try:
                parsed = validate_action(parsed)
            except ValueError as exc:
                logger.warning(
                    f"[Task #{task_id}] Action validation warning: {exc}"
                )
            # ── END PYDANTIC VALIDATION ──────────────────────────

            action_type = parsed.get("action", "final_answer")

            # ── HALLUCINATION GUARD ───────────────────────────────
            # Catch models that claim task completion without calling
            # a single tool on tasks that clearly require execution.
            has_tools = (
                self.allowed_tools is None or len(self.allowed_tools) > 0
            )
            if (
                action_type == "final_answer"
                and not tools_used
                and has_tools
                and self._is_action_task(task)
                and iteration < 2
            ):
                available = (
                    list(TOOL_REGISTRY.keys())[:6]
                    if self.allowed_tools is None
                    else self.allowed_tools[:6]
                )
                tool_list = ", ".join(available)
                task_title = task.get("title", "")

                logger.warning(
                    f"[Task #{task_id}] ⚠️ Hallucination guard: model "
                    f"returned final_answer on action task without any "
                    f"tool calls (iter {iteration})"
                )

                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        "STOP. You did NOT actually perform this task. "
                        "You just described what you would do, but nothing "
                        "was executed.\n\n"
                        f"Your task: {task_title}\n\n"
                        "You MUST call a tool to take real action. "
                        f"Available tools: {tool_list}\n\n"
                        "Example — to run a shell command:\n"
                        "```json\n"
                        '{"action": "tool_call", "tool": "shell", '
                        '"args": {"command": "ls -la"}}\n'
                        "```\n\n"
                        "Example — to search the web:\n"
                        "```json\n"
                        '{"action": "tool_call", "tool": "web_search", '
                        '"args": {"query": "your search here"}}\n'
                        "```\n\n"
                        "Respond with ONLY the JSON block. No explanation."
                    ),
                })

                await self._safe_log(
                    task_id, "system",
                    f"[hallucination_guard] Rejected premature final_answer, "
                    f"pushing model to use tools (iter {iteration})",
                    None, 0,
                )
                await self._save_checkpoint(
                    task_id, iteration + 1, messages, total_cost,
                    used_model, tier, tools_used, validation_retried,
                    completed_tool_ops, format_retries,
                )
                continue
            # ── END HALLUCINATION GUARD ───────────────────────────

            # ── FINAL ANSWER ──────────────────────────────────────────
            if action_type == "final_answer":
                result = parsed.get("result", content)

                # ── OUTPUT VALIDATION ──────────────────────────────
                if not validation_retried:
                    validation_error = self._validate_response(result, task)
                    if validation_error:
                        validation_retried = True
                        logger.warning(
                            f"[Task #{task_id}] ⚠️ Output validation failed: "
                            f"{validation_error}"
                        )
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"{validation_error}\n\n"
                                f"Please try again and provide a proper response."
                            ),
                        })
                        await self._safe_log(
                            task_id, "system",
                            f"[output_validation] {validation_error}",
                            None, 0,
                        )
                        await self._save_checkpoint(
                            task_id, iteration + 1, messages, total_cost,
                            used_model, tier, tools_used, validation_retried,
                            completed_tool_ops, format_retries,
                        )
                        continue
                # ── END OUTPUT VALIDATION ──────────────────────────
                # Persist memories
                raw_memories = parsed.get("memories", {})
                if raw_memories and isinstance(raw_memories, dict):
                    for key, value in raw_memories.items():
                        try:
                            await store_memory(
                                key, str(value),
                                category=self.name, goal_id=goal_id,
                            )
                        except Exception as exc:
                            logger.warning(
                                f"[Task #{task_id}] store_memory failed: {exc}"
                            )

                logger.info(
                    f"[Task #{task_id}] ✅ Agent answered after "
                    f"{iteration + 1} iteration(s)"
                )

                # Agent asked for subtask decomposition inside final_answer
                if parsed.get("subtasks"):
                    await self._clear_checkpoint_safe(task_id)
                    return {
                        "status":       "needs_subtasks",
                        "subtasks":     parsed["subtasks"],
                        "plan_summary": parsed.get("plan_summary", ""),
                        "model":        used_model,
                        "cost":         total_cost,
                        "tier":         tier,
                    }

                # Agent asked for clarification inside final_answer
                if parsed.get("needs_clarification"):
                    await self._clear_checkpoint_safe(task_id)
                    return {
                        "status":        "needs_clarification",
                        "clarification": parsed["needs_clarification"],
                        "model":         used_model,
                        "cost":          total_cost,
                        "tier":          tier,
                    }

                # ── Response grading (Phase 4) ──
                quality_score = None
                try:
                    quality_score = await grade_response(
                        task.get("title", ""),
                        task.get("description", ""),
                        result,
                    )
                    if quality_score is not None and task_id != "?":
                        await update_task(task_id, quality_score=quality_score)
                        # Update model stats with grade
                        await record_model_call(
                            model=used_model,
                            agent_type=self.name,
                            success=True,
                            grade=quality_score,
                        )
                except Exception as exc:
                    logger.debug(f"Response grading failed: {exc}")

                await self._clear_checkpoint_safe(task_id)
                return {
                    "status":        "completed",
                    "result":        result,
                    "model":         used_model,
                    "cost":          total_cost,
                    "tier":          tier,
                    "iterations":    iteration + 1,
                    "quality_score": quality_score,
                }

            # ── TOOL CALL ─────────────────────────────────────────────
            if action_type == "tool_call":
                tools_used = True
                tool_name = parsed.get("tool", "")
                tool_args = parsed.get("args", {})
                if not isinstance(tool_args, dict):
                    tool_args = {}

                # Validate access
                if (
                    self.allowed_tools is not None
                    and tool_name not in self.allowed_tools
                ):
                    tool_output = (
                        f"❌ Tool '{tool_name}' is not available to this "
                        f"agent. Allowed: {self.allowed_tools}"
                    )
                elif tool_name not in TOOL_REGISTRY:
                    tool_output = (
                        f"❌ Unknown tool '{tool_name}'. "
                        f"Available: {list(TOOL_REGISTRY.keys())}"
                    )
                else:
                    # ── Typed argument validation & coercion ──
                    arg_schema = _TOOL_SCHEMAS_BY_NAME.get(tool_name)
                    if arg_schema:
                        tool_args, arg_errors = validate_tool_args(
                            tool_name, tool_args, arg_schema,
                        )
                        if arg_errors:
                            err_msg = "; ".join(arg_errors)
                            logger.warning(
                                f"[Task #{task_id}] Tool arg validation: "
                                f"{err_msg}"
                            )
                            # Return error to LLM so it can fix the call
                            tool_output = (
                                f"❌ Argument error for tool '{tool_name}': "
                                f"{err_msg}\n\n"
                                f"Expected parameters: "
                                f"{json.dumps(arg_schema, indent=2)}"
                            )
                            messages.append(
                                {"role": "assistant", "content": content}
                            )
                            messages.append(
                                {"role": "user", "content": tool_output}
                            )
                            await self._safe_log(
                                task_id, "tool",
                                f"[{tool_name}] ARG_ERROR: {err_msg}",
                                None, 0,
                            )
                            await self._save_checkpoint(
                                task_id, iteration + 1, messages,
                                total_cost, used_model, tier,
                                tools_used, validation_retried,
                                completed_tool_ops, format_retries,
                            )
                            continue

                    # ── Idempotency check for side-effect tools ──
                    idem_key = self._tool_idempotency_key(
                        tool_name, tool_args,
                    )
                    cached = (
                        completed_tool_ops.get(idem_key)
                        if tool_name in SIDE_EFFECT_TOOLS
                        else None
                    )

                    if cached is not None:
                        tool_output = cached
                        logger.info(
                            f"[Task #{task_id}] ♻️ Idempotent skip: "
                            f"{tool_name} (cached result, "
                            f"{len(tool_output)} chars)"
                        )
                    else:
                        logger.info(
                            f"[Task #{task_id}] 🔧 {tool_name}("
                            f"{', '.join(f'{k}={repr(v)[:50]}' for k, v in tool_args.items())})"
                        )
                        try:
                            tool_output = await execute_tool(
                                tool_name, **tool_args,
                            )
                        except Exception as exc:
                            tool_output = f"❌ Tool execution error: {exc}"
                            logger.error(
                                f"[Task #{task_id}] Tool '{tool_name}' "
                                f"raised: {exc}"
                            )

                        # Record for idempotency (side-effect tools only)
                        if tool_name in SIDE_EFFECT_TOOLS:
                            completed_tool_ops[idem_key] = tool_output

                    # ── Log tool output to terminal for debugging ──
                    output_preview = tool_output[:500] if tool_output else "(empty)"
                    if tool_output and len(tool_output) > 500:
                        output_preview += f"\n... [{len(tool_output)} chars total]"
                    logger.info(
                        f"[Task #{task_id}] 📤 {tool_name} returned: "
                        f"{output_preview}"
                    )

                # Truncate
                if len(tool_output) > MAX_TOOL_OUTPUT_LENGTH:
                    tool_output = (
                        tool_output[:MAX_TOOL_OUTPUT_LENGTH]
                        + f"\n\n... [truncated — {len(tool_output)} chars total]"
                    )

                # Append turns
                                # Detect if the tool errored
                tool_failed = (
                    tool_output.startswith("❌")
                    or tool_output.startswith("🚫")
                    or "command not found" in tool_output
                    or "No such file" in tool_output
                    or "exit code" in tool_output
                    and "exit code 0" not in tool_output
                )

                # ── Mid-task escalation (Phase 4) ──
                if tool_failed:
                    consecutive_tool_failures += 1
                else:
                    consecutive_tool_failures = 0

                if (
                    not escalated
                    and consecutive_tool_failures >= ESCALATION_THRESHOLD
                    and iteration >= ESCALATION_THRESHOLD
                ):
                    next_tier = self._escalate_tier(tier)
                    if next_tier and next_tier in MODEL_TIERS:
                        logger.warning(
                            f"[Task #{task_id}] ⬆️ Escalating tier: "
                            f"'{tier}' → '{next_tier}' after "
                            f"{consecutive_tool_failures} consecutive "
                            f"tool failures"
                        )
                        tier = next_tier
                        escalated = True
                        await self._safe_log(
                            task_id, "system",
                            f"[escalation] Upgraded to tier '{tier}' "
                            f"after {consecutive_tool_failures} failures",
                            None, 0,
                        )

                if tool_failed:
                    recovery_guidance = (
                        f"## Tool Result (`{tool_name}`) — ERROR:\n\n"
                        f"```\n{tool_output}\n```\n\n"
                        f"The tool call failed. Try a DIFFERENT approach:\n"
                        f"- If a command wasn't found, use an alternative "
                        f"(e.g. `curl` instead of `gh`, "
                        f"`python3` instead of `python`)\n"
                        f"- If a file wasn't found, use `file_tree` to check "
                        f"what exists\n"
                        f"- If you're stuck, provide your best `final_answer` "
                        f"with what you know\n\n"
                        f"Respond with a JSON tool_call or final_answer. "
                        f"Iteration {iteration + 2}/{self.max_iterations}."
                    )
                else:
                    recovery_guidance = (
                        f"## Tool Result (`{tool_name}`):\n\n"
                        f"```\n{tool_output}\n```\n\n"
                        f"Continue working. Provide your `final_answer` when "
                        f"done, or call another tool. "
                        f"Iteration {iteration + 2}/{self.max_iterations}."
                    )

                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": recovery_guidance})

                await self._safe_log(
                    task_id, "tool",
                    f"[{tool_name}] {tool_output[:2000]}",
                    None, 0,
                )
                await self._save_checkpoint(
                    task_id, iteration + 1, messages, total_cost,
                    used_model, tier, tools_used, validation_retried,
                    completed_tool_ops, format_retries,
                )
                continue

            # ── CLARIFY ───────────────────────────────────────────────
            if action_type == "clarify":
                await self._clear_checkpoint_safe(task_id)
                return {
                    "status":        "needs_clarification",
                    "clarification": parsed.get("question", content),
                    "model":         used_model,
                    "cost":          total_cost,
                    "tier":          tier,
                }

            # ── DECOMPOSE ─────────────────────────────────────────────
            if action_type == "decompose":
                await self._clear_checkpoint_safe(task_id)
                return {
                    "status":       "needs_subtasks",
                    "subtasks":     parsed.get("subtasks", []),
                    "plan_summary": parsed.get("summary", ""),
                    "model":        used_model,
                    "cost":         total_cost,
                    "tier":         tier,
                }

            # ── UNKNOWN — nudge with concrete format examples ─────
            logger.warning(
                f"[Task #{task_id}] Unrecognized action "
                f"'{action_type}' on iteration {iteration + 1}. "
                f"Raw: {content[:200]}"
            )
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"ERROR: Unrecognized action '{action_type}'. "
                    f"You MUST use one of these exact formats:\n\n"
                    f"To call a tool:\n"
                    f"```json\n"
                    f'{{"action": "tool_call", "tool": "shell", '
                    f'"args": {{"command": "ls -la"}}}}\n'
                    f"```\n\n"
                    f"To give your final answer:\n"
                    f"```json\n"
                    f'{{"action": "final_answer", "result": '
                    f'"your complete answer here"}}\n'
                    f"```\n\n"
                    f"Respond with ONLY the JSON block. Nothing else."
                ),
            })
            await self._save_checkpoint(
                task_id, iteration + 1, messages, total_cost,
                used_model, tier, tools_used, validation_retried,
                completed_tool_ops, format_retries,
            )

        # ── exhausted iterations ──
        logger.warning(
            f"[Task #{task_id}] Agent '{self.name}' exhausted "
            f"{self.max_iterations} iterations"
        )
        await self._clear_checkpoint_safe(task_id)
        last = messages[-1].get("content", "") if messages else ""
        return {
            "status": "completed",
            "result": (
                f"[Completed after {self.max_iterations} iterations "
                f"without a final answer]\n\nLast context:\n{last[:3000]}"
            ),
            "model":      used_model,
            "cost":       total_cost,
            "tier":       tier,
            "iterations": self.max_iterations,
        }

    # ------------------------------------------------------------------ #
    #  Idempotency helpers                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tool_idempotency_key(tool_name: str, tool_args: dict) -> str:
        """Compute a short hash key for a tool call's identity.

        Used to skip re-execution of side-effect tools (write_file, shell,
        git_commit, etc.) when resuming from a checkpoint.
        """
        # Stable serialisation: sorted keys, no whitespace variance
        raw = f"{tool_name}|{json.dumps(tool_args, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ #
    #  Checkpointing helpers                                              #
    # ------------------------------------------------------------------ #
    async def _save_checkpoint(
        self,
        task_id,
        next_iteration: int,
        messages: list[dict],
        total_cost: float,
        used_model: str,
        tier: str,
        tools_used: bool,
        validation_retried: bool,
        completed_tool_ops: dict[str, str] | None = None,
        format_retries: int = 0,
    ) -> None:
        """Persist agent loop state so execution can resume after a crash."""
        if task_id == "?":
            return
        try:
            state = {
                "iteration": next_iteration,
                "messages": messages,
                "total_cost": total_cost,
                "used_model": used_model,
                "tier": tier,
                "tools_used": tools_used,
                "validation_retried": validation_retried,
                "format_retries": format_retries,
                "completed_tool_ops": completed_tool_ops or {},
            }
            await save_task_checkpoint(task_id, state)
            logger.debug(
                f"[Task #{task_id}] Checkpoint saved at iteration "
                f"{next_iteration}"
            )
        except Exception as exc:
            logger.warning(
                f"[Task #{task_id}] Checkpoint save failed: {exc}"
            )

    async def _clear_checkpoint_safe(self, task_id) -> None:
        """Clear checkpoint on successful completion — never raises."""
        if task_id == "?":
            return
        try:
            await clear_task_checkpoint(task_id)
        except Exception as exc:
            logger.warning(
                f"[Task #{task_id}] Checkpoint clear failed: {exc}"
            )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    async def _safe_log(
        self,
        task_id,
        role: str,
        content: str,
        model: str | None,
        cost: float,
    ) -> None:
        """Fire-and-forget conversation log — never breaks the loop."""
        try:
            await log_conversation(
                task_id, role, content, model, self.name, cost
            )
        except Exception as exc:
            logger.warning(f"[Task #{task_id}] log_conversation failed: {exc}")
