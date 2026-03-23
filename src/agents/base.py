# agents/base.py
"""
Base agent with iterative ReAct loop:
  Think → Act (tool or respond) → Observe → Think again
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
from typing import Callable

from ..collaboration.blackboard import get_or_create_blackboard, \
    format_blackboard_for_prompt
from ..context.onboarding import get_project_profile_for_task, \
    format_project_profile
from ..memory.preferences import get_user_preferences, format_preferences
from ..memory.rag import retrieve_context
from ..models.model_registry import get_registry
from ..core.router import (
    ModelRequirements,
    call_model,
    grade_response,
    select_model,
)
from ..infra.db import (
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
from ..tools import TOOL_REGISTRY, TOOL_SCHEMAS, get_tool_descriptions, execute_tool
from ..app.config import MAX_AGENT_ITERATIONS, MAX_TOOL_OUTPUT_LENGTH
from ..models.models import validate_action, validate_tool_args, validate_task_output
from ..infra.logging_config import get_logger
import litellm as _litellm

logger = get_logger("agents.base")


# Tools whose execution has side effects and should not be re-run on retry.
# Read-only tools (file_tree, read_file, git_log, etc.) are always re-executed.
SIDE_EFFECT_TOOLS: frozenset[str] = frozenset({
    "shell", "shell_stdin", "shell_sequential",
    "write_file", "edit_file", "patch_file", "apply_diff", "lint",
    "verify_deps", "run_code",
    "git_init", "git_commit", "git_branch", "git_rollback",
})

# Phase 5.6: Read-only tools whose results can be cached within a single
# agent execution. Cache is invalidated when any SIDE_EFFECT_TOOL runs.
CACHEABLE_READ_TOOLS: frozenset[str] = frozenset({
    "read_file", "file_tree", "git_status", "git_log", "git_diff",
    "web_search", "extract_url", "read_pdf", "read_docx",
    "read_spreadsheet", "extract_text",
})

# Max JSON format-correction retries before falling through to final_answer.
MAX_FORMAT_RETRIES: int = 2

# Mid-task escalation: after this many iterations with tool failures,
# escalate to the next tier up.
ESCALATION_THRESHOLD: int = 3


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

    # ── Phase 5: Execution pattern ──
    # "react_loop" (default) — multi-turn with tools
    # "single_shot" — one LLM call, no tool loop (planner, classifier)
    execution_pattern: str = "react_loop"

    # ── Phase 5: Self-reflection ──
    # If True, inject a "review your own output" prompt before accepting
    # final_answer. Costs one extra LLM call but catches obvious mistakes.
    enable_self_reflection: bool = False

    # ── Phase 5: Confidence-gated output ──
    # Minimum confidence (1-5) for final_answer. Below this → reviewer.
    min_confidence: int = 0  # 0 = disabled

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
            "To query another specialized agent (researcher, analyst, writer, coder):",
            "```json",
            "{",
            '  "action": "ask_agent",',
            '  "target": "<agent_type>",',
            '  "question": "<your question>"',
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

    def _check_tool_permission(self, tool_name: str) -> bool:
        """Check if this agent is permitted to use tool_name (Phase 8.1)."""
        try:
            from ..security.permissions import check_permission
            return check_permission(self.name, tool_name)
        except Exception:
            return True  # fail-open if permissions module unavailable

    def _escalate_requirements(self, reqs: ModelRequirements) -> ModelRequirements:
        """
        Escalate model requirements — increase quality floor.
        Replaces tier-based escalation with capability-aware escalation.
        """
        return reqs.escalate()

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
        # ── Phase 6.4: Ambient context injection ──
        try:
            from ..context.assembler import assemble_ambient_context
            ambient = await assemble_ambient_context(goal_id=goal_id, max_tokens=400)
            if ambient:
                parts.append(ambient)
        except Exception as exc:
            logger.debug(f"Ambient context injection failed (non-critical): {exc}")

        # ── Phase 12.6: Project profile injection ──
        try:
            project_profile = await get_project_profile_for_task(task)
            profile_block = format_project_profile(project_profile) if project_profile else ""
            if profile_block:
                parts.append(profile_block)
        except Exception as exc:
            logger.debug(f"Project profile injection failed (non-critical): {exc}")

        # ── Phase 13.1: Blackboard injection ──
        if goal_id:
            try:
                board = await get_or_create_blackboard(goal_id)
                bb_block = format_blackboard_for_prompt(board)
                if bb_block:
                    parts.append(bb_block)
            except Exception as exc:
                logger.debug(f"Blackboard injection failed (non-critical): {exc}")

        # ── Phase 13.2: Skill library injection ──
        try:
            from ..memory.skills import find_relevant_skills, format_skills_for_prompt
            task_text = f"{task.get('title', '')} {task.get('description', '')}"
            relevant_skills = await find_relevant_skills(task_text, limit=3)
            skills_block = format_skills_for_prompt(relevant_skills)
            if skills_block:
                parts.append(skills_block)
        except Exception as exc:
            logger.debug(f"Skill library injection failed (non-critical): {exc}")

        # ── Phase 11.3: RAG context injection ──
        try:
            rag_block = await retrieve_context(
                task=task, agent_type=self.name, max_tokens=2000,
            )
            if rag_block:
                parts.append(rag_block)
        except Exception as exc:
            logger.debug(f"RAG retrieval failed (non-critical): {exc}")

        # ── Phase 11.7: User preference injection ──
        try:
            prefs = await get_user_preferences()
            pref_block = format_preferences(prefs)
            if pref_block:
                parts.append(pref_block)
        except Exception as exc:
            logger.debug(f"Preference retrieval failed (non-critical): {exc}")

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
    def _parse_agent_response(self, content: str) -> dict | None:
        """
        Extract an action dict from the model's text.

        Phase 9.2 refactored pipeline:
        1. try json.loads (clean JSON)
        2. try fence extraction (```json``` blocks)
        3. one brace-depth scan (JSON buried in prose)
        4. explicit failure → return None (no silent fallback)

        Also handles:
        - Legacy action names (``tool`` → ``tool_call``, etc.)
        - Legacy ``{"status": "complete", ...}`` format

        Returns None when parsing fails — the caller is responsible
        for format retries or explicit failure handling.
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

        # Phase 9.2: Explicit failure — no silent fallback to final_answer.
        # The caller must handle None (format retry or explicit fail).
        return None

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
            "tool_call", "final_answer", "clarify", "decompose",
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

    def _get_context_window(self, model: str, tier_or_reqs=None) -> int:
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


    def _trim_messages_if_needed(
        self, messages: list[dict], model: str, tier_or_reqs=None,
    ) -> list[dict]:
        """
        If the conversation exceeds 80% of context, compress older exchanges.
        Accepts tier string or ModelRequirements for compat.
        """
        ctx_window = self._get_context_window(model, tier_or_reqs)
        threshold = int(ctx_window * 0.80)

        current = self._count_tokens(messages, model)
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
        if self._count_tokens(result, model) <= threshold:
            final = self._count_tokens(result, model)
            logger.info(f"Context compressed (truncate): {current} → {final} tokens")
            return result

        # Phase 2: drop oldest pairs
        while len(middle) >= 2:
            if self._count_tokens(head + middle + tail, model) <= threshold:
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
    # ── Phase 4.6: Progress streaming callback ──
    progress_callback: Callable | None = None

    async def execute(self, task: dict, progress_callback: Callable | None = None) -> dict:
        """
        Route to appropriate execution pattern, then run.
        progress_callback: async fn(task_id, iteration, max_iter, summary)
        """
        self.progress_callback = progress_callback
        # ── Phase 5: execution pattern routing ──
        if self.execution_pattern == "single_shot":
            return await self.execute_single_shot(task)
        return await self._execute_react_loop(task)

    async def _execute_react_loop(self, task: dict) -> dict:
        """ReAct loop with requirements-based model selection."""
        task_id = task.get("id", "?")
        goal_id = task.get("goal_id")

        # ── Parse task context ──
        _task_ctx = task.get("context")
        if isinstance(_task_ctx, str):
            try:
                _task_ctx = json.loads(_task_ctx)
            except (json.JSONDecodeError, TypeError):
                _task_ctx = {}
        if not isinstance(_task_ctx, dict):
            _task_ctx = {}
        model_override = _task_ctx.get("model_override")

        # ── Build initial ModelRequirements ──
        reqs = await self._build_model_requirements(task, _task_ctx)
        # Phase 9.2: Attach task_id for tracing in router
        reqs._task_id = int(task_id) if str(task_id).isdigit() else None
        if model_override:
            reqs.model_override = model_override

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
            messages = checkpoint.get("messages", [])
            start_iteration = checkpoint.get("iteration", 0)
            total_cost = checkpoint.get("total_cost", 0.0)
            used_model = checkpoint.get("used_model", "unknown")
            tools_used = checkpoint.get("tools_used", False)
            _compat_retried = checkpoint.get("validation_retried", False)
            custom_validation_retried = _compat_retried
            task_type_validation_retried = _compat_retried
            format_retries = checkpoint.get("format_retries", 0)
            completed_tool_ops: dict[str, str] = checkpoint.get(
                "completed_tool_ops", {}
            )

            # Restore reqs from checkpoint
            saved_reqs = checkpoint.get("reqs")
            if isinstance(saved_reqs, ModelRequirements):
                reqs = saved_reqs
            elif isinstance(saved_reqs, dict):
                # Checkpoint saved via dataclasses.asdict — reconstruct
                valid_fields = {f.name for f in dataclasses.fields(ModelRequirements)}
                reqs = ModelRequirements(
                    **{k: v for k, v in saved_reqs.items() if k in valid_fields}
                )
            else:
                # Very old checkpoint or missing — build fresh
                reqs = await self._build_model_requirements(task, _task_ctx)

            logger.info(
                f"[Task #{task_id}] Resuming from checkpoint "
                f"(iteration {start_iteration}, "
                f"{len(messages)} messages, ${total_cost:.4f} spent, "
                f"{len(completed_tool_ops)} cached tool ops)"
            )
        else:
            system_prompt = self._build_full_system_prompt(task)
            context = await self._build_context(task)

            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": context},
            ]

            total_cost = 0.0
            used_model = "unknown"
            tools_used = False
            custom_validation_retried = False
            task_type_validation_retried = False
            format_retries = 0
            completed_tool_ops: dict[str, str] = {}

        consecutive_tool_failures = 0
        escalated = False

        _progress_last_sent = 0.0

        for iteration in range(start_iteration, self.max_iterations):
            logger.info(
                f"[Task #{task_id}] Agent '{self.name}' iteration "
                f"{iteration + 1}/{self.max_iterations}"
            )

            # ── Phase 4.6: Progress streaming ──
            _now = time.time()
            if (self.progress_callback
                    and iteration > 0
                    and _now - _progress_last_sent >= 30):
                try:
                    # Summarize last action from messages
                    _last_action = ""
                    for _m in reversed(messages):
                        if _m.get("role") == "assistant":
                            _last_action = (_m.get("content") or "")[:100]
                            break
                    await self.progress_callback(
                        task_id, iteration + 1, self.max_iterations, _last_action
                    )
                    _progress_last_sent = _now
                except Exception:
                    pass

            # ── Update token estimates ──
            estimation_model = used_model if used_model != "unknown" else "gpt-4o-mini"
            reqs.estimated_input_tokens = self._count_tokens(
                messages, estimation_model
            )
            reqs.estimated_output_tokens = min(
                reqs.estimated_output_tokens, 4096,
            )

            # ── Trim context ── (now accepts reqs directly)
            messages = self._trim_messages_if_needed(
                messages, estimation_model, reqs,
            )

            # ── Tools ──
            litellm_tools = self._build_litellm_tools()
            if litellm_tools:
                reqs.needs_function_calling = True

            # ── Call LLM ──
            try:
                response = await call_model(
                    reqs,
                    messages,
                    tools=litellm_tools,
                )
            except Exception as exc:
                logger.error(f"[Task #{task_id}] Model call failed: {exc}")
                return {
                    "status": "completed",
                    "result": f"Agent failed after {iteration} iteration(s): {exc}",
                    "model": used_model,
                    "cost": total_cost,
                    "iterations": iteration,
                    "difficulty": reqs.difficulty,  # compat
                }

            content    = response.get("content", "")
            used_model = response.get("model", used_model)
            step_cost  = response.get("cost", 0)
            step_latency = response.get("latency", 0)
            total_cost += step_cost

            try:
                await record_model_call(
                    model=used_model,
                    agent_type=self.name,
                    success=True,
                    cost=step_cost,
                    latency=step_latency,
                )
            except Exception:
                pass

            if step_cost > 0:
                try:
                    await record_cost(step_cost)
                except Exception:
                    pass

            logger.debug(f"[Task #{task_id}] Raw response: {content[:200]}...")
            await self._safe_log(
                task_id, "assistant", content, used_model, step_cost
            )

            # ── Parse response ──
            fc_tool_calls = response.get("tool_calls")
            parsed = None
            if fc_tool_calls:
                parsed = self._parse_function_call_response(fc_tool_calls)
            if parsed is None:
                parsed = self._parse_agent_response(content)

            # ── FORMAT RETRY ──
            if parsed is None:
                if format_retries < MAX_FORMAT_RETRIES:
                    format_retries += 1
                    logger.warning(
                        f"[Task #{task_id}] JSON parse failed — "
                        f"retry {format_retries}/{MAX_FORMAT_RETRIES}"
                    )
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your response could not be parsed as valid JSON. "
                            "Please fix the formatting and try again.\n\n"
                            "You MUST respond with ONLY a valid JSON block:\n"
                            "```json\n"
                            '{"action": "tool_call", "tool": "...", "args": {...}}\n'
                            "```\nor:\n```json\n"
                            '{"action": "final_answer", "result": "..."}\n'
                            "```\nNo text before or after the JSON block."
                        ),
                    })
                    await self._save_checkpoint(
                        task_id, iteration + 1, messages, total_cost,
                        used_model, reqs, tools_used, custom_validation_retried or task_type_validation_retried,
                        completed_tool_ops, format_retries,
                    )
                    continue
                else:
                    parsed = {
                        "action": "final_answer",
                        "result": (
                            f"[Parse failure] Agent could not produce valid "
                            f"JSON after {MAX_FORMAT_RETRIES} retries. "
                            f"Raw output:\n{content[:2000]}"
                        ),
                    }

            try:
                parsed = validate_action(parsed)
            except ValueError as exc:
                logger.warning(f"[Task #{task_id}] Action validation warning: {exc}")

            action_type = parsed.get("action", "final_answer")

            # ── HALLUCINATION GUARD (unchanged) ──
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
                    f"[Task #{task_id}] ⚠️ Hallucination guard triggered"
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
                        "Respond with ONLY the JSON block. No explanation."
                    ),
                })
                await self._safe_log(
                    task_id, "system",
                    f"[hallucination_guard] Rejected premature final_answer",
                    None, 0,
                )
                await self._save_checkpoint(
                    task_id, iteration + 1, messages, total_cost,
                    used_model, reqs, tools_used, custom_validation_retried or task_type_validation_retried,
                    completed_tool_ops, format_retries,
                )
                continue

            # ── FINAL ANSWER ──
            if action_type == "final_answer":
                result = parsed.get("result", content)

                if not custom_validation_retried:
                    validation_error = self._validate_response(result, task)
                    if validation_error:
                        custom_validation_retried = True
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": f"{validation_error}\n\nPlease try again.",
                        })
                        await self._save_checkpoint(
                            task_id, iteration + 1, messages, total_cost,
                            used_model, reqs, tools_used,
                            custom_validation_retried or task_type_validation_retried,
                            completed_tool_ops, format_retries,
                        )
                        continue

                task_type_errors = validate_task_output(self.name, result)
                if task_type_errors and not task_type_validation_retried:
                    task_type_validation_retried = True
                    err_msg = "; ".join(task_type_errors)
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"Output quality issue: {err_msg}\n\nPlease revise.",
                    })
                    await self._save_checkpoint(
                        task_id, iteration + 1, messages, total_cost,
                        used_model, reqs, tools_used,
                        custom_validation_retried or task_type_validation_retried,
                        completed_tool_ops, format_retries,
                    )
                    continue

                # Memories
                raw_memories = parsed.get("memories", {})
                if raw_memories and isinstance(raw_memories, dict):
                    for key, value in raw_memories.items():
                        try:
                            await store_memory(
                                key, str(value),
                                category=self.name, goal_id=goal_id,
                            )
                        except Exception as exc:
                            logger.warning(f"store_memory failed: {exc}")

                logger.info(
                    f"[Task #{task_id}] ✅ Agent answered after "
                    f"{iteration + 1} iteration(s)"
                )

                # Debug: show what keys the parsed response has
                parsed_keys = list(parsed.keys()) if isinstance(parsed, dict) else "not-dict"
                has_subtasks = bool(parsed.get("subtasks")) if isinstance(parsed, dict) else False
                logger.debug(
                    f"[Task #{task_id}] Parsed keys: {parsed_keys}, "
                    f"has_subtasks={has_subtasks}, "
                    f"can_create_subtasks={self.can_create_subtasks}"
                )

                # Normalize subtask keys — LLMs sometimes use "tasks",
                # "steps", "plan" instead of "subtasks"
                subtasks = parsed.get("subtasks")
                if not subtasks and self.can_create_subtasks:
                    for alt_key in ("tasks", "steps", "plan", "sub_tasks"):
                        candidate = parsed.get(alt_key)
                        if isinstance(candidate, list) and candidate:
                            subtasks = candidate
                            logger.debug(
                                f"[Task #{task_id}] Found subtasks under "
                                f"alt key '{alt_key}' ({len(candidate)} items)"
                            )
                            break

                if subtasks:
                    await self._clear_checkpoint_safe(task_id)
                    return {
                        "status":       "needs_subtasks",
                        "subtasks":     subtasks,
                        "plan_summary": parsed.get("plan_summary", ""),
                        "model":        used_model,
                        "cost":         total_cost,
                        "difficulty":   reqs.difficulty,
                    }

                if parsed.get("needs_clarification"):
                    await self._clear_checkpoint_safe(task_id)
                    return {
                        "status":        "needs_clarification",
                        "clarification": parsed["needs_clarification"],
                        "model":         used_model,
                        "cost":          total_cost,
                        "difficulty":    reqs.difficulty,
                    }

                # Self-reflection
                if self.enable_self_reflection:
                    try:
                        reflection = await self._self_reflect(
                            task, result, reqs, used_model,
                        )
                        if reflection and reflection.get("verdict") == "fix":
                            corrected = reflection.get("corrected_result")
                            if corrected:
                                result = corrected
                    except Exception as exc:
                        logger.debug(f"Self-reflection error: {exc}")

                # Confidence gating
                confidence = parsed.get("confidence")
                if (
                    self.min_confidence > 0
                    and isinstance(confidence, (int, float))
                    and confidence < self.min_confidence
                ):
                    await self._clear_checkpoint_safe(task_id)
                    return {
                        "status":      "needs_review",
                        "result":      result,
                        "review_note": f"Agent confidence: {confidence}/5",
                        "model":       used_model,
                        "cost":        total_cost,
                        "difficulty":  reqs.difficulty,
                    }

                # Grading (skip trivial tasks)
                quality_score = None
                if reqs.difficulty >= 4:
                    try:
                        quality_score = await grade_response(
                            task.get("title", ""),
                            task.get("description", ""),
                            result,
                            generating_model=used_model,
                            task_name=reqs.task,
                        )
                        if quality_score is not None and task_id != "?":
                            await update_task(task_id, quality_score=quality_score)
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
                    "difficulty":    reqs.difficulty,
                    "iterations":    iteration + 1,
                    "quality_score": quality_score,
                }

            # ── TOOL CALL ──
            if action_type == "tool_call":
                tools_used = True
                tool_name = parsed.get("tool", "")
                tool_args = parsed.get("args", {})
                if not isinstance(tool_args, dict):
                    tool_args = {}

                if (
                    self.allowed_tools is not None
                    and tool_name not in self.allowed_tools
                ):
                    tool_output = (
                        f"❌ Tool '{tool_name}' not available. "
                        f"Allowed: {self.allowed_tools}"
                    )
                elif not self._check_tool_permission(tool_name):
                    tool_output = (
                        f"🚫 Tool '{tool_name}' not permitted for agent "
                        f"type '{self.name}' (security policy)."
                    )
                    logger.warning(
                        f"[Task #{task_id}] Permission denied: "
                        f"{self.name} → {tool_name}"
                    )
                elif tool_name not in TOOL_REGISTRY:
                    tool_output = (
                        f"❌ Unknown tool '{tool_name}'. "
                        f"Available: {list(TOOL_REGISTRY.keys())}"
                    )
                else:
                    arg_schema = _TOOL_SCHEMAS_BY_NAME.get(tool_name)
                    if arg_schema:
                        tool_args, arg_errors = validate_tool_args(
                            tool_name, tool_args, arg_schema,
                        )
                        if arg_errors:
                            err_msg = "; ".join(arg_errors)
                            tool_output = (
                                f"❌ Argument error for '{tool_name}': {err_msg}\n\n"
                                f"Expected: {json.dumps(arg_schema, indent=2)}"
                            )
                            messages.append({"role": "assistant", "content": content})
                            messages.append({"role": "user", "content": tool_output})
                            await self._save_checkpoint(
                                task_id, iteration + 1, messages, total_cost,
                                used_model, reqs, tools_used,
                                custom_validation_retried or task_type_validation_retried,
                                completed_tool_ops, format_retries,
                            )
                            continue

                    idem_key = self._tool_idempotency_key(tool_name, tool_args)

                    # Check caches: side-effect idempotency OR read-only result cache
                    cached = None
                    if tool_name in SIDE_EFFECT_TOOLS:
                        cached = completed_tool_ops.get(idem_key)
                    elif tool_name in CACHEABLE_READ_TOOLS:
                        cached = completed_tool_ops.get(f"rc:{idem_key}")

                    if cached is not None:
                        tool_output = cached
                        logger.debug(f"[Task #{task_id}] cache hit: {tool_name}")
                    else:
                        logger.info(
                            f"[Task #{task_id}] \U0001f527 {tool_name}("
                            f"{', '.join(f'{k}={repr(v)[:50]}' for k, v in tool_args.items())})"
                        )
                        try:
                            tool_output = await execute_tool(
                                tool_name, agent_type=self.name, **tool_args
                            )
                        except Exception as exc:
                            tool_output = f"\u274c Tool execution error: {exc}"

                        # Phase 8.4: Audit log tool execution
                        try:
                            from ..infra.audit import audit, ACTOR_AGENT, ACTION_TOOL_EXEC
                            _tid = int(task_id) if str(task_id).isdigit() else None
                            await audit(
                                actor=f"{ACTOR_AGENT}:{self.name}",
                                action=ACTION_TOOL_EXEC,
                                target=tool_name,
                                details=str(tool_args)[:500],
                                task_id=_tid,
                                goal_id=goal_id,
                            )
                        except Exception:
                            pass

                        # Phase 9.2: Trace tool execution
                        try:
                            _tid = int(task_id) if str(task_id).isdigit() else None
                            if _tid:
                                from ..infra.tracing import append_trace
                                await append_trace(
                                    task_id=_tid,
                                    entry_type="tool",
                                    input_summary=f"{tool_name}({', '.join(f'{k}={repr(v)[:30]}' for k, v in tool_args.items())})",
                                    output_summary=tool_output[:200] if tool_output else "",
                                )
                        except Exception:
                            pass

                        # Phase 9.1: Record tool call metric
                        try:
                            from ..infra.metrics import record_tool_call
                            record_tool_call(tool=tool_name)
                        except Exception:
                            pass

                        # Cache results
                        if tool_name in SIDE_EFFECT_TOOLS:
                            completed_tool_ops[idem_key] = tool_output
                            # Invalidate read-only cache on side effects
                            _to_remove = [k for k in completed_tool_ops if k.startswith("rc:")]
                            for k in _to_remove:
                                del completed_tool_ops[k]
                        elif tool_name in CACHEABLE_READ_TOOLS:
                            completed_tool_ops[f"rc:{idem_key}"] = tool_output

                if len(tool_output) > MAX_TOOL_OUTPUT_LENGTH:
                    tool_output = (
                        tool_output[:MAX_TOOL_OUTPUT_LENGTH]
                        + f"\n\n... [{len(tool_output)} chars total]"
                    )

                tool_failed = (
                    tool_output.startswith("❌")
                    or tool_output.startswith("🚫")
                    or "command not found" in tool_output
                    or "No such file" in tool_output
                    or ("exit code" in tool_output and "exit code 0" not in tool_output)
                )

                if tool_failed:
                    consecutive_tool_failures += 1
                else:
                    consecutive_tool_failures = 0

                # ── Mid-task escalation ── (NOW uses reqs.escalate())
                if (
                    not escalated
                    and consecutive_tool_failures >= ESCALATION_THRESHOLD
                    and iteration >= ESCALATION_THRESHOLD
                ):
                    old_tier = reqs.difficulty
                    reqs = self._escalate_requirements(reqs)
                    new_tier = reqs.difficulty
                    if new_tier != old_tier:
                        logger.warning(
                            f"[Task #{task_id}] ⬆️ Escalating: "
                            f"'{old_tier}' → '{new_tier}' after "
                            f"{consecutive_tool_failures} consecutive failures"
                        )
                        escalated = True
                        await self._safe_log(
                            task_id, "system",
                            f"[escalation] Upgraded quality after "
                            f"{consecutive_tool_failures} failures",
                            None, 0,
                        )

                if tool_failed:
                    recovery_guidance = (
                        f"## Tool Result (`{tool_name}`) — ERROR:\n\n"
                        f"```\n{tool_output}\n```\n\n"
                        f"The tool call failed. Try a DIFFERENT approach.\n"
                        f"Iteration {iteration + 2}/{self.max_iterations}."
                    )
                else:
                    recovery_guidance = (
                        f"## Tool Result (`{tool_name}`):\n\n"
                        f"```\n{tool_output}\n```\n\n"
                        f"Continue working. Iteration {iteration + 2}/{self.max_iterations}."
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
                    used_model, reqs, tools_used, custom_validation_retried or task_type_validation_retried,
                    completed_tool_ops, format_retries,
                )
                continue

            # ── ASK AGENT (inter-agent query) ──
            if action_type == "ask_agent":
                import asyncio as _asyncio
                target_type = parsed.get("target", "researcher")
                question = parsed.get("question", "")
                logger.info(
                    f"[Task #{task_id}] 🤝 ask_agent → {target_type}: "
                    f"{question[:80]}"
                )
                try:
                    from ..agents import get_agent as _get_agent
                    target_agent = _get_agent(target_type)
                    inline_task = {
                        "id": f"{task_id}_inline_{iteration}",
                        "title": f"[Inline query from {self.name}]",
                        "description": question,
                        "goal_id": goal_id,
                        "context": json.dumps({"tool_depth": 1}),
                    }
                    inline_result = await _asyncio.wait_for(
                        target_agent.execute(inline_task), timeout=300
                    )
                    agent_answer = inline_result.get("result", "(no answer)")
                    agent_cost = inline_result.get("cost", 0)
                    total_cost += agent_cost
                    tool_output = (
                        f"## Answer from {target_type} agent:\n\n{agent_answer}"
                    )
                    logger.info(
                        f"[Task #{task_id}] ✅ ask_agent from {target_type} "
                        f"completed (${agent_cost:.4f})"
                    )
                except _asyncio.TimeoutError:
                    tool_output = f"❌ ask_agent timeout: {target_type} did not respond within 5 minutes"
                    logger.warning(f"[Task #{task_id}] ask_agent timeout ({target_type})")
                except Exception as exc:
                    tool_output = f"❌ ask_agent error: {exc}"
                    logger.warning(f"[Task #{task_id}] ask_agent error: {exc}")

                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"{tool_output}\n\n"
                        f"Continue working. Iteration {iteration + 2}/{self.max_iterations}."
                    ),
                })
                await self._save_checkpoint(
                    task_id, iteration + 1, messages, total_cost,
                    used_model, reqs, tools_used,
                    custom_validation_retried or task_type_validation_retried,
                    completed_tool_ops, format_retries,
                )
                continue

            # ── CLARIFY / DECOMPOSE / UNKNOWN (unchanged) ──
            if action_type == "clarify":
                await self._clear_checkpoint_safe(task_id)
                return {
                    "status": "needs_clarification",
                    "clarification": parsed.get("question", content),
                    "model": used_model, "cost": total_cost, "difficulty": reqs.difficulty,
                }

            if action_type == "decompose":
                await self._clear_checkpoint_safe(task_id)
                return {
                    "status": "needs_subtasks",
                    "subtasks": parsed.get("subtasks", []),
                    "plan_summary": parsed.get("summary", ""),
                    "model": used_model, "cost": total_cost, "difficulty": reqs.difficulty,
                }

            # Unknown action
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"ERROR: Unrecognized action '{action_type}'. "
                    f"Use tool_call or final_answer only.\n\n"
                    f"```json\n"
                    f'{{"action": "tool_call", "tool": "shell", "args": {{"command": "ls"}}}}\n'
                    f"```\nor:\n```json\n"
                    f'{{"action": "final_answer", "result": "your answer"}}\n'
                    f"```"
                ),
            })
            await self._save_checkpoint(
                task_id, iteration + 1, messages, total_cost,
                used_model, reqs, tools_used, custom_validation_retried or task_type_validation_retried,
                completed_tool_ops, format_retries,
            )

        # ── Exhausted iterations ──
        await self._clear_checkpoint_safe(task_id)
        last = messages[-1].get("content", "") if messages else ""
        return {
            "status": "completed",
            "result": (
                f"[Completed after {self.max_iterations} iterations "
                f"without a final answer]\n\nLast context:\n{last[:3000]}"
            ),
            "model": used_model,
            "cost": total_cost,
            "difficulty": reqs.difficulty,
            "iterations": self.max_iterations,
        }

    async def _build_model_requirements(
        self, task: dict, task_ctx: dict,
    ) -> ModelRequirements:
        """
        Build ModelRequirements from task metadata + agent properties.
        Uses the new task-based routing while preserving all existing logic.
        """
        title = task.get("title", "").lower()
        description = task.get("description", "").lower()
        priority = task.get("priority", 5)

        # ── Start from curated AGENT_REQUIREMENTS template ──
        from src.core.router import AGENT_REQUIREMENTS
        import copy

        classification = task_ctx.get("classification", {})
        agent_type = classification.get("agent_type", self.name)

        template = AGENT_REQUIREMENTS.get(agent_type) or AGENT_REQUIREMENTS.get(
            self.name, ModelRequirements(task=agent_type, difficulty=5)
        )
        reqs = copy.deepcopy(template)
        reqs.agent_type = self.name
        reqs.priority = priority

        # Overlay classification signals (only upgrade, never downgrade)
        cls_difficulty = classification.get("difficulty", 5)
        reqs.difficulty = max(reqs.difficulty, cls_difficulty)

        if classification.get("needs_tools"):
            reqs.needs_function_calling = True
        if classification.get("needs_vision"):
            reqs.needs_vision = True
        if classification.get("needs_thinking"):
            reqs.needs_thinking = True
        if classification.get("local_only"):
            reqs.local_only = True

        # ── Adjust for task priority ──
        if priority >= 10:
            reqs.prefer_speed = True
            reqs.difficulty = max(reqs.difficulty, 6)
        elif priority <= 2:
            reqs.difficulty = max(1, reqs.difficulty - 2)

        # ── Detect personal/sensitive data ──
        sensitivity_keywords = [
            "personal", "private", "secret", "password",
            "credential", "my ", "my_", "home",
        ]
        if any(kw in f"{title} {description}" for kw in sensitivity_keywords):
            reqs.local_only = True

        if task_ctx.get("local_only"):
            reqs.local_only = True
        if task_ctx.get("prefer_quality"):
            reqs.prefer_quality = True
        if task_ctx.get("prefer_speed"):
            reqs.prefer_speed = True
            reqs.prefer_local = False

        # ── Model diversity ──
        exclude = task_ctx.get("exclude_models", [])
        if exclude:
            reqs.exclude_models = exclude

        # ── Estimate context size ──
        desc_len = len(task.get("description", ""))
        context_json = task.get("context", "{}")
        if isinstance(context_json, str):
            ctx_len = len(context_json)
        else:
            ctx_len = len(json.dumps(context_json))

        estimated_input = (desc_len + ctx_len) // 4  # rough char-to-token
        reqs.estimated_input_tokens = max(estimated_input, 1000)
        # Keep template's estimated_output_tokens — it's set per agent type
        # (e.g. coder=4000, planner=2000) for accurate speed scoring.

        # ── Tools needed? (agent-level override) ──
        # Only upgrade to function_calling, don't force it if the
        # template doesn't need it (e.g., planner, writer, reviewer).
        # The template already sets needs_function_calling=True for
        # agents that genuinely need tool use (coder, fixer, executor).
        if reqs.needs_function_calling:
            pass  # already set by template or classification
        elif self.allowed_tools and len(self.allowed_tools) > 0:
            # Agent explicitly declares tool list → it needs function calling
            reqs.needs_function_calling = True

        # ── Vision needed? (keyword override) ──
        if task_ctx.get("needs_vision"):
            reqs.needs_vision = True
        if any(kw in f"{title} {description}" for kw in [
            "screenshot", "image", "visual", "ui review", "layout",
            "diagram", "photo", "picture",
        ]):
            reqs.needs_vision = True

        # ── Thinking needed? ──
        if task_ctx.get("needs_thinking"):
            reqs.needs_thinking = True

        return reqs

    async def execute_single_shot(self, task: dict) -> dict:
        """Single LLM call with no tool loop. For planning/classification."""
        task_id = task.get("id", "?")

        _ss_ctx = task.get("context")
        if isinstance(_ss_ctx, str):
            try:
                _ss_ctx = json.loads(_ss_ctx)
            except (json.JSONDecodeError, TypeError):
                _ss_ctx = {}
        if not isinstance(_ss_ctx, dict):
            _ss_ctx = {}

        # Build requirements using the same method as react loop
        reqs = await self._build_model_requirements(task, _ss_ctx)

        _ss_model_override = _ss_ctx.get("model_override")
        if _ss_model_override:
            reqs.model_override = _ss_model_override

        system_prompt = self._build_full_system_prompt(task)
        context = await self._build_context(task)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": context},
        ]

        try:
            response = await call_model(reqs, messages)
        except Exception as exc:
            logger.error(f"[Task #{task_id}] Single-shot call failed: {exc}")
            return {
                "status": "completed",
                "result": f"Agent failed: {exc}",
                "model": "unknown", "cost": 0, "difficulty": reqs.difficulty,
            }

        content = response.get("content", "")
        used_model = response.get("model", "unknown")
        cost = response.get("cost", 0)

        parsed = self._parse_agent_response(content)
        if parsed is None:
            parsed = {"action": "final_answer", "result": content}

        action_type = parsed.get("action", "final_answer")

        if action_type == "decompose" or parsed.get("subtasks"):
            return {
                "status": "needs_subtasks",
                "subtasks": parsed.get("subtasks", []),
                "plan_summary": parsed.get("plan_summary", ""),
                "model": used_model, "cost": cost, "difficulty": reqs.difficulty,
            }

        return {
            "status": "completed",
            "result": parsed.get("result", content),
            "model": used_model, "cost": cost,
            "difficulty": reqs.difficulty, "iterations": 1,
        }

    async def _self_reflect(
        self, task: dict, result: str,
        tier_or_reqs=None, used_model: str = "",
    ) -> dict | None:
        """Review own output for errors. Accepts tier string or ModelRequirements."""
        try:
            # Build requirements for the reflection call
            if isinstance(tier_or_reqs, ModelRequirements):
                reflect_reqs = ModelRequirements(
                    task="reviewer",
                    difficulty=tier_or_reqs.difficulty,
                    agent_type="self_reflection",
                    estimated_input_tokens=800,
                    estimated_output_tokens=500,
                    prefer_speed=True,
                )
            else:
                # Legacy fallback — tier strings no longer used
                reflect_reqs = ModelRequirements(
                    task="reviewer",
                    difficulty=6,
                    agent_type="self_reflection",
                    estimated_input_tokens=800,
                    estimated_output_tokens=500,
                    prefer_speed=True,
                )

            messages = [
                {"role": "system", "content": (
                    "You are a careful reviewer. Check this response "
                    "for errors, omissions, or hallucinations. "
                    "If the response is good, respond: "
                    '{"verdict": "ok"}. '
                    "If there are issues, respond: "
                    '{"verdict": "fix", "issues": "description", '
                    '"corrected_result": "the fixed version"}.'
                )},
                {"role": "user", "content": (
                    f"Task: {task.get('title', '')}\n"
                    f"Description: {(task.get('description') or '')[:500]}\n\n"
                    f"Response to review:\n{result[:3000]}"
                )},
            ]
            response = await call_model(reflect_reqs, messages)
            raw = response.get("content", "").strip()
            parsed = self._try_parse_json(raw)
            if parsed and parsed.get("verdict") == "fix":
                return parsed
        except Exception as exc:
            logger.debug(f"Self-reflection failed: {exc}")
        return None

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
        reqs: ModelRequirements,
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
                "reqs": dataclasses.asdict(reqs),
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
