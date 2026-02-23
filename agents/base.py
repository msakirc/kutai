# agents/base.py
"""
Base agent with iterative ReAct loop:
  Think → Act (tool or respond) → Observe → Think again
"""
from __future__ import annotations

import json
import logging
import re

from router import call_model, classify_task
from db import (
    log_conversation,
    store_memory,
    recall_memory,
    get_completed_dependency_results,
)
from tools import TOOL_REGISTRY, get_tool_descriptions, execute_tool
from config import MAX_AGENT_ITERATIONS, AGENT_TIER_MAP, MAX_TOOL_OUTPUT_LENGTH

logger = logging.getLogger(__name__)


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

        # ── build messages ──
        system_prompt = self._build_full_system_prompt(task)
        context = await self._build_context(task)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": context},
        ]

        total_cost = 0.0
        used_model = "unknown"
        tools_used = False

        # ── iterative loop ──
        for iteration in range(self.max_iterations):
            logger.info(
                f"[Task #{task_id}] Agent '{self.name}' iteration "
                f"{iteration + 1}/{self.max_iterations}"
            )

            # Call LLM
            try:
                response = await call_model(tier, messages)
            except Exception as exc:
                logger.error(f"[Task #{task_id}] Model call failed: {exc}")
                return {
                    "status":     "complete",
                    "result":     f"Agent failed after {iteration} iteration(s): {exc}",
                    "model":      used_model,
                    "cost":       total_cost,
                    "tier":       tier,
                    "iterations": iteration,
                }

            content    = response.get("content", "")
            used_model = response.get("model", used_model)
            step_cost  = response.get("cost", 0)
            total_cost += step_cost

            logger.debug(f"[Task #{task_id}] Raw response: {content[:200]}...")

            # Log assistant turn
            await self._safe_log(
                task_id, "assistant", content, used_model, step_cost
            )

            # Parse
            parsed      = self._parse_agent_response(content)
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
                continue
            # ── END HALLUCINATION GUARD ───────────────────────────

            # ── FINAL ANSWER ──────────────────────────────────────────
            if action_type == "final_answer":
                result = parsed.get("result", content)

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
                    return {
                        "status":        "needs_clarification",
                        "clarification": parsed["needs_clarification"],
                        "model":         used_model,
                        "cost":          total_cost,
                        "tier":          tier,
                    }

                return {
                    "status":     "complete",
                    "result":     result,
                    "model":      used_model,
                    "cost":       total_cost,
                    "tier":       tier,
                    "iterations": iteration + 1,
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
                    logger.info(
                        f"[Task #{task_id}] 🔧 {tool_name}("
                        f"{', '.join(f'{k}={repr(v)[:50]}' for k, v in tool_args.items())})"
                    )
                    try:
                        tool_output = await execute_tool(tool_name, **tool_args)
                    except Exception as exc:
                        tool_output = f"❌ Tool execution error: {exc}"
                        logger.error(
                            f"[Task #{task_id}] Tool '{tool_name}' raised: {exc}"
                        )

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
                continue

            # ── CLARIFY ───────────────────────────────────────────────
            if action_type == "clarify":
                return {
                    "status":        "needs_clarification",
                    "clarification": parsed.get("question", content),
                    "model":         used_model,
                    "cost":          total_cost,
                    "tier":          tier,
                }

            # ── DECOMPOSE ─────────────────────────────────────────────
            if action_type == "decompose":
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

        # ── exhausted iterations ──
        logger.warning(
            f"[Task #{task_id}] Agent '{self.name}' exhausted "
            f"{self.max_iterations} iterations"
        )
        last = messages[-1].get("content", "") if messages else ""
        return {
            "status": "complete",
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
