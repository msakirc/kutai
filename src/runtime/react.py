"""ReAct multi-call loop — drives one task to completion via N LLM calls + tools.

Public API:
  await react.run(profile, task, progress_callback=None) -> dict

`profile` is duck-typed (BaseAgent instance or any object exposing the same
attribute surface): name, allowed_tools, max_iterations, enable_self_reflection,
min_confidence, can_create_subtasks, _suppress_clarification, get_system_prompt,
_partial_content (set by streaming layer for cancel-path recovery).

For Phase A.8, helper calls go through profile._X() delegates (BaseAgent
delegates to other src/runtime/ modules). Phase A.10/A.11 will replace those
with direct runtime imports once setup phase is also extracted.

State threaded through one task:
  - messages, total_cost, used_model, tools_used, tools_used_names,
    completed_tool_ops, format_corrections (mutated across iters)
  - consecutive_tool_failures, model_escalated, _progress_last_sent,
    _search_depth, _suppress_guards
  - effective_max_iterations, guard_burns, useful_iterations,
    empty_response_count
  - iter_tool_calls_seen (per-task tool-strip tracker; was self attribute)
  - profile._partial_content (streaming hook; cancel-path captures it)

Invariant — iteration budget vs checkpoint orthogonality (modularization doc,
task #1174 lesson). The for-range iterates fresh per attempt; checkpoint
restores conversational context only, not control flow.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from typing import Callable

from fatih_hoca.requirements import ModelRequirements
from ..app.config import MAX_TOOL_OUTPUT_LENGTH
from ..infra.db import (
    load_task_checkpoint, record_model_call, record_cost, store_memory,
)
from ..infra.logging_config import get_logger
from ..models.models import validate_action, validate_tool_args, validate_task_output
from ..tools import TOOL_REGISTRY, execute_tool

from .checkpoint import (
    safe_log_conversation, save_checkpoint, tool_idempotency_key,
)
from .context import build_system_prompt
from .escalation import escalate_requirements, trim_for_escalation
from .guards import (
    MAX_FORMAT_CORRECTIONS, MAX_SUB_CORRECTIONS,
    check_sub_iter_guards, get_search_depth,
)
from .parsing import parse_action, parse_function_call, unwrap_final_answer
from .reflection import self_reflect
from .tools import (
    CACHEABLE_READ_TOOLS, SIDE_EFFECT_TOOLS,
    TOOL_FAILURE_ESCALATION_THRESHOLD, TOOL_SCHEMAS_BY_NAME,
    build_litellm_tools, check_tool_permission, partition_tool_calls,
)
from .validation import validate_final_answer
from .window import (
    context_window_for, count_tokens, prune_tool_results, trim_if_needed,
)


async def safe_log_conversation_p(profile, task_id, role, content, model, cost):
    """Wrapper that pulls profile.name into the logger call."""
    await safe_log_conversation(task_id, role, content, model, cost, profile.name)

logger = get_logger("runtime.react")


async def run(profile, task: dict, progress_callback: Callable | None = None) -> dict:
    """ReAct loop with requirements-based model selection."""
    _start_time = time.time()
    task_id = task.get("id", "?")
    mission_id = task.get("mission_id")

    # ── Parse task context ──
    _task_ctx = task.get("context")
    if isinstance(_task_ctx, str):
        try:
            _task_ctx = json.loads(_task_ctx)
        except (json.JSONDecodeError, TypeError):
            _task_ctx = {}
    if not isinstance(_task_ctx, dict):
        _task_ctx = {}
    reqs = await profile._build_model_requirements(task, _task_ctx)
    # Phase 9.2: Attach task_id for tracing in router
    reqs._task_id = int(task_id) if str(task_id).isdigit() else None

    # ── attempt checkpoint recovery ──
    start_iteration = 0
    checkpoint = None
    # Skip checkpoint when this is a validation/grader retry. Presence
    # of _schema_error means a prior attempt failed and the hooks /
    # apply layers just injected a fresh nudge into ctx. The checkpoint
    # holds messages from that prior (bad) attempt — resuming from it
    # uses the OLD user prompt and the model never sees the retry
    # feedback, so it regenerates the same truncated output every
    # iteration. Mission 46 task #2888 looped 15+ times on an
    # identical ~26k-char truncation before DLQ (2026-04-24). Force
    # a fresh _build_context so enriched retry prompt lands.
    if _task_ctx.get("_schema_error"):
        logger.info(
            f"[Task #{task_id}] retry context detected "
            f"(_schema_error present) — skipping checkpoint, "
            f"rebuilding prompt with enriched nudge"
        )
    elif task_id != "?":
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
        tools_used_names: set[str] = set(checkpoint.get("tools_used_names", []))
        _compat_retried = checkpoint.get("validation_retried", False)
        custom_validation_retried = _compat_retried
        task_type_validation_retried = _compat_retried
        format_corrections = checkpoint.get("format_corrections",
                                            checkpoint.get("format_retries", 0))
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
            reqs = await profile._build_model_requirements(task, _task_ctx)

        logger.info(
            f"[Task #{task_id}] Resuming from checkpoint "
            f"(iteration {start_iteration}, "
            f"{len(messages)} messages, ${total_cost:.4f} spent, "
            f"{len(completed_tool_ops)} cached tool ops)"
        )

        # Validate recovered _prev_output from checkpoint context
        _recovered_prev = _task_ctx.get("_prev_output")
        if _recovered_prev:
            from dogru_mu_samet import assess as cq_assess, salvage as cq_salvage
            _rec_cq = cq_assess(_recovered_prev)
            if _rec_cq.is_degenerate:
                cleaned = cq_salvage(_recovered_prev)
                if cleaned:
                    _task_ctx["_prev_output"] = cleaned
                else:
                    _task_ctx.pop("_prev_output", None)
                logger.info(
                    f"[Task #{task_id}] Checkpoint _prev_output was degenerate, "
                    f"{'salvaged' if cleaned else 'discarded'}"
                )
    else:
        system_prompt = build_system_prompt(profile, task)
        context = await profile._build_context(task)

        logger.info(
            f"[Task #{task_id}] System prompt ({len(system_prompt)} chars):\n"
            f"{system_prompt}"
        )
        logger.info(
            f"[Task #{task_id}] User context ({len(context)} chars):\n"
            f"{context}"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": context},
        ]

        total_cost = 0.0
        used_model = "unknown"
        tools_used = False
        tools_used_names: set[str] = set()
        custom_validation_retried = False
        task_type_validation_retried = False
        format_corrections = 0
        completed_tool_ops: dict[str, str] = {}

    consecutive_tool_failures = 0
    model_escalated = False

    _progress_last_sent = time.time()
    _search_depth = get_search_depth(task)
    _suppress_guards = _task_ctx.get("suppress_guards", False)

    # Dynamic iteration budget (retry boost from exhaustion handler)
    effective_max_iterations = profile.max_iterations
    _boost = _task_ctx.get("iteration_budget_boost", 1.0)
    if _boost > 1.0:
        effective_max_iterations = min(int(profile.max_iterations * _boost), 12)
        logger.info(
            f"[Task #{task_id}] Iteration budget boosted: "
            f"{profile.max_iterations} → {effective_max_iterations}"
        )

    # Exhaustion tracking counters
    guard_burns = 0
    useful_iterations = 0
    empty_response_count = 0

    # Per-task cumulative tool-call tracker. Used to dynamically strip tools
    # like read_file once the agent has already invoked them — prevents
    # re-reading artifacts already present in the blackboard/history.
    iter_tool_calls_seen: set[str] = set()
    # Mirror onto profile so any external lookup (e.g. progress probes) can
    # still find it — but the authoritative state is local.
    profile._iter_tool_calls_seen = iter_tool_calls_seen

    try:
      # Budget is fresh per attempt. Checkpoint preserves messages/tool ops
      # for LLM context; it does not consume the new attempt's budget.
      for iteration in range(effective_max_iterations):
        # ── Check if task was cancelled while running ──
        if iteration > 0 and iteration % 2 == 0:
            try:
                from ..infra.db import get_task as _get_task
                _current = await _get_task(task_id)
                if _current and _current.get("status") == "cancelled":
                    logger.info(f"[Task #{task_id}] Cancelled by user, aborting")
                    return {"status": "cancelled", "result": "Task cancelled by user"}
            except Exception:
                pass

        logger.info(
            f"[Task #{task_id}] Agent '{profile.name}' iteration "
            f"{iteration + 1}/{effective_max_iterations}"
        )
        # ── Accumulated messages telemetry ──
        # ReAct loop appends assistant+tool_result pairs each iteration;
        # tool outputs can be 0.1-50kB each. The pruner trims when over
        # 80% ctx, but a 90k-token call may originate from accumulated
        # messages rather than _build_context alone. Log size + role
        # breakdown so retries-after-many-iterations are visible.
        try:
            _msg_counts: dict[str, int] = {}
            _msg_chars: dict[str, int] = {}
            for _m in messages:
                _role = _m.get("role", "?")
                _c = _m.get("content") or ""
                if not isinstance(_c, str):
                    _c = str(_c)
                _msg_counts[_role] = _msg_counts.get(_role, 0) + 1
                _msg_chars[_role] = _msg_chars.get(_role, 0) + len(_c)
            _total_chars = sum(_msg_chars.values())
            _breakdown = " ".join(
                f"{r}={_msg_counts[r]}#/{_msg_chars[r]}c"
                for r in sorted(_msg_chars.keys())
            )
            logger.info(
                f"[Task #{task_id}] messages state iter "
                f"{iteration + 1}: total={_total_chars}c "
                f"n={len(messages)} {_breakdown}"
            )
        except Exception as _exc:
            logger.debug(f"messages-size telemetry failed: {_exc!r}")
        # Per-task progress signal — orchestrator's no-progress
        # watchdog (src/core/heartbeat.py) keys off these bumps to
        # decide whether the dispatch is wedged.
        try:
            from src.core import heartbeat as _hb
            _hb.bump(task_id)
        except Exception:
            pass

        # ── Phase 4.6: Progress streaming ──
        _now = time.time()
        if (progress_callback
                and iteration > 0
                and _now - _progress_last_sent >= 15):
            try:
                # Summarize last action with meaningful context
                _last_action = ""
                for _m in reversed(messages):
                    if _m.get("role") == "assistant":
                        _content = _m.get("content") or ""
                        if _m.get("tool_calls"):
                            _tc = _m["tool_calls"][0]
                            _fn = _tc.get("function", {}).get("name", "tool")
                            _last_action = f"Using {_fn}..."
                        elif _content.lstrip().startswith(("{", "[", "```")):
                            import json as _pjson
                            import re as _re
                            # Strip markdown code fences before parsing
                            _raw = _re.sub(r"^```(?:json)?\s*|\s*```$", "", _content.strip())
                            try:
                                _parsed = _pjson.loads(_raw)
                                _action = _parsed.get("action", "")
                                if _action == "tool_call":
                                    _tool = _parsed.get("tool", "tool")
                                    _last_action = f"Using {_tool}..."
                                elif _action == "final_answer":
                                    _last_action = "Finalizing answer..."
                                elif _action:
                                    _last_action = f"{_action.replace('_', ' ').capitalize()}..."
                                else:
                                    _last_action = "Processing..."
                            except Exception:
                                _last_action = "Processing..."
                        else:
                            # Show a snippet of the LLM's reasoning
                            _snippet = _content.strip()[:80]
                            _last_action = f"Thinking: {_snippet}..." if _snippet else "Processing..."
                        break
                await progress_callback(
                    task_id, iteration + 1, effective_max_iterations, _last_action
                )
                _progress_last_sent = _now
            except Exception:
                pass

        # ── Inner correction loop ──
        # Guards and format corrections are handled as sub-iterations
        # within the SAME outer iteration, so they don't burn iteration budget.
        sub_corrections = 0

        while sub_corrections <= MAX_SUB_CORRECTIONS:
            # ── Update token estimates ──
            estimation_model = used_model if used_model != "unknown" else "gpt-4o-mini"
            reqs.estimated_input_tokens = count_tokens(
                messages, estimation_model
            )
            reqs.estimated_output_tokens = min(
                reqs.estimated_output_tokens, 4096,
            )

            # ── Prune oldest tool-result pairs if estimate exceeds
            # context budget (cheap char/3 guard before the heavier
            # compression below).  llama-server silently truncates long
            # prompts otherwise — we prefer to drop oldest tool output
            # visibly and log it.
            try:
                _ctx_win = context_window_for(estimation_model, reqs)
                messages = prune_tool_results(
                    messages,
                    ctx_window=_ctx_win,
                    estimated_output_tokens=reqs.estimated_output_tokens,
                    task_id=task_id,
                )
            except Exception as _prune_exc:
                logger.debug(f"[Task #{task_id}] prune skipped: {_prune_exc}")

            # ── Trim context ── (now accepts reqs directly)
            messages = trim_if_needed(
                messages, estimation_model, reqs,
            )

            # ── Tools ──
            # Hard guardrail: on the LAST iteration, strip all tools so the
            # LLM is forced to produce a text response (final_answer).
            # Small models ignore "LAST ITERATION" text warnings — this makes
            # it physically impossible to call tools on the final turn.
            #
            # Also strip tools when running low on time — local LLMs need
            # 120+ seconds to generate a full analysis.  Without this, the
            # agent wastes iterations on tool calls and then the task-level
            # timeout kills the final-answer LLM call mid-generation.
            is_last_iteration = (iteration + 1 >= effective_max_iterations)
            _elapsed = time.time() - _start_time
            _time_budget = getattr(profile, '_task_timeout', 300)
            _remaining = _time_budget - _elapsed
            if not is_last_iteration and _remaining < 120 and iteration > 0:
                logger.warning(
                    f"[Task #{task_id}] Forcing final answer: "
                    f"only {_remaining:.0f}s remaining (need 120s for answer)"
                )
                is_last_iteration = True
            if is_last_iteration:
                litellm_tools = None
                # Inject a system reminder that tools are gone
                # (only on first sub-correction pass to avoid duplicates)
                if sub_corrections == 0:
                    messages.append({
                        "role": "user",
                        "content": (
                            "FINAL ITERATION — no tools available. You MUST produce your "
                            "final answer NOW as plain text or JSON. Summarize everything "
                            "you have gathered so far."
                        ),
                    })
            else:
                # Dynamic tool strip: if the agent already read a file
                # earlier this task, drop ``read_file`` from the schema
                # so the LLM reuses the blackboard/prior tool results
                # instead of re-reading.
                _exclude: set[str] = set()
                if "read_file" in iter_tool_calls_seen:
                    _exclude.add("read_file")
                litellm_tools = build_litellm_tools(profile.allowed_tools, exclude=_exclude)
            if litellm_tools:
                reqs.needs_function_calling = True

            # ── Call LLM ──
            profile._partial_content = ""
            try:
                from src.core.llm_dispatcher import get_dispatcher, CallCategory
                response = await get_dispatcher().request(
                    CallCategory.MAIN_WORK,
                    task=reqs.effective_task or reqs.primary_capability,
                    agent_type=reqs.agent_type,
                    difficulty=reqs.difficulty,
                    messages=messages,
                    tools=litellm_tools,
                    needs_thinking=reqs.needs_thinking,
                    needs_function_calling=reqs.needs_function_calling,
                    needs_vision=reqs.needs_vision,
                    local_only=reqs.local_only,
                    prefer_speed=reqs.prefer_speed,
                    prefer_quality=reqs.prefer_quality,
                    prefer_local=reqs.prefer_local,
                    estimated_input_tokens=reqs.estimated_input_tokens,
                    estimated_output_tokens=reqs.estimated_output_tokens,
                    min_context=reqs.effective_context_needed,
                    priority=reqs.priority,
                    exclude_models=reqs.exclude_models or [],
                    remaining_budget=max(0.0, _remaining),
                    preselected_pick=task.get("preselected_pick") if iteration == 0 else None,
                    task_obj=task,
                    iteration_n=iteration,
                )
            except Exception as exc:
                # Let ModelCallFailed propagate — the orchestrator handles
                # it as an availability failure with backoff + wake signals.
                from src.core.router import ModelCallFailed
                _NON_RETRYABLE = (ModelCallFailed, AttributeError, TypeError,
                                  ImportError, NameError, KeyError)
                if isinstance(exc, _NON_RETRYABLE):
                    raise
                logger.error(f"[Task #{task_id}] Model call failed: {exc}")
                return {
                    "status": "failed",
                    "result": f"Agent failed after {iteration} iteration(s): {exc}",
                    "error": str(exc),
                    "model": used_model,
                    "cost": total_cost,
                    "iterations": iteration,
                    "difficulty": reqs.difficulty,
                }

            content    = response.get("content", "")
            used_model = response.get("model", used_model)
            step_cost  = response.get("cost", 0)
            step_latency = response.get("latency", 0)
            total_cost += step_cost

            try:
                await record_model_call(
                    model=used_model,
                    agent_type=profile.name,
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

            logger.info(f"[Task #{task_id}] Raw response ({len(content)} chars):\n{content}")

            # Skip empty responses — don't burn an iteration on nothing.
            if not content and not response.get("tool_calls"):
                logger.warning(
                    f"[Task #{task_id}] Empty response (0 chars, no tool_calls) "
                    f"— not counting as iteration {iteration + 1}/{effective_max_iterations}"
                )
                empty_response_count += 1
                if empty_response_count >= 3:
                    return {
                        "status": "failed",
                        "error": f"Model returned {empty_response_count} consecutive empty responses",
                        "model": used_model,
                        "cost": total_cost,
                    }
                continue  # retry same iteration
            empty_response_count = 0  # reset on non-empty

            await safe_log_conversation_p(profile, 
                task_id, "assistant", content, used_model, step_cost
            )

            # ── Parse response ──
            fc_tool_calls = response.get("tool_calls")
            parsed = None
            if fc_tool_calls:
                parsed = parse_function_call(fc_tool_calls)
            if parsed is None:
                parsed = parse_action(content)

            # ── FORMAT CORRECTION (sub-iteration) ──
            if parsed is None:
                # If the response is substantial natural-language text
                # (not a raw tool call or envelope), accept as final
                # answer rather than wasting a correction on format.
                _looks_like_tool = any(
                    marker in content[:100]
                    for marker in (
                        '"tool_call"', '"tool":', "<|function_call|>",
                        "```tool_code", '"write_file"', '"read_file"',
                    )
                )
                if len(content) > 200 and not _looks_like_tool:
                    result_text = unwrap_final_answer(content)
                    logger.info(
                        f"[Task #{task_id}] Accepting unparsed response "
                        f"as final answer ({len(content)} chars)"
                    )
                    parsed = {"action": "final_answer", "result": result_text}
                elif format_corrections < MAX_FORMAT_CORRECTIONS and sub_corrections < MAX_SUB_CORRECTIONS:
                    format_corrections += 1
                    sub_corrections += 1
                    logger.warning(
                        f"[Task #{task_id}] JSON parse failed — "
                        f"format-correction {format_corrections}/{MAX_FORMAT_CORRECTIONS}"
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
                    continue  # inner loop — re-prompt LLM
                else:
                    parsed = {
                        "action": "final_answer",
                        "result": (
                            f"[Parse failure] Agent could not produce valid "
                            f"JSON after {MAX_FORMAT_CORRECTIONS} format corrections. "
                            f"Raw output:\n{content[:2000]}"
                        ),
                    }

            try:
                parsed = validate_action(parsed)
            except ValueError as exc:
                logger.warning(f"[Task #{task_id}] Action validation warning: {exc}")

            # ── SUB-ITERATION GUARD CHECK ──
            correction = check_sub_iter_guards(
                profile=profile,
                parsed=parsed,
                iteration=iteration,
                tools_used=tools_used,
                tools_used_names=tools_used_names,
                task=task,
                search_depth=_search_depth,
                suppress_guards=_suppress_guards,
            )
            if correction and sub_corrections < MAX_SUB_CORRECTIONS:
                guard_burns += 1
                logger.warning(
                    f"[Task #{task_id}] [{correction.guard_name}] "
                    f"sub-correction {sub_corrections + 1}/{MAX_SUB_CORRECTIONS}"
                )
                await safe_log_conversation_p(profile, 
                    task_id, "system",
                    f"[{correction.guard_name}] sub-correction",
                    None, 0,
                )
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": correction.message})
                sub_corrections += 1
                continue  # inner loop — re-prompt LLM

            # ── CUSTOM VALIDATION (sub-iteration) ──
            action_type = parsed.get("action", "final_answer")
            if action_type == "final_answer":
                result = parsed.get("result", content)
                if not isinstance(result, str):
                    result = json.dumps(result, ensure_ascii=False, indent=2)

                if not custom_validation_retried and sub_corrections < MAX_SUB_CORRECTIONS:
                    validation_error = validate_final_answer(result, task)
                    if validation_error:
                        custom_validation_retried = True
                        sub_corrections += 1
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": f"{validation_error}\n\nPlease try again.",
                        })
                        continue  # inner loop — re-prompt LLM

                # ── TASK-TYPE VALIDATION (sub-iteration) ──
                task_type_errors = validate_task_output(profile.name, result)
                if task_type_errors and not task_type_validation_retried and sub_corrections < MAX_SUB_CORRECTIONS:
                    task_type_validation_retried = True
                    err_msg = "; ".join(task_type_errors)
                    sub_corrections += 1
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"Output quality issue: {err_msg}\n\nPlease revise.",
                    })
                    continue  # inner loop — re-prompt LLM

            break  # No guard/correction fired → proceed to action handling

        # Save checkpoint after inner loop completes
        await save_checkpoint(
            task_id, iteration + 1, messages, total_cost,
            used_model, reqs, tools_used,
            custom_validation_retried or task_type_validation_retried,
            completed_tool_ops, format_corrections,
            tools_used_names,
        )

        action_type = parsed.get("action", "final_answer")

        # ── FINAL ANSWER ──
        if action_type == "final_answer":
            result = parsed.get("result", content)
            # Ensure result is always a string — LLMs sometimes return
            # a dict/list as the result value instead of text.
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, indent=2)

            # Memories
            raw_memories = parsed.get("memories", {})
            if raw_memories and isinstance(raw_memories, dict):
                for key, value in raw_memories.items():
                    try:
                        await store_memory(
                            key, str(value),
                            category=profile.name, mission_id=mission_id,
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
                f"can_create_subtasks={profile.can_create_subtasks}"
            )

            # Normalize subtask keys — LLMs sometimes use "tasks",
            # "steps", "plan" instead of "subtasks"
            subtasks = parsed.get("subtasks")
            if not subtasks and profile.can_create_subtasks:
                for alt_key in ("tasks", "steps", "plan", "sub_tasks"):
                    candidate = parsed.get(alt_key)
                    if isinstance(candidate, list) and candidate:
                        subtasks = candidate
                        logger.debug(
                            f"[Task #{task_id}] Found subtasks under "
                            f"alt key '{alt_key}' ({len(candidate)} items)"
                        )
                        break

            # Workflow steps must produce output directly — never delegate
            _is_wf = bool(_task_ctx.get("is_workflow_step"))
            if subtasks and not _is_wf:
                return {
                    "status":       "needs_subtasks",
                    "subtasks":     subtasks,
                    "plan_summary": parsed.get("plan_summary", ""),
                    "model":        used_model,
                    "cost":         total_cost,
                    "difficulty":   reqs.difficulty,
                }
            elif subtasks and _is_wf:
                # Agent tried to create subtasks in a workflow step —
                # treat the plan_summary or result as the actual output
                logger.warning(
                    f"[Task #{task_id}] Planner tried to create subtasks "
                    f"in workflow step — using result directly"
                )
                result = parsed.get("plan_summary") or parsed.get("result", result)

            if parsed.get("needs_clarification"):
                return {
                    "status":        "needs_clarification",
                    "clarification": parsed["needs_clarification"],
                    "model":         used_model,
                    "cost":          total_cost,
                    "difficulty":    reqs.difficulty,
                }

            # Self-reflection
            if profile.enable_self_reflection:
                try:
                    reflection = await self_reflect(
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
                profile.min_confidence > 0
                and isinstance(confidence, (int, float))
                and confidence < profile.min_confidence
            ):
                return {
                    "status":      "needs_review",
                    "result":      result,
                    "review_note": f"Agent confidence: {confidence}/5",
                    "model":       used_model,
                    "cost":        total_cost,
                    "difficulty":  reqs.difficulty,
                }

            return {
                "status":           "completed",
                "result":           result,
                "model":            used_model,
                "cost":             total_cost,
                "difficulty":       reqs.difficulty,
                "iterations":       iteration + 1,
                "tools_used_names": sorted(tools_used_names),
                "generating_model": used_model,  # surfaced for post-hook scheduling
            }

        # ── TOOL CALL ──
        if action_type == "tool_call":
            tools_used = True
            tool_name = parsed.get("tool", "")
            tools_used_names.add(tool_name)
            iter_tool_calls_seen.add(tool_name)
            tool_args = parsed.get("args", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            # ── Intercept read_file for artifacts already in context ──
            # Workflow steps inject input artifacts into the prompt.
            # If the model tries to read_file for one of them, short-
            # circuit with a nudge — saves an iteration + tool call.
            if tool_name == "read_file" and _task_ctx.get("is_workflow_step"):
                _input_arts = _task_ctx.get("input_artifacts", [])
                _filepath = tool_args.get("filepath", "")
                _matched = any(
                    art in _filepath for art in _input_arts
                ) if _input_arts and _filepath else False
                if _matched:
                    tool_output = (
                        "This artifact is already included in your "
                        "context above under '## Context Artifacts'. "
                        "Use it directly — do NOT call read_file for it."
                    )
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"Tool Result ({tool_name}):\n{tool_output}",
                    })
                    await save_checkpoint(
                        task_id, iteration + 1, messages, total_cost,
                        used_model, reqs, tools_used,
                        custom_validation_retried or task_type_validation_retried,
                        completed_tool_ops, format_corrections,
                        tools_used_names,
                    )
                    continue

            if (
                profile.allowed_tools is not None
                and tool_name not in profile.allowed_tools
            ):
                tool_output = (
                    f"❌ Tool '{tool_name}' not available. "
                    f"Allowed: {profile.allowed_tools}"
                )
            elif not check_tool_permission(profile.name, tool_name):
                tool_output = (
                    f"🚫 Tool '{tool_name}' not permitted for agent "
                    f"type '{profile.name}' (security policy)."
                )
                logger.warning(
                    f"[Task #{task_id}] Permission denied: "
                    f"{profile.name} → {tool_name}"
                )
            elif tool_name not in TOOL_REGISTRY:
                tool_output = (
                    f"❌ Unknown tool '{tool_name}'. "
                    f"Available: {list(TOOL_REGISTRY.keys())}"
                )
            else:
                arg_schema = TOOL_SCHEMAS_BY_NAME.get(tool_name)
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
                        await save_checkpoint(
                            task_id, iteration + 1, messages, total_cost,
                            used_model, reqs, tools_used,
                            custom_validation_retried or task_type_validation_retried,
                            completed_tool_ops, format_corrections,
                        )
                        continue

                idem_key = tool_idempotency_key(tool_name, tool_args)

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
                        # Per-tool timeout: 120s for shell tools, 60s for others.
                        # Prevents a single hung tool from blocking the agent loop.
                        _tool_timeout = 120 if tool_name in (
                            "shell", "shell_stdin", "shell_sequential",
                        ) else 60
                        # Build task hints for context-aware tools
                        _hints = {
                            "agent_type": profile.name,
                            "search_depth": get_search_depth(task),
                            "shopping_sub_intent": task.get("shopping_sub_intent"),
                            "workspace_path": _task_ctx.get("workspace_path", ""),
                        }

                        tool_output = await asyncio.wait_for(
                            execute_tool(
                                tool_name, agent_type=profile.name, task_hints=_hints, **tool_args
                            ),
                            timeout=_tool_timeout,
                        )
                    except asyncio.TimeoutError:
                        tool_output = (
                            f"❌ Tool '{tool_name}' timed out after "
                            f"{_tool_timeout}s — try a simpler approach."
                        )
                    except Exception as exc:
                        tool_output = f"❌ Tool execution error: {exc}"

                    # Phase 8.4: Audit log tool execution
                    try:
                        from ..infra.audit import audit, ACTOR_AGENT, ACTION_TOOL_EXEC
                        _tid = int(task_id) if str(task_id).isdigit() else None
                        await audit(
                            actor=f"{ACTOR_AGENT}:{profile.name}",
                            action=ACTION_TOOL_EXEC,
                            target=tool_name,
                            details=str(tool_args)[:500],
                            task_id=_tid,
                            mission_id=mission_id,
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

                        # Phase E: Post-tool reindexing for file-modifying tools
                        if tool_name in ("write_file", "edit_file", "patch_file", "apply_diff"):
                            _target_file = tool_args.get("filepath", tool_args.get("path", ""))
                            if _target_file:
                                try:
                                    from ..parsing.code_embeddings import post_tool_reindex
                                    _repo = context.get("repo_path", "") if isinstance(context, dict) else ""
                                    await post_tool_reindex(_target_file, root_path=_repo)
                                except Exception:
                                    pass
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
                useful_iterations += 1

            # ── Mid-task escalation ── (uses reqs.escalate())
            if (
                not model_escalated
                and consecutive_tool_failures >= TOOL_FAILURE_ESCALATION_THRESHOLD
                and iteration >= TOOL_FAILURE_ESCALATION_THRESHOLD
            ):
                old_tier = reqs.difficulty
                reqs = escalate_requirements(reqs)
                new_tier = reqs.difficulty
                if new_tier != old_tier:
                    logger.warning(
                        f"[Task #{task_id}] ⬆️ model-escalation: "
                        f"'{old_tier}' → '{new_tier}' after "
                        f"{consecutive_tool_failures} consecutive failures"
                    )
                    model_escalated = True
                    await safe_log_conversation_p(profile, 
                        task_id, "system",
                        f"[escalation] Upgraded quality after "
                        f"{consecutive_tool_failures} failures",
                        None, 0,
                    )
                    # Reset context for the better model
                    messages = trim_for_escalation(
                        messages, iteration, effective_max_iterations,
                    )

            if tool_failed:
                recovery_guidance = (
                    f"## Tool Result (`{tool_name}`) — ERROR:\n\n"
                    f"```\n{tool_output}\n```\n\n"
                    f"The tool call failed. Try a DIFFERENT approach.\n"
                    f"Iteration {iteration + 2}/{effective_max_iterations}."
                )
            else:
                recovery_guidance = (
                    f"## Tool Result (`{tool_name}`):\n\n"
                    f"```\n{tool_output}\n```\n\n"
                    f"{'LAST ITERATION — you MUST respond with final_answer now. Do NOT call any more tools.' if iteration + 2 >= effective_max_iterations else 'Continue working.'} Iteration {iteration + 2}/{effective_max_iterations}."
                )

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": recovery_guidance})

            await safe_log_conversation_p(profile, 
                task_id, "tool",
                f"[{tool_name}] {tool_output[:2000]}",
                None, 0,
            )
            await save_checkpoint(
                task_id, iteration + 1, messages, total_cost,
                used_model, reqs, tools_used, custom_validation_retried or task_type_validation_retried,
                completed_tool_ops, format_corrections,
            )
            continue

        # ── MULTI TOOL CALL (parallel read-only, sequential side-effect) ──
        if action_type == "multi_tool_call":
            tools_used = True
            tool_list = parsed.get("tools", [])

            # Validate each tool call
            validated: list[tuple[str, dict, str | None]] = []
            for tc in tool_list:
                t_name = tc.get("tool", "")
                t_args = tc.get("args", {})
                if not isinstance(t_args, dict):
                    t_args = {}
                tools_used_names.add(t_name)
                iter_tool_calls_seen.add(t_name)

                if profile.allowed_tools is not None and t_name not in profile.allowed_tools:
                    validated.append((t_name, t_args, f"❌ Tool '{t_name}' not available."))
                elif not check_tool_permission(profile.name, t_name):
                    validated.append((t_name, t_args, f"🚫 Tool '{t_name}' not permitted."))
                elif t_name not in TOOL_REGISTRY:
                    validated.append((t_name, t_args, f"❌ Unknown tool '{t_name}'."))
                else:
                    validated.append((t_name, t_args, None))

            to_execute = [(n, a) for n, a, err in validated if err is None]
            errors = [(n, err) for n, _, err in validated if err is not None]

            parallel_group, sequential_group = partition_tool_calls(
                [{"tool": n, "args": a} for n, a in to_execute]
            )

            results: list[tuple[str, dict, str]] = []

            # --- Parallel group (read-only) ---
            if parallel_group:
                async def _exec_one(tc_item: dict) -> tuple[str, dict, str]:
                    _tn, _ta = tc_item["tool"], tc_item["args"]
                    _timeout = 120 if _tn in ("shell", "shell_stdin", "shell_sequential") else 60
                    _hints = {
                        "agent_type": profile.name,
                        "search_depth": _search_depth,
                        "shopping_sub_intent": task.get("shopping_sub_intent"),
                        "workspace_path": _task_ctx.get("workspace_path", ""),
                    }
                    try:
                        out = await asyncio.wait_for(
                            execute_tool(_tn, agent_type=profile.name, task_hints=_hints, **_ta),
                            timeout=_timeout,
                        )
                    except asyncio.TimeoutError:
                        out = f"❌ Tool '{_tn}' timed out after {_timeout}s"
                    except Exception as exc:
                        out = f"❌ Tool execution error: {exc}"
                    return _tn, _ta, out

                par_results = await asyncio.gather(
                    *[_exec_one(tc_item) for tc_item in parallel_group],
                    return_exceptions=True,
                )
                for r in par_results:
                    if isinstance(r, Exception):
                        results.append(("unknown", {}, f"❌ Parallel error: {r}"))
                    else:
                        results.append(r)

            # --- Sequential group (side-effect) ---
            for tc_item in sequential_group:
                _tn, _ta = tc_item["tool"], tc_item["args"]
                _timeout = 120 if _tn in ("shell", "shell_stdin", "shell_sequential") else 60
                _hints = {
                    "agent_type": profile.name,
                    "search_depth": _search_depth,
                    "shopping_sub_intent": task.get("shopping_sub_intent"),
                    "workspace_path": _task_ctx.get("workspace_path", ""),
                }
                try:
                    out = await asyncio.wait_for(
                        execute_tool(_tn, agent_type=profile.name, task_hints=_hints, **_ta),
                        timeout=_timeout,
                    )
                except asyncio.TimeoutError:
                    out = f"❌ Tool '{_tn}' timed out after {_timeout}s"
                except Exception as exc:
                    out = f"❌ Tool execution error: {exc}"
                results.append((_tn, _ta, out))

            # Add pre-validation errors
            for n, err in errors:
                results.append((n, {}, err))

            # Audit, metrics, caching per tool result
            for t_name, t_args, t_output in results:
                # Audit log
                try:
                    from ..infra.audit import audit, ACTOR_AGENT, ACTION_TOOL_EXEC
                    _tid = int(task_id) if str(task_id).isdigit() else None
                    await audit(
                        actor=f"{ACTOR_AGENT}:{profile.name}",
                        action=ACTION_TOOL_EXEC,
                        target=t_name,
                        details=str(t_args)[:500],
                        task_id=_tid,
                        mission_id=mission_id,
                    )
                except Exception:
                    pass

                # Metrics
                try:
                    from ..infra.metrics import record_tool_call
                    record_tool_call(tool=t_name)
                except Exception:
                    pass

                # Cache results
                if t_name in SIDE_EFFECT_TOOLS:
                    idem_key = tool_idempotency_key(t_name, t_args)
                    completed_tool_ops[idem_key] = t_output
                    _to_remove = [k for k in completed_tool_ops if k.startswith("rc:")]
                    for k in _to_remove:
                        del completed_tool_ops[k]
                elif t_name in CACHEABLE_READ_TOOLS:
                    idem_key = tool_idempotency_key(t_name, t_args)
                    completed_tool_ops[f"rc:{idem_key}"] = t_output

            # Build combined result message
            result_parts = []
            tool_failures = 0
            for t_name, t_args, t_output in results:
                if len(t_output) > MAX_TOOL_OUTPUT_LENGTH:
                    t_output = (
                        t_output[:MAX_TOOL_OUTPUT_LENGTH]
                        + f"\n\n... [{len(t_output)} chars total]"
                    )
                key_arg = next(iter(t_args.values()), "") if t_args else ""
                if isinstance(key_arg, str) and len(key_arg) > 60:
                    key_arg = key_arg[:60]
                result_parts.append(
                    f"## Tool Result (`{t_name}` → {key_arg}):\n\n"
                    f"```\n{t_output}\n```"
                )
                if t_output.startswith("❌") or t_output.startswith("🚫"):
                    tool_failures += 1

            if tool_failures > 0:
                consecutive_tool_failures += tool_failures
            else:
                consecutive_tool_failures = 0
                useful_iterations += 1

            is_next_last = (iteration + 2 >= effective_max_iterations)
            combined = "\n\n".join(result_parts)
            combined += (
                f"\n\n{'LAST ITERATION — you MUST respond with final_answer now.' if is_next_last else 'Continue working.'}"
                f" Iteration {iteration + 2}/{effective_max_iterations}."
            )

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": combined})

            await safe_log_conversation_p(profile, 
                task_id, "tool",
                f"[multi:{len(results)} tools] {', '.join(n for n, _, _ in results)}",
                None, 0,
            )
            await save_checkpoint(
                task_id, iteration + 1, messages, total_cost,
                used_model, reqs, tools_used, custom_validation_retried or task_type_validation_retried,
                completed_tool_ops, format_corrections,
            )
            continue

        # ── ASK AGENT (inter-agent query) ──
        if action_type == "ask_agent":
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
                    "title": f"[Inline query from {profile.name}]",
                    "description": question,
                    "mission_id": mission_id,
                    "context": json.dumps({"tool_depth": 1}),
                }
                inline_result = await asyncio.wait_for(
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
            except asyncio.TimeoutError:
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
                    f"{'LAST ITERATION — you MUST respond with final_answer now. Do NOT call any more tools.' if iteration + 2 >= effective_max_iterations else 'Continue working.'} Iteration {iteration + 2}/{effective_max_iterations}."
                ),
            })
            await save_checkpoint(
                task_id, iteration + 1, messages, total_cost,
                used_model, reqs, tools_used,
                custom_validation_retried or task_type_validation_retried,
                completed_tool_ops, format_corrections,
            )
            continue

        # ── CLARIFY / DECOMPOSE / UNKNOWN ──
        # NOTE: Blocked clarification (suppress_clarification=True) is now
        # handled as a sub-iteration guard — see _check_sub_iteration_guards.
        # Only the non-suppressed return path remains here.
        if action_type == "clarify":
            return {
                "status": "needs_clarification",
                "clarification": parsed.get("question", content),
                "model": used_model, "cost": total_cost, "difficulty": reqs.difficulty,
            }

        if action_type == "decompose":
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
        await save_checkpoint(
            task_id, iteration + 1, messages, total_cost,
            used_model, reqs, tools_used, custom_validation_retried or task_type_validation_retried,
            completed_tool_ops, format_corrections,
        )

    except asyncio.CancelledError:
        # Timeout — save checkpoint so previous work survives for retry.
        # Include any partial LLM output captured via streaming.
        try:
            partial = getattr(profile, "_partial_content", "")
            if partial and len(partial) > 50:
                messages.append({"role": "assistant", "content": partial})
                logger.info(
                    f"[Task #{task_id}] Saved {len(partial)} chars of "
                    f"partial LLM output from interrupted generation"
                )
            await save_checkpoint(
                task_id, iteration, messages, total_cost,
                used_model, reqs, tools_used,
                False, completed_tool_ops, format_corrections,
                tools_used_names,
            )
            logger.info(
                f"[Task #{task_id}] Timeout checkpoint saved at iteration {iteration}"
            )
        except Exception:
            pass
        raise  # re-raise so orchestrator's TimeoutError handler fires

    # ── Exhausted iterations ──
    # Checkpoint is preserved here — orchestrator will clear it on final completion.

    # Classify exhaustion reason
    if guard_burns >= effective_max_iterations * 0.5:
        exhaustion_reason = "guards"
    elif consecutive_tool_failures >= TOOL_FAILURE_ESCALATION_THRESHOLD:
        exhaustion_reason = "tool_failures"
    else:
        exhaustion_reason = "budget"

    # Extract last meaningful assistant response for the result.
    # Do NOT truncate before unwrapping — truncation breaks JSON parsing.
    last_assistant = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            last_assistant = msg["content"]
            break
    # Try to parse as JSON and extract "result" field — the LLM often
    # wraps its answer in {"action": "final_answer", "result": "..."}
    if last_assistant:
        parsed_final = parse_action(last_assistant)
        if parsed_final and parsed_final.get("result"):
            last_assistant = parsed_final["result"]
        elif '"result"' in last_assistant and '"final_answer"' in last_assistant:
            # JSON parse failed (truncated by context trimming?) — regex fallback
            import re as _re
            m = _re.search(r'"result"\s*:\s*"((?:[^"\\]|\\.)*)', last_assistant)
            if m:
                try:
                    last_assistant = m.group(1).encode().decode('unicode_escape')
                except Exception:
                    last_assistant = m.group(1)
    # Truncate AFTER unwrapping — preserve the actual content.
    # 8000 chars is well above _SUMMARY_THRESHOLD (3000) so the post-hook
    # will always create a summary for large artifacts.
    if len(last_assistant) > 8000:
        last_assistant = last_assistant[:8000]

    logger.warning(
        f"[Task #{task_id}] Exhausted iterations | "
        f"reason={exhaustion_reason} "
        f"guard_burns={guard_burns} "
        f"useful={useful_iterations}/{effective_max_iterations}"
    )

    return {
        "status": "exhausted",
        "result": last_assistant or "",
        "exhaustion_reason": exhaustion_reason,
        "guard_burns": guard_burns,
        "useful_iterations": useful_iterations,
        "model": used_model,
        "cost": total_cost,
        "difficulty": reqs.difficulty,
        "iterations": effective_max_iterations,
        "tools_used_names": sorted(tools_used_names),
    }
