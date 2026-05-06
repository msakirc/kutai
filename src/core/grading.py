"""
Task grading — structured binary evaluation.

Replaces the old 1-5 numeric grading with a structured YES/NO prompt.
All grading calls go through the LLM dispatcher as OVERHEAD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.infra.logging_config import get_logger
from src.infra.times import utc_now, db_now, to_db

logger = get_logger("core.grading")

GRADING_SYSTEM = (
    "You are a strict evaluator. Reply ONLY with the requested fields, "
    "one per line. Do not add explanation or commentary."
)

GRADING_PROMPT = """Evaluate this task result.

Task: {title}
Description: {description}
Result: {response}

Reply with EXACTLY these fields, one per line:
RELEVANT: YES or NO
COMPLETE: YES or NO
VERDICT: PASS or FAIL
WELL_FORMED: PASS or FAIL (no repeated sections, no garbage, structurally sound)
COHERENT: PASS or FAIL (output makes logical sense end-to-end)
SITUATION: one line, what type of problem was solved
STRATEGY: one line, what approach worked
TOOLS: comma-separated list of tools used effectively
PREFERENCE: one-line user preference signal observed in this task, or NONE
INSIGHT: one-line reusable learning from this task, or NONE"""


@dataclass
class GradeResult:
    passed: bool
    relevant: Optional[bool] = None
    complete: Optional[bool] = None
    well_formed: Optional[bool] = None
    coherent: Optional[bool] = None
    situation: str = ""
    strategy: str = ""
    tools: list[str] = field(default_factory=list)
    preference: str = ""
    insight: str = ""
    raw: str = ""


_THINKING_PATTERNS = [
    # <think>...</think> blocks (Qwen, DeepSeek-R1 style)
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    # "Thinking Process:" / "Thought Process:" preambles until a double newline or a KEY:
    re.compile(
        r"(?:^|\n)(?:thinking|thought)\s+process\s*:.*?(?=\n\s*\n|\n[A-Z_]{2,}\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    # "## Thinking" / "### Reasoning" markdown sections
    re.compile(
        r"(?:^|\n)#{1,6}\s*(?:thinking|reasoning|thought|analysis)\b.*?(?=\n#{1,6}\s|\n[A-Z_]{2,}\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    # Numbered-bullet analyze/evaluate preamble (Qwen3.5-A3B, Gemma style):
    #   "1. **Analyze the Request:** ..."   "2. Evaluate the Result ..."
    # Strip everything from the first such bullet up to the first structured KEY: line
    # or end of input. Matches only when the numbered bullet starts near the top.
    re.compile(
        r"(?:^|\n)\s*\d+\.\s+\*{0,2}(?:analyze|evaluate|assess|review|consider|examine)\b.*?(?=\n[A-Z_]{2,}\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
]


_FIELD_LINE_RE = re.compile(r"^[A-Z_]{2,}\s*:", re.MULTILINE)


def _tail_fields_region(text: str) -> str:
    """Return the suffix starting at the first structured KEY: line in the
    LAST contiguous run of such lines. If none found, return original text.

    Thinking-model output often puts final structured fields at the end after
    a reasoning blob. Parsing only the tail region avoids regex collisions
    with echoed keys like `Task:`, `Description:` inside the preamble.
    """
    matches = list(_FIELD_LINE_RE.finditer(text))
    if not matches:
        return text
    # Walk backwards — find the start of the last contiguous KEY: run.
    # Adjacent means ≤3 non-KEY lines between two KEY: lines.
    start = matches[-1].start()
    for i in range(len(matches) - 2, -1, -1):
        between = text[matches[i].end():matches[i + 1].start()]
        if between.count("\n") <= 3:
            start = matches[i].start()
        else:
            break
    return text[start:]


def _strip_thinking(raw: str) -> str:
    """Remove visible chain-of-thought from grader output.

    Some graders (thinking models without reasoning suppressed) emit
    `<think>...</think>` blocks or `Thinking Process:` preambles before
    the structured fields. Strip these so parsing can still find VERDICT.
    """
    stripped = raw
    for pattern in _THINKING_PATTERNS:
        stripped = pattern.sub("", stripped)
    return stripped.strip()


def _parse_yes_no(text: str, key: str) -> Optional[bool]:
    """Extract a YES/NO value for a given key. Line-anchored, last-match wins
    (thinking models may echo keys inside reasoning; the final line is truth)."""
    pattern = rf"^\s*{key}\s*:\s*(YES|NO|PASS|FAIL)\b"
    matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
    if not matches:
        return None
    val = matches[-1].upper()
    return val in ("YES", "PASS")


_NONE_VARIANTS = frozenset({"none", "n/a", "na", "no", "nil", "null", "-", "not applicable"})


def _is_none_value(val: str) -> bool:
    """Check if a parsed field value is a 'none' variant from the LLM."""
    if not val:
        return True
    normalized = val.strip().rstrip(".").lower()
    return normalized in _NONE_VARIANTS or normalized.startswith("no ")


# Pollution markers — grader-prose echoes / template leak / chain-of-thought.
# A SITUATION/STRATEGY value matching any of these is a parse failure, not data.
_POLLUTION_RE = re.compile(
    r"(?:"
    r"\bone line\b"                       # echo of prompt template
    r"|comma-separated list"              # template hint copied verbatim
    r"|\b(?:STRATEGY|TOOLS|PREFERENCE|INSIGHT|SITUATION)\s*:"  # multi-field swallow
    r"|^\s*\*"                            # bullet leak
    r"|\bWait,"                           # CoT marker
    r"|I am evaluating"                   # CoT marker
    r"|looking at the .{0,40}(?:prompt|result|output|task)"
    r"|Task Context\s*:"                  # CoT marker
    r"|Observation\s*:"                   # CoT marker
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def _sanitize_field(value: str, max_len: int = 400) -> str:
    """Drop the value if it looks like grader prose / CoT leak instead of a clean line.

    Returns "" on rejection, original (trimmed) value otherwise.
    """
    if not value:
        return ""
    v = value.strip()
    if len(v) > max_len:
        return ""
    if _POLLUTION_RE.search(v):
        return ""
    return v


def _parse_text_field(text: str, key: str) -> str:
    """Extract a free-text value for a given key from grader output.

    Captures everything after KEY: until the next uppercase KEY: marker
    or end of string. Handles values that wrap across multiple lines.
    """
    # Line-anchored start, last match wins. Value spans until next KEY: or EOF.
    pattern = rf"^\s*{key}\s*:\s*(.+?)(?=\n[A-Z_]{{2,}}\s*:|\Z)"
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if not matches:
        return ""
    value = matches[-1].strip()
    value = re.sub(r'\s*\n\s*', ' ', value)
    return value


def parse_grade_response(raw: str) -> GradeResult:
    """Parse structured grader output into a GradeResult.

    Parsing cascade (most structured → least):
      1. All 6 fields via regex
      2. If SITUATION/STRATEGY/TOOLS fail → grade still valid, skill fields empty
      3. If RELEVANT/COMPLETE fail → derive from VERDICT
      4. If VERDICT not found → scan for bare PASS/FAIL keyword
      5. Nothing → raise ValueError

    Thinking-model preambles (`<think>...</think>`, `Thinking Process: ...`)
    are stripped before parsing so reasoning leak doesn't hide a valid grade.
    """
    raw = _strip_thinking(raw)
    # Focus parsing on the last contiguous run of KEY: lines. Thinking models
    # put the final structured answer at the end after a reasoning blob, and
    # echoed keys in the preamble (e.g. "Task:", "Description:") must not
    # collide with field regexes.
    raw = _tail_fields_region(raw)
    relevant = _parse_yes_no(raw, "RELEVANT")
    complete = _parse_yes_no(raw, "COMPLETE")
    verdict = _parse_yes_no(raw, "VERDICT")
    well_formed = _parse_yes_no(raw, "WELL_FORMED")
    coherent = _parse_yes_no(raw, "COHERENT")

    # Skill extraction fields (optional — never block grading)
    situation = _sanitize_field(_parse_text_field(raw, "SITUATION"))
    strategy = _sanitize_field(_parse_text_field(raw, "STRATEGY"))
    tools_raw = _parse_text_field(raw, "TOOLS")
    # Reject tools list if it swallowed multiple fields.
    if tools_raw and _POLLUTION_RE.search(tools_raw):
        tools_raw = ""
    tools = [t.strip() for t in tools_raw.split(",") if t.strip()] if tools_raw else []

    # Piggybacked learning fields (optional)
    preference = _sanitize_field(_parse_text_field(raw, "PREFERENCE"))
    if _is_none_value(preference):
        preference = ""
    insight = _sanitize_field(_parse_text_field(raw, "INSIGHT"))
    if _is_none_value(insight):
        insight = ""

    # Cascade 1: VERDICT present
    if verdict is not None:
        effective_passed = verdict if well_formed is not False else False
        return GradeResult(
            passed=effective_passed, relevant=relevant, complete=complete,
            well_formed=well_formed, coherent=coherent,
            situation=situation, strategy=strategy, tools=tools,
            preference=preference, insight=insight, raw=raw,
        )

    # Cascade 2: derive from RELEVANT + COMPLETE
    if relevant is not None and complete is not None:
        derived = relevant and complete
        effective_passed = derived if well_formed is not False else False
        return GradeResult(
            passed=effective_passed, relevant=relevant, complete=complete,
            well_formed=well_formed, coherent=coherent,
            situation=situation, strategy=strategy, tools=tools,
            preference=preference, insight=insight, raw=raw,
        )

    # Cascade 3: bare PASS/FAIL keyword anywhere (last resort)
    # Strip WELL_FORMED/COHERENT lines first — their PASS/FAIL values
    # must not be mistaken for a verdict on the task itself.
    _stripped = re.sub(
        r'(?:WELL_FORMED|COHERENT)\s*:\s*(?:PASS|FAIL)\b', '', raw, flags=re.IGNORECASE,
    )
    bare = re.search(r'\bPASS\b', _stripped, re.IGNORECASE)
    if bare:
        effective_passed = True if well_formed is not False else False
        return GradeResult(passed=effective_passed, well_formed=well_formed, coherent=coherent,
                           situation=situation, strategy=strategy, tools=tools,
                           preference=preference, insight=insight, raw=raw)
    bare_fail = re.search(r'\bFAIL\b', _stripped, re.IGNORECASE)
    if bare_fail:
        return GradeResult(passed=False, well_formed=well_formed, coherent=coherent,
                           situation=situation, strategy=strategy, tools=tools,
                           preference=preference, insight=insight, raw=raw)

    raise ValueError(f"grader incapable: could not parse VERDICT, RELEVANT, or COMPLETE from output: {raw[:150]}")


async def grade_task(task: dict) -> GradeResult:
    """Grade a task's output via dispatcher OVERHEAD.

    Fatih Hoca picks the best available model; the grading system only
    tells the dispatcher what to EXCLUDE (the generating model and
    previously-failed graders), never what to USE.

    Args:
        task: Task dict with title, description, result, context

    Returns:
        GradeResult

    Raises:
        ValueError: grader parse failure (QualityError equivalent)
        RuntimeError: grading call failed (AvailabilityError equivalent)
    """
    import json
    # dispatcher import retained for backward compat with other callers in this module

    ctx = task.get("context", "{}")
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}

    generating_model = ctx.get("generating_model", "")
    grade_excluded = ctx.get("grade_excluded_models", [])
    all_excluded = list(set([generating_model] + grade_excluded))

    result_text = task.get("result", "")
    if not result_text or len(str(result_text).strip()) < 10:
        return GradeResult(passed=False, raw="auto-fail: trivial/empty output")

    from dogru_mu_samet import assess as cq_assess
    _grade_cq = cq_assess(str(result_text))
    if _grade_cq.is_degenerate:
        return GradeResult(passed=False, raw=f"auto-fail: {_grade_cq.summary}")

    import general_beckman
    import time as _time
    import uuid as _uuid
    from src.core.llm_dispatcher import _task_result_to_request_response

    messages = [
        {"role": "system", "content": GRADING_SYSTEM},
        {
            "role": "user",
            "content": GRADING_PROMPT.format(
                title=task.get("title", "")[:100],
                description=task.get("description", "")[:500],
                # Previously 4000 — truncated mid-JSON for shopping outputs
                # (26KB candidate lists). Grader saw cut-off structure and
                # rightly said WELL_FORMED=FAIL, triggering retry loops.
                # 30000 fits full candidate JSON; summarization post-hook
                # handles anything larger.
                response=str(result_text)[:30000],
            ),
        },
    ]

    graded_task_id = task.get("id")

    # Try with current exclusion list, then one retry with the grader model
    # that failed added to exclusions. Thinking-model reasoning leaks cannot
    # be fixed by server reload (swap cost >> retry cost).
    exclusions = list(all_excluded)
    last_raw = ""
    last_grader: str = ""
    for attempt in (0, 1):
        _suffix = f"{_time.monotonic_ns() % 1_000_000:06d}-{_uuid.uuid4().hex[:6]}"
        spec = {
            "title": f"grader:task#{graded_task_id}:{_suffix}",
            "description": "Grading review of task output",
            "agent_type": "reviewer",
            "kind": "overhead",
            "priority": 1,
            "context": {
                "llm_call": {
                    "raw_dispatch": True,
                    "call_category": "overhead",
                    "task": "reviewer",
                    "agent_type": "reviewer",
                    "difficulty": 3,
                    "messages": messages,
                    "failures": [],
                    "estimated_input_tokens": 800,
                    "estimated_output_tokens": 600,
                    "prefer_speed": True,
                    "exclude_models": exclusions,
                },
            },
        }
        task_result = await general_beckman.enqueue(
            spec,
            parent_id=graded_task_id,
            await_inline=True,
        )

        if task_result.status == "failed":
            logger.warning(
                f"grader enqueue failed attempt={attempt}: {task_result.error}",
                task_id=graded_task_id,
            )
            return GradeResult(
                passed=False,
                raw=f"auto-fail: grader call failed ({task_result.error})",
            )

        response = _task_result_to_request_response(task_result)

        raw_content = response.get("content", "")
        if isinstance(raw_content, list):
            raw_content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw_content
            )
        raw_str = str(raw_content)
        last_raw = raw_str
        last_grader = response.get("model", "")

        logger.debug(
            f"grader raw response ({len(raw_str)} chars) attempt={attempt} "
            f"model={last_grader}: {raw_str[:300]}",
            task_id=task.get("id"),
        )
        try:
            result = parse_grade_response(raw_str)
            # Annotate which grader model produced the verdict (separate from
            # the generating model in context). GraderAgent surfaces this.
            result.raw = raw_str
            return result
        except ValueError as e:
            logger.warning(
                f"grader parse failed attempt={attempt} model={last_grader} "
                f"full raw ({len(raw_str)} chars): {raw_str[:2000]}",
                task_id=task.get("id"),
            )
            if last_grader and last_grader not in exclusions:
                exclusions.append(last_grader)
            # Fall through to retry (attempt 1). On attempt 1 failure, exit loop.
            _last_error = str(e)

    # Both attempts failed — auto-fail the source task's grade rather than
    # raising. Raising would kill the grader task itself and cascade a DLQ
    # entry against the source. A false-negative is recoverable (worker
    # retries); a dead grader task is not.
    return GradeResult(
        passed=False,
        raw=f"auto-fail: grader_incapable after 2 attempts (last model={last_grader}): {last_raw[:300]}",
    )


async def apply_grade_result(task_id: int, verdict: GradeResult) -> None:
    """Apply grade outcome to a task. Handles PASS and FAIL.

    PASS: transition to completed, record model quality, trigger skill extraction.
    FAIL: increment attempts, add model to exclusions, retry or DLQ.
    """
    import json
    from src.infra.db import get_task, update_task
    from src.core.state_machine import transition_task
    from src.core.retry import compute_retry_timing

    task = await get_task(task_id)
    if not task:
        logger.warning(f"apply_grade_result: task #{task_id} not found")
        return

    ctx = task.get("context", "{}")
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}

    if verdict.passed:
        await transition_task(
            task_id, "completed",
            completed_at=db_now(),
        )

        # Record model quality feedback
        try:
            from src.infra.db import record_model_call
            await record_model_call(
                model=ctx.get("generating_model", ""),
                agent_type=task.get("agent_type", "executor"),
                success=True,
            )
        except Exception:
            pass

        # Workflow-step exemplar capture (replaces old grader-prose skill mint).
        # If the task title carries a [X.Y] step id, store the result + tool
        # sequence under (workflow, step_id, agent_type). Future runs of the
        # same step retrieve worked examples directly. Non-workflow tasks no
        # longer auto-mint skills — those are now seeds-only.
        try:
            from src.memory.workflow_exemplars import (
                extract_step_id, capture_exemplar,
            )
            agent_type = task.get("agent_type", "executor")
            title = task.get("title", "")
            step_id = extract_step_id(title)
            if step_id:
                # Look up workflow name from mission, if present.
                workflow_name = None
                mission_id = task.get("mission_id")
                if mission_id:
                    try:
                        from src.infra.db import get_db
                        db = await get_db()
                        async with db.execute(
                            "SELECT workflow FROM missions WHERE id=?",
                            (mission_id,),
                        ) as cur:
                            row = await cur.fetchone()
                            if row:
                                workflow_name = row[0]
                    except Exception:
                        pass
                result_text = task.get("result", "") or ""
                quality = task.get("quality_score") or 1.0
                if result_text and len(str(result_text).strip()) >= 50:
                    await capture_exemplar(
                        workflow=workflow_name,
                        step_id=step_id,
                        agent_type=agent_type,
                        result=str(result_text),
                        quality_score=float(quality),
                        task_id=task_id,
                        mission_id=mission_id,
                    )
        except Exception as e:
            logger.debug(f"workflow exemplar capture failed: {e}")

        # Telegram notification for non-silent tasks
        try:
            _is_silent = ctx.get("silent", False)
            if not _is_silent:
                from src.app.telegram_bot import get_telegram
                bot = get_telegram()
                if bot:
                    await bot.send_notification(
                        f"✅ Görev #{task_id} derecelendirildi ve tamamlandı\n"
                        f"**{task.get('title', '')[:60]}**"
                    )
        except Exception:
            pass

        # Track injection success for skills that were injected into this task
        try:
            injected = ctx.get("injected_skills", [])
            if injected:
                from src.memory.skills import record_injection_success
                await record_injection_success(injected)
        except Exception:
            pass

        # Preference extraction — piggybacked from grading output
        if verdict.preference:
            try:
                from src.memory.preferences import store_preference
                await store_preference(
                    preference=verdict.preference,
                    category="grader_observed",
                    chat_id=ctx.get("chat_id", "default"),
                    confidence=0.8,
                )
            except Exception as e:
                logger.debug(f"preference storage failed: {e}")

        # Insight extraction — piggybacked from grading output
        if verdict.insight:
            try:
                from src.memory.episodic import store_insight
                await store_insight(
                    insight_text=verdict.insight,
                    agent_type=task.get("agent_type", "executor"),
                    task_id=task_id,
                    task_title=task.get("title", ""),
                )
            except Exception as e:
                logger.debug(f"insight storage failed: {e}")

        logger.info(f"grade PASS | task_id={task_id}")
    else:
        # VERDICT=FAIL — worker quality failure
        generating_model = ctx.get("generating_model", "")
        from src.core.retry import RetryContext
        retry_ctx = RetryContext.from_task(task)
        decision = retry_ctx.record_failure("quality", model=generating_model)

        # Bonus attempt: if terminal but task made real progress,
        # grant extra attempts instead of DLQ (same logic as orchestrator).
        _MAX_BONUS = 2
        if decision.action == "terminal":
            bonus_count = ctx.get("_bonus_count", 0)
            if bonus_count < _MAX_BONUS:
                try:
                    # Check workspace files for progress
                    output_names = ctx.get("output_artifacts", [])
                    mission_id = ctx.get("mission_id") or task.get("mission_id")
                    has_progress = False
                    if mission_id and output_names:
                        import os
                        from src.tools.workspace import WORKSPACE_DIR
                        artifact_dir = os.path.join(WORKSPACE_DIR, f"mission_{mission_id}")
                        for name in output_names:
                            for ext in (".md", ".json", ".txt"):
                                fpath = os.path.join(artifact_dir, f"{name}{ext}")
                                if os.path.isfile(fpath) and os.path.getsize(fpath) > 200:
                                    has_progress = True
                                    break
                    if has_progress:
                        ctx["_bonus_count"] = bonus_count + 1
                        retry_ctx.max_worker_attempts += 1
                        decision = retry_ctx.record_failure("quality", model=generating_model)
                        logger.info(
                            f"grade bonus attempt | task_id={task_id} "
                            f"bonus={bonus_count + 1}/{_MAX_BONUS}"
                        )
                except Exception:
                    pass

        if decision.action == "terminal":
            ctx.update(retry_ctx.to_context_patch())
            await transition_task(
                task_id, "failed",
                context=json.dumps(ctx),
                **retry_ctx.to_db_fields(),
            )
            try:
                from src.infra.dead_letter import quarantine_task
                await quarantine_task(
                    task_id=task_id,
                    mission_id=task.get("mission_id"),
                    error=f"Quality gate failed after {retry_ctx.worker_attempts} attempts",
                    error_category="quality",
                    original_agent=task.get("agent_type", "executor"),
                    attempts_snapshot=retry_ctx.worker_attempts,
                )
            except Exception as e:
                logger.warning(f"DLQ quarantine failed: {e}")

            # Notify on terminal failure
            try:
                if not ctx.get("silent"):
                    from src.app.telegram_bot import get_telegram
                    bot = get_telegram()
                    if bot:
                        await bot.send_notification(
                            f"❌ Görev #{task_id} kalite kontrolünden geçemedi → DLQ\n"
                            f"**{task.get('title', '')[:60]}**"
                        )
            except Exception:
                pass

            logger.warning(f"grade FAIL terminal | task_id={task_id} attempts={retry_ctx.worker_attempts}")
        else:
            next_retry = None
            if decision.action == "delayed":
                from datetime import timedelta
                next_retry = to_db(utc_now() + timedelta(seconds=decision.delay_seconds))

            retry_ctx.next_retry_at = next_retry
            retry_ctx.grade_attempts = 0  # reset grade attempts on worker retry
            ctx.update(retry_ctx.to_context_patch())
            await transition_task(
                task_id, "pending",
                context=json.dumps(ctx),
                **retry_ctx.to_db_fields(),
            )

            # Notify on retry
            try:
                if not ctx.get("silent"):
                    from src.app.telegram_bot import get_telegram
                    bot = get_telegram()
                    if bot:
                        await bot.send_notification(
                            f"🔄 Görev #{task_id} çıktısı reddedildi, farklı model ile tekrar deniyor\n"
                            f"**{task.get('title', '')[:60]}**"
                        )
            except Exception:
                pass

            logger.info(f"grade FAIL retry | task_id={task_id} attempts={retry_ctx.worker_attempts}")


# drain_ungraded_tasks was removed in the post-hook extraction refactor
# (2026-04-21). Grading is now a first-class Beckman-scheduled task via
# RequestPostHook("grade") + GraderAgent; `grade_task()` above remains the
# unit of work that the GraderAgent wraps.
