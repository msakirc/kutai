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


def _parse_yes_no(text: str, key: str) -> Optional[bool]:
    """Extract a YES/NO value for a given key from grader output."""
    pattern = rf"{key}\s*:\s*(YES|NO|PASS|FAIL)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    val = match.group(1).upper()
    return val in ("YES", "PASS")


_NONE_VARIANTS = frozenset({"none", "n/a", "na", "no", "nil", "null", "-", "not applicable"})


def _is_none_value(val: str) -> bool:
    """Check if a parsed field value is a 'none' variant from the LLM."""
    if not val:
        return True
    normalized = val.strip().rstrip(".").lower()
    return normalized in _NONE_VARIANTS or normalized.startswith("no ")


def _parse_text_field(text: str, key: str) -> str:
    """Extract a free-text value for a given key from grader output.

    Captures everything after KEY: until the next uppercase KEY: marker
    or end of string. Handles values that wrap across multiple lines.
    """
    pattern = rf"{key}\s*:\s*(.+?)(?=\n[A-Z]{{2,}}\s*:|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    # Collapse internal newlines + whitespace into single spaces
    value = match.group(1).strip()
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
    """
    relevant = _parse_yes_no(raw, "RELEVANT")
    complete = _parse_yes_no(raw, "COMPLETE")
    verdict = _parse_yes_no(raw, "VERDICT")
    well_formed = _parse_yes_no(raw, "WELL_FORMED")
    coherent = _parse_yes_no(raw, "COHERENT")

    # Skill extraction fields (optional — never block grading)
    situation = _parse_text_field(raw, "SITUATION")
    strategy = _parse_text_field(raw, "STRATEGY")
    tools_raw = _parse_text_field(raw, "TOOLS")
    tools = [t.strip() for t in tools_raw.split(",") if t.strip()] if tools_raw else []

    # Piggybacked learning fields (optional)
    preference = _parse_text_field(raw, "PREFERENCE")
    if _is_none_value(preference):
        preference = ""
    insight = _parse_text_field(raw, "INSIGHT")
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

    # Cascade 3: bare PASS/FAIL keyword anywhere (last resort — VERDICT/RELEVANT/COMPLETE not found)
    bare = re.search(r'\bPASS\b', raw, re.IGNORECASE)
    if bare:
        effective_passed = True if well_formed is not False else False
        return GradeResult(passed=effective_passed, well_formed=well_formed, coherent=coherent,
                           situation=situation, strategy=strategy, tools=tools,
                           preference=preference, insight=insight, raw=raw)
    bare_fail = re.search(r'\bFAIL\b', raw, re.IGNORECASE)
    if bare_fail:
        return GradeResult(passed=False, well_formed=well_formed, coherent=coherent,
                           situation=situation, strategy=strategy, tools=tools,
                           preference=preference, insight=insight, raw=raw)

    raise ValueError(f"grader incapable: could not parse VERDICT, RELEVANT, or COMPLETE from output: {raw[:150]}")


async def grade_task(task: dict, grader_model: str) -> GradeResult:
    """Grade a task's output using a specific model via dispatcher OVERHEAD.

    Args:
        task: Task dict with title, description, result, context
        grader_model: litellm_name of the model to use for grading

    Returns:
        GradeResult

    Raises:
        ValueError: grader parse failure (QualityError equivalent)
        RuntimeError: grading call failed (AvailabilityError equivalent)
    """
    import json
    from src.core.llm_dispatcher import get_dispatcher, CallCategory
    from src.core.router import ModelRequirements

    ctx = task.get("context", "{}")
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}

    generating_model = ctx.get("generating_model", "")
    grade_excluded = ctx.get("grade_excluded_models", [])
    all_excluded = list(set([generating_model] + grade_excluded))

    reqs = ModelRequirements(
        task="reviewer",
        difficulty=3,
        priority=1,
        estimated_input_tokens=800,
        estimated_output_tokens=100,
        prefer_speed=True,
        exclude_models=all_excluded,
        model_override=grader_model,
    )

    result_text = task.get("result", "")
    if not result_text or len(str(result_text).strip()) < 10:
        return GradeResult(passed=False, raw="auto-fail: trivial/empty output")

    from content_quality import assess as cq_assess
    _grade_cq = cq_assess(str(result_text))
    if _grade_cq.is_degenerate:
        return GradeResult(passed=False, raw=f"auto-fail: {_grade_cq.summary}")

    dispatcher = get_dispatcher()
    response = await dispatcher.request(
        CallCategory.OVERHEAD,
        reqs,
        messages=[
            {"role": "system", "content": GRADING_SYSTEM},
            {
                "role": "user",
                "content": GRADING_PROMPT.format(
                    title=task.get("title", "")[:100],
                    description=task.get("description", "")[:500],
                    response=str(result_text)[:4000],
                ),
            },
        ],
    )

    raw_content = response.get("content", "")
    if isinstance(raw_content, list):
        raw_content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_content
        )

    raw_str = str(raw_content)
    logger.debug(
        f"grader raw response ({len(raw_str)} chars): {raw_str[:300]}",
        task_id=task.get("id"),
        grader_model=grader_model,
    )
    return parse_grade_response(raw_str)


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

        # Skill extraction — uses verdict fields when available, mechanical fallback otherwise
        iterations = ctx.get("iterations", 1) or 1
        tools_used = ctx.get("tools_used_names", [])
        if iterations >= 2 and tools_used:
            try:
                from src.memory.skills import add_skill
                agent_type = task.get("agent_type", "executor")
                title = task.get("title", "")

                skill_name = f"auto:{agent_type}:{title[:40]}"

                if verdict.situation:
                    # Rich extraction from grader output
                    await add_skill(
                        name=skill_name,
                        description=verdict.situation,
                        strategy_summary=verdict.strategy or f"Used {', '.join(tools_used[:5])}",
                        tools_used=verdict.tools or sorted(tools_used),
                        avg_iterations=iterations,
                        source_grade="great",
                        source_task_id=task_id,
                    )
                else:
                    # Mechanical fallback — still better than nothing
                    await add_skill(
                        name=skill_name,
                        description=f"Task: {title[:100]}. Agent: {agent_type}.",
                        strategy_summary=f"Used {', '.join(sorted(tools_used)[:5])} in {iterations} iterations",
                        tools_used=sorted(tools_used),
                        avg_iterations=iterations,
                        source_grade="great",
                        source_task_id=task_id,
                    )
            except Exception as e:
                logger.debug(f"skill extraction failed: {e}")

        # Telegram notification for non-silent tasks
        try:
            _is_silent = ctx.get("silent", False)
            if not _is_silent:
                from src.app.telegram_bot import get_bot
                bot = get_bot()
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
                    from src.app.telegram_bot import get_bot
                    bot = get_bot()
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
                    from src.app.telegram_bot import get_bot
                    bot = get_bot()
                    if bot:
                        await bot.send_notification(
                            f"🔄 Görev #{task_id} çıktısı reddedildi, farklı model ile tekrar deniyor\n"
                            f"**{task.get('title', '')[:60]}**"
                        )
            except Exception:
                pass

            logger.info(f"grade FAIL retry | task_id={task_id} attempts={retry_ctx.worker_attempts}")


async def drain_ungraded_tasks(new_model: str) -> int:
    """Grade all ungraded tasks that the new model can grade.

    Called from on_model_swap() and idle path. The model can grade any task
    NOT generated by itself and not in its grade_excluded_models.

    Returns number of tasks graded.
    """
    import json
    from datetime import timedelta
    from src.infra.db import get_db, update_task
    from src.core.retry import compute_retry_timing

    db = await get_db()
    cursor = await db.execute(
        """SELECT * FROM tasks
           WHERE status = 'ungraded'
           AND (next_retry_at IS NULL OR next_retry_at <= datetime('now'))"""
    )
    tasks = [dict(row) for row in await cursor.fetchall()]

    if not tasks:
        return 0

    graded = 0
    for task in tasks:
        ctx_str = task.get("context") or "{}"
        try:
            ctx = json.loads(ctx_str)
        except (json.JSONDecodeError, TypeError):
            ctx = {}

        generating_model = ctx.get("generating_model", "")
        if generating_model == new_model:
            continue  # can't self-grade

        if new_model in ctx.get("grade_excluded_models", []):
            continue  # this grader already failed for this task

        task_id = task["id"]

        # Re-check status — another drain may have graded this task
        from src.infra.db import get_task as _get_task
        fresh = await _get_task(task_id)
        if not fresh or fresh.get("status") != "ungraded":
            continue

        try:
            verdict = await grade_task(task, new_model)
            await apply_grade_result(task_id, verdict)
            graded += 1
        except ValueError:
            # Grader parse failure (QualityError)
            g_attempts = (task.get("grade_attempts") or 0) + 1
            max_g = task.get("max_grade_attempts") or 3
            ctx.setdefault("grade_excluded_models", []).append(new_model)

            if g_attempts >= max_g:
                from src.core.state_machine import transition_task
                await transition_task(
                    task_id, "completed",
                    grade_attempts=g_attempts,
                    completed_at=db_now(),
                    context=json.dumps(ctx),
                )
                logger.warning(f"grading waived (parse failures) | task_id={task_id} grade_attempts={g_attempts}")
                graded += 1
            else:
                await update_task(
                    task_id,
                    grade_attempts=g_attempts,
                    context=json.dumps(ctx),
                )
                logger.info(f"grader parse fail, will retry | task_id={task_id} grade_attempts={g_attempts}")
        except Exception as e:
            # Availability error — backoff, stay ungraded
            last_delay = ctx.get("last_avail_delay", 0)
            decision = compute_retry_timing("availability", last_avail_delay=last_delay)

            if decision.action == "terminal":
                from src.core.state_machine import transition_task
                try:
                    from src.infra.dead_letter import quarantine_task
                    await transition_task(
                        task_id, "failed",
                        failed_in_phase="grading",
                    )
                    await quarantine_task(
                        task_id=task_id,
                        mission_id=task.get("mission_id"),
                        error=f"Grading availability exhausted: {e}",
                        error_category="availability",
                    )
                except Exception as dlq_err:
                    logger.warning(f"grading DLQ failed: {dlq_err}")
                logger.warning(f"grading availability DLQ | task_id={task_id}")
            else:
                ctx["last_avail_delay"] = decision.delay_seconds
                next_retry = to_db(utc_now() + timedelta(seconds=decision.delay_seconds))
                await update_task(
                    task_id,
                    next_retry_at=next_retry,
                    retry_reason="availability",
                    context=json.dumps(ctx),
                )
                logger.info(f"grading availability backoff | task_id={task_id} delay={decision.delay_seconds}")

    if graded:
        logger.info(f"drain_ungraded | graded={graded} total_checked={len(tasks)}")
    return graded
