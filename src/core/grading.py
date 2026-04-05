"""
Task grading — structured binary evaluation.

Replaces the old 1-5 numeric grading with a structured YES/NO prompt.
All grading calls go through the LLM dispatcher as OVERHEAD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("core.grading")

GRADING_PROMPT = """Evaluate this task response.

Task: {title}
Description: {description}
Response: {response}

Answer each with YES or NO only:
RELEVANT: Does the response address the task?
COMPLETE: Does it contain a concrete deliverable, not just a plan or description?
VERDICT: Should this response be accepted?"""


@dataclass
class GradeResult:
    passed: bool
    relevant: Optional[bool] = None
    complete: Optional[bool] = None
    raw: str = ""
    score: float = 0.0

    def __post_init__(self):
        self.score = 4.0 if self.passed else 2.0


def _parse_yes_no(text: str, key: str) -> Optional[bool]:
    """Extract a YES/NO value for a given key from grader output."""
    pattern = rf"{key}\s*:\s*(YES|NO|PASS|FAIL)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    val = match.group(1).upper()
    return val in ("YES", "PASS")


def parse_grade_response(raw: str) -> GradeResult:
    """Parse structured grader output into a GradeResult.

    Parse priority:
      1. VERDICT → use directly
      2. If no VERDICT but RELEVANT+COMPLETE → derive (both YES = PASS)
      3. If nothing parses → raise ValueError (grader incapable)
    """
    relevant = _parse_yes_no(raw, "RELEVANT")
    complete = _parse_yes_no(raw, "COMPLETE")
    verdict = _parse_yes_no(raw, "VERDICT")

    if verdict is not None:
        return GradeResult(passed=verdict, relevant=relevant, complete=complete, raw=raw)

    if relevant is not None and complete is not None:
        return GradeResult(passed=(relevant and complete), relevant=relevant, complete=complete, raw=raw)

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

    dispatcher = get_dispatcher()
    response = await dispatcher.request(
        CallCategory.OVERHEAD,
        reqs,
        messages=[{
            "role": "user",
            "content": GRADING_PROMPT.format(
                title=task.get("title", "")[:100],
                description=task.get("description", "")[:200],
                response=str(result_text)[:2000],
            ),
        }],
    )

    raw_content = response.get("content", "")
    if isinstance(raw_content, list):
        raw_content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_content
        )

    return parse_grade_response(str(raw_content))


async def apply_grade_result(task_id: int, verdict: GradeResult) -> None:
    """Apply grade outcome to a task. Handles PASS and FAIL.

    PASS: transition to completed, record model quality, trigger skill extraction.
    FAIL: increment attempts, add model to exclusions, retry or DLQ.
    """
    import json
    from datetime import datetime
    from src.infra.db import get_task, update_task
    from src.core.state_machine import transition_task
    from src.core.retry import (
        compute_retry_timing, update_exclusions_on_failure,
    )

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
            quality_score=verdict.score,
            completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Record model quality feedback
        try:
            from src.infra.db import record_model_call
            await record_model_call(
                model=ctx.get("generating_model", ""),
                agent_type=task.get("agent_type", "executor"),
                success=True,
                grade=verdict.score,
            )
        except Exception:
            pass

        # Skill extraction for deferred grades
        iterations = task.get("iterations", 1) or 1
        tools_used = ctx.get("tools_used_names", [])
        if iterations >= 2 and tools_used and verdict.score >= 4.0:
            try:
                from src.memory.skills import add_skill
                agent_type = task.get("agent_type", "executor")
                title = task.get("title", "")
                await add_skill(
                    name=f"auto:{agent_type}:{title[:40]}",
                    description=f"Task: {title[:100]}. Agent: {agent_type}.",
                    strategy_summary=f"Used {', '.join(sorted(tools_used)[:5])} in {iterations} iterations",
                    tools_used=sorted(tools_used),
                    avg_iterations=iterations,
                    source_grade="great",
                    source_task_id=task_id,
                )
            except Exception as e:
                logger.debug(f"deferred skill extraction failed: {e}")

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

        logger.info(f"grade PASS | task_id={task_id} score={verdict.score}")
    else:
        # VERDICT=FAIL — worker quality failure
        generating_model = ctx.get("generating_model", "")
        attempts = (task.get("attempts") or 0) + 1
        max_attempts = task.get("max_attempts") or 6

        update_exclusions_on_failure(ctx, generating_model, attempts)
        decision = compute_retry_timing("quality", attempts=attempts, max_attempts=max_attempts)

        if decision.action == "terminal":
            await transition_task(
                task_id, "failed",
                failed_in_phase="worker",
                attempts=attempts,
                context=json.dumps(ctx),
            )
            try:
                from src.infra.dead_letter import quarantine_task
                await quarantine_task(
                    task_id=task_id,
                    mission_id=task.get("mission_id"),
                    error=f"Quality gate failed after {attempts} attempts",
                    error_category="quality",
                    original_agent=task.get("agent_type", "executor"),
                    retry_count=attempts,
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

            logger.warning(f"grade FAIL terminal | task_id={task_id} attempts={attempts}")
        else:
            next_retry = None
            if decision.action == "delayed":
                from datetime import timedelta
                next_retry = (datetime.now() + timedelta(seconds=decision.delay_seconds)).strftime("%Y-%m-%d %H:%M:%S")

            await transition_task(
                task_id, "pending",
                attempts=attempts,
                grade_attempts=0,
                next_retry_at=next_retry,
                retry_reason="quality",
                context=json.dumps(ctx),
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

            logger.info(f"grade FAIL retry | task_id={task_id} attempts={attempts}")


async def drain_ungraded_tasks(new_model: str) -> int:
    """Grade all ungraded tasks that the new model can grade.

    Called from on_model_swap() and idle path. The model can grade any task
    NOT generated by itself and not in its grade_excluded_models.

    Returns number of tasks graded.
    """
    import json
    from datetime import datetime, timedelta
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
                    quality_score=None,
                    grade_attempts=g_attempts,
                    completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
                next_retry = (datetime.now() + timedelta(seconds=decision.delay_seconds)).strftime("%Y-%m-%d %H:%M:%S")
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
