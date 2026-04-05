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
        return GradeResult(passed=True, raw="auto-pass: trivial output")

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
