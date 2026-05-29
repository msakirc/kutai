"""Code review post-hook — LLM judges a build step's emitted code.

Mirrors src.core.grading shape but with a code-review-flavoured prompt.
Used by ``CodeReviewerAgent`` as a post-hook on build steps. Pass/fail
verdict drives the existing retry-with-feedback pipeline; failed
reviews carry concrete issue text so the source agent's next attempt
sees what to fix.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("core.code_review")


CODE_REVIEW_SYSTEM = (
    "You are a strict code reviewer. Inspect the emitted code below for "
    "correctness, security, error handling, and completeness against the "
    "task's stated requirements. Be concrete and actionable. Do not approve "
    "code that is a stub, scaffold, or placeholder when real implementation "
    "was requested. Reply ONLY in the requested format."
)

CODE_REVIEW_PROMPT = """Review this build-step output.

Task: {title}
Description: {description}
Declared files (the agent claims to have produced these): {produces}
Output:
{response}

Reply with EXACTLY this format, in this order:

ISSUES:
- <one concrete issue per bullet, including file path + line/symbol + suggested fix>
- <another>
- (use the literal word NONE if no issues found)

VERDICT: PASS or FAIL

Notes:
- VERDICT must be FAIL if ANY of these are true: missing implementation,
  hardcoded secret, SQL injection, missing auth check, broken imports,
  syntax error, returning fake/placeholder data ("TODO", "abc123", "uuid"
  literal in body), claimed file not actually written.
- Otherwise VERDICT may be PASS even with low/medium issues; severity is
  the source's retry feedback, not your verdict.
"""


@dataclass
class CodeReviewResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    raw: str = ""


_VERDICT_RE = re.compile(r"^\s*VERDICT\s*:\s*(PASS|FAIL)\b", re.IGNORECASE | re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.+)$", re.MULTILINE)


def parse_code_review_response(raw: str) -> CodeReviewResult:
    """Parse the reviewer's structured output. PASS/FAIL line + bullet issues."""
    if not raw or len(raw.strip()) < 4:
        return CodeReviewResult(passed=False, raw=raw or "")

    verdict_match = _VERDICT_RE.findall(raw)
    if verdict_match:
        passed = verdict_match[-1].upper() == "PASS"
    else:
        # Fallback: bare PASS/FAIL keyword
        if re.search(r"\bFAIL\b", raw, re.IGNORECASE):
            passed = False
        elif re.search(r"\bPASS\b", raw, re.IGNORECASE):
            passed = True
        else:
            passed = False  # ambiguous → fail closed

    # Extract issues block bounded by the ISSUES: header on top and VERDICT: at bottom
    issues_section = ""
    lower = raw.lower()
    issues_idx = lower.find("issues:")
    verdict_idx = lower.find("verdict:")
    if issues_idx >= 0:
        end = verdict_idx if verdict_idx > issues_idx else len(raw)
        issues_section = raw[issues_idx + len("issues:") : end]
    bullets = [b.strip() for b in _BULLET_RE.findall(issues_section) if b.strip()]
    # Filter out the literal NONE marker
    bullets = [b for b in bullets if b.upper() != "NONE"]

    return CodeReviewResult(passed=passed, issues=bullets[:30], raw=raw)


def build_code_review_spec(source: dict, exclusions: list):
    """Pure builder for the code-review reviewer child (SP3). Returns a spec
    dict, or a CodeReviewResult auto-fail for trivial/degenerate source."""
    import json as _json
    import time as _time
    import uuid as _uuid

    result_text = source.get("result", "")
    if not result_text or len(str(result_text).strip()) < 10:
        return CodeReviewResult(passed=False, raw="auto-fail: trivial/empty output")

    ctx = source.get("context", "{}")
    if isinstance(ctx, str):
        try:
            ctx = _json.loads(ctx)
        except (ValueError, TypeError):
            ctx = {}
    produces = ctx.get("produces") or []

    messages = [
        {"role": "system", "content": CODE_REVIEW_SYSTEM},
        {"role": "user", "content": CODE_REVIEW_PROMPT.format(
            title=str(source.get("title", ""))[:100],
            description=str(source.get("description", ""))[:500],
            produces=_json.dumps(produces),
            response=str(result_text)[:30000],
        )},
    ]
    _suffix = f"{_time.monotonic_ns() % 1_000_000:06d}-{_uuid.uuid4().hex[:6]}"
    return {
        "title": f"code_reviewer:task#{source.get('id')}:{_suffix}",
        "description": "Code review of build-step output",
        "agent_type": "reviewer",
        "kind": "overhead",
        "priority": 1,
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "overhead",
            "task": "reviewer",
            "agent_type": "reviewer",
            "difficulty": 4,
            "messages": messages,
            "failures": [],
            "estimated_input_tokens": 1500,
            "estimated_output_tokens": 800,
            "exclude_models": list(exclusions),
        }},
    }


async def code_review_task(source: dict) -> CodeReviewResult:
    """Run an LLM code review over ``source``'s emitted output.

    Goes through dispatcher OVERHEAD (Beckman enqueue, raw_dispatch). On
    infra failure returns ``passed=False`` with a short reason rather
    than raising — the caller (CodeReviewerAgent) treats anything that
    isn't a clean PASS as fail-with-feedback so the source retries.
    """
    import json
    import time as _time
    import uuid as _uuid

    import general_beckman
    from src.core.llm_dispatcher import _task_result_to_request_response

    ctx = source.get("context", "{}")
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}

    result_text = source.get("result", "")
    if not result_text or len(str(result_text).strip()) < 10:
        return CodeReviewResult(passed=False, raw="auto-fail: trivial/empty output")

    # Reject degenerate output before round-tripping a reviewer model
    try:
        from dogru_mu_samet import assess as cq_assess
        _cq = cq_assess(str(result_text))
        if _cq.is_degenerate:
            return CodeReviewResult(passed=False, raw=f"auto-fail: {_cq.summary}")
    except Exception:
        pass

    # The model that wrote the source output should not also review it.
    generating_model = ctx.get("generating_model", "")
    review_excluded = list(ctx.get("review_excluded_models") or [])
    exclusions = list({m for m in [generating_model, *review_excluded] if m})

    produces = ctx.get("produces") or []

    messages = [
        {"role": "system", "content": CODE_REVIEW_SYSTEM},
        {
            "role": "user",
            "content": CODE_REVIEW_PROMPT.format(
                title=str(source.get("title", ""))[:100],
                description=str(source.get("description", ""))[:500],
                produces=json.dumps(produces),
                response=str(result_text)[:30000],
            ),
        },
    ]

    graded_task_id = source.get("id")
    _suffix = f"{_time.monotonic_ns() % 1_000_000:06d}-{_uuid.uuid4().hex[:6]}"
    spec = {
        "title": f"code_reviewer:task#{graded_task_id}:{_suffix}",
        "description": "Code review of build-step output",
        "agent_type": "reviewer",
        "kind": "overhead",
        "priority": 1,
        "context": {
            "llm_call": {
                "raw_dispatch": True,
                "call_category": "overhead",
                "task": "reviewer",
                "agent_type": "reviewer",
                "difficulty": 4,
                "messages": messages,
                "failures": [],
                "estimated_input_tokens": 1500,
                "estimated_output_tokens": 800,
                "exclude_models": exclusions,
            },
        },
    }

    try:
        task_result = await general_beckman.enqueue(
            spec, parent_id=graded_task_id, await_inline=True,
        )
    except Exception as e:
        logger.warning(
            f"code review enqueue raised: {e!r}", source_id=graded_task_id,
        )
        return CodeReviewResult(
            passed=False, raw=f"auto-fail: code-review call exception ({e})",
        )

    if task_result.status == "failed":
        logger.warning(
            f"code review enqueue failed: {task_result.error}",
            source_id=graded_task_id,
        )
        return CodeReviewResult(
            passed=False,
            raw=f"auto-fail: code-review call failed ({task_result.error})",
        )

    response = _task_result_to_request_response(task_result)
    raw_content = response.get("content", "")
    if isinstance(raw_content, list):
        raw_content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in raw_content
        )
    return parse_code_review_response(str(raw_content or ""))
