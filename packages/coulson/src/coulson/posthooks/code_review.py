"""Code review post-hook — LLM judges a build step's emitted code.

Mirrors src.core.grading shape but with a code-review-flavoured prompt.
Used by the ``posthook.code_review.resume`` continuation handler via
``build_code_review_spec``.  Pass/fail verdict drives the existing
retry-with-feedback pipeline; failed reviews carry concrete issue text
so the source agent's next attempt sees what to fix.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.infra.logging_config import get_logger
from finch import build_messages

logger = get_logger("coulson.posthooks.code_review")


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

    try:
        from dogru_mu_samet import assess as cq_assess
        _cq = cq_assess(str(result_text))
        if _cq.is_degenerate:
            return CodeReviewResult(passed=False, raw=f"auto-fail: {_cq.summary}")
    except Exception:
        pass

    ctx = source.get("context", "{}")
    if isinstance(ctx, str):
        try:
            ctx = _json.loads(ctx)
        except (ValueError, TypeError):
            ctx = {}
    produces = ctx.get("produces") or []

    # NO TRUNCATION of any reviewer input, ever — same rule as the grader
    # (src/core/grading.py). The description is the review contract and the
    # response is the artifact under review; lopping either makes the reviewer
    # judge a partial spec against a partial artifact and emit false verdicts.
    messages = build_messages("code_review", {
        "title": str(source.get("title", "")),
        "description": str(source.get("description", "")),
        "produces": _json.dumps(produces),
        "response": str(result_text),
    })
    _suffix = f"{_time.monotonic_ns() % 1_000_000:06d}-{_uuid.uuid4().hex[:6]}"
    # Estimate from the ACTUAL prompt size so selection fits the context window.
    _prompt_chars = sum(len(m["content"]) for m in messages)
    _est_input = max(1500, _prompt_chars // 4)
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
            "estimated_input_tokens": _est_input,
            "estimated_output_tokens": 800,
            "exclude_models": list(exclusions),
        }},
    }


