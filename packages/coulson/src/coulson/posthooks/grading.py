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
from prompt_foundry import build_messages

logger = get_logger("coulson.posthooks.grading")


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


def build_grading_spec(source: dict, exclusions: list):
    """Pure builder for the grading reviewer child (SP3).

    Returns a ready Beckman spec dict when the source is gradeable, OR a
    GradeResult (auto-fail) when the source is trivial/empty/degenerate
    (caller short-circuits to apply that verdict without enqueueing a child).
    """
    import time as _time
    import uuid as _uuid

    result_text = source.get("result", "")
    if not result_text or len(str(result_text).strip()) < 10:
        return GradeResult(passed=False, raw="auto-fail: trivial/empty output")

    from dogru_mu_samet import assess as cq_assess
    _cq = cq_assess(str(result_text))
    if _cq.is_degenerate:
        return GradeResult(passed=False, raw=f"auto-fail: {_cq.summary}")

    # NO TRUNCATION of any grader input, ever. The description IS the
    # grading contract and the result IS the artifact under judgment;
    # lopping either makes the grader judge a partial spec against a
    # partial artifact and emit false verdicts. #289700 (2026-06-04): a
    # 1329-char 5-section charter instruction cut at 500 chars dropped
    # sections 4-5, and the grader reported the artifact "added a sixth
    # section" — 36% of i2p steps (94/259) have instructions past the old
    # cap. An oversized input is a model-selection/capacity concern, never
    # solved by silent truncation here.
    messages = build_messages("grading", {
        "title": str(source.get("title", "")),
        "description": str(source.get("description", "")),
        "response": str(result_text),
    })
    _suffix = f"{_time.monotonic_ns() % 1_000_000:06d}-{_uuid.uuid4().hex[:6]}"
    # Derive the input estimate from the ACTUAL prompt size (~4 chars/token).
    # Now that nothing is truncated, a large artifact must steer selection to a
    # model whose context window actually fits it — otherwise the call-level
    # context cap becomes the new silent-truncation point we just removed.
    _prompt_chars = sum(len(m["content"]) for m in messages)
    _est_input = max(800, _prompt_chars // 4)
    return {
        "title": f"grader:task#{source.get('id')}:{_suffix}",
        "description": "Grading review of task output",
        "agent_type": "reviewer",
        "kind": "overhead",
        "priority": 1,
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "overhead",
            "task": "reviewer",
            "agent_type": "reviewer",
            "difficulty": 3,
            "messages": messages,
            "failures": [],
            "estimated_input_tokens": _est_input,
            "estimated_output_tokens": 600,
            "prefer_speed": True,
            "exclude_models": list(exclusions),
        }},
    }
