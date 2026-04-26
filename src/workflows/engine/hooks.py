"""Pre/post execution hooks for workflow steps in the orchestrator.

Handles artifact injection, output storage, conditional evaluation,
template expansion triggers, and CodingPipeline delegation detection.
"""
from __future__ import annotations

import json
from typing import Optional

from src.infra.logging_config import get_logger
from .artifacts import ArtifactStore, format_artifacts_for_prompt, get_phase_summaries, CONTEXT_BUDGETS, _TIER_ORDER
from .conditions import evaluate_condition, resolve_group
from .policies import ReviewTracker
from .quality_gates import evaluate_gate, format_gate_result

logger = get_logger("workflows.engine.hooks")

async def _llm_summarize(text: str, artifact_name: str) -> str | None:
    """Summarize a large artifact using the LLM (OVERHEAD call)."""
    from ...core.llm_dispatcher import get_dispatcher, CallCategory

    max_input = 16000
    truncated_text = text[:max_input]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise summarizer. Produce a summary that "
                "preserves ALL key facts, decisions, and data points. "
                "Target: under 400 words. No filler."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Summarize this '{artifact_name}' artifact. Keep every "
                f"important fact, number, name, and decision:\n\n"
                f"{truncated_text}"
            ),
        },
    ]

    response = await get_dispatcher().request(
        CallCategory.OVERHEAD,
        task="summarizer",
        difficulty=2,
        messages=messages,
        prefer_speed=True,
        prefer_local=True,
        estimated_input_tokens=min(len(text) // 4, 4000),
        estimated_output_tokens=500,
    )
    summary = response.get("content", "").strip()
    if summary and len(summary) > 50:
        from dogru_mu_samet import assess as cq_assess
        cq = cq_assess(summary)
        if not cq.is_degenerate:
            return summary
        logger.warning(f"[LLM Summary] degenerate output for '{artifact_name}': {cq.summary}")
    return None


def _extract_json_string_field(text: str, key: str) -> str | None:
    """Recover a string field's value from possibly-truncated JSON.

    Walks char-by-char from the first ``"key": "`` match, honoring JSON
    escape rules. Returns the raw escaped-string body up to the first
    unescaped closing ``"`` — or, if the input is truncated mid-string,
    everything captured so far. Returns ``None`` only when the key isn't
    found at all.

    Why this exists: the older regex fallback required a downstream
    sentinel (``"memories"``, ``"subtasks"``, or a closing ``}``) and
    failed silently on LLM outputs that ran out of budget mid-string,
    leaving the raw broken envelope as the artifact (observed task 2890
    user_stories, 4132-char truncated triple-escaped output).
    """
    import re as _re
    m = _re.search(rf'"{key}"\s*:\s*"', text)
    if not m:
        return None
    i = m.end()
    out: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            # Capture the escape pair verbatim — unescape happens after.
            if i + 1 < len(text):
                out.append(ch + text[i + 1])
                i += 2
                continue
            out.append(ch)
            break
        if ch == '"':
            return "".join(out)
        out.append(ch)
        i += 1
    # Truncated mid-string — return everything captured. The downstream
    # quality check / schema validator will reject if it's unsalvageable;
    # surfacing the partial body is strictly better than emitting the
    # raw `{"action":"final_answer","result":"...` envelope.
    return "".join(out)


def _unescape_json_string(s: str) -> str:
    """Reverse JSON string escapes. Tolerant of the triple-backslash mess
    Qwen3.5-9B produces on long nested arrays — falls back to manual
    pairwise replacement when strict json decoding fails."""
    try:
        return json.loads(f'"{s}"')
    except (json.JSONDecodeError, ValueError):
        return (
            s.replace("\\\\", "\\")
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
        )


def canonicalize_for_retry(text: str, max_depth: int = 4) -> str:
    """Collapse multi-layer JSON escape compounding to canonical form.

    When ``_prev_output`` is fed back into a retry prompt, models like
    Qwen3.5-9B re-wrap it in their own ``final_answer`` envelope and
    re-escape inner quotes — every retry adds another escape layer, so
    by attempt 3-4 the output is unparseable triple-backslashed soup.

    Strategy: repeatedly try to parse ``text`` as JSON. If it parses to
    a list/dict, re-dump with ``ensure_ascii=False``. If it parses to a
    string (i.e. the input was a JSON-string holding more JSON), recurse
    on that string. Bounded by ``max_depth`` to keep pathological inputs
    from looping. Non-JSON text is returned unchanged.

    This is the structural counterpart to ``_unwrap_envelope``: unwrap
    strips the outer envelope, canonicalize collapses inner escape
    compounding so the next retry prompt sees clean JSON.
    """
    if not isinstance(text, str):
        return text
    current = text.strip()
    for _ in range(max_depth):
        if not current:
            return current
        first = current.lstrip()[:1]
        # Only attempt JSON parse if it actually looks like JSON. Plain
        # markdown / prose passes through untouched.
        if first not in ("[", "{", '"'):
            return current
        try:
            parsed = json.loads(current)
        except (json.JSONDecodeError, ValueError):
            return current
        if isinstance(parsed, str):
            # One escape layer peeled. If the inner string is itself JSON,
            # the loop continues; otherwise we hit the ``first not in``
            # guard next iteration.
            current = parsed.strip()
            continue
        if isinstance(parsed, (list, dict)):
            return json.dumps(parsed, ensure_ascii=False)
        return current
    return current


def _unwrap_envelope(text: str) -> str:
    """Strip JSON envelopes, model tokens, and degenerate repetition.

    Handles:
      - final_answer envelopes: {"action": "final_answer", "result": "..."}
      - tool_call envelopes: {"action": "tool_call", "tool": "write_file",
        "args": {"content": "..."}}
      - Model-specific tokens: <|function_call|>, <|function_result|>
      - Markdown code fences
      - Degenerate repetition (same section repeated 3+ times)
    """
    import re as _re

    # ── Strip model-specific tokens ──
    stripped = text.strip()
    for token in ("<|function_call|>", "<|function_result|>",
                  "<|im_start|>", "<|im_end|>"):
        stripped = stripped.replace(token, "")
    stripped = stripped.strip()

    # ── Strip markdown code fences ──
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        stripped = stripped.rsplit("```", 1)[0].strip()

    # ── Try JSON parse for known envelope types ──
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            # final_answer envelope
            if "result" in obj:
                val = obj["result"]
                stripped = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
            # write_file tool_call envelope — extract the file content
            elif obj.get("action") == "tool_call" and obj.get("tool") == "write_file":
                args = obj.get("args", {})
                if isinstance(args, dict) and "content" in args:
                    val = args["content"]
                    stripped = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass

    # ── Fallback for broken / truncated JSON ──
    # The structural extractor walks the string char-by-char respecting
    # JSON escape rules and returns the partial body when truncated.
    # Replaces the older regex pair which silently failed on outputs
    # without a closing-marker sentinel.
    if '"result"' in stripped and '"final_answer"' in stripped:
        body = _extract_json_string_field(stripped, "result")
        if body is not None:
            stripped = _unescape_json_string(body)

    if '"write_file"' in stripped and '"content"' in stripped:
        body = _extract_json_string_field(stripped, "content")
        if body is not None:
            stripped = _unescape_json_string(body)

    return stripped


def _detect_repetition_ratio(text: str) -> float:
    """Return 0.0–1.0 indicating how much of the text is repetitive.

    .. deprecated::
        Use ``dogru_mu_samet.assess()`` instead. All callsites now use
        the dogru_mu_samet package. Kept for reference.

    Splits into ## sections, normalizes headers, and counts how many
    sections share a normalized header with at least one other section.
    Does NOT modify the text — purely a signal.
    """
    import re as _re2

    sections = text.split("\n## ")
    if len(sections) < 5:
        return 0.0

    norm_headers: list[str] = []
    for sec in sections[1:]:  # skip content before first ##
        header = sec.split("\n", 1)[0].strip()
        norm = _re2.sub(
            r'\s+(summary|examples?|notes|details)\s*$', '',
            header.lower(),
        ).strip()
        norm_headers.append(norm)

    from collections import Counter
    counts = Counter(norm_headers)
    duplicated = sum(c - 1 for c in counts.values() if c > 1)
    return duplicated / len(norm_headers) if norm_headers else 0.0


def _is_empty_required_value(val) -> bool:
    """Decide if a required-field value is a placeholder rather than real
    content. Used by validate_artifact_schema to guard against constrained-
    decoding passes that satisfy "field present" with empty containers
    (mission 46 task 2964: ``[{feature_id: {}, feature_name: {}, ...}]`` —
    schema validator passed because all fields existed, but every value
    was an empty object).

    Empty placeholders considered missing:
      * ``None``
      * empty string / whitespace-only string
      * empty dict ``{}``
      * empty list ``[]``

    Real content (boolean ``False``, number ``0``, non-empty string, etc.)
    is NOT empty even though it is falsy in Python.
    """
    if val is None:
        return True
    if isinstance(val, str):
        return not val.strip()
    if isinstance(val, dict):
        return len(val) == 0
    if isinstance(val, list):
        return len(val) == 0
    return False


def validate_artifact_schema(output_value: str, schema: dict) -> tuple[bool, str]:
    """Validate an artifact output against its schema definition.

    Returns (is_valid, error_message). error_message is empty if valid.
    """
    if not schema:
        return True, ""

    # Unwrap final_answer JSON envelope if present — agents sometimes
    # wrap their results in {"action": "final_answer", "result": "..."}
    # Uses _unwrap_envelope which handles malformed JSON (unescaped quotes).
    if isinstance(output_value, str):
        output_value = _unwrap_envelope(output_value)

    for artifact_name, rules in schema.items():
        if not isinstance(rules, dict):
            continue  # skip non-artifact entries like max_output_chars
        schema_type = rules.get("type", "string")

        if schema_type == "object":
            required = rules.get("required_fields", [])
            # Try JSON first
            try:
                data = json.loads(output_value) if isinstance(output_value, str) else output_value
                if isinstance(data, dict):
                    # If agent wrapped output in the artifact name, unwrap
                    if artifact_name in data and isinstance(data[artifact_name], dict):
                        data = data[artifact_name]
                    missing = [f for f in required if f not in data]
                    if missing:
                        return False, f"Missing required fields in '{artifact_name}': {missing}"
                    # Empty-value guard: a constrained-decoding pass can
                    # satisfy "field present" with placeholder ``{}`` /
                    # ``""`` / ``[]`` for every required field — schema
                    # PRESENCE check passed but downstream consumers got
                    # nothing usable. Reject empties as if the field
                    # were missing. Mission 46 task 2964 produced
                    # ``[{feature_id: {}, feature_name: {}, ...}]`` —
                    # validator accepted, downstream broke.
                    empty_value_fields = [
                        f for f in required
                        if _is_empty_required_value(data.get(f))
                    ]
                    if empty_value_fields:
                        return False, (
                            f"Required fields in '{artifact_name}' have empty "
                            f"placeholder values: {empty_value_fields}. Each "
                            f"required field must hold real content."
                        )
                    continue  # this artifact passed
            except (json.JSONDecodeError, TypeError):
                pass
            # Fallback: accept text/markdown if required fields appear as keywords
            # Small LLMs often produce structured text, not JSON.
            # Check each word of multi-word fields independently — e.g.
            # "per_competitor" matches if both "per" and "competitor"
            # appear anywhere in the text, since LLMs rephrase freely.
            if required:
                import re as _re_obj
                # Normalize: remove apostrophes, replace dashes/underscores with space
                def _norm(s):
                    s = s.lower().replace("'", "").replace("\u2019", "")
                    return _re_obj.sub(r"[\u2010-\u2015\u2212\-_]", " ", s)
                text_norm = _norm(str(output_value))
                missing = []
                _QUALIFIER_SUFFIXES = {
                    "level", "rate", "count", "status", "type",
                    "value", "score", "index", "ratio",
                }
                for f in required:
                    words = _norm(f).split()
                    # Qualifier-only suffixes (level, rate, etc.) are
                    # informational; small LLMs drop them ("High
                    # confidence" == "confidence_level: High"). Check
                    # only the non-qualifier core words.
                    core = [w for w in words if w not in _QUALIFIER_SUFFIXES]
                    if not core:
                        core = words
                    if not all(w in text_norm for w in core):
                        missing.append(f)
                if missing:
                    return False, f"'{artifact_name}' missing content about: {missing}"

        elif schema_type == "array":
            # Try JSON first — the output may be a raw JSON array, a JSON
            # object containing the array as a field, or markdown text with
            # embedded JSON.
            try:
                data = json.loads(output_value) if isinstance(output_value, str) else output_value
                # If it's a dict, look for the artifact key or any list value
                if isinstance(data, dict):
                    data = data.get(artifact_name) or next(
                        (v for v in data.values() if isinstance(v, list)), None
                    )
                if isinstance(data, list):
                    min_items = rules.get("min_items", 0)
                    if len(data) < min_items:
                        return False, f"'{artifact_name}' has {len(data)} items, need >= {min_items}"
                    item_fields = rules.get("item_fields", [])
                    if item_fields and data:
                        for i, item in enumerate(data):
                            if isinstance(item, dict):
                                missing = [f for f in item_fields if f not in item]
                                if missing:
                                    return False, f"Item {i} in '{artifact_name}' missing fields: {missing}"
                                # Empty-value guard (same rationale as object).
                                empty_fields = [
                                    f for f in item_fields
                                    if _is_empty_required_value(item.get(f))
                                ]
                                if empty_fields:
                                    return False, (
                                        f"Item {i} in '{artifact_name}' has empty "
                                        f"placeholder values: {empty_fields}. Each "
                                        f"required field must hold real content."
                                    )
                    continue  # passed
            except (json.JSONDecodeError, TypeError):
                # Try extracting JSON from markdown code blocks
                import re as _re2
                _json_block = _re2.search(r'```(?:json)?\s*\n([\s\S]*?)\n```', str(output_value))
                if _json_block:
                    try:
                        data = json.loads(_json_block.group(1))
                        if isinstance(data, dict):
                            data = data.get(artifact_name) or next(
                                (v for v in data.values() if isinstance(v, list)), None
                            )
                        if isinstance(data, list):
                            if len(data) >= rules.get("min_items", 0):
                                continue  # passed
                    except (json.JSONDecodeError, TypeError):
                        pass
            # Fallback: accept text if it has numbered/bulleted/table/JSON items
            min_items = rules.get("min_items", 0)
            if min_items > 0:
                import re as _re
                text = str(output_value)
                # Numbered/bulleted lists
                items = _re.findall(r'(?:^|\n)\s*(?:\d+[\.\)]|\-|\*)\s+\S', text)
                # Markdown table data rows (exclude header separator lines like |---|)
                if len(items) < min_items:
                    table_rows = [
                        line for line in text.split("\n")
                        if line.strip().startswith("|")
                        and not _re.match(r'^\s*\|[\s\-:]+\|', line)  # skip separator
                        and not _re.match(r'^\s*\|\s*:?-', line)  # skip separator variant
                    ]
                    # Exclude header row (first table row) from count
                    if len(table_rows) > 1:
                        items = table_rows[1:]  # skip header
                # Truncated JSON arrays: count complete {...} objects
                # (models often truncate mid-array at output token limit)
                if len(items) < min_items:
                    item_fields = rules.get("item_fields", [])
                    if item_fields and text.lstrip().startswith("["):
                        json_items = _re.findall(r'\{[^{}]{20,}\}', text)
                        if len(json_items) >= min_items:
                            items = json_items
                if len(items) < min_items:
                    return False, f"'{artifact_name}' has ~{len(items)} list items, need >= {min_items}"

        elif schema_type == "string":
            min_length = rules.get("min_length", 1)
            if not output_value or len(str(output_value).strip()) < min_length:
                return False, f"'{artifact_name}' is too short (min {min_length} chars)"

        elif schema_type == "markdown":
            required_sections = rules.get("required_sections", [])
            text = str(output_value)
            text_lower = text.lower()
            # Normalize numbered headers: "## 1. Vision" → "## Vision"
            import re as _re
            text_normalized = _re.sub(
                r'^(#{1,4})\s*\d+[\.\)]\s*',
                r'\1 ',
                text_lower,
                flags=_re.MULTILINE,
            )
            # Check for actual markdown headers (## Section or ### Section),
            # not just substring mentions like "Vision (streamlining...)"
            missing = []
            for s in required_sections:
                s_lower = s.lower()
                # Accept: ## Vision, ### Vision, # Vision, **Vision**, Vision\n---
                has_header = (
                    f"# {s_lower}" in text_normalized
                    or f"**{s_lower}**" in text_normalized
                    or f"\n{s_lower}\n" in text_normalized
                )
                if not has_header:
                    missing.append(s)
            if missing:
                return False, f"'{artifact_name}' missing sections: {missing}"

    return True, ""

# ── Disguised failure detection ────────────────────────────────────────────

# Phrases that indicate the agent failed but wrapped it in final_answer
_FAILURE_PHRASES = [
    # Direct failure statements
    "cannot be completed",
    "cannot be performed",
    "cannot proceed",
    "could not be performed",
    "could not be completed",
    "could not produce a final answer",
    "unable to complete",
    "cannot complete",
    "cannot execute",
    "cannot run",
    # Tool/environment failures
    "shell tool is blocked",
    "shell tool is disabled",
    "shell tool is completely",
    "shell tool was blocked",
    "shell tool has failed",
    "shell is blocked",
    "tool is temporarily disabled",
    "tool is blocked",
    "tool is disabled",
    "tool block",
    "tool failed",
    "tool is completely",
    "tool is non-functional",
    "tools are available",  # "no alternative tools are available"
    # Status keywords
    "execution blocked",
    "status: blocked",
    "status: failed",
    "cannot proceed until",
    "blocked - cannot",
    "critical blocker",
    "no test execution possible",
    "cannot be generated without",
    "not run - shell",
    "not run -",
    "workspace appears empty",
    "workspace is inaccessible",
    "inaccessible due to",
    # Verification failures
    "verification status: failed",
    "verification failed",
    "task failed",
    "failed / incomplete",
    "creation failed",
    "suite creation failed",
    # Missing/incomplete outputs
    "documentation is incomplete",
    "no artifact",
    "artifact cannot be generated",
    "not yet created",
    "placeholder text",
]

# Output starts with raw JSON tool_call — agent never produced final_answer
_RAW_TOOL_CALL_PREFIXES = [
    '{"action":"tool_call"',
    '{"action": "tool_call"',
    '{"action":"toolcall"',
    '{"action": "toolcall"',
]

# Phrases that look like failure but are legitimate analysis
_FALSE_POSITIVE_PHRASES = [
    "competitor",       # "competitor failed to..." is analysis, not failure
    "risk if wrong",    # risk assessment
    "failure mode",     # design docs discussing failure modes
    "error handling",   # architecture discussing error handling
    "error states",     # UX discussing error states
    "empty states",     # UX design for empty states
]

# Hard failure indicators — these override false positives. Each must
# be specific enough that a positive sentence form ("the shell tool is
# working fine") doesn't trip detection. Bare ``"shell tool"`` was
# removed 2026-04-27 because it was matching legit references to the
# tool in design docs and post-mortems.
_HARD_FAILURE_PHRASES = [
    "shell tool is blocked",
    "shell tool is disabled",
    "shell tool was blocked",
    "tool is blocked",
    "tool is disabled",
    "cannot proceed until",
    "execution blocked",
    "_ctx is not defined",
    "not in allowlist",
]


_NEGATION_PRECEDERS = (
    "not ", "no ", "n't ", "non-", "without ", "never ",
    "isn't ", "wasn't ", "aren't ", "weren't ",
)


_SENTENCE_BOUNDARY_CHARS = ".!?\n;"


def _phrase_matches_unnegated(text_lower: str, phrase: str) -> bool:
    """Return True iff ``phrase`` appears in ``text_lower`` AT LEAST ONCE
    without a negation token earlier in the SAME sentence.

    Mission 57 task 4373 surfaced the false positive: a problem-statement
    artifact contained ``"is not a critical blocker"`` and the substring
    ``"critical blocker"`` (in ``_FAILURE_PHRASES``) tripped detection
    five retries in a row. The output was structurally fine — the
    negated phrase was descriptive content, not a failure report.

    Heuristic: for each occurrence, walk back to the nearest sentence
    boundary (``. ! ? \\n ;``) or 80 chars, whichever is closer. Check
    that window for a negation token. If ALL occurrences are negated,
    treat as no match. Otherwise at least one positive use exists ->
    match. Sentence-bounded so a negation in an EARLIER sentence
    doesn't mask a positive use later in the text.
    """
    if phrase not in text_lower:
        return False
    start = 0
    while True:
        idx = text_lower.find(phrase, start)
        if idx == -1:
            return False  # all checked occurrences were negated
        # Find the start of the current sentence.
        win_start = max(0, idx - 80)
        for j in range(idx - 1, win_start - 1, -1):
            if text_lower[j] in _SENTENCE_BOUNDARY_CHARS:
                win_start = j + 1
                break
        window = text_lower[win_start:idx]
        if not any(neg in window for neg in _NEGATION_PRECEDERS):
            return True
        start = idx + 1


def _is_disguised_failure(output: str) -> bool:
    """Detect if a 'completed' result is actually a failure report."""
    if not output or len(output) < 10:
        return False

    stripped = output.strip()

    # Raw tool_call JSON as the result — agent never produced an answer
    for prefix in _RAW_TOOL_CALL_PREFIXES:
        if prefix in stripped[:100]:
            return True

    lower = output.lower()

    # Hard failure indicators — always a failure regardless of context.
    # Still negation-aware: ``"the shell tool was working"`` shouldn't
    # be flagged just because ``"shell tool"`` is hard-list.
    if any(_phrase_matches_unnegated(lower, p) for p in _HARD_FAILURE_PHRASES):
        return True

    # Check for false positives — legitimate analysis about failures
    has_false_positive = any(fp in lower for fp in _FALSE_POSITIVE_PHRASES)

    # Count failure indicators (negation-aware).
    hits = sum(
        1 for phrase in _FAILURE_PHRASES
        if _phrase_matches_unnegated(lower, phrase)
    )

    # 2+ failure phrases = almost certainly a disguised failure
    if hits >= 2:
        return True

    # 1 failure phrase without false positive context
    if hits >= 1 and not has_false_positive:
        return True

    return False



def _structural_summary(text: str, target: int = 1500) -> str:
    """Fallback summary without LLM — keep headings + first lines."""
    if len(text) <= target:
        return text
    lines = text.split("\n")
    out: list[str] = []
    total = 0
    for line in lines:
        s = line.strip()
        if s.startswith("#") or (out and not out[-1].startswith("#") and not s):
            out.append(s)
            total += len(s) + 1
        elif s and (not out or out[-1].startswith("#")):
            out.append(s[:200])
            total += min(len(s), 200) + 1
        if total >= target:
            break
    return "\n".join(out) if out else text[:target]


# ── Module-level singleton ─────────────────────────────────────────────────

_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Return the module-level ArtifactStore singleton (lazy init)."""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore(use_db=True)
    return _artifact_store


# ── Helper functions ───────────────────────────────────────────────────────


async def should_skip_workflow_step(task: dict) -> tuple[bool, str]:
    """Evaluate the step's ``skip_when_expr`` against loaded artifacts.

    Returns ``(True, reason)`` when the step should short-circuit as
    skipped, else ``(False, "")``. Safe default is to NOT skip on any
    parse or lookup error — misconfigured expressions should not silently
    suppress work. Current supported expression shape:

        <artifact_name>.<dot.path> != '<literal>'
        <artifact_name>.<dot.path> == '<literal>'

    Runtime-loaded — the artifact store is consulted each time the task
    is dispatched so skip conditions reflect post-completion state of
    dependencies, not whatever was in flight when the task was expanded.
    """
    ctx = _parse_context(task)
    mission_id = ctx.get("mission_id") or task.get("mission_id")
    if mission_id is None:
        return False, ""

    expr = ctx.get("skip_when_expr")
    # Legacy rescue: tasks expanded before the skip_when_expr context
    # field was added won't have it. Look the step's skip_when up in
    # the live workflow JSON. Works for the shopping_v2 synth_one /
    # clarify_variant steps that have DB rows pre-dating the fix.
    if (not expr or not isinstance(expr, str)) and ctx.get("is_workflow_step"):
        try:
            step_id = ctx.get("workflow_step_id")
            if step_id:
                from src.infra.db import get_db as _get_db
                _db = await _get_db()
                _cur = await _db.execute(
                    "SELECT context FROM missions WHERE id = ?", (mission_id,),
                )
                _row = await _cur.fetchone()
                await _cur.close()
                _mctx = {}
                if _row and _row[0]:
                    try:
                        _mctx = json.loads(_row[0])
                        if isinstance(_mctx, str):
                            _mctx = json.loads(_mctx)
                    except (json.JSONDecodeError, TypeError):
                        _mctx = {}
                _wf_name = (
                    _mctx.get("workflow_name")
                    if isinstance(_mctx, dict) else None
                ) or "i2p_v3"
                from src.workflows.engine.loader import load_workflow
                _wf = load_workflow(_wf_name)
                _step = _wf.get_step(step_id)
                if _step:
                    _sw = _step.get("skip_when")
                    if isinstance(_sw, str):
                        expr = _sw
        except Exception as _e:
            logger.debug(
                f"[Workflow Hook] skip_when step lookup failed: {_e}"
            )
    if not expr or not isinstance(expr, str):
        return False, ""

    import re as _re
    m = _re.match(
        r"\s*([A-Za-z_][\w]*)((?:\.[A-Za-z_][\w]*)+)\s*(==|!=)\s*'([^']*)'\s*$",
        expr,
    )
    if not m:
        logger.warning(
            f"[Workflow Hook] skip_when_expr {expr!r} unparseable — not skipping"
        )
        return False, ""

    artifact_name = m.group(1)
    path = m.group(2).lstrip(".").split(".")
    op = m.group(3)
    literal = m.group(4)

    try:
        store = get_artifact_store()
        raw_artifact = await store.retrieve(mission_id, artifact_name)
    except Exception as exc:
        logger.debug(
            f"[Workflow Hook] skip_when lookup failed for {artifact_name}: {exc}"
        )
        return False, ""

    if raw_artifact is None:
        # Artifact not produced yet — can't evaluate; let step run.
        return False, ""

    try:
        data = json.loads(raw_artifact) if isinstance(raw_artifact, str) else raw_artifact
    except (json.JSONDecodeError, TypeError):
        return False, ""

    current = data
    for segment in path:
        if isinstance(current, dict):
            current = current.get(segment)
        else:
            current = None
            break

    matched = (op == "==" and current == literal) or (op == "!=" and current != literal)
    if matched:
        return True, f"{artifact_name}.{'.'.join(path)} {op} {literal!r}"
    return False, ""


def is_workflow_step(context: dict) -> bool:
    """Check whether the task context marks this as a workflow step."""
    return bool(context.get("is_workflow_step"))


def extract_output_artifact_names(context: dict) -> list[str]:
    """Get output_artifacts list from context, defaulting to empty."""
    return context.get("output_artifacts", [])


def _parse_context(task: dict) -> dict:
    """Parse task context, handling both dict and JSON string forms."""
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    if not isinstance(ctx, dict):
        ctx = {}
    return ctx


# ──────────────────────────────────────────────────────────────────
# Removed 2026-04-27 (handoff item D, follow-up to commit f9507bb):
# `enrich_task_description` and `pre_execute_workflow_step` had ZERO
# production callers since the Task 13 orchestrator trim (~2026-04-
# 20). Live prompt build runs through `BaseAgent._build_context`
# (deps + chain context + missing-artifact NOTE + schema retry hint).
# Phase-summary injection that the dead path provided is NOT replic-
# ated elsewhere — accepted loss per architectural pick A. If a fut-
# ure mission shows steps starved for prior-phase summary context,
# port `_generate_phase_summary` calls into `_build_context`.
# ──────────────────────────────────────────────────────────────────



_review_tracker = ReviewTracker()


_POST_EXECUTE_SLOW_MS = 250


async def post_execute_workflow_step(task: dict, result: dict) -> None:
    """Post-hook: store output artifacts, evaluate conditional groups,
    trigger template expansion, and track review cycles.

    If the task is not a workflow step, returns immediately.

    Instrumented with a wall-clock timer that logs a WARNING whenever
    a single invocation exceeds ``_POST_EXECUTE_SLOW_MS``. Pump-tick
    starvation (Telegram "Query is too old" bursts on 2026-04-24) was
    traced to synchronous CPU-heavy work inside this hook — schema
    validation regex on 20-30k-char outputs, ``dogru_mu_samet.assess``,
    JSON round-trips, ``_unwrap_envelope`` on malformed long strings.
    The warning surfaces hot invocations without speculatively moving
    things into ``run_in_executor``; once profile data confirms a
    specific offender, it can be migrated.
    """
    import time as _time
    _t0 = _time.perf_counter()
    try:
        return await _post_execute_workflow_step_impl(task, result)
    finally:
        _elapsed_ms = (_time.perf_counter() - _t0) * 1000
        if _elapsed_ms > _POST_EXECUTE_SLOW_MS:
            ctx_for_log = _parse_context(task)
            output_len = 0
            try:
                ov = result.get("result") if isinstance(result, dict) else None
                if isinstance(ov, str):
                    output_len = len(ov)
            except Exception:
                pass
            logger.warning(
                f"[Workflow Hook] post_execute slow: "
                f"{_elapsed_ms:.0f}ms "
                f"task_id={task.get('id','?')} "
                f"step={ctx_for_log.get('workflow_step_id','?')} "
                f"output_chars={output_len}"
            )


async def _post_execute_workflow_step_impl(task: dict, result: dict) -> None:
    ctx = _parse_context(task)
    if not is_workflow_step(ctx):
        return

    mission_id = ctx.get("mission_id") or task.get("mission_id")
    output_names = extract_output_artifact_names(ctx)
    step_id = ctx.get("workflow_step_id", "")

    if not mission_id:
        return

    store = get_artifact_store()

    if not output_names:
        # No output artifacts — skip artifact storage but still run
        # phase completion check below.
        workflow_phase = ctx.get("workflow_phase")
        if workflow_phase:
            await _check_phase_completion(mission_id, workflow_phase)
        return
    output_value = result.get("result", "")

    # ── Unwrap envelope from the agent result ──
    output_value = _unwrap_envelope(output_value)

    # ── Recover artifact content from workspace files ──
    # If the agent wrote output to files (via write_file), the result text
    # may be a short summary.  Collect ALL matching workspace files and
    # combine them with the result — this ensures schema validation sees
    # the full content even when multiple output artifacts exist.
    if mission_id and output_names:
        try:
            import os
            from ...tools.workspace import WORKSPACE_DIR
            artifact_dir = os.path.join(WORKSPACE_DIR, f"mission_{mission_id}")
            file_parts = []
            for name in output_names:
                for ext in (".json", ".md", ".txt"):
                    fpath = os.path.join(artifact_dir, f"{name}{ext}")
                    if os.path.isfile(fpath):
                        with open(fpath, "r", encoding="utf-8") as f:
                            file_content = f.read()
                        file_content = _unwrap_envelope(file_content)
                        from dogru_mu_samet import assess as cq_assess, salvage as cq_salvage
                        cq = cq_assess(file_content)
                        if cq.is_degenerate:
                            cleaned = cq_salvage(file_content)
                            if cleaned:
                                logger.info(
                                    f"[Workflow Hook] Salvaged degenerate "
                                    f"workspace file '{name}{ext}' "
                                    f"({len(file_content)} -> {len(cleaned)} chars)"
                                )
                                try:
                                    with open(fpath, "w", encoding="utf-8") as wf:
                                        wf.write(cleaned)
                                except OSError:
                                    pass
                                file_parts.append(cleaned)
                            else:
                                logger.warning(
                                    f"[Workflow Hook] Deleting unsalvageable "
                                    f"workspace file '{name}{ext}' "
                                    f"({len(file_content)} chars, {cq.summary})"
                                )
                                try:
                                    os.remove(fpath)
                                except OSError:
                                    pass
                            break
                        if len(file_content) > 200:
                            file_parts.append(file_content)
                            logger.info(
                                f"[Workflow Hook] Found artifact '{name}' "
                                f"in workspace ({len(file_content)} chars)"
                            )
                        break
            if file_parts:
                # Prefer the agent's current result over stale workspace
                # files from earlier failed attempts. Previously the code
                # picked whichever was LONGER, which let a 26k-char
                # wrapped-and-truncated file from a prior DLQ'd attempt
                # overwrite the current attempt's clean 6k JSON (observed
                # on mission 46 task 2888 2026-04-24: retry produced
                # valid MoSCoW but the artifact file was the prior
                # truncated wrapper, poisoning downstream 2889).
                # Rule: only adopt the workspace file when the current
                # result is empty OR fails to json-parse AND the file
                # DOES parse cleanly.
                best_file = max(file_parts, key=len)

                def _parses_ok(s: str) -> bool:
                    try:
                        json.loads(s)
                        return True
                    except (json.JSONDecodeError, TypeError):
                        return False

                current_ok = bool(output_value) and _parses_ok(output_value)
                if not current_ok and _parses_ok(best_file):
                    output_value = best_file
        except Exception as e:
            logger.debug(f"[Workflow Hook] Workspace artifact recovery failed: {e}")

    # ── Detect fake completions ──
    # Small LLMs wrap failure reports in final_answer. Detect and reject.
    if output_value and _is_disguised_failure(output_value):
        result["status"] = "failed"
        result["error"] = "Agent reported completion but output indicates failure"
        logger.warning(
            f"[Workflow Hook] Step '{step_id}' detected as disguised failure — "
            f"overriding to failed for retry"
        )
        return  # Don't store garbage artifacts

    # ── Final quality gate before storing ──
    if output_value:
        from dogru_mu_samet import assess as cq_assess
        _artifact_schema = ctx.get("artifact_schema", {})
        _step_max = _artifact_schema.get("max_output_chars", 20_000)
        cq = cq_assess(output_value, max_size=_step_max)
        if cq.is_degenerate:
            result["status"] = "failed"
            result["error"] = f"Degenerate content rejected: {cq.summary}"
            logger.warning(
                f"[Workflow Hook] Step '{step_id}' output rejected: "
                f"{cq.summary} ({len(output_value)} chars)"
            )
            return

    for name in output_names:
        await store.store(mission_id, name, output_value)
        logger.info(
            f"[Workflow Hook] Post-execute: stored artifact '{name}' "
            f"for mission {mission_id} ({len(output_value)} chars)"
        )

    # ── Auto-summarize large artifacts ──
    # Structural summary is stored immediately (fast, no LLM).
    # LLM upgrade is queued for the orchestrator to process between cycles.
    _SUMMARY_THRESHOLD = 3000
    _MIN_SUMMARY_LEN = 50
    if output_value and len(output_value) > _SUMMARY_THRESHOLD:
        for name in output_names:
            summary = _structural_summary(output_value)
            if summary and len(summary) >= _MIN_SUMMARY_LEN:
                summary_name = f"{name}_summary"
                await store.store(mission_id, summary_name, summary)
                logger.info(
                    f"[Workflow Hook] Structural summary '{name}' -> '{summary_name}' "
                    f"({len(output_value)} -> {len(summary)} chars)"
                )
            # LLM-upgrade summary is scheduled by Beckman as a post-hook
            # task after grade passes (packages/general_beckman/apply.py::
            # _apply_posthook_verdict) — no need to queue here.

    # ── Write artifacts to disk in mission directory ──
    if output_value and mission_id:
        try:
            from ...tools.workspace import WORKSPACE_DIR
            import os
            artifact_dir = os.path.join(WORKSPACE_DIR, f"mission_{mission_id}")
            os.makedirs(artifact_dir, exist_ok=True)
            for name in output_names:
                file_path = os.path.join(artifact_dir, f"{name}.md")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(output_value)
                logger.debug(f"[Workflow Hook] Wrote artifact to {file_path}")
        except Exception as e:
            logger.debug(f"[Workflow Hook] Could not write artifact to disk: {e}")

    # ── Validate artifact schema ──
    # Gate is on `artifact_schema` AND "this task is the LLM-driven worker
    # that's supposed to produce the artifact". Mechanical sibling tasks
    # (workflow_advance, git_commit, clarify, etc.) inherit artifact_schema
    # from their parent step's ctx but DON'T produce artifact content —
    # they do bookkeeping. The pre-existing gate `if artifact_schema and
    # output_value` skipped them by accident (empty output → false). My
    # earlier "empty output must fail" tightening (commit 2d6d82c) broke
    # them by removing that incidental skip; mission 46 tasks 3797 + 3804
    # DLQ'd as mechanical workflow_advance with "schema requires
    # backend_design_compilation / technical_design_document" even though
    # their LLM siblings (2923, 2924) had already produced the artifacts.
    #
    # Right gate: validate only when this task IS the artifact producer
    # (no executor, agent isn't mechanical/grader/artifact_summarizer).
    # For producer tasks, an empty output is still a real failure (not
    # the silent bypass that let task 2921 ghost-complete on attempt 2).
    artifact_schema = ctx.get("artifact_schema")
    _agent_type = (task.get("agent_type") or ctx.get("agent_type") or "")
    _executor = (task.get("executor") or ctx.get("executor") or "")
    _is_producer = (
        _executor != "mechanical"
        and _agent_type not in ("mechanical", "grader", "artifact_summarizer")
    )
    # Skip schema validation when the agent emitted a clarify action on a
    # ``triggers_clarification`` step. Such steps produce a HUMAN
    # QUESTION, not a structured artifact — the question text lives in
    # ``result.question`` (or ``result.clarification``), not in
    # ``result.result`` which would normally hold the artifact body. The
    # ``triggers_clarification`` override below at line ~1140 then routes
    # to needs_clarification correctly. Without this skip, mission 57
    # step 0.5 (human_clarification_request) burned 5 retries because
    # the schema demanded ``array, min_items: 3`` but output_value was
    # empty (clarify text in the wrong field).
    _is_clarify_action = bool(
        ctx.get("triggers_clarification")
        and (
            (isinstance(result, dict) and result.get("status") == "needs_clarification")
            or (isinstance(result, dict) and (result.get("question") or "").strip())
            or (isinstance(result, dict) and (result.get("clarification") or "").strip())
        )
    )
    if artifact_schema and _is_producer and not _is_clarify_action:
        if not output_value or not str(output_value).strip():
            is_valid = False
            error_msg = (
                "empty output — the agent returned no artifact content for "
                f"a step that requires schema {list(artifact_schema.keys())}"
            )
        else:
            is_valid, error_msg = validate_artifact_schema(output_value, artifact_schema)
        if not is_valid:
            attempts = task.get("worker_attempts", 0)
            logger.warning(
                f"[Workflow Hook] Artifact schema validation failed for step "
                f"'{step_id}': {error_msg}. Attempt {attempts}"
            )
            # Store the error AND the previous output in context so the
            # enriched prompt on next attempt shows what went wrong and what
            # was already produced — preventing the agent from burning
            # iterations re-reading files it already wrote.
            try:
                from ...infra.db import update_task
                new_ctx = dict(ctx)
                new_ctx["_schema_error"] = error_msg
                # Canonicalize before storing — collapses any escape
                # layers Qwen / Mistral / etc. emit so the NEXT retry's
                # prompt shows clean JSON, not soup that the model will
                # re-escape into deeper compounding.
                new_ctx["_prev_output"] = canonicalize_for_retry(
                    output_value
                )[:6000]
                # Stamp for the NEXT attempt — _retry_or_dlq increments
                # worker_attempts before re-queuing. The reader gates on
                # match against the live worker_attempts.
                new_ctx["_schema_error_for_attempt"] = int(attempts) + 1
                await update_task(
                    task.get("id"),
                    context=json.dumps(new_ctx),
                )
            except Exception as e:
                logger.debug(f"[Workflow Hook] Could not update task context: {e}")
            # Signal failure — the unified retry/DLQ path in the orchestrator
            # decides whether to retry, delay, or give up.
            result["status"] = "failed"
            result["error"] = f"Schema validation: {error_msg}"

    # ── Force needs_clarification for human-gate steps ──
    # Steps with triggers_clarification=true bypass LLM's clarify action.
    # Only fires ONCE — if clarification_history already has answers,
    # the human already responded and the step should complete normally.
    #
    # The agent's clarify question may live in ``result.question``,
    # ``result.clarification``, or ``result.result`` depending on which
    # parser path produced it. Pick the first non-empty source so the
    # override doesn't silently skip when the agent picked the wrong
    # field name (mission 57 task 4376 — clarify text was in
    # ``result.question`` while ``output_value`` was empty).
    _clarify_text = output_value
    if not _clarify_text and isinstance(result, dict):
        _clarify_text = (
            (result.get("question") or "").strip()
            or (result.get("clarification") or "").strip()
        )
    if (ctx.get("triggers_clarification")
            and _clarify_text
            and not ctx.get("clarification_history")):
        from dogru_mu_samet import assess as cq_assess
        _clar_cq = cq_assess(_clarify_text)
        if _clar_cq.is_degenerate:
            result["status"] = "failed"
            result["error"] = (
                f"Clarification question was degenerate ({_clar_cq.summary}), "
                f"retrying instead of sending garbled text to human"
            )
            logger.warning(
                f"[Workflow Hook] Step '{step_id}' clarification rejected: "
                f"{_clar_cq.summary}"
            )
            return
        result["status"] = "needs_clarification"
        result["clarification"] = _clarify_text
        logger.info(
            f"[Workflow Hook] Step '{step_id}' triggers_clarification — "
            f"overriding result status to needs_clarification"
        )

    # ── Store clarification_answers artifact when human-gate step completes ──
    # Second run (after user answered): clarification_history exists, step completes.
    # Store user_clarification as the clarification_answers artifact so downstream
    # steps (e.g. idea_brief_compilation) can consume it.
    if (ctx.get("triggers_clarification")
            and ctx.get("clarification_history")):
        user_clarification = ctx.get("user_clarification", "")
        if user_clarification and mission_id:
            await store.store(mission_id, "clarification_answers", user_clarification)
            # Also write to disk
            try:
                from ...tools.workspace import WORKSPACE_DIR
                import os
                artifact_dir = os.path.join(WORKSPACE_DIR, f"mission_{mission_id}")
                os.makedirs(artifact_dir, exist_ok=True)
                file_path = os.path.join(artifact_dir, "clarification_answers.md")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(user_clarification)
            except Exception:
                pass
            logger.info(
                f"[Workflow Hook] Step '{step_id}' completed with clarification — "
                f"stored 'clarification_answers' artifact ({len(user_clarification)} chars)"
            )

    # ── Check conditional group triggers ──
    await _check_conditional_triggers(mission_id, output_names, store)

    # ── Check template expansion trigger ──
    if "implementation_backlog" in output_names:
        await _trigger_template_expansion(mission_id, output_value)

    # ── Track review status ──
    status = result.get("status", "completed")
    if status in ("needs_review", "failed"):
        action = _review_tracker.record_failure(step_id)
        if action == "escalate":
            logger.warning(
                f"[Workflow Hook] Step '{step_id}' exceeded max review "
                f"cycles — escalating to needs_clarification"
            )

    # ── Check phase completion for checkpoint/resume support ──
    workflow_phase = ctx.get("workflow_phase")
    if mission_id and workflow_phase:
        await _check_phase_completion(mission_id, workflow_phase)


async def _check_phase_completion(mission_id: int, phase_id: str) -> bool:
    """Detect when all tasks in a workflow phase are done and checkpoint it.

    Returns True if the phase is complete, False otherwise.
    """
    try:
        from ...infra.db import get_tasks_for_mission, get_workflow_checkpoint, upsert_workflow_checkpoint
    except ImportError as exc:
        logger.debug(f"[Workflow Hook] Phase completion check skipped (import): {exc}")
        return False

    try:
        tasks = await get_tasks_for_mission(mission_id)
    except Exception as exc:
        logger.debug(f"[Workflow Hook] Could not fetch tasks for mission {mission_id}: {exc}")
        return False

    terminal_states = {"completed", "skipped", "cancelled"}
    phase_tasks = []
    for t in tasks:
        ctx = _parse_context(t)
        if ctx.get("workflow_phase") == phase_id:
            phase_tasks.append(t)

    if not phase_tasks:
        return False

    all_done = all(t.get("status") in terminal_states for t in phase_tasks)
    if not all_done:
        return False

    # Phase complete — update checkpoint
    try:
        checkpoint = await get_workflow_checkpoint(mission_id)
        completed = checkpoint["completed_phases"] if checkpoint else []
        workflow_name = checkpoint["workflow_name"] if checkpoint else ""

        if phase_id not in completed:
            completed.append(phase_id)

        await upsert_workflow_checkpoint(
            mission_id=mission_id,
            workflow_name=workflow_name,
            current_phase=phase_id,
            completed_phases=completed,
        )
        logger.info(
            f"[Workflow Hook] Phase '{phase_id}' complete for mission {mission_id} "
            f"({len(phase_tasks)} tasks). Checkpoint updated."
        )
    except Exception as exc:
        logger.debug(f"[Workflow Hook] Could not update checkpoint: {exc}")

    # Generate a summary artifact for the completed phase
    await _generate_phase_summary(mission_id, phase_id, phase_tasks)

    # ── Evaluate quality gate ──
    await _evaluate_phase_gate(mission_id, phase_id, workflow_name)

    return True


async def _evaluate_phase_gate(
    mission_id: int, phase_id: str, workflow_name: str = ""
) -> None:
    """Evaluate the quality gate for a completed phase and store the result.

    Loads the gate definition from the workflow JSON if available,
    falling back to hardcoded gates (v1/v2 compatibility).
    """
    store = get_artifact_store()
    try:
        phase_num = phase_id.replace("phase_", "")

        # Try to load gate definition from workflow JSON metadata
        gate_def = None
        if workflow_name:
            try:
                from .loader import load_workflow
                wf = load_workflow(workflow_name)
                gates = wf.metadata.get("quality_gates", {})
                gate_def = gates.get(phase_id)
            except Exception as exc:
                logger.debug(
                    f"[Workflow Hook] Could not load workflow gate "
                    f"for {phase_id}: {exc}"
                )

        passed, details = await evaluate_gate(
            mission_id, phase_id, store, gate_def=gate_def
        )

        # Store gate result as artifact
        result_text = format_gate_result(phase_id, passed, details)
        await store.store(mission_id, f"phase_{phase_num}_gate_result", result_text)

        if details:  # Only log if there was actually a gate
            if passed:
                logger.info(
                    f"[Workflow Hook] Quality gate for '{phase_id}' PASSED "
                    f"(mission {mission_id})"
                )
            else:
                logger.warning(
                    f"[Workflow Hook] Quality gate for '{phase_id}' FAILED "
                    f"(mission {mission_id}): {result_text}"
                )
    except Exception as exc:
        logger.debug(f"[Workflow Hook] Quality gate evaluation failed: {exc}")


async def _generate_phase_summary(
    mission_id: int, phase_id: str, phase_tasks: list[dict]
) -> None:
    """Build a structured summary from a completed phase's output artifacts.

    The summary is stored as ``phase_{N}_summary`` in the artifact store so
    that subsequent phases can receive it as context.
    """
    from .status import PHASE_NAMES

    store = get_artifact_store()

    # Collect output artifact names from all phase tasks
    output_names: list[str] = []
    for t in phase_tasks:
        ctx = _parse_context(t)
        output_names.extend(ctx.get("output_artifacts", []))

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique_names: list[str] = []
    for name in output_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    # Fetch artifact contents
    artifact_contents = await store.collect(mission_id, unique_names)

    # Build summary text
    phase_name = PHASE_NAMES.get(phase_id, phase_id)
    # Extract phase number for the artifact key
    try:
        phase_num = phase_id.split("_", 1)[1]
    except IndexError:
        phase_num = phase_id

    names_with_content = [
        n for n in unique_names if artifact_contents.get(n)
    ]
    artifact_count = len(names_with_content)

    lines: list[str] = [
        f"## Phase {phase_num}: {phase_name} — Summary",
        f"**Key outputs:** {', '.join(names_with_content) if names_with_content else 'none'}",
        f"**Artifacts produced:** {artifact_count}",
        "",
    ]

    for name in names_with_content:
        content = artifact_contents[name] or ""
        excerpt = content[:200]
        if len(content) > 200:
            excerpt += "..."
        lines.append(f"### {name}\n{excerpt}")
        lines.append("")

    summary_text = "\n".join(lines).rstrip()

    from dogru_mu_samet import assess as cq_assess
    _phase_cq = cq_assess(summary_text)
    if _phase_cq.is_degenerate:
        logger.warning(
            f"[Workflow Hook] Phase summary degenerate ({_phase_cq.summary}), "
            f"using structural fallback"
        )
        summary_text = _structural_summary(summary_text)

    summary_artifact_name = f"phase_{phase_num}_summary"
    await store.store(mission_id, summary_artifact_name, summary_text)
    logger.info(
        f"[Workflow Hook] Generated summary for '{phase_id}' "
        f"({artifact_count} artifacts) -> '{summary_artifact_name}'"
    )


async def _check_conditional_triggers(
    mission_id: int, output_names: list[str], store: ArtifactStore
) -> None:
    """Evaluate conditional groups when their trigger artifact is produced."""
    try:
        from .loader import load_workflow

        # Try loading the workflow used by this mission
        workflow_name = "i2p_v3"  # fallback
        try:
            from ...infra.db import get_mission
            mission = await get_mission(mission_id)
            if mission:
                m_ctx = mission.get("context", "{}")
                if isinstance(m_ctx, str):
                    m_ctx = json.loads(m_ctx)
                workflow_name = m_ctx.get("workflow_name", "i2p_v3")
        except Exception:
            pass
        wf = load_workflow(workflow_name)
    except Exception:
        logger.debug("[Workflow Hook] Could not load workflow for conditional eval")
        return

    for group in wf.conditional_groups:
        condition_artifact = group.get("condition_artifact", "")
        if condition_artifact not in output_names:
            continue

        artifact_value = await store.retrieve(mission_id, condition_artifact)
        if artifact_value is None:
            continue

        condition_check = group.get("condition_check", "")
        result_bool = evaluate_condition(condition_check, artifact_value)
        included, excluded = resolve_group(group, artifact_value)

        logger.info(
            f"[Workflow Hook] Conditional group '{group.get('group_id')}': "
            f"condition={result_bool}, include={len(included)}, "
            f"exclude={len(excluded)} steps"
        )

        # Update task statuses in DB for excluded steps
        if excluded:
            try:
                from ...infra.db import update_task_by_context_field, propagate_skips

                for step in excluded:
                    logger.warning(
                        "TASK SKIPPED (conditional group exclusion)",
                        step=step,
                        group=group.get("group_id"),
                        mission_id=mission_id,
                    )
                    await update_task_by_context_field(
                        mission_id=mission_id,
                        field="workflow_step_id",
                        value=step,
                        status="skipped",
                        error=f"excluded by conditional group '{group.get('group_id')}'",
                    )
                # Cascade skips to downstream dependents
                skipped_count = await propagate_skips(mission_id)
                if skipped_count:
                    logger.info(
                        f"[Workflow Hook] Cascaded skip to {skipped_count} dependent tasks"
                    )
            except (ImportError, Exception) as e:
                logger.debug(
                    f"[Workflow Hook] Could not skip excluded steps: {e}"
                )


async def _trigger_template_expansion(mission_id: int, backlog_text: str) -> None:
    """Expand feature_implementation_template for each feature in backlog.

    Respects ``depends_on_features`` from the backlog: the first task of a
    dependent feature won't start until the last task of its prerequisite
    feature completes.  After all features are expanded, inserts a
    cross-feature integration test step.
    """
    import json as _json

    try:
        features = _json.loads(backlog_text)
        if not isinstance(features, list):
            logger.debug("[Workflow Hook] implementation_backlog is not a list")
            return
    except (ValueError, TypeError):
        logger.debug("[Workflow Hook] Could not parse implementation_backlog as JSON")
        return

    try:
        from .loader import load_workflow
        from .expander import expand_template, expand_steps_to_tasks
        from ...infra.db import add_task as insert_task, update_task

        # Try the workflow used by this mission, fall back to i2p_v3 then i2p_v2
        workflow_name = "i2p_v3"
        try:
            from ...infra.db import get_mission
            mission = await get_mission(mission_id)
            if mission:
                m_ctx = mission.get("context", "{}")
                if isinstance(m_ctx, str):
                    m_ctx = _json.loads(m_ctx)
                workflow_name = m_ctx.get("workflow_name", "i2p_v3")
        except Exception:
            pass

        wf = load_workflow(workflow_name)
        template = wf.get_template("feature_implementation_template")
        if not template:
            logger.warning("[Workflow Hook] feature_implementation_template not found")
            return

        # Track feature_id → (first_task_id, last_task_id) for cross-feature deps
        feature_task_range: dict[str, tuple[int, int]] = {}

        # Idempotency check (handoff item P): mission 46 phase 8 had
        # two waves of feat tasks (4015-4023 + 4040-onward) when
        # workflow_advance fired twice for the parent step (or when a
        # reset re-triggered the expansion). Each feature's expanded
        # tasks share a ``[8.{fid}.<step>]`` title prefix, so a single
        # SQL pre-check tells us which fids are already expanded for
        # this mission. We skip those fids on this call.
        already_expanded: set[str] = set()
        try:
            from ...infra.db import get_db
            _db = await get_db()
            _cur = await _db.execute(
                """SELECT title FROM tasks
                    WHERE mission_id = ?
                      AND title LIKE '[8.%]%'""",
                (mission_id,),
            )
            for _row in await _cur.fetchall():
                _t = _row[0] if isinstance(_row, tuple) else _row["title"]
                if not _t or not _t.startswith("[8."):
                    continue
                # Title is "[8.<fid>.<stepname>] ..." — extract fid.
                _close_bracket = _t.find("]")
                if _close_bracket <= 3:
                    continue
                _step_id = _t[1:_close_bracket]  # e.g. "8.feat1.write_code"
                _parts = _step_id.split(".", 2)
                if len(_parts) >= 2:
                    already_expanded.add(_parts[1])
            await _cur.close()
        except Exception as _exc:
            logger.debug(
                f"[Workflow Hook] feature-template idempotency check "
                f"failed: {_exc!r} — proceeding without skip"
            )

        for feature in features:
            if not isinstance(feature, dict):
                continue
            fid = feature.get("id", feature.get("feature_id", "unknown"))
            fname = feature.get("name", feature.get("feature_name", "Unnamed"))

            if fid in already_expanded:
                logger.info(
                    f"[Workflow Hook] feature '{fid}' already expanded "
                    f"for mission #{mission_id} — skipping (handoff P)"
                )
                continue

            expanded = expand_template(
                template,
                params={"feature_id": fid, "feature_name": fname},
                prefix=f"8.{fid}.",
            )

            tasks = expand_steps_to_tasks(
                expanded, mission_id=mission_id, initial_context={}
            )

            # Batch insert with rollback on failure
            inserted_ids = []
            try:
                for t in tasks:
                    t.pop("depends_on_steps", None)
                    task_id = await insert_task(**t)
                    inserted_ids.append(task_id)
            except Exception as insert_err:
                # Rollback: cancel partially inserted tasks
                for tid in inserted_ids:
                    try:
                        await update_task(tid, status="cancelled")
                    except Exception:
                        pass
                logger.error(
                    f"[Workflow Hook] Partial expansion rollback for '{fid}': {insert_err}"
                )
                continue  # Skip this feature, try next one

            if inserted_ids:
                feature_task_range[fid] = (inserted_ids[0], inserted_ids[-1])

            logger.info(
                f"[Workflow Hook] Expanded template for feature '{fid}' "
                f"({len(expanded)} steps \u2192 {len(tasks)} tasks)"
            )

        # ── Wire cross-feature dependencies ──
        # If feature B depends_on_features: ["A"], then B's first task
        # should wait until A's last task completes.
        for feature in features:
            if not isinstance(feature, dict):
                continue
            fid = feature.get("id", feature.get("feature_id", "unknown"))
            dep_features = feature.get("depends_on_features", [])
            if not dep_features or fid not in feature_task_range:
                continue

            first_task_id = feature_task_range[fid][0]
            prerequisite_task_ids = []
            for dep_fid in dep_features:
                if dep_fid in feature_task_range:
                    prerequisite_task_ids.append(feature_task_range[dep_fid][1])

            if prerequisite_task_ids:
                try:
                    dep_json = _json.dumps(prerequisite_task_ids)
                    await update_task(first_task_id, depends_on=dep_json)
                    logger.info(
                        f"[Workflow Hook] Feature '{fid}' first task #{first_task_id} "
                        f"depends on tasks {prerequisite_task_ids} (cross-feature)"
                    )
                except Exception as dep_err:
                    logger.debug(
                        f"[Workflow Hook] Could not set cross-feature deps: {dep_err}"
                    )

        # ── Insert cross-feature integration test step ──
        # Runs after ALL features are done — tests interactions between features
        if len(feature_task_range) >= 2:
            all_last_tasks = [last for _, last in feature_task_range.values()]
            feature_names = []
            for f in features:
                if isinstance(f, dict):
                    feature_names.append(
                        f.get("name", f.get("feature_name", f.get("id", "?")))
                    )
            try:
                integration_task_id = await insert_task(
                    title="[8.integration] Cross-feature integration tests",
                    description=(
                        f"Test interactions between all implemented features: "
                        f"{', '.join(feature_names)}. "
                        f"Verify: shared data flows correctly between features, "
                        f"auth/permissions work across feature boundaries, "
                        f"navigation between features works, "
                        f"no conflicts in shared resources (DB, API routes, state). "
                        f"Run the full test suite and report results."
                    ),
                    mission_id=mission_id,
                    agent_type="test_generator",
                    tier="auto",
                    priority=7,
                    depends_on=all_last_tasks,
                    context={
                        "workflow_step_id": "8.integration",
                        "workflow_phase": "phase_8",
                        "is_workflow_step": True,
                        "difficulty": 6,
                        "tools_hint": ["shell", "read_file", "write_file", "coverage",
                                       "query_codebase", "codebase_map"],
                        "output_artifacts": ["integration_test_results"],
                    },
                )
                logger.info(
                    f"[Workflow Hook] Created cross-feature integration test "
                    f"task #{integration_task_id} (depends on {len(all_last_tasks)} features)"
                )
            except Exception as integ_err:
                logger.debug(
                    f"[Workflow Hook] Could not create integration test task: {integ_err}"
                )

    except (ImportError, Exception) as e:
        logger.debug(f"[Workflow Hook] Template expansion failed: {e}")
