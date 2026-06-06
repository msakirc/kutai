"""Z7 A5 — brand_voice_lint posthook handler.

Lints produced copy artifacts against brand-voice rules from
``docs/templates/brand_voices/<audience>.md``.

Handler contract
----------------
``handle(task, result) -> dict``

Returns one of:

- ``{"status": "ok"}``                    — all checks passed
- ``{"status": "skip", "reason": "..."}`` — graceful skip (absent audience
  metadata or missing voice doc)
- ``{"status": "failed", "reason": "...", "violations": [...],
    "founder_action_id": int | None}``    — at least one blocker violation

Violations shape
----------------
Each violation::

    {
        "severity": "blocker" | "warning" | "info",
        "check":    str,          # "prohibited_term" | "sentence_length" | ...
        "detail":   str,          # human-readable description
        "excerpt":  str,          # offending text excerpt (≤120 chars)
    }

Severity policy
---------------
- prohibited term hit          → **blocker**
- FK reading level out of band → **warning**
- avg sentence length drift    → **warning**
- we/you pronoun ratio drift   → **warning**
- tone score low (LLM pass)    → **info**

Blockers only trigger ``status="failed"``. Warnings and info annotations are
returned in ``violations`` even on ``status="ok"`` so the apply layer can
surface them.

Founder-action annotation
-------------------------
When there are blocker violations, a ``founder_action`` review card is created
(``kind="generic"``) with the violations list as ``instructions``. The card id
is returned under ``founder_action_id`` — apply.py surfacing is additive;
the posthook's failed status already re-queues the source step for retry.

LLM tone pass
-------------
Routed through ``husam.run`` (non-pump, no sibling-deadlock) per the CPS SP4a
migration. The tone call is best-effort — if husam.run raises, we degrade to
``info``-level note "tone_pass_skipped" and continue.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthooks.brand_voice_lint")

# ─── Flesch-Kincaid constants ──────────────────────────────────────────────
# Grade = 0.39 × (words/sentences) + 11.8 × (syllables/words) − 15.59
_FK_A = 0.39
_FK_B = 11.8
_FK_C = 15.59

# ─── Sentence splitter ────────────────────────────────────────────────────
_SENTENCE_END = re.compile(r"[.!?]+[\s\"')\]]*(?=[A-Z\s]|\Z)", re.MULTILINE)

# ─── Word tokeniser (basic) ───────────────────────────────────────────────
_WORD_RE = re.compile(r"\b[a-zA-Z']+\b")

# ─── Pronoun patterns ─────────────────────────────────────────────────────
_WE_RE = re.compile(r"\b(we|us|our|ours|ourselves)\b", re.IGNORECASE)
_YOU_RE = re.compile(r"\b(you|your|yours|yourself|yourselves)\b", re.IGNORECASE)

# ─── 200-word window for tone signals ─────────────────────────────────────
_WINDOW_WORDS = 200

# ─── Drift thresholds ─────────────────────────────────────────────────────
_SENTENCE_DRIFT = 0.25   # ±25% of target triggers warning
_FK_TOLERANCE = 0.0      # must be at or below max (no band)
_PRONOUN_DRIFT = 0.05    # ±5% allowed slack around we_ratio_max


# ---------------------------------------------------------------------------
# Syllable counting (heuristic — no NLTK dependency)
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    """Heuristic syllable counter for FK readability."""
    word = word.lower().strip("'")
    if not word:
        return 0
    # Strip trailing silent 'e'
    if word.endswith("e") and len(word) > 2:
        word = word[:-1]
    count = len(re.findall(r"[aeiou]+", word))
    return max(1, count)


# ---------------------------------------------------------------------------
# Text statistics
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (heuristic)."""
    # Split on . ! ? followed by whitespace + capital or end of string
    parts = _SENTENCE_END.split(text)
    # Also add back any trailing fragments
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences if sentences else [text]


def _text_stats(text: str) -> dict[str, Any]:
    """Compute word count, sentence count, syllables, pronoun counts."""
    words = _WORD_RE.findall(text)
    sentences = _split_sentences(text)

    word_count = len(words)
    sentence_count = max(1, len(sentences))
    syllable_count = sum(_count_syllables(w) for w in words)

    avg_sentence_len = word_count / sentence_count

    # Flesch-Kincaid grade level
    if word_count == 0:
        fk_grade = 0.0
    else:
        fk_grade = round(
            _FK_A * (word_count / sentence_count)
            + _FK_B * (syllable_count / word_count)
            - _FK_C,
            2,
        )

    we_count = len(_WE_RE.findall(text))
    you_count = len(_YOU_RE.findall(text))

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": round(avg_sentence_len, 2),
        "fk_grade": fk_grade,
        "we_count": we_count,
        "you_count": you_count,
    }


# ---------------------------------------------------------------------------
# Individual lint checks
# ---------------------------------------------------------------------------

def _check_prohibited_terms(text: str, prohibited: list[str]) -> list[dict]:
    """Check each prohibited_terms entry against text.

    Entries starting/ending with ``/`` are treated as regex patterns.
    Exact-string entries are matched case-sensitively.
    Returns list of violation dicts.
    """
    violations: list[dict] = []
    for term in prohibited:
        if term.startswith("/") and term.endswith("/") and len(term) > 2:
            pattern_str = term[1:-1]
            try:
                pat = re.compile(pattern_str)
            except re.error:
                logger.debug("brand_voice_lint: invalid prohibited regex: %r", term)
                continue
            m = pat.search(text)
            if m:
                excerpt = text[max(0, m.start() - 20): m.end() + 20].strip()
                violations.append({
                    "severity": "blocker",
                    "check": "prohibited_term",
                    "detail": f"Prohibited pattern matched: {term}",
                    "excerpt": excerpt[:120],
                })
        else:
            if term in text:
                idx = text.index(term)
                excerpt = text[max(0, idx - 20): idx + len(term) + 20].strip()
                violations.append({
                    "severity": "blocker",
                    "check": "prohibited_term",
                    "detail": f"Prohibited term found: {term!r}",
                    "excerpt": excerpt[:120],
                })
    return violations


def _check_sentence_length(
    stats: dict, target: int
) -> list[dict]:
    """Warn if avg sentence length drifts more than ±25% from target."""
    violations: list[dict] = []
    avg = stats["avg_sentence_len"]
    low = target * (1.0 - _SENTENCE_DRIFT)
    high = target * (1.0 + _SENTENCE_DRIFT)
    if avg < low or avg > high:
        violations.append({
            "severity": "warning",
            "check": "sentence_length",
            "detail": (
                f"Avg sentence length {avg:.1f} words is outside "
                f"±25% of target {target} "
                f"(allowed range {low:.1f}–{high:.1f})"
            ),
            "excerpt": "",
        })
    return violations


def _check_flesch_kincaid(
    stats: dict, fk_max: float
) -> list[dict]:
    """Warn if Flesch-Kincaid grade level exceeds the ceiling."""
    violations: list[dict] = []
    grade = stats["fk_grade"]
    if grade > fk_max + _FK_TOLERANCE:
        violations.append({
            "severity": "warning",
            "check": "flesch_kincaid",
            "detail": (
                f"Flesch-Kincaid grade {grade:.1f} exceeds "
                f"max allowed {fk_max:.1f}"
            ),
            "excerpt": "",
        })
    return violations


def _check_pronoun_ratio(
    stats: dict, we_ratio_max: float | None
) -> list[dict]:
    """Warn if we/(we+you) ratio exceeds the ceiling."""
    if we_ratio_max is None:
        return []
    violations: list[dict] = []
    we = stats["we_count"]
    you = stats["you_count"]
    total_pronouns = we + you
    if total_pronouns < 5:
        # Not enough data to be meaningful
        return []
    ratio = we / total_pronouns
    ceiling = we_ratio_max + _PRONOUN_DRIFT
    if ratio > ceiling:
        violations.append({
            "severity": "warning",
            "check": "pronoun_ratio",
            "detail": (
                f"we/(we+you) ratio {ratio:.2f} exceeds allowed "
                f"max {we_ratio_max:.2f} "
                f"(we={we}, you={you})"
            ),
            "excerpt": "",
        })
    return violations


def _check_tone_signals_mechanical(
    text: str,
    required_signals: list[str],
    forbidden_signals: list[str],
) -> list[dict]:
    """Mechanically check tone keyword presence per 200-word window.

    required_signals: at least one must appear per 200-word block.
    forbidden_signals: none may appear anywhere.
    """
    violations: list[dict] = []

    # Check forbidden signals across whole text
    for signal in forbidden_signals:
        pattern = re.compile(r"\b" + re.escape(signal) + r"\b", re.IGNORECASE)
        m = pattern.search(text)
        if m:
            excerpt = text[max(0, m.start() - 20): m.end() + 20].strip()
            violations.append({
                "severity": "blocker",
                "check": "tone_forbidden",
                "detail": f"Forbidden tone signal found: {signal!r}",
                "excerpt": excerpt[:120],
            })

    # Check required signals per 200-word window
    if required_signals:
        words = _WORD_RE.findall(text)
        # Find all word positions in original text for window slicing
        word_positions = list(_WORD_RE.finditer(text))
        total_windows = max(1, (len(words) + _WINDOW_WORDS - 1) // _WINDOW_WORDS)
        for w_idx in range(total_windows):
            start_word = w_idx * _WINDOW_WORDS
            end_word = min(start_word + _WINDOW_WORDS, len(word_positions))
            if start_word >= len(word_positions):
                break
            # Get text slice for this window
            char_start = word_positions[start_word].start()
            char_end = word_positions[end_word - 1].end()
            window_text = text[char_start:char_end]

            found_any = any(
                re.search(r"\b" + re.escape(sig) + r"\b", window_text, re.IGNORECASE)
                for sig in required_signals
            )
            if not found_any:
                preview = window_text[:80].strip()
                violations.append({
                    "severity": "blocker",
                    "check": "tone_required",
                    "detail": (
                        f"No required tone signal ({', '.join(required_signals)!r}) "
                        f"found in 200-word window {w_idx + 1}/{total_windows}"
                    ),
                    "excerpt": preview[:120],
                })

    return violations


# ---------------------------------------------------------------------------
# LLM tone pass
# ---------------------------------------------------------------------------

_TONE_SYSTEM = (
    "You are a brand-voice tone reviewer. Score the provided text's tone match "
    "against the stated voice profile on a scale of 0–10 (10 = perfect match). "
    "Identify the 1–2 sections most misaligned with the voice. "
    "Reply ONLY in JSON with exactly: "
    '{"score": <int 0-10>, "flagged_sections": [{"excerpt": "...", "reason": "..."}]}'
)

_TONE_PROMPT = """Brand voice profile: {profile_name}
Voice body guidance:
{voice_body}

Text to score:
{text}

Return JSON with score (0-10) and up to 2 flagged sections."""


async def _run_llm_tone_pass(
    text: str,
    voice_display_name: str,
    voice_body_md: str,
    source_task_id: int | None,
) -> list[dict]:
    """Run LLM tone-match scoring via husam.run (single-call worker).

    Returns a list of ``info``-severity violations for low-scoring sections.
    Degrades gracefully to a single info note on any infrastructure error.
    """
    try:
        import husam
    except ImportError as exc:
        logger.debug("brand_voice_lint: husam import failed: %s", exc)
        return [{
            "severity": "info",
            "check": "tone_pass_skipped",
            "detail": f"LLM tone pass skipped (import error: {exc})",
            "excerpt": "",
        }]

    truncated_text = text[:4000]
    truncated_body = (voice_body_md or "")[:800]

    messages = [
        {"role": "system", "content": _TONE_SYSTEM},
        {
            "role": "user",
            "content": _TONE_PROMPT.format(
                profile_name=voice_display_name,
                voice_body=truncated_body,
                text=truncated_text,
            ),
        },
    ]

    _suffix = f"{time.monotonic_ns() % 1_000_000:06d}-{uuid.uuid4().hex[:6]}"
    spec = {
        "title": f"brand_voice_tone:task#{source_task_id}:{_suffix}",
        "description": "LLM tone-match scoring for brand_voice_lint",
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
                "estimated_output_tokens": 200,
            },
        },
    }

    try:
        resp = await husam.run(spec)
    except Exception as exc:
        logger.warning("brand_voice_lint: tone husam call raised: %r", exc)
        return [{
            "severity": "info",
            "check": "tone_pass_skipped",
            "detail": f"LLM tone pass unavailable: {exc}",
            "excerpt": "",
        }]

    raw_content = resp.get("content", "")
    if isinstance(raw_content, list):
        raw_content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in raw_content
        )
    raw_content = str(raw_content or "").strip()

    # Parse JSON response
    try:
        # Extract JSON from possible surrounding text
        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if not json_match:
            raise ValueError("no JSON found")
        parsed = json.loads(json_match.group())
        score = int(parsed.get("score", 10))
        flagged = parsed.get("flagged_sections") or []
    except Exception as exc:
        logger.debug("brand_voice_lint: tone parse error: %s | raw: %r", exc, raw_content[:200])
        return [{
            "severity": "info",
            "check": "tone_score",
            "detail": "Tone-match score could not be parsed from LLM response",
            "excerpt": "",
        }]

    violations: list[dict] = []
    if score < 6:
        for section in flagged[:2]:
            if isinstance(section, dict):
                violations.append({
                    "severity": "info",
                    "check": "tone_score",
                    "detail": (
                        f"Tone score {score}/10 — flagged section: "
                        f"{section.get('reason', '')}"
                    ),
                    "excerpt": str(section.get("excerpt", ""))[:120],
                })
        if not violations:
            violations.append({
                "severity": "info",
                "check": "tone_score",
                "detail": f"Tone-match score {score}/10 is below threshold (6)",
                "excerpt": "",
            })
    return violations


# ---------------------------------------------------------------------------
# Founder-action annotation
# ---------------------------------------------------------------------------

async def _annotate_founder_action(
    mission_id: int | None,
    task_id: int | None,
    audience: str,
    violations: list[dict],
) -> int | None:
    """Create a founder_action review card with violations as instructions.

    Best-effort — never raises; returns action id or None.
    """
    if not mission_id:
        return None
    try:
        from src.founder_actions import create as fa_create

        blocker_count = sum(1 for v in violations if v.get("severity") == "blocker")
        warning_count = sum(1 for v in violations if v.get("severity") == "warning")
        info_count = sum(1 for v in violations if v.get("severity") == "info")

        summary_line = (
            f"brand_voice_lint({audience!r}): "
            f"{blocker_count} blocker(s), {warning_count} warning(s), {info_count} info"
        )

        instructions = []
        for v in violations[:20]:
            sev = v.get("severity", "info").upper()
            check = v.get("check", "")
            detail = v.get("detail", "")
            excerpt = v.get("excerpt", "")
            line = f"[{sev}] {check}: {detail}"
            if excerpt:
                line += f" | excerpt: {excerpt!r}"
            instructions.append(line)

        action = await fa_create(
            mission_id=mission_id,
            kind="generic",
            title=summary_line,
            why=(
                f"Brand-voice lint found {blocker_count + warning_count} issue(s) "
                f"in '{audience}' voice copy. Blocker violations must be resolved "
                "before this step advances."
            ),
            instructions=instructions,
            blocking_task_id=task_id,
            notify_telegram=False,  # posthook surfacing is via the normal verdict path
            urgent=(blocker_count > 0),
        )
        return int(getattr(action, "id", 0) or 0) or None
    except Exception as exc:
        logger.debug(
            "brand_voice_lint: founder_action create failed: %s", exc,
            task_id=task_id,
        )
        return None


# ---------------------------------------------------------------------------
# Public handler
# ---------------------------------------------------------------------------

async def handle(task: dict, result: dict) -> dict:
    """Brand-voice lint posthook.

    Reads ``audience`` from ``task["context"]["brand_voice_audience"]`` (or
    ``task["context"]["audience"]``). Loads the corresponding voice doc.
    Runs all lint checks. If blocker violations found: creates a founder_action
    card and returns ``status="failed"``.
    """
    t0 = time.monotonic()
    task_id: int | None = task.get("id")
    mission_id: int | None = task.get("mission_id")

    # ── Parse task context ────────────────────────────────────────────────
    ctx_raw = task.get("context", "{}")
    if isinstance(ctx_raw, str):
        try:
            ctx: dict = json.loads(ctx_raw)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    elif isinstance(ctx_raw, dict):
        ctx = ctx_raw
    else:
        ctx = {}

    audience: str = (
        str(ctx.get("brand_voice_audience") or ctx.get("audience") or "").strip()
    )

    # ── Graceful skip: no audience metadata ───────────────────────────────
    if not audience:
        logger.info(
            "brand_voice_lint: no audience metadata — skip",
            task_id=task_id, mission_id=mission_id,
        )
        return {"status": "skip", "reason": "no audience metadata on task"}

    # ── Load brand voice doc ──────────────────────────────────────────────
    from src.ops.brand_voice import load_brand_voice
    voice = load_brand_voice(audience)

    if voice is None:
        logger.info(
            "brand_voice_lint: no voice doc for audience %r — skip",
            audience, task_id=task_id,
        )
        return {
            "status": "skip",
            "reason": f"no brand-voice doc found for audience={audience!r}",
        }

    # ── Extract artifact text to lint ─────────────────────────────────────
    # Prefer result["result"]; fall back to task result field.
    text_raw = result.get("result") or result.get("text") or ""
    if not isinstance(text_raw, str):
        text_raw = str(text_raw)
    text = text_raw.strip()

    if not text:
        logger.info(
            "brand_voice_lint: empty artifact text — skip",
            task_id=task_id, audience=audience,
        )
        return {"status": "skip", "reason": "artifact text is empty"}

    # ── Compute statistics once ───────────────────────────────────────────
    stats = _text_stats(text)

    # ── Run lint checks ───────────────────────────────────────────────────
    all_violations: list[dict] = []

    # 1. Prohibited terms (blocker on hit)
    all_violations.extend(
        _check_prohibited_terms(text, voice.prohibited_terms)
    )

    # 2. Tone signal presence (mechanical pass — required=blocker, forbidden=blocker)
    all_violations.extend(
        _check_tone_signals_mechanical(
            text, voice.tone_required_signals, voice.tone_forbidden_signals
        )
    )

    # 3. Avg sentence length (warning if outside ±25%)
    all_violations.extend(
        _check_sentence_length(stats, voice.target_avg_sentence_length_words)
    )

    # 4. Flesch-Kincaid reading level (warning if above max)
    all_violations.extend(
        _check_flesch_kincaid(stats, voice.flesch_kincaid_reading_level_max)
    )

    # 5. we/you pronoun ratio (warning if above ceiling)
    all_violations.extend(
        _check_pronoun_ratio(stats, voice.we_ratio_max)
    )

    # 6. LLM tone pass (info, best-effort)
    tone_violations = await _run_llm_tone_pass(
        text=text,
        voice_display_name=voice.display_name or audience,
        voice_body_md=voice.raw_body_md,
        source_task_id=task_id,
    )
    all_violations.extend(tone_violations)

    # ── Determine outcome ─────────────────────────────────────────────────
    blocker_violations = [v for v in all_violations if v.get("severity") == "blocker"]
    has_blockers = bool(blocker_violations)

    logger.info(
        "brand_voice_lint complete",
        task_id=task_id,
        audience=audience,
        violations_total=len(all_violations),
        blockers=len(blocker_violations),
        duration_s=round(time.monotonic() - t0, 3),
    )

    if not has_blockers:
        return {
            "status": "ok",
            "violations": all_violations,
            "stats": stats,
            "audience": audience,
        }

    # ── Annotate founder_action review card ───────────────────────────────
    fa_id = await _annotate_founder_action(
        mission_id=mission_id,
        task_id=task_id,
        audience=audience,
        violations=all_violations,
    )

    reason = (
        f"brand_voice_lint({audience!r}): {len(blocker_violations)} blocker violation(s). "
        + "; ".join(
            v.get("detail", "")[:80]
            for v in blocker_violations[:3]
        )
    )[:500]

    return {
        "status": "failed",
        "reason": reason,
        "violations": all_violations,
        "stats": stats,
        "audience": audience,
        "founder_action_id": fa_id,
    }
