# Memory Cleanup & Piggybacked Learning

Date: 2026-04-08
Status: Design
Scope: Phases 2-4 of the memory roadmap (see `docs/issues/memory-redesign-context.md` section 4)

## Problem

Phase 1 fixed prompt bloat (context gating, unified grading, threshold tuning). But the **quality of stored knowledge** is still degraded by four noise sources — fake HyDE, dead summaries, keyword preferences, fake insights. These pollute the vector space, making every future retrieval slightly worse.

Meanwhile, the grading prompt already reads full task+response. Two useful signals — user preferences and reusable insights — could be extracted from the same LLM call at zero additional cost using the progressive parsing infrastructure built in Phase 1.

## Goals

1. Remove all four noise sources from the vector space
2. Replace preferences and insights with piggybacked extraction from the grading prompt
3. Wire conversation summaries into the convo layer (they're stored but never read)
4. Enable the cross-encoder reranker now that stored data is cleaner
5. Fix `_parse_text_field` regex for multiline field support
6. Add test coverage for `apply_grade_result` PASS/FAIL paths

## Non-Goals

- Temporal validity (#9) — separate effort with contradiction detection
- Bilingual query expansion (#10) — independent, lower priority
- Precomputed essentials (#11) — independent feature
- Weight validation (#12) — needs instrumentation + production data

---

## Part 1: Remove Fake HyDE

**File:** `src/memory/rag.py`

Delete `_hyde_expand()` (lines 274-301) and its call in `retrieve_context()` (line 444-446). Remove `HYDE_ENABLED` config flag (line 80).

Raw query embedding is strictly better. The hardcoded template `"The task was completed successfully..."` biases every query toward generic boilerplate language, degrading retrieval precision.

Real HyDE requires an LLM call before retrieval — not viable on local hardware. If cloud models are connected later, real HyDE can be added as a separate feature.

---

## Part 2: Wire Conversation Summaries into RAG

**Files:** `src/memory/conversations.py`, `src/memory/rag.py`, `src/agents/base.py`

Currently `maybe_summarize()` stores summaries in the `semantic` collection with `type: conversation_summary`, but nothing reads them. The `"convo"` context layer in `_build_context()` only uses raw `recent_conversation` from task_context.

### Design

**Keep** `maybe_summarize()` — it works correctly. **Change** where summaries are stored: from `semantic` to `conversations` collection, so they're colocated with the exchanges they summarize.

**Update the `"convo"` layer** in `_build_context()` to use a two-tier approach:
1. **Last 1-2 raw exchanges** from `task_context["recent_conversation"]` — for immediate follow-ups
2. **Relevant summaries** queried from the `conversations` collection filtered by `type: conversation_summary` — for longer-term conversation memory

The convo layer's token budget (from context_policy) is split: ~30% for raw recent, ~70% for summaries. If no summaries match, raw exchanges get the full budget.

**Add `conversations` to RAG collection gating** for agent types that use `"convo"` policy:
- `assistant` already has `conversations` in RAG_COLLECTIONS — good
- `researcher` has `"convo"` in policy but no `conversations` in RAG_COLLECTIONS — add it

### Changes

| Location | Change |
|----------|--------|
| `conversations.py:276` | Change collection from `"semantic"` to `"conversations"` |
| `conversations.py:279` | Add `"type": "conversation_summary"` already present — keep |
| `base.py:_format_conversation()` | Query summaries from conversations collection, merge with raw exchanges |
| `rag.py:RAG_COLLECTIONS` | Add `"conversations"` to `researcher` entry |

---

## Part 3: Replace Keyword Preferences with Piggybacked Grading

**Files:** `src/memory/preferences.py`, `src/core/grading.py`

### What gets removed

- `detect_preferences()` (lines 212-278) — never called, dead code
- `_extract_patterns()` (lines 281-361) — keyword matching with ~50% false positives
- The dead `"prefs"` injection block in `base.py` (guard `if "prefs" in policy:` is unreachable — no policy includes `"prefs"`)

### What gets added

**New grading prompt field:**
```
PREFERENCE: one-line user preference signal observed in this task, or NONE
```

Added after TOOLS in `GRADING_PROMPT`. The progressive parsing cascade already handles missing fields gracefully — if the LLM can't produce it, preference stays empty.

**New `GradeResult` field:**
```python
preference: str = ""  # e.g. "User prefers Turkish responses"
```

**New preference storage in `apply_grade_result()`:**

On PASS, if `verdict.preference` is non-empty and not "NONE"/"none":
```python
from src.memory.preferences import store_preference
await store_preference(
    preference=verdict.preference,
    category="grader_observed",
    chat_id=ctx.get("chat_id", "default"),
    confidence=0.8,
)
```

This reuses the existing `store_preference()` function — it handles deduplication via deterministic doc_id hashing.

### What stays

- `record_feedback()` — useful for explicit user feedback signals
- `store_preference()` — now called from grading instead of keyword detection
- `get_user_preferences()` — reads from vector store, works regardless of how preferences were stored
- `format_preferences()` — formatting for prompt injection

### Preference injection

Currently no policy includes `"prefs"`. After piggybacked extraction produces real preferences, add `"prefs"` to policies where it matters:
- `assistant` — personalizing responses
- `writer` — matching user's communication style

Don't add to task-focused profiles (coder, fixer, executor) — preferences for those come through skills.

---

## Part 4: Replace Fake Insights with Piggybacked Grading

**Files:** `src/memory/episodic.py`, `src/core/grading.py`, `src/core/orchestrator.py`

### What gets removed

- `extract_and_store_insight()` (episodic.py:296-349) — just reformats `f"Insight from {agent_type}: Task '{title}' — {result_preview}"`, not real extraction
- Its call from `store_task_result()` (episodic.py, line 93 area)

### What gets added

**New grading prompt field:**
```
INSIGHT: one-line reusable learning from this task, or NONE
```

Added after PREFERENCE in `GRADING_PROMPT`.

**New `GradeResult` field:**
```python
insight: str = ""  # e.g. "Turkish e-commerce sites require User-Agent header"
```

**New insight storage in `apply_grade_result()`:**

On PASS, if `verdict.insight` is non-empty and not "NONE"/"none":
```python
from src.memory.episodic import store_insight
await store_insight(
    insight_text=verdict.insight,
    agent_type=task.get("agent_type", "executor"),
    task_id=task_id,
    task_title=task.get("title", ""),
)
```

**New `store_insight()` function** in episodic.py — a clean replacement:
```python
async def store_insight(
    insight_text: str,
    agent_type: str,
    task_id: int,
    task_title: str = "",
) -> str | None:
    """Store a grader-extracted insight in the semantic collection."""
    if not is_ready() or not insight_text:
        return None
    
    metadata = {
        "type": "cross_agent_insight",
        "agent_type": agent_type,
        "task_title": task_title[:200],
        "source": "grader_extraction",
        "importance": 7,  # higher than the old 6 — these are real insights
        "timestamp": time.time(),
    }
    doc_id = f"insight-{task_id}-{int(time.time())}"
    return await embed_and_store(
        text=insight_text,
        metadata=metadata,
        collection="semantic",
        doc_id=doc_id,
    )
```

Key difference from old `extract_and_store_insight`: the `text` field is the actual insight from the LLM, not a reformatted title. This produces meaningful embeddings.

---

## Part 5: Enable Reranker

**File:** `src/memory/rag.py`

Change `RERANKER_ENABLED = False` to `RERANKER_ENABLED = True`.

**Selective triggering:** The existing code at line 358 already short-circuits when `not results`. Add a minimum threshold — only rerank when 3+ results exist (reranking 1-2 results has no value):

```python
if not RERANKER_ENABLED or len(results) < 3:
    return results
```

**Lazy loading:** The existing implementation already lazy-loads the CrossEncoder on first use and caches it. If loading fails (dependency missing), it silently returns unranked results. No changes needed to the loading logic.

**Performance:** `ms-marco-MiniLM-L-6-v2` runs on CPU. Expected ~200ms for 5 results. No VRAM cost. `sentence-transformers` is already a dependency (used for embeddings).

---

## Part 6: Fix _parse_text_field for Multiline

**File:** `src/core/grading.py`

Current regex: `rf"{key}\s*:\s*(.+)"` — only captures content on the same line.

New regex — capture until next `KEY:` pattern or end of string:
```python
def _parse_text_field(text: str, key: str) -> str:
    pattern = rf"{key}\s*:\s*(.+?)(?=\n[A-Z]+\s*:|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()
```

This handles cases where small LLMs wrap long TOOLS lists across lines:
```
TOOLS: smart_search, web_search,
  api_call, scraper
```

The `(?=\n[A-Z]+\s*:|$)` lookahead stops at the next field marker or end of string.

---

## Part 7: Test Coverage for apply_grade_result

**File:** `tests/test_grading.py`

Add tests for:

1. **PASS path with rich verdict** — skill extraction uses `verdict.situation`/`strategy`/`tools`, preference stored, insight stored
2. **PASS path with empty verdict** — mechanical skill fallback, no preference/insight stored
3. **PASS with iterations < 2** — no skill extraction (gate check)
4. **FAIL path with retries remaining** — retry scheduling, model exclusion
5. **FAIL path terminal** — DLQ quarantine, notification
6. **grade_task with empty result** — auto-fail at <10 chars

Mocking strategy: patch `transition_task`, `get_task`, `record_model_call`, `add_skill`, `store_preference`, `store_insight`, `quarantine_task`, `compute_retry_timing`, `update_exclusions_on_failure`.

---

## Updated Grading Prompt

After all changes, the full prompt becomes:

```
Evaluate this task result.

Task: {title}
Description: {description}
Result: {response}

RELEVANT: YES or NO
COMPLETE: YES or NO
VERDICT: PASS or FAIL
SITUATION: one line, what type of problem was solved
STRATEGY: one line, what approach worked
TOOLS: comma-separated list of tools used effectively
PREFERENCE: one-line user preference signal observed in this task, or NONE
INSIGHT: one-line reusable learning from this task, or NONE
```

This adds ~30 tokens to the prompt. The parsing cascade handles each field independently — any subset can succeed or fail without affecting the others.

---

## Changes Summary

| File | Change |
|------|--------|
| `src/memory/rag.py` | Remove `_hyde_expand()`, `HYDE_ENABLED`. Enable reranker with 3+ gate. |
| `src/memory/conversations.py` | Change summary storage from `semantic` to `conversations` collection |
| `src/memory/preferences.py` | Remove `detect_preferences()`, `_extract_patterns()` |
| `src/memory/episodic.py` | Remove `extract_and_store_insight()`. Add `store_insight()`. |
| `src/core/grading.py` | Add PREFERENCE/INSIGHT to prompt, GradeResult, parsing. Fix multiline regex. |
| `src/agents/base.py` | Update `_format_conversation()` to query summaries. Remove dead prefs block. Add `"prefs"` to assistant/writer policies via context_policy. |
| `src/memory/context_policy.py` | Add `"prefs"` to `assistant` and `writer` policies |
| `src/core/orchestrator.py` | Remove call to `extract_and_store_insight()` if still present |
| `tests/test_grading.py` | Add apply_grade_result PASS/FAIL tests, multiline regex tests, new fields |
| `tests/test_rag_thresholds.py` | Update for removed HyDE, enabled reranker |
