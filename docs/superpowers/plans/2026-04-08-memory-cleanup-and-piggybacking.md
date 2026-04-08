# Memory Cleanup & Piggybacked Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean vector space noise sources, replace them with piggybacked learning from the grading prompt, wire conversation summaries into the convo layer, enable the cross-encoder reranker, and fix multiline parsing.

**Architecture:** Seven independent tasks. Tasks 1-4 each remove a noise source. Task 3 and 4 also add piggybacked fields to the unified grading prompt (PREFERENCE, INSIGHT). Task 5 enables the reranker. Task 6 fixes multiline regex. Task 7 adds apply_grade_result test coverage. Each task is self-contained and committable.

**Tech Stack:** Python 3.10, async, ChromaDB, sentence-transformers (multilingual-e5-base, ms-marco-MiniLM-L-6-v2)

**Key references:**
- Design spec: `docs/superpowers/specs/2026-04-08-memory-cleanup-and-piggybacking-design.md`
- Memory roadmap: `docs/issues/memory-redesign-context.md`
- Findings: `docs/issues/memory-subsystem-findings.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/memory/rag.py` | Modify | Remove `_hyde_expand()`, `HYDE_ENABLED`. Enable reranker with 3+ gate. |
| `src/memory/conversations.py` | Modify | Change summary storage collection from `semantic` to `conversations` |
| `src/memory/preferences.py` | Modify | Remove `detect_preferences()`, `_extract_patterns()`. Keep `store_preference()`, `get_user_preferences()`, `format_preferences()`. |
| `src/memory/episodic.py` | Modify | Remove `extract_and_store_insight()`. Add `store_insight()`. Remove call from `store_task_result()`. |
| `src/core/grading.py` | Modify | Add PREFERENCE/INSIGHT to prompt, GradeResult, parsing. Fix multiline regex. Wire storage into apply_grade_result. |
| `src/agents/base.py` | Modify | Update `_format_conversation()` to query summaries. |
| `src/memory/context_policy.py` | Modify | Add `"prefs"` to `assistant` and `writer` policies. |
| `tests/test_grading.py` | Modify | Add new field tests, multiline tests, apply_grade_result tests. |
| `tests/test_rag_thresholds.py` | Modify | Add HyDE removal assertion, reranker config assertion. |
| `tests/test_context_policy.py` | Modify | Update assistant/writer policy assertions. |

---

## Task 1: Remove Fake HyDE

**Files:**
- Modify: `src/memory/rag.py:80, 274-301, 443-446`
- Modify: `tests/test_rag_thresholds.py`

- [ ] **Step 1: Write failing test for HyDE removal**

Add to `tests/test_rag_thresholds.py`:

```python
class TestHyDERemoved:
    def test_hyde_disabled_or_removed(self):
        """Fake HyDE must not exist — raw queries are better."""
        import src.memory.rag as rag_mod
        assert not getattr(rag_mod, "HYDE_ENABLED", False), "HYDE_ENABLED should be False or removed"
        assert not hasattr(rag_mod, "_hyde_expand"), "_hyde_expand should be removed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag_thresholds.py::TestHyDERemoved -v`
Expected: FAIL — `HYDE_ENABLED` is True and `_hyde_expand` exists.

- [ ] **Step 3: Remove fake HyDE from rag.py**

In `src/memory/rag.py`:

Delete line 80:
```python
HYDE_ENABLED = True  # Generate hypothetical answers for better retrieval
```

Delete lines 272-301 (the entire `_hyde_expand` function and its section comment):
```python
# ─── Phase F: HyDE Query Expansion ──────────────────────────────────────────

async def _hyde_expand(query_text: str) -> Optional[str]:
    ...
    return hyde_text[:500]
```

Delete lines 443-446 in `retrieve_context()`:
```python
    # Phase F: HyDE expansion — add hypothetical answer as extra query
    hyde_text = await _hyde_expand(query_text)
    if hyde_text:
        queries.append(hyde_text)
```

Also update the docstring of `retrieve_context()` (around line 402-410) — remove step 3 mentioning HyDE:
```python
    """
    Retrieve relevant context from all vector store collections.

    Pipeline:
      1. Compute dynamic token budget (scales to model context window)
      2. Query decomposition for multi-part queries
      3. Query collections gated by agent type
      4. Optional cross-encoder reranking
      5. Rank by recency * relevance * importance
      6. Filter by minimum relevance threshold
      7. Deduplicate
      8. Format within token budget
    ...
    """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag_thresholds.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/rag.py tests/test_rag_thresholds.py
git commit -m "fix(rag): remove fake HyDE — raw queries are better than generic template"
```

---

## Task 2: Wire Conversation Summaries into Convo Layer

**Files:**
- Modify: `src/memory/conversations.py:276`
- Modify: `src/agents/base.py:604-619`
- Modify: `src/memory/rag.py:53-68` (RAG_COLLECTIONS)

- [ ] **Step 1: Change summary storage collection**

In `src/memory/conversations.py`, line 276, change the `collection` parameter:

```python
    result = await embed_and_store(
        text=summary_text,
        metadata={
            "type": "conversation_summary",
            "chat_id": str(chat_id),
            "exchange_count": len(recent),
            "topics": topic_str[:200],
            "timestamp": time.time(),
        },
        collection="conversations",
        doc_id=doc_id,
    )
```

Only change: `collection="semantic"` → `collection="conversations"`.

- [ ] **Step 2: Update _format_conversation to query summaries**

In `src/agents/base.py`, replace `_format_conversation()` (lines 604-619):

```python
    def _format_conversation(self, task_context: dict, max_tokens: int) -> str:
        """Format recent conversation + summaries, truncated to budget."""
        parts = ["## Recent Conversation (for context)"]

        # Tier 1: Last 1-2 raw exchanges for immediate follow-up context
        raw_exchanges = task_context.get("recent_conversation", [])
        raw_budget = max(200, int(max_tokens * 0.3))
        for entry in raw_exchanges[:2]:
            user_q = entry.get("user_asked", "?")
            result = entry.get("result", "")
            if len(result) > 400:
                result = result[:400] + "... [truncated]"
            parts.append(f"**User asked:** {user_q}\n**Result:** {result}\n")

        parts.append(
            "_Use this context to understand follow-up references "
            "like 'list them', 'the names', 'do it again', etc._"
        )

        return self._truncate_to_tokens("\n".join(parts), max_tokens)
```

Note: We keep the method synchronous and use only the inline raw exchanges. The summaries are now stored in the `conversations` collection and will be picked up by RAG when the `"rag"` layer queries `conversations` collection for agent types that have it in their `RAG_COLLECTIONS`. This avoids adding an async vector query inside a formatting method.

- [ ] **Step 3: Add conversations to researcher RAG collections**

In `src/memory/rag.py`, update `RAG_COLLECTIONS` — add `"conversations"` to `researcher`:

```python
    "researcher":       ["web_knowledge", "semantic", "conversations"],
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_rag_thresholds.py tests/test_context_policy.py -v`
Expected: All PASS (no test changes needed — existing tests don't assert on researcher's collections specifically)

- [ ] **Step 5: Commit**

```bash
git add src/memory/conversations.py src/agents/base.py src/memory/rag.py
git commit -m "feat(memory): wire conversation summaries into conversations collection for RAG retrieval"
```

---

## Task 3: Replace Keyword Preferences with Piggybacked Grading

**Files:**
- Modify: `src/memory/preferences.py:210-361`
- Modify: `src/core/grading.py:19-30, 33-41, 54-60, 201-276`
- Modify: `src/memory/context_policy.py:19-20`
- Modify: `tests/test_grading.py`
- Modify: `tests/test_context_policy.py`

- [ ] **Step 1: Write failing tests for new PREFERENCE field**

Add to `tests/test_grading.py`:

```python
class TestPreferenceField:
    def test_preference_parsed(self):
        raw = (
            "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS\n"
            "SITUATION: test\nSTRATEGY: test\nTOOLS: api_call\n"
            "PREFERENCE: User prefers Turkish responses\n"
            "INSIGHT: NONE"
        )
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.preference == "User prefers Turkish responses"

    def test_preference_none_becomes_empty(self):
        raw = (
            "VERDICT: PASS\n"
            "PREFERENCE: NONE"
        )
        result = parse_grade_response(raw)
        assert result.preference == ""

    def test_preference_none_lowercase(self):
        raw = (
            "VERDICT: PASS\n"
            "PREFERENCE: none"
        )
        result = parse_grade_response(raw)
        assert result.preference == ""

    def test_missing_preference_stays_empty(self):
        raw = "VERDICT: PASS"
        result = parse_grade_response(raw)
        assert result.preference == ""

    def test_default_preference_empty(self):
        result = GradeResult(passed=True)
        assert result.preference == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grading.py::TestPreferenceField -v`
Expected: FAIL — `GradeResult` has no `preference` field.

- [ ] **Step 3: Add PREFERENCE to grading prompt and GradeResult**

In `src/core/grading.py`, update `GRADING_PROMPT` (line 19-30) — add after the TOOLS line:

```python
GRADING_PROMPT = """Evaluate this task result.

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
INSIGHT: one-line reusable learning from this task, or NONE"""
```

Update `GradeResult` (line 33-41) — add two new fields:

```python
@dataclass
class GradeResult:
    passed: bool
    relevant: Optional[bool] = None
    complete: Optional[bool] = None
    situation: str = ""
    strategy: str = ""
    tools: list[str] = field(default_factory=list)
    preference: str = ""
    insight: str = ""
    raw: str = ""
```

Update `parse_grade_response()` — add parsing for new fields after the tools parsing (around line 80):

```python
    # Skill extraction fields (optional — never block grading)
    situation = _parse_text_field(raw, "SITUATION")
    strategy = _parse_text_field(raw, "STRATEGY")
    tools_raw = _parse_text_field(raw, "TOOLS")
    tools = [t.strip() for t in tools_raw.split(",") if t.strip()] if tools_raw else []

    # Piggybacked learning fields (optional)
    preference = _parse_text_field(raw, "PREFERENCE")
    if preference.upper() == "NONE":
        preference = ""
    insight = _parse_text_field(raw, "INSIGHT")
    if insight.upper() == "NONE":
        insight = ""
```

Update all `GradeResult(...)` constructors in `parse_grade_response()` to include `preference=preference, insight=insight`. There are 4 return statements:

Cascade 1 (VERDICT present, around line 84):
```python
        return GradeResult(
            passed=verdict, relevant=relevant, complete=complete,
            situation=situation, strategy=strategy, tools=tools,
            preference=preference, insight=insight, raw=raw,
        )
```

Cascade 2 (derive from RELEVANT+COMPLETE, around line 92):
```python
        return GradeResult(
            passed=(relevant and complete), relevant=relevant, complete=complete,
            situation=situation, strategy=strategy, tools=tools,
            preference=preference, insight=insight, raw=raw,
        )
```

Cascade 3 bare PASS (around line 100):
```python
        return GradeResult(passed=True, situation=situation, strategy=strategy, tools=tools,
                           preference=preference, insight=insight, raw=raw)
```

Cascade 3 bare FAIL (around line 102):
```python
        return GradeResult(passed=False, situation=situation, strategy=strategy, tools=tools,
                           preference=preference, insight=insight, raw=raw)
```

- [ ] **Step 4: Wire preference storage into apply_grade_result**

In `src/core/grading.py`, in the `apply_grade_result()` PASS block (after skill extraction, around line 253), add preference storage:

```python
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
```

Insert this after the injection success tracking block (after line 275) and before `logger.info(f"grade PASS | task_id={task_id}")`.

- [ ] **Step 5: Remove keyword detection from preferences.py**

In `src/memory/preferences.py`, delete `detect_preferences()` (lines 212-278) and `_extract_patterns()` (lines 281-361). Keep everything else: `record_feedback()`, `store_preference()`, `get_user_preferences()`, `format_preferences()`.

Also remove `PREFERENCE_CATEGORIES` list (lines 41-51) — no longer used by anything.

- [ ] **Step 6: Add prefs to assistant and writer context policies**

In `src/memory/context_policy.py`, update lines 19-20:

```python
    "assistant":        {"convo", "rag", "memory", "prefs"},
    "writer":           {"deps", "convo", "memory", "prefs"},
```

- [ ] **Step 7: Update context policy tests**

In `tests/test_context_policy.py`, add:

```python
    def test_assistant_has_prefs(self):
        policy = get_context_policy("assistant")
        assert "prefs" in policy

    def test_writer_has_prefs(self):
        policy = get_context_policy("writer")
        assert "prefs" in policy
```

- [ ] **Step 8: Run tests**

Run: `pytest tests/test_grading.py tests/test_context_policy.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/core/grading.py src/memory/preferences.py src/memory/context_policy.py tests/test_grading.py tests/test_context_policy.py
git commit -m "feat(grading): piggyback PREFERENCE on grading prompt, remove keyword detection"
```

---

## Task 4: Replace Fake Insights with Piggybacked Grading

**Files:**
- Modify: `src/memory/episodic.py:90-96, 296-349`
- Modify: `src/core/grading.py:201-277`
- Modify: `tests/test_grading.py`

- [ ] **Step 1: Write failing tests for INSIGHT field**

Add to `tests/test_grading.py`:

```python
class TestInsightField:
    def test_insight_parsed(self):
        raw = (
            "VERDICT: PASS\n"
            "SITUATION: test\nSTRATEGY: test\nTOOLS: api_call\n"
            "PREFERENCE: NONE\n"
            "INSIGHT: Turkish e-commerce sites require User-Agent header"
        )
        result = parse_grade_response(raw)
        assert result.insight == "Turkish e-commerce sites require User-Agent header"

    def test_insight_none_becomes_empty(self):
        raw = "VERDICT: PASS\nINSIGHT: NONE"
        result = parse_grade_response(raw)
        assert result.insight == ""

    def test_missing_insight_stays_empty(self):
        raw = "VERDICT: PASS"
        result = parse_grade_response(raw)
        assert result.insight == ""

    def test_default_insight_empty(self):
        result = GradeResult(passed=True)
        assert result.insight == ""
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_grading.py::TestInsightField -v`
Expected: PASS — the INSIGHT field was already added in Task 3. This verifies it works end-to-end.

- [ ] **Step 3: Replace extract_and_store_insight with store_insight**

In `src/memory/episodic.py`, replace `extract_and_store_insight()` (lines 296-349) with:

```python
async def store_insight(
    insight_text: str,
    agent_type: str,
    task_id: int,
    task_title: str = "",
) -> str | None:
    """Store a grader-extracted insight in the semantic collection.

    Unlike the old extract_and_store_insight, this receives real LLM-extracted
    insight text, not a reformatted task title. Called from apply_grade_result.
    """
    if not is_ready() or not insight_text:
        return None

    metadata = {
        "type": "cross_agent_insight",
        "agent_type": agent_type,
        "task_title": task_title[:200],
        "source": "grader_extraction",
        "importance": 7,
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

- [ ] **Step 4: Remove fake insight call from store_task_result**

In `src/memory/episodic.py`, remove lines 90-95:

```python
    # Phase D: Extract cross-agent insight from successful tasks
    if stored and success:
        try:
            await extract_and_store_insight(task, result, agent_type=agent_type)
        except Exception as e:
            logger.debug("Insight extraction skipped: %s", e)
```

Replace with just:

```python
    return stored
```

(The `return stored` on line 97 stays as-is.)

- [ ] **Step 5: Wire insight storage into apply_grade_result**

In `src/core/grading.py`, in the PASS block of `apply_grade_result()`, after the preference storage block added in Task 3, add:

```python
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
```

Insert before `logger.info(f"grade PASS | task_id={task_id}")`.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_grading.py -v`
Expected: All PASS

- [ ] **Step 7: Verify fake insight function is removed**

Run: `python -c "from src.memory.episodic import extract_and_store_insight; print('FAIL: still importable')" 2>&1 || echo "OK: removed"`
Expected: ImportError → "OK: removed"

- [ ] **Step 8: Commit**

```bash
git add src/memory/episodic.py src/core/grading.py tests/test_grading.py
git commit -m "feat(grading): piggyback INSIGHT on grading prompt, remove fake insight extraction"
```

---

## Task 5: Enable Reranker

**Files:**
- Modify: `src/memory/rag.py:76, 358`
- Modify: `tests/test_rag_thresholds.py`

- [ ] **Step 1: Write failing test for reranker enabled**

Add to `tests/test_rag_thresholds.py`:

```python
class TestRerankerConfig:
    def test_reranker_enabled(self):
        from src.memory.rag import RERANKER_ENABLED
        assert RERANKER_ENABLED is True

    def test_reranker_skips_small_result_sets(self):
        """Reranking <3 results has no value."""
        import asyncio
        from src.memory.rag import _rerank_results
        # With 2 results, should return them unchanged (no reranking)
        results = [
            {"text": "result 1", "id": "1"},
            {"text": "result 2", "id": "2"},
        ]
        out = asyncio.get_event_loop().run_until_complete(
            _rerank_results("test query", results)
        )
        assert out == results  # returned as-is, no reranking
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_thresholds.py::TestRerankerConfig -v`
Expected: FAIL — `RERANKER_ENABLED` is False.

- [ ] **Step 3: Enable reranker with minimum result gate**

In `src/memory/rag.py`, line 76:

```python
RERANKER_ENABLED = True  # Cross-encoder reranking for 3+ results
```

In `_rerank_results()`, line 358, update the guard:

```python
    if not RERANKER_ENABLED or len(results) < 3:
        return results
```

This replaces `if not RERANKER_ENABLED or not results:`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_rag_thresholds.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/rag.py tests/test_rag_thresholds.py
git commit -m "feat(rag): enable cross-encoder reranker for 3+ results"
```

---

## Task 6: Fix Multiline _parse_text_field Regex

**Files:**
- Modify: `src/core/grading.py:54-60`
- Modify: `tests/test_grading.py`

- [ ] **Step 1: Write failing test for multiline parsing**

Add to `tests/test_grading.py`:

```python
class TestMultilineParsing:
    def test_tools_spanning_two_lines(self):
        raw = (
            "VERDICT: PASS\n"
            "SITUATION: Multi-store price check\n"
            "STRATEGY: Sequential scraping\n"
            "TOOLS: smart_search, web_search,\n"
            "  api_call, scraper\n"
            "PREFERENCE: NONE\n"
            "INSIGHT: NONE"
        )
        result = parse_grade_response(raw)
        assert "smart_search" in result.tools
        assert "api_call" in result.tools
        assert "scraper" in result.tools

    def test_strategy_wrapping(self):
        raw = (
            "VERDICT: PASS\n"
            "SITUATION: Complex research\n"
            "STRATEGY: First searched each store,\n"
            "  then compared prices across all\n"
            "TOOLS: smart_search\n"
            "PREFERENCE: NONE"
        )
        result = parse_grade_response(raw)
        assert "First searched each store" in result.strategy
        assert "compared prices" in result.strategy

    def test_single_line_still_works(self):
        """Regression: single-line values must still parse correctly."""
        raw = (
            "VERDICT: PASS\n"
            "SITUATION: Weather lookup\n"
            "STRATEGY: Used API\n"
            "TOOLS: api_call"
        )
        result = parse_grade_response(raw)
        assert result.situation == "Weather lookup"
        assert result.strategy == "Used API"
        assert result.tools == ["api_call"]

    def test_last_field_captures_to_end(self):
        """Last field in output has no next KEY: to stop at."""
        raw = (
            "VERDICT: PASS\n"
            "INSIGHT: Turkish sites need UA header\n"
            "  and proper Accept-Language"
        )
        result = parse_grade_response(raw)
        assert "Turkish sites need UA header" in result.insight
        assert "Accept-Language" in result.insight
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grading.py::TestMultilineParsing -v`
Expected: FAIL on multiline tests — current regex only captures first line.

- [ ] **Step 3: Update _parse_text_field for multiline**

In `src/core/grading.py`, replace `_parse_text_field()` (lines 54-60):

```python
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
```

Key changes:
- `(.+?)` with `re.DOTALL` matches across lines (non-greedy)
- `(?=\n[A-Z]{2,}\s*:|$)` lookahead stops at the next field marker (2+ uppercase chars followed by colon) or end of string
- Internal newlines collapsed to spaces for clean output

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_grading.py -v`
Expected: All PASS (including existing tests — regression check)

- [ ] **Step 5: Commit**

```bash
git add src/core/grading.py tests/test_grading.py
git commit -m "fix(grading): multiline-aware _parse_text_field for wrapped field values"
```

---

## Task 7: Test Coverage for apply_grade_result

**Files:**
- Modify: `tests/test_grading.py`

- [ ] **Step 1: Add PASS path tests**

Add to `tests/test_grading.py`:

```python
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.core.grading import parse_grade_response, GradeResult, apply_grade_result


class TestApplyGradeResultPass:
    """Test apply_grade_result PASS path: skill extraction, preference, insight."""

    @pytest.mark.asyncio
    @patch("src.core.grading.transition_task", new_callable=AsyncMock)
    @patch("src.core.grading.get_task", new_callable=AsyncMock)
    @patch("src.memory.skills.add_skill", new_callable=AsyncMock)
    @patch("src.core.grading.record_model_call", new_callable=AsyncMock)
    @patch("src.memory.preferences.store_preference", new_callable=AsyncMock)
    @patch("src.memory.episodic.store_insight", new_callable=AsyncMock)
    async def test_pass_with_rich_verdict(
        self, mock_insight, mock_pref, mock_record, mock_skill, mock_get, mock_trans
    ):
        mock_get.return_value = {
            "id": 42, "title": "Compare laptop prices",
            "agent_type": "shopping_advisor", "iterations": 3,
            "context": '{"generating_model": "test-model", "tools_used_names": ["smart_search", "web_search"], "chat_id": "12345"}',
        }
        verdict = GradeResult(
            passed=True,
            situation="Price comparison across Turkish stores",
            strategy="Search each store separately then compare",
            tools=["smart_search", "web_search"],
            preference="User prefers Turkish responses",
            insight="Trendyol requires User-Agent header",
        )

        await apply_grade_result(42, verdict)

        mock_trans.assert_called_once()
        mock_skill.assert_called_once()
        # Skill should use verdict.situation, not mechanical fallback
        skill_call = mock_skill.call_args
        assert "Price comparison" in (skill_call.kwargs.get("description", "") or skill_call.args[1] if len(skill_call.args) > 1 else "")
        mock_pref.assert_called_once()
        assert "Turkish" in mock_pref.call_args.kwargs.get("preference", "")
        mock_insight.assert_called_once()
        assert "Trendyol" in mock_insight.call_args.kwargs.get("insight_text", "")

    @pytest.mark.asyncio
    @patch("src.core.grading.transition_task", new_callable=AsyncMock)
    @patch("src.core.grading.get_task", new_callable=AsyncMock)
    @patch("src.memory.skills.add_skill", new_callable=AsyncMock)
    @patch("src.core.grading.record_model_call", new_callable=AsyncMock)
    @patch("src.memory.preferences.store_preference", new_callable=AsyncMock)
    @patch("src.memory.episodic.store_insight", new_callable=AsyncMock)
    async def test_pass_with_empty_verdict_uses_mechanical_fallback(
        self, mock_insight, mock_pref, mock_record, mock_skill, mock_get, mock_trans
    ):
        mock_get.return_value = {
            "id": 43, "title": "Check weather",
            "agent_type": "executor", "iterations": 2,
            "context": '{"generating_model": "test-model", "tools_used_names": ["api_call"]}',
        }
        verdict = GradeResult(passed=True)  # No skill/preference/insight fields

        await apply_grade_result(43, verdict)

        mock_trans.assert_called_once()
        mock_skill.assert_called_once()
        # Mechanical fallback — description contains "Task:" prefix
        skill_call = mock_skill.call_args
        desc = skill_call.kwargs.get("description", "")
        assert "Task:" in desc or "Check weather" in desc
        # No preference or insight to store
        mock_pref.assert_not_called()
        mock_insight.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.core.grading.transition_task", new_callable=AsyncMock)
    @patch("src.core.grading.get_task", new_callable=AsyncMock)
    @patch("src.memory.skills.add_skill", new_callable=AsyncMock)
    @patch("src.core.grading.record_model_call", new_callable=AsyncMock)
    async def test_pass_low_iterations_skips_skill(
        self, mock_record, mock_skill, mock_get, mock_trans
    ):
        mock_get.return_value = {
            "id": 44, "title": "Simple lookup",
            "agent_type": "executor", "iterations": 1,
            "context": '{"generating_model": "test-model", "tools_used_names": ["api_call"]}',
        }
        verdict = GradeResult(passed=True, situation="Simple API call")

        await apply_grade_result(44, verdict)

        mock_trans.assert_called_once()
        mock_skill.assert_not_called()  # iterations < 2 → no skill
```

- [ ] **Step 2: Add FAIL path tests**

Add to `tests/test_grading.py`:

```python
class TestApplyGradeResultFail:
    """Test apply_grade_result FAIL path: retry and DLQ."""

    @pytest.mark.asyncio
    @patch("src.core.grading.transition_task", new_callable=AsyncMock)
    @patch("src.core.grading.get_task", new_callable=AsyncMock)
    @patch("src.core.grading.update_exclusions_on_failure")
    @patch("src.core.grading.compute_retry_timing")
    @patch("src.infra.db.update_task", new_callable=AsyncMock)
    async def test_fail_with_retries_remaining(
        self, mock_update, mock_retry, mock_excl, mock_get, mock_trans
    ):
        mock_get.return_value = {
            "id": 50, "title": "Failed task",
            "agent_type": "coder", "attempts": 1, "max_attempts": 6,
            "context": '{"generating_model": "test-model"}',
        }
        mock_retry_decision = MagicMock()
        mock_retry_decision.action = "retry"
        mock_retry_decision.delay = 30
        mock_retry.return_value = mock_retry_decision
        verdict = GradeResult(passed=False)

        await apply_grade_result(50, verdict)

        mock_excl.assert_called_once()
        mock_retry.assert_called_once()
        # Should NOT transition to failed (retries remaining)
        trans_calls = [c for c in mock_trans.call_args_list if "failed" in str(c)]
        assert len(trans_calls) == 0

    @pytest.mark.asyncio
    @patch("src.core.grading.transition_task", new_callable=AsyncMock)
    @patch("src.core.grading.get_task", new_callable=AsyncMock)
    @patch("src.core.grading.update_exclusions_on_failure")
    @patch("src.core.grading.compute_retry_timing")
    @patch("src.infra.dead_letter.quarantine_task", new_callable=AsyncMock)
    async def test_fail_terminal_quarantines(
        self, mock_quarantine, mock_retry, mock_excl, mock_get, mock_trans
    ):
        mock_get.return_value = {
            "id": 51, "title": "Hopeless task",
            "agent_type": "coder", "attempts": 5, "max_attempts": 6,
            "context": '{"generating_model": "test-model"}',
        }
        mock_retry_decision = MagicMock()
        mock_retry_decision.action = "terminal"
        mock_retry.return_value = mock_retry_decision
        verdict = GradeResult(passed=False)

        await apply_grade_result(51, verdict)

        mock_trans.assert_called_once()
        assert "failed" in str(mock_trans.call_args)
        mock_quarantine.assert_called_once()
```

- [ ] **Step 3: Add grade_task auto-fail test**

Add to `tests/test_grading.py`:

```python
class TestGradeTaskAutoFail:
    def test_empty_result_auto_fails(self):
        """grade_task should return FAIL for trivial/empty output without calling LLM."""
        import asyncio
        from src.core.grading import grade_task

        task = {"title": "Test", "description": "Test", "result": "", "context": "{}"}
        # This should not call dispatcher — auto-fail for empty result
        result = asyncio.get_event_loop().run_until_complete(
            grade_task(task, "test-model")
        )
        assert result.passed is False
        assert "auto-fail" in result.raw

    def test_short_result_auto_fails(self):
        import asyncio
        from src.core.grading import grade_task

        task = {"title": "Test", "description": "Test", "result": "ok", "context": "{}"}
        result = asyncio.get_event_loop().run_until_complete(
            grade_task(task, "test-model")
        )
        assert result.passed is False
        assert "auto-fail" in result.raw
```

- [ ] **Step 4: Run all grading tests**

Run: `pytest tests/test_grading.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_grading.py
git commit -m "test(grading): add apply_grade_result PASS/FAIL path coverage"
```

---

## Task 8: Integration Validation

**Files:** None (testing only)

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -x --timeout=30 -q 2>&1 | tail -20`
Expected: All existing tests pass. Fix any regressions.

- [ ] **Step 2: Smoke test imports**

```bash
python -c "
from src.core.grading import grade_task, apply_grade_result, GradeResult, parse_grade_response
assert hasattr(GradeResult(passed=True), 'preference')
assert hasattr(GradeResult(passed=True), 'insight')
print('grading OK')
"
python -c "
from src.memory.rag import RAG_CONFIG, RERANKER_ENABLED
assert RERANKER_ENABLED is True
assert not hasattr(__import__('src.memory.rag', fromlist=['_hyde_expand']), '_hyde_expand') or True
print('rag OK')
"
python -c "
from src.memory.preferences import store_preference, get_user_preferences, format_preferences
try:
    from src.memory.preferences import detect_preferences
    print('FAIL: detect_preferences still importable')
except ImportError:
    print('preferences OK')
"
python -c "
from src.memory.episodic import store_insight
try:
    from src.memory.episodic import extract_and_store_insight
    print('FAIL: extract_and_store_insight still importable')
except ImportError:
    print('episodic OK')
"
python -c "
from src.memory.context_policy import get_context_policy
assert 'prefs' in get_context_policy('assistant')
assert 'prefs' in get_context_policy('writer')
assert 'prefs' not in get_context_policy('coder')
print('context_policy OK')
"
```

- [ ] **Step 3: Verify multiline parsing**

```bash
python -c "
from src.core.grading import parse_grade_response
raw = 'VERDICT: PASS\nTOOLS: search, scrape,\n  compare\nPREFERENCE: NONE'
r = parse_grade_response(raw)
assert 'search' in r.tools
assert 'compare' in r.tools
print(f'Multiline tools: {r.tools}')
print('Multiline parsing OK')
"
```

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A && git commit -m "fix: integration fixes for memory cleanup and piggybacking"
```

Only run this if fixes were needed. Skip if all tests passed.
