# Memory & Context Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix prompt bloat on 8K-context local LLMs by unifying grading (restoring skill extraction), gating context injection by task type, and raising retrieval thresholds.

**Architecture:** Three independent phases. Phase 1 unifies the two grading systems into one with progressive skill extraction. Phase 2 adds a context policy module that gates which layers `_build_context()` injects per task type, with hard token budgets. Phase 3 raises RAG/skill thresholds and adds collection gating.

**Tech Stack:** Python 3.10, async, aiosqlite, ChromaDB, sentence-transformers (multilingual-e5-base)

**Key references:**
- Design spec: `docs/superpowers/specs/2026-04-07-memory-context-redesign-design.md`
- Decisions & constraints: `docs/issues/memory-redesign-context.md`
- Context layer inventory: `docs/issues/context-layers-reference.md`
- Skill system docs: `docs/skill-system.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/core/grading.py` | Modify | Unified grading prompt, progressive parsing, updated GradeResult, skill extraction in apply_grade_result |
| `src/core/router.py` | Modify | Remove `grade_response()` and `GRADING_PROMPT` (dead code) |
| `src/core/orchestrator.py` | Modify | Remove score-based checks, rewrite skill extraction to use verdict fields, remove quality_score from result dicts |
| `src/agents/base.py` | Modify | Remove quality_score from result dicts, refactor `_build_context()` to use context policy |
| `src/memory/context_policy.py` | Create | Context policy map, heuristic overrides, budget calculator |
| `src/memory/rag.py` | Modify | New thresholds, collection gating, reduced top_k, accept max_tokens param |
| `src/memory/skills.py` | Modify | New match threshold |
| `tests/test_grading.py` | Modify | Update tests for new GradeResult, add progressive parsing tests |
| `tests/test_context_policy.py` | Create | Tests for policy map, heuristics, budget calculator |
| `tests/test_rag_thresholds.py` | Create | Tests for new thresholds and collection gating |
| `docs/skill-system.md` | Modify | Update to reflect unified grading |

---

## Task 1: Update GradeResult and Parsing

**Files:**
- Modify: `src/core/grading.py:1-71`
- Modify: `tests/test_grading.py`

- [ ] **Step 1: Write failing tests for new GradeResult fields and progressive parsing**

```python
# tests/test_grading.py — replace entire file
import pytest
from src.core.grading import parse_grade_response, GradeResult


class TestGradeResult:
    def test_passed_has_no_score_field(self):
        """GradeResult must not have a score field."""
        result = GradeResult(passed=True)
        assert not hasattr(result, "score")

    def test_passed_with_skill_fields(self):
        result = GradeResult(
            passed=True,
            situation="Price comparison across Turkish stores",
            strategy="Search each store separately then compare",
            tools=["smart_search", "web_search"],
        )
        assert result.passed is True
        assert result.situation == "Price comparison across Turkish stores"
        assert result.tools == ["smart_search", "web_search"]

    def test_default_skill_fields_empty(self):
        result = GradeResult(passed=False)
        assert result.situation == ""
        assert result.strategy == ""
        assert result.tools == []


class TestParseGradeResponse:
    def test_full_output_with_skill_fields(self):
        raw = (
            "RELEVANT: YES\n"
            "COMPLETE: YES\n"
            "VERDICT: PASS\n"
            "SITUATION: Price comparison across Turkish stores\n"
            "STRATEGY: Search each store separately then compare\n"
            "TOOLS: smart_search, web_search"
        )
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.relevant is True
        assert result.complete is True
        assert result.situation == "Price comparison across Turkish stores"
        assert result.strategy == "Search each store separately then compare"
        assert result.tools == ["smart_search", "web_search"]

    def test_verdict_pass_without_skill_fields(self):
        raw = "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS"
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.situation == ""
        assert result.strategy == ""
        assert result.tools == []

    def test_verdict_fail(self):
        raw = "RELEVANT: YES\nCOMPLETE: NO\nVERDICT: FAIL"
        result = parse_grade_response(raw)
        assert result.passed is False
        assert result.complete is False

    def test_all_yes_no(self):
        raw = "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: YES"
        result = parse_grade_response(raw)
        assert result.passed is True

    def test_verdict_no(self):
        raw = "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: NO"
        result = parse_grade_response(raw)
        assert result.passed is False

    def test_derive_from_relevant_complete_when_no_verdict(self):
        raw = "RELEVANT: YES\nCOMPLETE: YES"
        result = parse_grade_response(raw)
        assert result.passed is True

    def test_derive_fail_when_relevant_no(self):
        raw = "RELEVANT: NO\nCOMPLETE: YES"
        result = parse_grade_response(raw)
        assert result.passed is False

    def test_bare_pass_fallback(self):
        raw = "I think this response is good and should be accepted. PASS"
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.relevant is None
        assert result.complete is None

    def test_bare_fail_fallback(self):
        raw = "The response does not address the task at all. FAIL"
        result = parse_grade_response(raw)
        assert result.passed is False

    def test_bare_pass_case_insensitive(self):
        raw = "Overall this is a pass from me."
        result = parse_grade_response(raw)
        assert result.passed is True

    def test_unparseable_raises(self):
        with pytest.raises(ValueError, match="grader incapable"):
            parse_grade_response("Here is my analysis of the task response quality metrics")

    def test_case_insensitive_fields(self):
        raw = "relevant: yes\ncomplete: Yes\nverdict: PASS"
        result = parse_grade_response(raw)
        assert result.passed is True

    def test_with_reasoning_noise(self):
        raw = (
            "The response looks good.\n"
            "RELEVANT: YES\n"
            "I think it is complete.\n"
            "COMPLETE: YES\n"
            "VERDICT: PASS\n"
            "SITUATION: Weather lookup for Istanbul\n"
            "STRATEGY: Used weather API directly\n"
            "TOOLS: api_call"
        )
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.situation == "Weather lookup for Istanbul"
        assert result.tools == ["api_call"]

    def test_partial_skill_fields(self):
        """If only SITUATION parses but STRATEGY doesn't, keep what we got."""
        raw = (
            "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS\n"
            "SITUATION: Currency conversion task"
        )
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.situation == "Currency conversion task"
        assert result.strategy == ""
        assert result.tools == []

    def test_tools_with_spaces(self):
        raw = (
            "VERDICT: PASS\n"
            "SITUATION: test\n"
            "STRATEGY: test\n"
            "TOOLS: smart_search , web_search , api_call"
        )
        result = parse_grade_response(raw)
        assert result.tools == ["smart_search", "web_search", "api_call"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grading.py -v`
Expected: Multiple FAILs — `GradeResult` still has `score`, no `situation`/`strategy`/`tools` fields, no bare PASS/FAIL fallback.

- [ ] **Step 3: Update GradeResult dataclass and parsing**

Replace the GradeResult dataclass and parsing in `src/core/grading.py`:

```python
# Replace lines 19-71 of src/core/grading.py

GRADING_PROMPT = """Evaluate this task result.

Task: {title}
Description: {description}
Result: {response}

RELEVANT: YES or NO
COMPLETE: YES or NO
VERDICT: PASS or FAIL
SITUATION: one line, what type of problem was solved
STRATEGY: one line, what approach worked
TOOLS: comma-separated list of tools used effectively"""


@dataclass
class GradeResult:
    passed: bool
    relevant: Optional[bool] = None
    complete: Optional[bool] = None
    situation: str = ""
    strategy: str = ""
    tools: list[str] = field(default_factory=list)
    raw: str = ""
```

Add `field` to imports:

```python
from dataclasses import dataclass, field
```

Replace parsing functions:

```python
def _parse_yes_no(text: str, key: str) -> Optional[bool]:
    """Extract a YES/NO value for a given key from grader output."""
    pattern = rf"{key}\s*:\s*(YES|NO|PASS|FAIL)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    val = match.group(1).upper()
    return val in ("YES", "PASS")


def _parse_text_field(text: str, key: str) -> str:
    """Extract a free-text value for a given key from grader output."""
    pattern = rf"{key}\s*:\s*(.+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


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

    # Skill extraction fields (optional — never block grading)
    situation = _parse_text_field(raw, "SITUATION")
    strategy = _parse_text_field(raw, "STRATEGY")
    tools_raw = _parse_text_field(raw, "TOOLS")
    tools = [t.strip() for t in tools_raw.split(",") if t.strip()] if tools_raw else []

    # Cascade 1: VERDICT present
    if verdict is not None:
        return GradeResult(
            passed=verdict, relevant=relevant, complete=complete,
            situation=situation, strategy=strategy, tools=tools, raw=raw,
        )

    # Cascade 2: derive from RELEVANT + COMPLETE
    if relevant is not None and complete is not None:
        return GradeResult(
            passed=(relevant and complete), relevant=relevant, complete=complete,
            situation=situation, strategy=strategy, tools=tools, raw=raw,
        )

    # Cascade 3: bare PASS/FAIL keyword anywhere
    bare = re.search(r'\bPASS\b', raw, re.IGNORECASE)
    if bare:
        return GradeResult(passed=True, situation=situation, strategy=strategy, tools=tools, raw=raw)
    bare_fail = re.search(r'\bFAIL\b', raw, re.IGNORECASE)
    if bare_fail:
        return GradeResult(passed=False, situation=situation, strategy=strategy, tools=tools, raw=raw)

    raise ValueError(f"grader incapable: could not parse VERDICT, RELEVANT, or COMPLETE from output: {raw[:150]}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grading.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/grading.py tests/test_grading.py
git commit -m "feat(grading): unified prompt with progressive skill extraction"
```

---

## Task 2: Update apply_grade_result for Skill Extraction

**Files:**
- Modify: `src/core/grading.py:142-293`

- [ ] **Step 1: Write failing test for skill extraction from verdict fields**

Add to `tests/test_grading.py`:

```python
from unittest.mock import patch, AsyncMock


class TestApplyGradeSkillExtraction:
    """Skill extraction should use verdict.situation/strategy/tools when available."""

    @patch("src.core.grading.transition_task", new_callable=AsyncMock)
    @patch("src.core.grading.get_task", new_callable=AsyncMock)
    @patch("src.memory.skills.add_skill", new_callable=AsyncMock)
    @patch("src.core.grading.record_model_call", new_callable=AsyncMock)
    async def test_skill_extraction_from_verdict_fields(
        self, mock_record, mock_add_skill, mock_get_task, mock_transition
    ):
        import asyncio
        mock_get_task.return_value = {
            "id": 42,
            "title": "Compare laptop prices",
            "agent_type": "shopping_advisor",
            "iterations": 3,
            "context": '{"generating_model": "test", "tools_used_names": ["smart_search", "web_search"]}',
        }

        verdict = GradeResult(
            passed=True,
            situation="Price comparison across Turkish stores",
            strategy="Search each store separately then compare",
            tools=["smart_search", "web_search"],
        )

        from src.core.grading import apply_grade_result
        await apply_grade_result(42, verdict)

        mock_add_skill.assert_called_once()
        call_kwargs = mock_add_skill.call_args
        # Should use verdict.situation as description, not mechanical fallback
        assert "Price comparison" in call_kwargs.kwargs.get("description", "") or \
               "Price comparison" in (call_kwargs.args[1] if len(call_kwargs.args) > 1 else "")

    @patch("src.core.grading.transition_task", new_callable=AsyncMock)
    @patch("src.core.grading.get_task", new_callable=AsyncMock)
    @patch("src.memory.skills.add_skill", new_callable=AsyncMock)
    @patch("src.core.grading.record_model_call", new_callable=AsyncMock)
    async def test_mechanical_fallback_when_no_situation(
        self, mock_record, mock_add_skill, mock_get_task, mock_transition
    ):
        import asyncio
        mock_get_task.return_value = {
            "id": 43,
            "title": "Check weather",
            "agent_type": "executor",
            "iterations": 2,
            "context": '{"generating_model": "test", "tools_used_names": ["api_call"]}',
        }

        verdict = GradeResult(passed=True)  # No skill fields

        from src.core.grading import apply_grade_result
        await apply_grade_result(43, verdict)

        mock_add_skill.assert_called_once()
        call_kwargs = mock_add_skill.call_args
        # Should use mechanical fallback
        desc = call_kwargs.kwargs.get("description", "") or (call_kwargs.args[1] if len(call_kwargs.args) > 1 else "")
        assert "Task:" in desc or "Used " in call_kwargs.kwargs.get("strategy_summary", "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grading.py::TestApplyGradeSkillExtraction -v`
Expected: FAIL — current code checks `verdict.score >= 4.0` which no longer exists.

- [ ] **Step 3: Rewrite skill extraction in apply_grade_result**

In `src/core/grading.py`, replace the PASS block in `apply_grade_result()` (lines 167-220):

```python
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
        iterations = task.get("iterations", 1) or 1
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

        logger.info(f"grade PASS | task_id={task_id}")
```

Also remove the `grade=verdict.score` argument from `record_model_call` — the function accepts it but we no longer have a numeric score. Pass no `grade` argument (it defaults to `None`).

Also update `transition_task` call — remove `quality_score=verdict.score`. The column still exists in DB but we stop writing to it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grading.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/grading.py tests/test_grading.py
git commit -m "feat(grading): skill extraction uses verdict fields, removes score"
```

---

## Task 3: Remove Dead Grader and Score References

**Files:**
- Modify: `src/core/router.py:1496-1610` — remove `grade_response()` and its `GRADING_PROMPT`
- Modify: `src/core/orchestrator.py:2228-2299` — remove score-based skill extraction (now handled by grading.py)
- Modify: `src/agents/base.py:1592-1670` — remove `quality_score` from result dicts
- Modify: `src/app/telegram_bot.py:2761` — remove quality score display
- Remove: `tests/test_llm_dispatcher.py:429-445` — test for dead `grade_response`

- [ ] **Step 1: Remove `grade_response()` and its `GRADING_PROMPT` from router.py**

In `src/core/router.py`, find and delete the `GRADING_PROMPT` constant (around line 1496-1503) and the entire `grade_response()` function (around lines 1506-1610). These are dead code — confirmed no imports from `src/`.

- [ ] **Step 2: Remove score-based skill extraction from orchestrator.py**

In `src/core/orchestrator.py`, replace the quality score notification block and skill extraction block (lines 2228-2299):

```python
        # ── Injection success tracking ──
        # (skill extraction now happens in grading.py:apply_grade_result)
        try:
            injected = task_ctx_parsed.get("injected_skills", [])
            if injected and result.get("status") != "ungraded":
                from ..memory.skills import record_injection_success
                await record_injection_success(injected)
        except Exception:
            pass
```

This removes:
- The `quality_score` notification (lines 2228-2239) — grading.py now sends its own notification
- The `worth_capturing` skill extraction block (lines 2243-2289) — moved to grading.py
- Keeps injection success tracking but gates on non-ungraded (ungraded tasks track success when graded later)

- [ ] **Step 3: Remove quality_score from base.py result dicts**

In `src/agents/base.py`:

Line 1593: Remove `quality_score = None`

Line 1612: Remove `quality_score = verdict.score`

Line 1621: Remove `"quality_score": quality_score,` from the return dict

Line 1668: Remove `"quality_score": quality_score,` from the return dict

The result dicts should no longer contain `quality_score`. The field stays in the DB schema (no migration needed — it just stops being written).

- [ ] **Step 4: Remove quality score display from telegram_bot.py**

In `src/app/telegram_bot.py`, around line 2761, find `quality = task.get('quality_score', 0) or 0` and remove any quality score display logic that uses it. Replace with a simple pass or remove the block.

- [ ] **Step 5: Remove dead test for grade_response**

In `tests/test_llm_dispatcher.py`, remove the `test_grade_response_uses_low_priority` test method (around lines 429-445).

In `tests/test_web_search_integration.py`, around line 339, remove or update the `@patch("src.agents.base.grade_response", ...)` decorator if it causes import errors.

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/test_grading.py tests/test_llm_dispatcher.py -v`
Expected: All PASS, no import errors for removed functions.

- [ ] **Step 7: Commit**

```bash
git add src/core/router.py src/core/orchestrator.py src/agents/base.py src/app/telegram_bot.py tests/test_llm_dispatcher.py tests/test_web_search_integration.py
git commit -m "refactor(grading): remove dead grade_response and quality_score references"
```

---

## Task 4: Create Context Policy Module

**Files:**
- Create: `src/memory/context_policy.py`
- Create: `tests/test_context_policy.py`

- [ ] **Step 1: Write tests for context policy**

```python
# tests/test_context_policy.py
import pytest
from src.memory.context_policy import (
    get_context_policy,
    apply_heuristics,
    compute_layer_budgets,
    DEFAULT_POLICY,
)


class TestGetContextPolicy:
    def test_known_profile(self):
        policy = get_context_policy("executor")
        assert policy == {"deps", "skills", "api"}

    def test_shopping_advisor(self):
        policy = get_context_policy("shopping_advisor")
        assert policy == {"skills", "convo"}

    def test_reviewer_empty(self):
        policy = get_context_policy("reviewer")
        assert policy == set()

    def test_unknown_returns_default(self):
        policy = get_context_policy("nonexistent_agent_type")
        assert policy == DEFAULT_POLICY


class TestApplyHeuristics:
    def test_tools_hint_adds_skills_and_api(self):
        task = {"context": {"tools_hint": ["smart_search"]}}
        result = apply_heuristics(task, set())
        assert "skills" in result
        assert "api" in result

    def test_depends_on_adds_deps(self):
        task = {"depends_on": "[1, 2]"}
        result = apply_heuristics(task, set())
        assert "deps" in result

    def test_followup_adds_convo(self):
        task = {"context": {"is_followup": True}}
        result = apply_heuristics(task, set())
        assert "convo" in result

    def test_mission_adds_board(self):
        task = {"mission_id": 5}
        result = apply_heuristics(task, set())
        assert "board" in result

    def test_does_not_mutate_input(self):
        original = {"skills"}
        task = {"mission_id": 5}
        result = apply_heuristics(task, original)
        assert "board" not in original
        assert "board" in result

    def test_no_heuristics_returns_copy(self):
        policy = {"skills", "rag"}
        task = {}
        result = apply_heuristics(task, policy)
        assert result == policy
        assert result is not policy


class TestComputeLayerBudgets:
    def test_8k_executor(self):
        budgets = compute_layer_budgets(8192, {"deps", "skills", "api"})
        total = sum(budgets.values())
        # Should not exceed 40% of context
        assert total <= int(8192 * 0.40) + 1  # +1 for rounding
        # deps should get most (weight 5)
        assert budgets["deps"] > budgets["skills"]
        assert budgets["skills"] > budgets["api"]

    def test_empty_layers(self):
        budgets = compute_layer_budgets(8192, set())
        assert budgets == {}

    def test_single_layer_gets_full_budget(self):
        budgets = compute_layer_budgets(8192, {"rag"})
        assert budgets["rag"] == int(8192 * 0.40)

    def test_32k_model_more_budget(self):
        budgets_8k = compute_layer_budgets(8192, {"deps", "skills"})
        budgets_32k = compute_layer_budgets(32768, {"deps", "skills"})
        assert budgets_32k["deps"] > budgets_8k["deps"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_context_policy.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement context_policy.py**

```python
# src/memory/context_policy.py
"""
Context gating — decides which context layers to inject per task type.

Each task profile maps to a set of layer IDs. Heuristic overrides
adjust based on task metadata. Budget calculator distributes tokens
across active layers by priority weight.
"""
from __future__ import annotations

import json
from typing import Optional

CONTEXT_POLICIES: dict[str, set[str]] = {
    "executor":         {"deps", "skills", "api"},
    "coder":            {"deps", "skills", "profile", "rag"},
    "implementer":      {"deps", "prior", "skills", "profile", "board"},
    "fixer":            {"deps", "skills", "rag", "profile"},
    "researcher":       {"skills", "rag", "api", "convo"},
    "shopping_advisor": {"skills", "convo"},
    "assistant":        {"convo", "rag", "memory"},
    "writer":           {"deps", "convo", "memory"},
    "planner":          {"deps", "board", "ambient", "memory"},
    "architect":        {"deps", "profile", "board", "rag"},
    "reviewer":         set(),
    "summarizer":       {"deps"},
    "analyst":          {"deps", "rag", "board"},
    "error_recovery":   {"deps", "rag", "skills"},
    "router":           set(),
    "visual_reviewer":  set(),
    "test_generator":   {"deps", "skills", "profile"},
}

DEFAULT_POLICY: set[str] = {"deps", "skills", "rag"}

# Priority weights for budget distribution.
# Higher weight = more tokens allocated.
LAYER_WEIGHTS: dict[str, int] = {
    "deps": 5, "prior": 4, "skills": 3, "rag": 3,
    "convo": 2, "board": 2, "profile": 1, "ambient": 1,
    "api": 1, "memory": 1, "prefs": 1,
}

# Fraction of model context reserved for injected context.
# Remaining 60% is for system prompt + model reasoning.
CONTEXT_FRACTION = 0.40


def get_context_policy(agent_type: str) -> set[str]:
    """Return the context layer set for a given agent/task profile."""
    return set(CONTEXT_POLICIES.get(agent_type, DEFAULT_POLICY))


def apply_heuristics(task: dict, policy: set[str]) -> set[str]:
    """Adjust policy based on task metadata. Returns new set, never mutates input."""
    p = set(policy)

    # Parse task context
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    if not isinstance(ctx, dict):
        ctx = {}

    # Workflow steps with tools_hint → ensure skills + api
    if ctx.get("tools_hint"):
        p.add("skills")
        p.add("api")

    # Tasks with dependencies always get deps
    if task.get("depends_on"):
        p.add("deps")

    # Follow-up tasks always get conversation
    if ctx.get("is_followup"):
        p.add("convo")

    # Mission tasks always get blackboard
    if task.get("mission_id"):
        p.add("board")

    return p


def compute_layer_budgets(model_context: int, active_layers: set[str]) -> dict[str, int]:
    """Distribute token budget across active layers by priority weight.

    Returns a dict mapping layer ID to its max token allocation.
    """
    if not active_layers:
        return {}

    available = int(model_context * CONTEXT_FRACTION)

    active_weights = {k: LAYER_WEIGHTS[k] for k in active_layers if k in LAYER_WEIGHTS}
    total_weight = sum(active_weights.values()) or 1

    return {
        layer: int(available * w / total_weight)
        for layer, w in active_weights.items()
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_context_policy.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/context_policy.py tests/test_context_policy.py
git commit -m "feat(context): add context policy module with gating, heuristics, budgets"
```

---

## Task 5: Refactor _build_context to Use Context Policy

**Files:**
- Modify: `src/agents/base.py:347-577`

This is the largest change. We rewrite `_build_context()` to use the policy module. Each layer becomes a helper function with a `max_tokens` parameter.

- [ ] **Step 1: Write test for gated context building**

Add to existing test file or create `tests/test_build_context.py`:

```python
# tests/test_build_context.py
import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock


class TestBuildContextGating:
    """Verify that _build_context respects context policies."""

    @pytest.mark.asyncio
    @patch("src.memory.context_policy.get_context_policy", return_value=set())
    async def test_reviewer_gets_no_extra_context(self, mock_policy):
        """Reviewer policy is empty — only task core + task_ctx should appear."""
        from src.agents.base import BaseAgent

        agent = BaseAgent.__new__(BaseAgent)
        agent.name = "reviewer"
        agent.allowed_tools = []

        task = {
            "title": "Grade this response",
            "description": "Check if it addresses the task",
            "agent_type": "reviewer",
            "context": "{}",
        }

        with patch.object(agent, "_estimate_tokens", side_effect=lambda s: len(s) // 4):
            result = await agent._build_context(task)

        # Should contain task title but NOT RAG, skills, preferences, etc.
        assert "Grade this response" in result
        # These should NOT appear since policy is empty
        assert "## Recalled Knowledge" not in result
        assert "## Execution Recipes" not in result
        assert "## User Preferences" not in result

    @pytest.mark.asyncio
    @patch("src.memory.context_policy.get_context_policy", return_value={"skills"})
    @patch("src.memory.skills.find_relevant_skills", new_callable=AsyncMock, return_value=[])
    async def test_executor_with_skills_only(self, mock_skills, mock_policy):
        """When policy only includes skills, RAG should not be called."""
        from src.agents.base import BaseAgent

        agent = BaseAgent.__new__(BaseAgent)
        agent.name = "executor"
        agent.allowed_tools = []

        task = {
            "title": "Run API call",
            "description": "Call weather API",
            "agent_type": "executor",
            "context": "{}",
        }

        with patch("src.memory.rag.retrieve_context", new_callable=AsyncMock) as mock_rag:
            with patch.object(agent, "_estimate_tokens", side_effect=lambda s: len(s) // 4):
                await agent._build_context(task)
            # RAG should NOT have been called
            mock_rag.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_build_context.py -v`
Expected: FAIL — current `_build_context` doesn't use policy gating.

- [ ] **Step 3: Refactor _build_context in base.py**

Replace `_build_context()` method (lines 347-577) in `src/agents/base.py`:

```python
    async def _build_context(self, task: dict) -> str:
        """
        Assemble the user message with task info and policy-gated context layers.
        Each layer respects its allocated token budget.
        """
        import json
        from ..memory.context_policy import (
            get_context_policy, apply_heuristics, compute_layer_budgets,
        )

        parts: list[str] = []

        # ── Task description (PRIMARY — always injected) ──
        parts.append(
            f"## Task (PRIMARY — this is what you must do)\n"
            f"**{task.get('title', 'Untitled')}**\n"
            f"{task.get('description', '')}"
        )

        # ── Parse task.context ──
        task_context = task.get("context")
        if isinstance(task_context, str):
            try:
                task_context = json.loads(task_context)
            except (json.JSONDecodeError, TypeError):
                task_context = {}
        if not isinstance(task_context, dict):
            task_context = {}

        # ── Task context fields (always injected if present) ──
        if "workspace_snapshot" in task_context:
            parts.append(
                f"## Current Workspace State\n{task_context['workspace_snapshot']}"
            )
        if "tool_result" in task_context:
            parts.append(
                f"## Prior Tool Result\n{task_context['tool_result']}"
            )
        if "user_clarification" in task_context:
            answer = task_context["user_clarification"]
            history = task_context.get("clarification_history", [])
            parts.append(
                f"## User Clarification\n"
                f"You previously asked for clarification. The user answered: **{answer}**\n"
                f"Do NOT ask for clarification again. Use this answer and proceed with the task."
            )
            if len(history) > 1:
                parts.append(f"Previous answers: {history}")

        _skip = {"workspace_snapshot", "tool_result", "prior_steps", "tool_depth",
                 "recent_conversation", "user_clarification", "clarification_history"}
        extra = {k: v for k, v in task_context.items() if k not in _skip and not k.startswith("_")}
        if extra:
            parts.append(
                f"## Additional Context\n{json.dumps(extra, indent=2)}"
            )

        # ── Determine active layers and budgets ──
        agent_type = task.get("agent_type") or self.name
        policy = get_context_policy(agent_type)
        policy = apply_heuristics(task, policy)

        model_ctx = task_context.get("model_context_length", 4096)
        budgets = compute_layer_budgets(model_ctx, policy)

        mission_id = task.get("mission_id")

        # ── Gated layers — each respects its token budget ──

        if "deps" in policy:
            block = await self._fetch_deps(task, max_tokens=budgets.get("deps", 2000))
            if block:
                parts.append(block)

        if "prior" in policy:
            block = self._format_prior_steps(task_context, max_tokens=budgets.get("prior", 1500))
            if block:
                parts.append(block)

        if "convo" in policy:
            block = self._format_conversation(task_context, max_tokens=budgets.get("convo", 800))
            if block:
                parts.append(block)

        if "ambient" in policy:
            try:
                from ..context.assembler import assemble_ambient_context
                ambient = await assemble_ambient_context(
                    mission_id=mission_id,
                    max_tokens=min(budgets.get("ambient", 400), 400),
                )
                if ambient:
                    parts.append(ambient)
            except Exception as exc:
                logger.debug(f"Ambient context failed: {exc}")

        if "profile" in policy:
            try:
                project_profile = await get_project_profile_for_task(task)
                profile_block = format_project_profile(project_profile) if project_profile else ""
                if profile_block:
                    # Truncate to budget
                    budget = budgets.get("profile", 500)
                    truncated = self._truncate_to_tokens(profile_block, budget)
                    parts.append(truncated)
            except Exception as exc:
                logger.debug(f"Project profile failed: {exc}")

        if "board" in policy and mission_id:
            try:
                board = await get_or_create_blackboard(mission_id)
                bb_block = format_blackboard_for_prompt(board)
                if bb_block:
                    budget = budgets.get("board", 500)
                    truncated = self._truncate_to_tokens(bb_block, budget)
                    parts.append(truncated)
            except Exception as exc:
                logger.debug(f"Blackboard failed: {exc}")

        if "skills" in policy:
            try:
                from ..memory.skills import (
                    find_relevant_skills, format_skills_for_prompt,
                    get_tools_to_inject, record_injection,
                )
                task_text = f"{task.get('title', '')} {task.get('description', '')}"
                budget = budgets.get("skills", 800)
                relevant_skills = await find_relevant_skills(task_text, limit=3)
                if relevant_skills:
                    skills_block = format_skills_for_prompt(relevant_skills, budget)
                    if skills_block:
                        parts.append(skills_block)

                    extra_tools = get_tools_to_inject(relevant_skills)
                    if extra_tools and self.allowed_tools is not None:
                        for tool in extra_tools:
                            if tool not in self.allowed_tools:
                                self.allowed_tools.append(tool)

                    skill_names = [s["name"] for s in relevant_skills]
                    await record_injection(skill_names)
                    try:
                        _ctx = json.loads(task.get("context", "{}"))
                        _ctx["injected_skills"] = skill_names
                        task["context"] = json.dumps(_ctx)
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("Skill injection failed: %s", exc)

        if "api" in policy:
            try:
                api_enrichment = task_context.get("api_enrichment")
                if api_enrichment:
                    budget = budgets.get("api", 300)
                    truncated = self._truncate_to_tokens(api_enrichment, budget)
                    parts.append(truncated)
            except Exception as exc:
                logger.debug("API enrichment failed: %s", exc)

        if "rag" in policy:
            try:
                budget = budgets.get("rag", 2000)
                rag_block = await retrieve_context(
                    task=task, agent_type=self.name,
                    max_tokens=budget,
                )
                if rag_block:
                    parts.append(rag_block)
            except Exception as exc:
                logger.debug(f"RAG retrieval failed: {exc}")

        if "prefs" in policy:
            try:
                prefs = await get_user_preferences()
                pref_block = format_preferences(prefs)
                if pref_block:
                    budget = budgets.get("prefs", 200)
                    truncated = self._truncate_to_tokens(pref_block, budget)
                    parts.append(truncated)
            except Exception as exc:
                logger.debug(f"Preference retrieval failed: {exc}")

        if "memory" in policy:
            try:
                memories = await recall_memory(mission_id=mission_id, limit=10)
                if memories:
                    budget = budgets.get("memory", 500)
                    mem_parts = ["## Project Memory"]
                    for mem in memories:
                        mem_value = mem.get('value', '')
                        if not isinstance(mem_value, str):
                            mem_value = str(mem_value)
                        mem_parts.append(f"- **{mem.get('key', 'unknown')}**: {mem_value[:300]}")
                    mem_block = "\n".join(mem_parts)
                    truncated = self._truncate_to_tokens(mem_block, budget)
                    parts.append(truncated)
            except Exception as exc:
                logger.debug(f"Memory recall failed: {exc}")

        return "\n\n".join(parts)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Rough truncation: ~4 chars per token."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... [truncated to budget]"

    async def _fetch_deps(self, task: dict, max_tokens: int) -> str:
        """Fetch dependency results, truncated to budget."""
        depends_on = task.get("depends_on")
        if isinstance(depends_on, str):
            try:
                depends_on = json.loads(depends_on)
            except (json.JSONDecodeError, TypeError):
                depends_on = []
        if not depends_on:
            return ""
        try:
            dep_results = await get_completed_dependency_results(depends_on)
        except Exception as exc:
            logger.warning(f"Failed to fetch dependency results: {exc}")
            return ""
        if not dep_results:
            return ""

        parts = ["## Results from Previous Steps"]
        budget_chars = max_tokens * 4
        used = len(parts[0])
        per_dep = max(500, (budget_chars - used) // max(len(dep_results), 1))

        for dep_id, dep in dep_results.items():
            text = dep.get("result") or "(no result)"
            if len(text) > per_dep:
                text = text[:per_dep] + "\n... (truncated)"
            entry = f"### Step #{dep_id}: {dep.get('title', 'Unknown')}\n{text}"
            parts.append(entry)

        result = "\n".join(parts)
        return self._truncate_to_tokens(result, max_tokens)

    def _format_prior_steps(self, task_context: dict, max_tokens: int) -> str:
        """Format inline prior steps, truncated to budget."""
        if "prior_steps" not in task_context:
            return ""
        parts = ["## Results from Prior Steps (Inline)"]
        per_step = max(400, (max_tokens * 4) // max(len(task_context["prior_steps"]), 1))
        for step in task_context["prior_steps"]:
            result = step.get("result", "")
            if len(result) > per_step:
                result = result[:per_step] + "\n... [truncated]"
            parts.append(
                f"### Step: {step.get('title', 'Unknown')} "
                f"(Status: {step.get('status', '?')})\n{result}"
            )
        return self._truncate_to_tokens("\n".join(parts), max_tokens)

    def _format_conversation(self, task_context: dict, max_tokens: int) -> str:
        """Format recent conversation, truncated to budget."""
        if "recent_conversation" not in task_context:
            return ""
        parts = ["## Recent Conversation (for context)"]
        for entry in task_context["recent_conversation"]:
            user_q = entry.get("user_asked", "?")
            result = entry.get("result", "")
            if len(result) > 600:
                result = result[:600] + "... [truncated]"
            parts.append(f"**User asked:** {user_q}\n**Result:** {result}\n")
        parts.append(
            "_Use this context to understand follow-up references "
            "like 'list them', 'the names', 'do it again', etc._"
        )
        return self._truncate_to_tokens("\n".join(parts), max_tokens)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_build_context.py tests/test_context_policy.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `pytest tests/ -x --timeout=30 -q`
Expected: No new failures.

- [ ] **Step 6: Commit**

```bash
git add src/agents/base.py
git commit -m "feat(context): refactor _build_context to use policy-gated layers with budgets"
```

---

## Task 6: Threshold Tuning and RAG Collection Gating

**Files:**
- Modify: `src/memory/rag.py:45-50, 428-450`
- Modify: `src/memory/skills.py:30`
- Create: `tests/test_rag_thresholds.py`

- [ ] **Step 1: Write tests for new thresholds and collection gating**

```python
# tests/test_rag_thresholds.py
import pytest
from src.memory.rag import RAG_CONFIG, get_rag_collections


class TestRAGConfig:
    def test_relevance_threshold_raised(self):
        assert RAG_CONFIG["min_relevance"] >= 0.72

    def test_top_k_reduced(self):
        assert RAG_CONFIG["top_k_per_collection"] <= 2


class TestRAGCollectionGating:
    def test_coder_gets_errors_and_codebase(self):
        collections = get_rag_collections("coder")
        assert "errors" in collections
        assert "codebase" in collections
        assert "shopping" not in collections

    def test_shopping_advisor_gets_shopping(self):
        collections = get_rag_collections("shopping_advisor")
        assert "shopping" in collections
        assert "errors" not in collections

    def test_unknown_gets_default(self):
        collections = get_rag_collections("nonexistent")
        assert "episodic" in collections
        assert "semantic" in collections

    def test_assistant_gets_semantic_and_conversations(self):
        collections = get_rag_collections("assistant")
        assert "semantic" in collections
        assert "conversations" in collections


class TestSkillThreshold:
    def test_match_threshold_raised(self):
        from src.memory.skills import MATCH_SIMILARITY_THRESHOLD
        assert MATCH_SIMILARITY_THRESHOLD >= 0.75
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_thresholds.py -v`
Expected: FAIL — old thresholds, no `RAG_CONFIG` dict, no `get_rag_collections()`.

- [ ] **Step 3: Update RAG thresholds and add collection gating**

In `src/memory/rag.py`, replace the scattered threshold constants (around lines 42-50) with a config dict:

```python
# Replace individual constants with config dict
RAG_CONFIG = {
    "min_relevance": 0.72,
    "top_k_per_collection": 2,
    "max_budget": 12000,
    "min_budget": 800,
    "budget_fraction": 0.15,
    "dedup_threshold": 0.85,
}

RAG_COLLECTIONS: dict[str, list[str]] = {
    "coder":            ["errors", "codebase"],
    "fixer":            ["errors", "codebase", "episodic"],
    "implementer":      ["errors", "codebase"],
    "researcher":       ["web_knowledge", "semantic"],
    "shopping_advisor": ["shopping"],
    "assistant":        ["semantic", "conversations"],
    "writer":           ["semantic"],
    "error_recovery":   ["errors", "episodic"],
    "test_generator":   ["errors", "codebase"],
    "planner":          ["episodic", "semantic"],
    "architect":        ["episodic", "semantic"],
    "analyst":          ["semantic", "web_knowledge"],
}

RAG_DEFAULT_COLLECTIONS = ["episodic", "semantic"]


def get_rag_collections(agent_type: str) -> list[str]:
    """Return which ChromaDB collections to query for a given agent type."""
    return RAG_COLLECTIONS.get(agent_type, RAG_DEFAULT_COLLECTIONS)
```

Then update the query section (around lines 428-450) to use `get_rag_collections()` and `RAG_CONFIG["top_k_per_collection"]`:

```python
    # ── 1. Query collections gated by agent type ──
    collections_to_query = get_rag_collections(agent_type)
    top_k = RAG_CONFIG["top_k_per_collection"]

    all_raw_results = []
    for q in queries:
        for col_name in collections_to_query:
            results = await _vs_query(text=q, collection=col_name, top_k=top_k)
            all_raw_results.extend(results)
```

Update the relevance filter in `_rank_results()` to use `RAG_CONFIG["min_relevance"]` instead of `RAG_MIN_RELEVANCE`.

Update `retrieve_context()` signature to accept `agent_type` (it may already) and pass it to the collection gating function.

Also update the old `RAG_MIN_RELEVANCE` references — search for them and replace with `RAG_CONFIG["min_relevance"]`.

- [ ] **Step 4: Update skill match threshold**

In `src/memory/skills.py`, line 30:

```python
MATCH_SIMILARITY_THRESHOLD = 0.75  # was 0.6
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_rag_thresholds.py -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -x --timeout=30 -q`
Expected: No new failures. Existing RAG tests may need threshold adjustments if they assert on the old 0.5 value.

- [ ] **Step 7: Commit**

```bash
git add src/memory/rag.py src/memory/skills.py tests/test_rag_thresholds.py
git commit -m "feat(rag): raise thresholds, add collection gating, reduce top_k"
```

---

## Task 7: Update Documentation

**Files:**
- Modify: `docs/skill-system.md`

- [ ] **Step 1: Update skill-system.md**

Update the following sections:

**Capture section** (around line 27-48): Replace the grader flow to reflect unified grading:

```markdown
### Capture (after task completion)

```
Task completes → Unified grader evaluates (PASS/FAIL)
  │
  ├─ FAIL → nothing captured
  │
  └─ PASS → grader also outputs (when model can produce them):
       SITUATION: "Comparing laptop prices across Turkish stores"
       STRATEGY: "Search each store separately then compare"
       TOOLS: smart_search, web_search
             │
             ▼
       Check ChromaDB: does a similar skill exist? (cosine similarity >= 0.85)
             │
             ├─ Yes → add this strategy to existing skill's strategy list
             │
             └─ No  → create new skill, embed description in ChromaDB
```

**Graceful degradation**: If SITUATION/STRATEGY/TOOLS are empty (small LLM couldn't produce them), falls back to task metadata (title + agent_type + tools used). Parsing is progressive — binary PASS/FAIL always works, skill fields are bonus.

**Unified grading**: Both immediate and deferred grading use the same prompt. The deferred grading gap (where most tasks got only mechanical skill entries) is fixed.
```

**Thresholds section** (around line 162-174): Update `MATCH_SIMILARITY_THRESHOLD` from 0.6 to 0.75.

**Known limitations section** (around line 199-208): Remove limitation #1 (deferred grading gap — now fixed). Update the section.

**Future work section** (around line 209-214): Remove "Address deferred grading gap" item.

**Files table** (around line 117-128): Remove `src/core/router.py:1468-1476` entry (grading prompt removed from router). Add `src/memory/context_policy.py` entry.

- [ ] **Step 2: Commit**

```bash
git add docs/skill-system.md
git commit -m "docs: update skill-system.md for unified grading and new thresholds"
```

---

## Task 8: Integration Validation

**Files:** None (testing only)

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -x --timeout=30 -v 2>&1 | head -100
```

Expected: All existing tests pass. Fix any regressions.

- [ ] **Step 2: Smoke test imports**

```bash
python -c "from src.core.grading import grade_task, apply_grade_result, GradeResult, parse_grade_response; print('grading OK')"
python -c "from src.memory.context_policy import get_context_policy, apply_heuristics, compute_layer_budgets; print('context_policy OK')"
python -c "from src.memory.rag import RAG_CONFIG, get_rag_collections; print('rag OK')"
python -c "from src.memory.skills import MATCH_SIMILARITY_THRESHOLD; assert MATCH_SIMILARITY_THRESHOLD >= 0.75; print('skills OK')"
```

Expected: All print OK.

- [ ] **Step 3: Verify grade_response is fully removed**

```bash
python -c "
try:
    from src.core.router import grade_response
    print('FAIL: grade_response still importable')
except ImportError:
    print('OK: grade_response removed')
"
```

Expected: "OK: grade_response removed"

- [ ] **Step 4: Verify context policy gating**

```bash
python -c "
from src.memory.context_policy import get_context_policy, compute_layer_budgets
p = get_context_policy('reviewer')
assert p == set(), f'reviewer should be empty, got {p}'
p = get_context_policy('executor')
b = compute_layer_budgets(8192, p)
total = sum(b.values())
assert total <= 3277, f'8K budget overflow: {total}'
print(f'executor on 8K: {b}, total={total}')
print('Context gating verified')
"
```

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: integration test fixes for memory context redesign"
```
