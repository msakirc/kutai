# Agent Iteration Exhaustion Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the cascade of bugs causing every agent task to exhaust all iterations, produce truncated results that fail workflow validation, and lose summaries — then fix artifact injection so downstream steps get the right content.

**Architecture:** Seven fixes across three layers: (1) agent layer — keyword validation, JSON unwrapping, truncation limits, cacheable tools, (2) orchestrator — timeout recovery truncation, (3) workflow hooks — defensive unwrapping, summary-first artifact fetching, missing DB function for conditional skips. All fixes are independent and can be implemented in any order.

**Tech Stack:** Python 3.10, asyncio, unittest

---

## Root Cause Chain

```
LLM produces valid 5000+ char result with "Sources:" (plural)
  → validate_task_output rejects: "source:" not found (singular match fails)
  → `continue` on last iteration exits loop (no iterations left)
  → "Exhausted iterations" path truncates to 3000 chars, keeps JSON wrapper
  → _parse_agent_response fails on truncated JSON
  → Workflow hook receives: `{"action": "final_answer", "result": "## Research...<cut>`
  → Artifact stored at exactly 3000 chars — below _SUMMARY_THRESHOLD (3000)
  → No summary created (> 3000 check fails at boundary)
  → validate_artifact_schema: table rows broken by JSON escaping → "~0 list items"
  → Hook sets status="failed", orchestrator retries entire task
  → Cycle repeats 3-6 times, burning ALL iterations each time
  → Downstream steps fetch full (mangled) artifact, no summary available
```

## File Map

| File | Changes |
|------|---------|
| `src/models/models.py:283-293` | Fix research keyword list |
| `src/agents/base.py:2007-2029` | Fix truncation: unwrap JSON first, increase limit to 8000 |
| `src/agents/base.py:59-63` | Add `smart_search` to CACHEABLE_READ_TOOLS |
| `src/core/orchestrator.py:1610-1628` | Fix timeout recovery truncation |
| `src/workflows/engine/hooks.py:20-26` | Defensive JSON unwrapping in validate_artifact_schema |
| `src/workflows/engine/hooks.py:453-460` | Summary-first artifact fetching in pre_execute hook |
| `src/infra/db.py` | Add missing `update_task_by_context_field` function |
| `tests/test_iteration_exhaustion.py` | New: all tests for these fixes |

---

### Task 1: Fix `validate_task_output` Research Keywords

Local LLMs consistently write `"Sources:"` (plural) or `"**Sources:**"` (bold markdown), but the validation only checks for `"source:"` (singular with colon). The substring `"source:"` does not match `"sources:"` because the `s` comes before the colon.

**Files:**
- Modify: `src/models/models.py:283-293`
- Test: `tests/test_iteration_exhaustion.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_iteration_exhaustion.py`:

```python
"""Tests for agent iteration exhaustion fixes."""

import json
import unittest


class TestValidateTaskOutput(unittest.TestCase):
    """Tests for validate_task_output research keyword matching."""

    def test_sources_plural_passes(self):
        """Local LLMs write 'Sources:' (plural) — must pass validation."""
        from src.models.models import validate_task_output
        result = "## Research\n\n**Sources:**\n- Wikipedia\n- Reddit"
        errors = validate_task_output("researcher", result)
        self.assertEqual(errors, [])

    def test_sources_bold_markdown_passes(self):
        """'**Sources:**' in markdown bold must pass."""
        from src.models.models import validate_task_output
        result = "## Analysis\n\n**Sources:**\n1. App Store reviews"
        errors = validate_task_output("researcher", result)
        self.assertEqual(errors, [])

    def test_based_on_passes(self):
        """'based on' is a common LLM phrasing for source attribution."""
        from src.models.models import validate_task_output
        result = "Based on analysis of competitor reviews, the market shows..."
        errors = validate_task_output("researcher", result)
        self.assertEqual(errors, [])

    def test_references_plural_passes(self):
        """'references:' (plural) must pass."""
        from src.models.models import validate_task_output
        result = "## Study\n\nReferences:\n- Smith et al."
        errors = validate_task_output("researcher", result)
        self.assertEqual(errors, [])

    def test_url_still_passes(self):
        """URL-based validation still works."""
        from src.models.models import validate_task_output
        result = "See https://example.com for details."
        errors = validate_task_output("researcher", result)
        self.assertEqual(errors, [])

    def test_no_source_still_fails(self):
        """Result with no source indicators should still fail."""
        from src.models.models import validate_task_output
        result = "The market is growing rapidly."
        errors = validate_task_output("researcher", result)
        self.assertGreater(len(errors), 0)

    def test_analyst_same_rules(self):
        """Analyst uses same 'research' category."""
        from src.models.models import validate_task_output
        result = "## Analysis\n\n**Sources:**\n- Market data"
        errors = validate_task_output("analyst", result)
        self.assertEqual(errors, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -m pytest tests/test_iteration_exhaustion.py::TestValidateTaskOutput -v`
Expected: `test_sources_plural_passes`, `test_sources_bold_markdown_passes`, `test_based_on_passes`, `test_references_plural_passes` FAIL. Others PASS.

- [ ] **Step 3: Fix the keyword list**

In `src/models/models.py`, replace lines 285-288:

```python
        has_source = any(kw in result.lower() for kw in [
            "source:", "reference:", "according to",
            "documentation", "found that", "article",
        ])
```

With:

```python
        has_source = any(kw in result.lower() for kw in [
            "source", "reference", "according to",
            "documentation", "found that", "article",
            "based on", "cited", "survey", "review",
        ])
```

Changed `"source:"` → `"source"` (no colon) so it matches `source:`, `sources:`, `**Sources:**`, etc. Same for `"reference"`. Added `"based on"`, `"cited"`, `"survey"`, `"review"` — all common LLM source-attribution phrases.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -m pytest tests/test_iteration_exhaustion.py::TestValidateTaskOutput -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/models.py tests/test_iteration_exhaustion.py
git commit -m "fix: validate_task_output accepts 'Sources:' (plural) and common attribution phrases"
```

---

### Task 2: Fix 3000-Char Truncation in Exhausted Iterations Path

The "exhausted iterations" path truncates to 3000 chars BEFORE trying to unwrap the JSON. This breaks `_parse_agent_response` because the JSON is cut mid-string. Additionally, exactly-3000-char artifacts fall below the `_SUMMARY_THRESHOLD` (> 3000), so no summary is ever created for them.

Fix: unwrap first, then truncate to 8000 (well above summary threshold).

**Files:**
- Modify: `src/agents/base.py:2007-2029`
- Test: `tests/test_iteration_exhaustion.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_iteration_exhaustion.py`:

```python
class TestExhaustedIterationsResult(unittest.TestCase):
    """Tests for the exhausted-iterations fallback path."""

    def test_json_wrapped_result_is_unwrapped(self):
        """When last assistant msg is JSON-wrapped final_answer, extract the result."""
        from src.agents.base import BaseAgent

        agent = BaseAgent.__new__(BaseAgent)
        # Simulate a 6000-char result wrapped in JSON
        inner = "## Research Report\n\n" + "x" * 5900
        wrapped = json.dumps({"action": "final_answer", "result": inner})
        # The agent should extract `inner`, not return the JSON wrapper
        parsed = agent._parse_agent_response(wrapped)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["result"], inner)

    def test_truncated_json_recovered_via_regex(self):
        """If JSON is truncated, regex fallback extracts partial result."""
        inner = "## Report\n\n" + "A" * 8000
        wrapped = json.dumps({"action": "final_answer", "result": inner})
        truncated = wrapped[:3000]  # breaks JSON

        # Regex extraction should recover partial content without wrapper
        import re
        m = re.search(r'"result"\s*:\s*"((?:[^"\\]|\\.)*)', truncated)
        self.assertIsNotNone(m)
        recovered = m.group(1)
        # Should start with the actual content, not JSON wrapper
        self.assertTrue(recovered.startswith("## Report"))
```

- [ ] **Step 2: Run test to verify the regex approach works**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -m pytest tests/test_iteration_exhaustion.py::TestExhaustedIterationsResult -v`
Expected: PASS (these test the approach, not the bug)

- [ ] **Step 3: Fix the exhausted iterations path**

In `src/agents/base.py`, replace the exhausted iterations block (lines 2007-2029):

```python
        # ── Exhausted iterations ──
        await self._clear_checkpoint_safe(task_id)
        # Extract last meaningful assistant response for the result
        last_assistant = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant = msg["content"][:3000]
                break
        # Try to parse as JSON and extract "result" field — the LLM often
        # wraps its answer in {"action": "final_answer", "result": "..."}
        if last_assistant:
            parsed_final = self._parse_agent_response(last_assistant)
            if parsed_final and parsed_final.get("result"):
                last_assistant = parsed_final["result"]
        return {
            "status": "completed",
            "result": last_assistant or "Task completed but could not produce a final answer.",
            "model": used_model,
            "cost": total_cost,
            "difficulty": reqs.difficulty,
            "iterations": self.max_iterations,
            "tools_used_names": sorted(tools_used_names),
        }
```

With:

```python
        # ── Exhausted iterations ──
        await self._clear_checkpoint_safe(task_id)
        # Extract last meaningful assistant response for the result.
        # Do NOT truncate before unwrapping — truncation breaks JSON parsing.
        last_assistant = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant = msg["content"]
                break
        # Try to parse as JSON and extract "result" field — the LLM often
        # wraps its answer in {"action": "final_answer", "result": "..."}
        if last_assistant:
            parsed_final = self._parse_agent_response(last_assistant)
            if parsed_final and parsed_final.get("result"):
                last_assistant = parsed_final["result"]
            elif '"result"' in last_assistant and '"final_answer"' in last_assistant:
                # JSON parse failed (truncated by context trimming?) — regex fallback
                import re as _re
                m = _re.search(r'"result"\s*:\s*"((?:[^"\\]|\\.)*)', last_assistant)
                if m:
                    try:
                        last_assistant = m.group(1).encode().decode('unicode_escape')
                    except Exception:
                        last_assistant = m.group(1)
        # Truncate AFTER unwrapping — preserve the actual content.
        # 8000 chars is well above _SUMMARY_THRESHOLD (3000) so the post-hook
        # will always create a summary for large artifacts.
        if len(last_assistant) > 8000:
            last_assistant = last_assistant[:8000]
        return {
            "status": "completed",
            "result": last_assistant or "Task completed but could not produce a final answer.",
            "model": used_model,
            "cost": total_cost,
            "difficulty": reqs.difficulty,
            "iterations": self.max_iterations,
            "tools_used_names": sorted(tools_used_names),
        }
```

- [ ] **Step 4: Run tests**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -m pytest tests/test_iteration_exhaustion.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/base.py
git commit -m "fix: unwrap JSON before truncating in exhausted-iterations path, increase limit to 8000"
```

---

### Task 3: Fix Timeout Recovery Truncation

Same truncation bug exists in the orchestrator's timeout recovery path — 3 strategies all do `[:3000]`.

**Files:**
- Modify: `src/core/orchestrator.py:1610-1628`
- Test: `tests/test_iteration_exhaustion.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_iteration_exhaustion.py`:

```python
class TestTimeoutRecoveryTruncation(unittest.TestCase):
    """Timeout recovery should unwrap JSON before truncating."""

    def test_timeout_recovery_unwraps_json(self):
        """JSON-wrapped final_answer should be unwrapped during recovery."""
        inner = "## Research\n\n" + "B" * 5000
        wrapped = json.dumps({"action": "final_answer", "result": inner})

        # Simulate what timeout recovery should do
        try:
            parsed = json.loads(wrapped)
            result = parsed.get("result", wrapped)[:8000]
        except (json.JSONDecodeError, TypeError):
            result = wrapped[:8000]

        self.assertEqual(result, inner)
        self.assertFalse(result.startswith("{"))
```

- [ ] **Step 2: Fix the timeout recovery path**

In `src/core/orchestrator.py`, find Strategy 1 (around line 1614-1616). Replace:

```python
                                if "final_answer" in c and len(c) > 100:
                                    partial_result = c[:3000]
                                    break
```

With:

```python
                                if "final_answer" in c and len(c) > 100:
                                    try:
                                        _p = json.loads(c)
                                        partial_result = _p.get("result", c)[:8000]
                                    except (json.JSONDecodeError, TypeError):
                                        partial_result = c[:8000]
                                    break
```

Find Strategy 2 (around line 1622). Replace `[:3000]` with `[:8000]`.

Find Strategy 3 (around line 1628). Replace `[:3000]` with `[:8000]`.

- [ ] **Step 3: Verify import**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -c "from src.core.orchestrator import Orchestrator; print('OK')"`

- [ ] **Step 4: Commit**

```bash
git add src/core/orchestrator.py
git commit -m "fix: timeout recovery unwraps JSON and increases truncation limit to 8000"
```

---

### Task 4: Defensive JSON Unwrapping in Artifact Schema Validation

Belt-and-suspenders: even after upstream fixes, `validate_artifact_schema` should handle JSON-wrapped content gracefully. This prevents future regressions.

**Files:**
- Modify: `src/workflows/engine/hooks.py:20-26`
- Test: `tests/test_iteration_exhaustion.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_iteration_exhaustion.py`:

```python
class TestArtifactSchemaValidation(unittest.TestCase):
    """Tests for validate_artifact_schema handling of JSON-wrapped content."""

    def test_array_validation_with_table_rows(self):
        """Normal markdown table should pass array validation."""
        from src.workflows.engine.hooks import validate_artifact_schema
        content = (
            "## Competitors\n\n"
            "| Name | Rating | Notes |\n"
            "|------|--------|-------|\n"
            "| App A | 4.5 | Good UX |\n"
            "| App B | 3.8 | Slow |\n"
            "| App C | 4.2 | Expensive |\n"
        )
        schema = {"competitor_list": {"type": "array", "min_items": 1}}
        is_valid, err = validate_artifact_schema(content, schema)
        self.assertTrue(is_valid, f"Should pass but got: {err}")

    def test_array_validation_with_json_escaped_table(self):
        """JSON-escaped table (\\n instead of newlines) should be unwrapped."""
        from src.workflows.engine.hooks import validate_artifact_schema
        inner = (
            "## Competitors\n\n"
            "| Name | Rating |\n"
            "|------|--------|\n"
            "| App A | 4.5 |\n"
            "| App B | 3.8 |\n"
        )
        wrapped = json.dumps({"action": "final_answer", "result": inner})
        schema = {"competitor_list": {"type": "array", "min_items": 1}}
        is_valid, err = validate_artifact_schema(wrapped, schema)
        self.assertTrue(is_valid, f"JSON-wrapped content should be unwrapped: {err}")

    def test_object_validation_with_json_wrapped(self):
        """JSON-wrapped content should pass object field validation."""
        from src.workflows.engine.hooks import validate_artifact_schema
        inner = "## Research\n\npain_points: Users struggle with...\ncurrent_tools: They use..."
        wrapped = json.dumps({"action": "final_answer", "result": inner})
        schema = {"audience_data": {
            "type": "object",
            "required_fields": ["pain_points", "current_tools"],
        }}
        is_valid, err = validate_artifact_schema(wrapped, schema)
        self.assertTrue(is_valid, f"Should find keywords after unwrapping: {err}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -m pytest tests/test_iteration_exhaustion.py::TestArtifactSchemaValidation -v`
Expected: `test_array_validation_with_json_escaped_table` and `test_object_validation_with_json_wrapped` FAIL

- [ ] **Step 3: Add JSON unwrapping to validate_artifact_schema**

In `src/workflows/engine/hooks.py`, at the top of `validate_artifact_schema` (after the `if not schema: return True, ""` check, around line 26), add:

```python
    # Unwrap final_answer JSON envelope if present — agents sometimes
    # wrap their results in {"action": "final_answer", "result": "..."}
    if isinstance(output_value, str) and '"final_answer"' in output_value:
        try:
            _envelope = json.loads(output_value)
            if isinstance(_envelope, dict) and "result" in _envelope:
                output_value = _envelope["result"]
        except (json.JSONDecodeError, TypeError):
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -m pytest tests/test_iteration_exhaustion.py tests/test_workflow_hooks.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/workflows/engine/hooks.py tests/test_iteration_exhaustion.py
git commit -m "fix: unwrap final_answer JSON envelope in artifact schema validation"
```

---

### Task 5: Summary-First Artifact Fetching

Downstream workflow steps currently fetch the full artifact by exact name from `input_artifacts`. They never check if a `{name}_summary` exists. When a full artifact exceeds the tier's budget, `format_artifacts_for_prompt` blindly truncates it — losing structure and meaning.

Fix: In the pre-execute hook, for each input artifact, check if a summary exists. **Use the summary when the full artifact exceeds the tier's budget; use the full artifact when it fits.**

**Files:**
- Modify: `src/workflows/engine/hooks.py:453-460`
- Test: `tests/test_iteration_exhaustion.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_iteration_exhaustion.py`:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


class TestSummaryFirstFetching(unittest.TestCase):
    """Downstream steps should prefer summaries when full artifact exceeds budget."""

    def test_summary_used_when_artifact_exceeds_budget(self):
        """When full artifact > tier budget, summary should be used instead."""
        from src.workflows.engine.artifacts import CONTEXT_BUDGETS

        full_content = "## Full Research\n\n" + "X" * 10000  # 10k chars
        summary_content = "## Summary\n\nKey findings: X, Y, Z."  # 40 chars

        # reference tier budget is 3000 — full_content (10k) exceeds it
        budget = CONTEXT_BUDGETS["reference"]
        self.assertGreater(len(full_content), budget)
        self.assertLess(len(summary_content), budget)

        # The logic: if len(full) > budget and summary exists, use summary
        should_use_summary = len(full_content) > budget and summary_content
        self.assertTrue(should_use_summary)

    def test_full_artifact_used_when_fits_budget(self):
        """When full artifact fits in tier budget, use it directly."""
        from src.workflows.engine.artifacts import CONTEXT_BUDGETS

        full_content = "## Short Report\n\nDone."  # 23 chars
        summary_content = "Short report done."

        budget = CONTEXT_BUDGETS["primary"]  # 8000
        self.assertLess(len(full_content), budget)

        # The logic: if len(full) <= budget, use full even if summary exists
        should_use_summary = len(full_content) > budget and summary_content
        self.assertFalse(should_use_summary)

    def test_full_artifact_used_when_no_summary(self):
        """When no summary exists, fall back to full artifact (truncated by formatter)."""
        full_content = "## Big Report\n\n" + "Y" * 10000
        summary_content = None

        budget = 3000
        should_use_summary = len(full_content) > budget and summary_content
        self.assertFalse(should_use_summary)  # no summary, must use full
```

- [ ] **Step 2: Run tests**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -m pytest tests/test_iteration_exhaustion.py::TestSummaryFirstFetching -v`
Expected: PASS (logic tests)

- [ ] **Step 3: Implement summary-first fetching in pre_execute_workflow_step**

In `src/workflows/engine/hooks.py`, find the artifact fetching block in `pre_execute_workflow_step` (around lines 456-460):

```python
    # Fetch artifacts from store
    store = get_artifact_store()
    artifact_contents: dict[str, Optional[str]] = {}
    if mission_id is not None and input_artifact_names:
        artifact_contents = await store.collect(mission_id, input_artifact_names)
```

Replace with:

```python
    # Fetch artifacts from store — prefer summaries when full artifact
    # exceeds the tier budget (summaries preserve meaning better than
    # blind truncation).
    store = get_artifact_store()
    artifact_contents: dict[str, Optional[str]] = {}
    if mission_id is not None and input_artifact_names:
        context_strategy = ctx.get("context_strategy")
        for name in input_artifact_names:
            full = await store.retrieve(mission_id, name)
            if full is None:
                artifact_contents[name] = None
                continue

            # Determine this artifact's tier budget
            budget = CONTEXT_BUDGETS["default"]
            if isinstance(context_strategy, dict):
                for tier in _TIER_ORDER:
                    if name in context_strategy.get(tier, []):
                        budget = CONTEXT_BUDGETS[tier]
                        break

            # Use summary if full artifact exceeds the tier budget
            if len(full) > budget:
                summary = await store.retrieve(mission_id, f"{name}_summary")
                if summary:
                    artifact_contents[name] = summary
                    continue

            artifact_contents[name] = full
```

Also add the imports at the top of the function (or ensure they're available). `CONTEXT_BUDGETS` and `_TIER_ORDER` are already importable from `artifacts.py` — add to the import at the top of `hooks.py`:

Find the existing import (it should already import `ArtifactStore` and `format_artifacts_for_prompt`). Add `CONTEXT_BUDGETS` and `_TIER_ORDER`:

```python
from .artifacts import ArtifactStore, format_artifacts_for_prompt, CONTEXT_BUDGETS, _TIER_ORDER
```

If the import is split across the file (lazy imports), add `CONTEXT_BUDGETS` and `_TIER_ORDER` where `format_artifacts_for_prompt` is imported.

- [ ] **Step 4: Run all tests**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -m pytest tests/test_iteration_exhaustion.py tests/test_workflow_hooks.py -v`
Expected: ALL PASS

- [ ] **Step 5: Verify import**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -c "from src.workflows.engine.hooks import pre_execute_workflow_step; print('OK')"`

- [ ] **Step 6: Commit**

```bash
git add src/workflows/engine/hooks.py src/workflows/engine/artifacts.py tests/test_iteration_exhaustion.py
git commit -m "feat: summary-first artifact fetching — use summaries when full artifact exceeds tier budget"
```

---

### Task 6: Add `smart_search` to CACHEABLE_READ_TOOLS

Minor efficiency fix — prevents wasting iterations on duplicate searches within a single agent execution.

**Files:**
- Modify: `src/agents/base.py:59-63`

- [ ] **Step 1: Add smart_search to the frozenset**

In `src/agents/base.py`, find `CACHEABLE_READ_TOOLS` (line 59):

```python
CACHEABLE_READ_TOOLS: frozenset[str] = frozenset({
    "read_file", "file_tree", "git_status", "git_log", "git_diff",
    "web_search", "extract_url", "read_pdf", "read_docx",
    "read_spreadsheet", "extract_text",
})
```

Replace with:

```python
CACHEABLE_READ_TOOLS: frozenset[str] = frozenset({
    "read_file", "file_tree", "git_status", "git_log", "git_diff",
    "web_search", "smart_search", "extract_url", "read_pdf", "read_docx",
    "read_spreadsheet", "extract_text",
})
```

- [ ] **Step 2: Verify import**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -c "from src.agents.base import CACHEABLE_READ_TOOLS; assert 'smart_search' in CACHEABLE_READ_TOOLS; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add src/agents/base.py
git commit -m "fix: add smart_search to CACHEABLE_READ_TOOLS to prevent duplicate searches"
```

---

### Task 7: Add Missing `update_task_by_context_field` to db.py

The conditional group logic in hooks.py tries to skip excluded workflow steps by calling `update_task_by_context_field`, but this function was never implemented in `db.py`. The import fails silently and excluded steps are never skipped, causing the workflow to execute steps that should be bypassed.

The function needs to: find tasks by `mission_id` where a JSON field in the `context` column matches a value, and update their status.

**Files:**
- Modify: `src/infra/db.py` (add after `update_task` around line 1076)
- Test: `tests/test_iteration_exhaustion.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_iteration_exhaustion.py`:

```python
class TestUpdateTaskByContextField(unittest.TestCase):
    """Tests for update_task_by_context_field DB function."""

    def test_function_exists(self):
        """The function should be importable from db."""
        from src.infra.db import update_task_by_context_field
        self.assertTrue(callable(update_task_by_context_field))

    def test_propagate_skips_exists(self):
        """propagate_skips should already exist."""
        from src.infra.db import propagate_skips
        self.assertTrue(callable(propagate_skips))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -m pytest tests/test_iteration_exhaustion.py::TestUpdateTaskByContextField::test_function_exists -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement the function**

In `src/infra/db.py`, add after the `update_task` function (around line 1076):

```python
async def update_task_by_context_field(
    mission_id: int, field: str, value: str, **kwargs
):
    """Update tasks matching a JSON context field within a mission.

    Uses SQLite's json_extract to find tasks where
    ``context->>'$.{field}' = value`` and applies the given updates.

    Example::

        await update_task_by_context_field(
            mission_id=30,
            field="workflow_step_id",
            value="1.3",
            status="skipped",
        )
    """
    _validate_columns(kwargs, _TASK_COLUMNS, "tasks")
    db = await get_db()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [mission_id, value]
    await db.execute(
        f"UPDATE tasks SET {sets} "
        f"WHERE mission_id = ? AND json_extract(context, '$.{field}') = ?",
        values,
    )
    await db.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -m pytest tests/test_iteration_exhaustion.py::TestUpdateTaskByContextField -v`
Expected: ALL PASS

- [ ] **Step 5: Verify the import from hooks works**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -c "from src.infra.db import update_task_by_context_field, propagate_skips; print('OK')"`

- [ ] **Step 6: Commit**

```bash
git add src/infra/db.py tests/test_iteration_exhaustion.py
git commit -m "feat: add update_task_by_context_field for workflow conditional group skipping"
```

---

### Task 8: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -m pytest tests/test_iteration_exhaustion.py tests/test_workflow_hooks.py -v`
Expected: ALL PASS

- [ ] **Step 2: Run import checks on all modified files**

```bash
cd /c/Users/sakir/Dropbox/Workspaces/kutay
python -c "from src.models.models import validate_task_output; print('models OK')"
python -c "from src.agents.base import BaseAgent; print('base OK')"
python -c "from src.core.orchestrator import Orchestrator; print('orchestrator OK')"
python -c "from src.workflows.engine.hooks import validate_artifact_schema; print('hooks OK')"
```

- [ ] **Step 3: Run broader test suite for regressions**

```bash
cd /c/Users/sakir/Dropbox/Workspaces/kutay && python -m pytest tests/ -x --timeout=30 -q
```
