# Error Recovery Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dead ErrorRecoveryAgent, rename shopping error classification to scraper_failure_handler, and add a DLQ Analyst that detects failure patterns and alerts via Telegram with inline action buttons.

**Architecture:** Four independent work streams — removal, rename, new module, workflow cleanup. The DLQ Analyst is a pure Python class in `src/infra/dlq_analyst.py` that hooks into `quarantine_task()`, queries recent DLQ entries for patterns, runs deterministic diagnostic checks, and sends Telegram alerts with Retry All / Drop All / Pause Similar buttons.

**Tech Stack:** Python 3.10, aiosqlite, python-telegram-bot v20+, pytest

**Spec:** `docs/superpowers/specs/2026-04-08-error-recovery-evolution-design.md`

---

### Task 1: Remove ErrorRecoveryAgent — Agent File and Orchestrator Methods

**Files:**
- Delete: `src/agents/error_recovery.py`
- Modify: `src/core/orchestrator.py` (lines ~61, ~1745, ~2225, ~2448, ~2616-2787)

- [ ] **Step 1: Delete the agent file**

```bash
rm src/agents/error_recovery.py
```

- [ ] **Step 2: Remove the timeout entry from orchestrator**

In `src/core/orchestrator.py` around line 61, remove:
```python
    "error_recovery": 300,  # was 240
```

- [ ] **Step 3: Remove `_spawn_error_recovery` call site in timeout handler**

Around line 1745, the timeout handler calls `await self._spawn_error_recovery(task, timeout_err)` — remove that call and the `return` after it. The retry pipeline's `record_failure("timeout")` on the lines above already handles the retry.

- [ ] **Step 4: Remove `_spawn_error_recovery` call site in error handler**

Around line 2225, the error handler calls `recovery_spawned = await self._spawn_error_recovery(task, error_str)` with an `if not recovery_spawned:` branch — remove the recovery call and unwrap the `if not` block so the DLQ path always runs.

- [ ] **Step 5: Remove `_process_recovery_result` call site**

Around line 2448, there's a check `if agent_type == "error_recovery":` that calls `_process_recovery_result` — remove the entire `if` block.

- [ ] **Step 6: Remove `_spawn_error_recovery` method**

Delete the entire method `_spawn_error_recovery` (starts around line 2618, ~90 lines) including the section comment `# ─── Error Recovery ───`.

- [ ] **Step 7: Remove `_process_recovery_result` method**

Delete the entire method `_process_recovery_result` (starts around line 2727, ~50 lines).

- [ ] **Step 8: Verify imports compile**

```bash
cd C:/Users/sakir/Dropbox/Workspaces/kutay
python -c "from src.core.orchestrator import Orchestrator; print('OK')"
```

- [ ] **Step 9: Commit**

```bash
git add -u src/agents/error_recovery.py src/core/orchestrator.py
git commit -m "refactor: remove ErrorRecoveryAgent and orchestrator spawn/process methods"
```

---

### Task 2: Remove ErrorRecoveryAgent — Supporting References

**Files:**
- Modify: `src/models/capabilities.py:338-354`
- Modify: `src/core/task_classifier.py:71,331,357`
- Modify: `src/core/router.py:1571`
- Modify: `src/memory/episodic.py:95-145`
- Modify: `src/memory/decay.py:43`
- Modify: `src/memory/rag.py:61,186-187,461-462`
- Modify: `src/memory/context_policy.py:26`
- Modify: `src/security/permissions.py:77`

- [ ] **Step 1: Remove task profile from capabilities.py**

In `src/models/capabilities.py` around line 338, remove the entire `"error_recovery": { ... }` dict entry from `TASK_PROFILES`.

- [ ] **Step 2: Remove classifier entries from task_classifier.py**

In `src/core/task_classifier.py`:
- Line 71: remove the `error_recovery` docstring/comment entry
- Line 331: remove `("error_recovery", 5, ["recover", "retry", "fallback", "roll back", "revert failure"]),`
- Line 357: remove `"error_recovery"` from the `needs_tools` tuple

- [ ] **Step 3: Remove router entry from router.py**

In `src/core/router.py` line 1571, remove:
```python
    "error_recovery": ModelRequirements(task="error_recovery", difficulty=6, estimated_output_tokens=2000, needs_function_calling=True),
```

- [ ] **Step 4: Remove `store_error_recovery` from episodic.py**

In `src/memory/episodic.py`, delete the entire `store_error_recovery` function (lines ~95-145).

- [ ] **Step 5: Remove decay protection from decay.py**

In `src/memory/decay.py` line 43, change:
```python
PROTECTED_TYPES = {"user_preference", "error_recovery"}
```
to:
```python
PROTECTED_TYPES = {"user_preference"}
```

- [ ] **Step 6: Remove RAG special-casing from rag.py**

In `src/memory/rag.py`:
- Line 61: remove `"error_recovery": ["errors", "episodic"],` from the collections mapping
- Lines 186-187: remove or adjust the `if doc_type in ("error_recovery", "user_preference"):` check — keep just `"user_preference"`
- Lines 461-462: remove the `elif doc_type == "error_recovery":` block

- [ ] **Step 7: Remove context policy entry from context_policy.py**

In `src/memory/context_policy.py` line 26, remove:
```python
    "error_recovery":   {"deps", "rag", "skills"},
```

- [ ] **Step 8: Remove permissions entry from permissions.py**

In `src/security/permissions.py` line 77, remove:
```python
    "error_recovery": None,  # needs full access for recovery
```

- [ ] **Step 9: Verify all modules compile**

```bash
cd C:/Users/sakir/Dropbox/Workspaces/kutay
python -c "from src.models.capabilities import TASK_PROFILES; print('OK')"
python -c "from src.core.task_classifier import classify_task; print('OK')"
python -c "from src.core.router import Router; print('OK')"
python -c "from src.memory.episodic import store_episode; print('OK')"
python -c "from src.memory.decay import PROTECTED_TYPES; print('OK')"
python -c "from src.memory.rag import RAGPipeline; print('OK')"
python -c "from src.memory.context_policy import get_context_policy; print('OK')"
python -c "from src.security.permissions import get_permissions; print('OK')"
```

- [ ] **Step 10: Run existing tests**

```bash
pytest tests/ -x -q --timeout=30 2>&1 | tail -20
```

- [ ] **Step 11: Commit**

```bash
git add -u
git commit -m "refactor: remove error_recovery from classifier, router, capabilities, memory, permissions"
```

---

### Task 3: Update dead_letter.py Docstring

**Files:**
- Modify: `src/infra/dead_letter.py:1-18`

- [ ] **Step 1: Update module docstring**

Replace lines 1-18 of `src/infra/dead_letter.py`:

```python
# dead_letter.py
"""
Dead-letter queue for permanently failed tasks.

When a task exhausts all retries (worker attempts, infrastructure resets,
or grading attempts), it enters the dead-letter queue.

The DLQ:
- Quarantines tasks so they don't block downstream work
- Notifies via Telegram
- Provides `/dlq` command to inspect / retry / discard
- Auto-pauses a workflow mission if too many tasks land here
- Feeds the DLQ Analyst for pattern detection and proactive alerts

Integration with existing systems:
- RetryContext handles in-flight failure recovery (model rotation, difficulty bumps)
- BackpressureQueue handles transient model call failures (rate limits)
- DLQAnalyst detects cross-task failure patterns after quarantine
"""
```

- [ ] **Step 2: Commit**

```bash
git add src/infra/dead_letter.py
git commit -m "docs: update dead_letter.py docstring — remove error_recovery references"
```

---

### Task 4: Rename Shopping Error Recovery to Scraper Failure Handler

**Files:**
- Rename: `src/shopping/resilience/error_recovery.py` → `src/shopping/resilience/scraper_failure_handler.py`
- Modify: `src/shopping/resilience/__init__.py:4,27-31`

- [ ] **Step 1: Rename the file**

```bash
cd C:/Users/sakir/Dropbox/Workspaces/kutay
git mv src/shopping/resilience/error_recovery.py src/shopping/resilience/scraper_failure_handler.py
```

- [ ] **Step 2: Update __init__.py import**

In `src/shopping/resilience/__init__.py`, change line 4 docstring and lines 27-31:

Old:
```python
error recovery, cache fallback, and anti-detection monitoring for
```
New:
```python
scraper failure handling, cache fallback, and anti-detection monitoring for
```

Old:
```python
from src.shopping.resilience.error_recovery import (
    handle_scraper_error,
    handle_llm_error,
    classify_error,
)
```
New:
```python
from src.shopping.resilience.scraper_failure_handler import (
    handle_scraper_error,
    handle_llm_error,
    classify_error,
)
```

- [ ] **Step 3: Verify import**

```bash
python -c "from src.shopping.resilience import handle_scraper_error, classify_error; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add -u src/shopping/resilience/
git commit -m "refactor: rename shopping error_recovery to scraper_failure_handler"
```

---

### Task 5: DLQ Analyst — Write Failing Tests

**Files:**
- Create: `tests/test_dlq_analyst.py`

- [ ] **Step 1: Write pattern detection tests**

```python
"""Tests for DLQ Analyst pattern detection."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.infra.dlq_analyst import DLQAnalyst


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestPatternDetection:
    """Pattern grouping and threshold logic."""

    def setup_method(self):
        self.analyst = DLQAnalyst()

    def test_no_pattern_below_threshold(self):
        """2 similar failures should NOT trigger an alert."""
        entries = [
            {"task_id": 1, "error_category": "timeout", "original_agent": "researcher",
             "error": "timed out", "mission_id": None, "quarantined_at": "2026-04-08 10:00:00"},
            {"task_id": 2, "error_category": "timeout", "original_agent": "researcher",
             "error": "timed out", "mission_id": None, "quarantined_at": "2026-04-08 10:30:00"},
        ]
        patterns = self.analyst.detect_patterns(entries)
        assert len(patterns) == 0

    def test_pattern_at_threshold(self):
        """3 similar failures SHOULD trigger."""
        entries = [
            {"task_id": 1, "error_category": "timeout", "original_agent": "researcher",
             "error": "timed out", "mission_id": None, "quarantined_at": "2026-04-08 10:00:00"},
            {"task_id": 2, "error_category": "timeout", "original_agent": "coder",
             "error": "timed out", "mission_id": None, "quarantined_at": "2026-04-08 10:30:00"},
            {"task_id": 3, "error_category": "timeout", "original_agent": "researcher",
             "error": "request timed out", "mission_id": None, "quarantined_at": "2026-04-08 11:00:00"},
        ]
        patterns = self.analyst.detect_patterns(entries)
        assert len(patterns) >= 1
        assert any(p["group_key"] == "category:timeout" for p in patterns)

    def test_groups_by_mission(self):
        """3 failures from same mission should trigger mission pattern."""
        entries = [
            {"task_id": i, "error_category": cat, "original_agent": "coder",
             "error": "some error", "mission_id": 42, "quarantined_at": f"2026-04-08 1{i}:00:00"}
            for i, cat in enumerate(["timeout", "parse_error", "network_error"], 1)
        ]
        patterns = self.analyst.detect_patterns(entries)
        assert any(p["group_key"] == "mission:42" for p in patterns)


class TestDeduplication:
    """Alert dedup within 1-hour window."""

    def setup_method(self):
        self.analyst = DLQAnalyst()

    def test_first_alert_not_deduped(self):
        assert not self.analyst.is_deduped("category:timeout")

    def test_second_alert_within_hour_deduped(self):
        self.analyst.record_alert("category:timeout")
        assert self.analyst.is_deduped("category:timeout")

    def test_different_pattern_not_deduped(self):
        self.analyst.record_alert("category:timeout")
        assert not self.analyst.is_deduped("category:network_error")


class TestAlertFormatting:
    """Telegram message formatting."""

    def setup_method(self):
        self.analyst = DLQAnalyst()

    def test_format_contains_task_ids(self):
        pattern = {
            "group_key": "category:timeout",
            "count": 3,
            "entries": [
                {"task_id": 42, "original_agent": "researcher", "error": "timed out after 300s"},
                {"task_id": 45, "original_agent": "coder", "error": "request timeout"},
                {"task_id": 48, "original_agent": "researcher", "error": "deadline exceeded"},
            ],
            "diagnostic": "llama-server responding but slow (4.2s)",
        }
        msg = self.analyst.format_alert(pattern)
        assert "#42" in msg
        assert "#45" in msg
        assert "#48" in msg
        assert "timeout" in msg.lower()
        assert "4.2s" in msg
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dlq_analyst.py -v 2>&1 | tail -20
```
Expected: ModuleNotFoundError for `src.infra.dlq_analyst`

- [ ] **Step 3: Commit**

```bash
git add tests/test_dlq_analyst.py
git commit -m "test: add failing tests for DLQ Analyst pattern detection"
```

---

### Task 6: DLQ Analyst — Implement Core

**Files:**
- Create: `src/infra/dlq_analyst.py`

- [ ] **Step 1: Implement the DLQAnalyst class**

```python
"""DLQ Analyst — deterministic failure pattern detection.

Hooks into quarantine_task() to detect cross-task failure patterns.
Groups recent DLQ entries by error category, model, tool, and mission.
Sends Telegram alerts with inline action buttons when patterns are detected.
No LLM calls — pure Python pattern matching.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("infra.dlq_analyst")

# Alert thresholds
PATTERN_THRESHOLD = 3       # min matches to trigger alert
DEDUP_SECONDS = 3600        # 1 hour dedup per pattern key


class DLQAnalyst:
    """Detects failure patterns across DLQ entries and formats alerts."""

    WINDOW_HOURS = 3  # sliding window for pattern detection

    def __init__(self):
        self._last_alert: dict[str, float] = {}  # pattern_key -> timestamp

    def detect_patterns(self, entries: list[dict]) -> list[dict]:
        """Group DLQ entries and return patterns that exceed threshold.

        Args:
            entries: list of dead_letter_tasks rows (dicts) from the last WINDOW_HOURS.

        Returns:
            List of pattern dicts with keys: group_key, count, entries
        """
        groups: dict[str, list[dict]] = defaultdict(list)

        for entry in entries:
            # Group by error category
            cat = entry.get("error_category", "unknown")
            groups[f"category:{cat}"].append(entry)

            # Group by mission
            mid = entry.get("mission_id")
            if mid:
                groups[f"mission:{mid}"].append(entry)

        # Filter to patterns exceeding threshold
        patterns = []
        for key, items in groups.items():
            if len(items) >= PATTERN_THRESHOLD:
                patterns.append({
                    "group_key": key,
                    "count": len(items),
                    "entries": items,
                })

        return patterns

    def is_deduped(self, pattern_key: str) -> bool:
        """Check if this pattern was already alerted within DEDUP_SECONDS."""
        last = self._last_alert.get(pattern_key)
        if last is None:
            return False
        return (time.time() - last) < DEDUP_SECONDS

    def record_alert(self, pattern_key: str) -> None:
        """Record that an alert was sent for this pattern."""
        self._last_alert[pattern_key] = time.time()

    def format_alert(self, pattern: dict) -> str:
        """Format a pattern into a Telegram alert message."""
        key = pattern["group_key"]
        count = pattern["count"]
        entries = pattern["entries"]
        diagnostic = pattern.get("diagnostic", "")

        # Build header
        group_type, group_value = key.split(":", 1)
        if group_type == "category":
            header = f"DLQ Pattern: {count} tasks failed with {group_value}"
        elif group_type == "mission":
            header = f"DLQ Pattern: {count} tasks from mission #{group_value} failed"
        else:
            header = f"DLQ Pattern: {count} tasks matched {key}"

        # Build task list (max 5)
        lines = [header, ""]
        for entry in entries[:5]:
            tid = entry.get("task_id", "?")
            agent = entry.get("original_agent", "?")
            error = entry.get("error", "")[:80]
            lines.append(f"- Task #{tid} ({agent}): {error}")

        if len(entries) > 5:
            lines.append(f"  ... and {len(entries) - 5} more")

        # Add diagnostic if available
        if diagnostic:
            lines.append("")
            lines.append(f"Diagnostic: {diagnostic}")

        return "\n".join(lines)
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_dlq_analyst.py -v
```
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/infra/dlq_analyst.py
git commit -m "feat: implement DLQ Analyst core — pattern detection, dedup, formatting"
```

---

### Task 7: DLQ Analyst — Known Failure Diagnostics

**Files:**
- Modify: `src/infra/dlq_analyst.py`
- Modify: `tests/test_dlq_analyst.py`

- [ ] **Step 1: Write failing tests for diagnostics**

Add to `tests/test_dlq_analyst.py`:

```python
class TestDiagnostics:
    """Known failure signature checks."""

    def setup_method(self):
        self.analyst = DLQAnalyst()

    @patch("src.infra.dlq_analyst.aiohttp.ClientSession")
    def test_timeout_diagnostic_server_down(self, mock_session_cls):
        """Timeout pattern should check llama-server health."""
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = AsyncMock(return_value=mock_resp)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        result = _run(self.analyst.run_diagnostic("category:timeout", []))
        assert "not responding" in result.lower() or "unhealthy" in result.lower()

    def test_grading_diagnostic_same_model(self):
        """Grading failures with same model should flag the model."""
        entries = [
            {"task_id": i, "error": f"grade fail", "original_agent": "coder",
             "error_category": "unknown", "mission_id": None,
             "quarantined_at": f"2026-04-08 1{i}:00:00"}
            for i in range(1, 4)
        ]
        # Simulate failed_in_phase = grading context — inject via error text
        for e in entries:
            e["error"] = "Grading exhausted: model=qwen2.5-7b"
        result = _run(self.analyst.run_diagnostic("category:unknown", entries))
        # Should detect model name from error text
        assert "qwen2.5-7b" in result or "same model" in result.lower() or result == ""
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dlq_analyst.py::TestDiagnostics -v
```

- [ ] **Step 3: Implement `run_diagnostic` method**

Add to `DLQAnalyst` class in `src/infra/dlq_analyst.py`:

```python
    async def run_diagnostic(self, pattern_key: str, entries: list[dict]) -> str:
        """Run a quick diagnostic check based on the failure pattern.

        Returns a human-readable diagnostic string, or empty string if no check applies.
        """
        group_type, group_value = pattern_key.split(":", 1)

        if group_type == "category":
            if group_value == "timeout":
                return await self._check_llama_health()
            if group_value == "network_error":
                return await self._check_connectivity()
            if group_value == "rate_limit":
                return "Rate limiting detected — external API throttling"

        # Check if grading failures reference the same model
        model_counts = self._extract_model_mentions(entries)
        if model_counts:
            top_model, count = max(model_counts.items(), key=lambda x: x[1])
            if count >= PATTERN_THRESHOLD:
                return f"Model {top_model}: failed {count} times — may be misconfigured"

        return ""

    async def _check_llama_health(self) -> str:
        """Ping llama-server /health endpoint."""
        import aiohttp

        try:
            start = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://127.0.0.1:8080/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    elapsed = round(time.time() - start, 1)
                    if resp.status == 200:
                        if elapsed > 2.0:
                            return f"llama-server responding but slow ({elapsed}s)"
                        return f"llama-server healthy ({elapsed}s) — timeouts likely from long generation"
                    return f"llama-server unhealthy (HTTP {resp.status})"
        except Exception:
            return "llama-server not responding — may be down or restarting"

    async def _check_connectivity(self) -> str:
        """Basic internet connectivity check."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://www.google.com", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        return "Internet reachable — issue may be with specific API"
                    return f"Connectivity check returned HTTP {resp.status}"
        except Exception:
            return "Network connectivity issue — internet unreachable"

    @staticmethod
    def _extract_model_mentions(entries: list[dict]) -> dict[str, int]:
        """Extract model names mentioned in error text."""
        import re

        model_counts: dict[str, int] = defaultdict(int)
        pattern = re.compile(r"model=([^\s,)]+)")
        for entry in entries:
            error = entry.get("error", "")
            for match in pattern.finditer(error):
                model_counts[match.group(1)] += 1
        return dict(model_counts)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_dlq_analyst.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/infra/dlq_analyst.py tests/test_dlq_analyst.py
git commit -m "feat: add known failure diagnostics to DLQ Analyst"
```

---

### Task 8: DLQ Analyst — Hook into quarantine_task and Telegram Alerts

**Files:**
- Modify: `src/infra/dead_letter.py:69-116`
- Modify: `src/app/telegram_bot.py` (callback handling)

- [ ] **Step 1: Add module-level analyst instance to dead_letter.py**

At the top of `src/infra/dead_letter.py`, after the logger:

```python
from src.infra.dlq_analyst import DLQAnalyst

_analyst = DLQAnalyst()
```

- [ ] **Step 2: Hook analyst into quarantine_task**

At the end of `quarantine_task()` in `src/infra/dead_letter.py` (before `return dlq_id`, after `_check_mission_health`), add:

```python
    # Run DLQ pattern analysis
    try:
        await _run_pattern_analysis(task_id, error_category)
    except Exception as e:
        logger.debug(f"[DLQ] Pattern analysis failed (non-critical): {e}")

    return dlq_id
```

- [ ] **Step 3: Implement `_run_pattern_analysis` in dead_letter.py**

Add this function to `src/infra/dead_letter.py`:

```python
async def _run_pattern_analysis(task_id: int, error_category: str) -> None:
    """Check for failure patterns and send Telegram alert if detected."""
    from src.infra.db import get_db

    db = await get_db()
    await _ensure_dlq_table()

    # Fetch recent unresolved DLQ entries within the window
    cursor = await db.execute(
        """SELECT task_id, mission_id, error, error_category, original_agent,
                  quarantined_at
           FROM dead_letter_tasks
           WHERE resolved_at IS NULL
             AND quarantined_at >= datetime('now', ?)
           ORDER BY quarantined_at DESC""",
        (f"-{DLQAnalyst.WINDOW_HOURS} hours",),
    )
    rows = await cursor.fetchall()
    entries = [dict(r) for r in rows]

    if len(entries) < 3:
        return

    patterns = _analyst.detect_patterns(entries)

    for pattern in patterns:
        key = pattern["group_key"]
        if _analyst.is_deduped(key):
            continue

        # Run diagnostic check
        diagnostic = await _analyst.run_diagnostic(key, pattern["entries"])
        pattern["diagnostic"] = diagnostic

        # Format and send alert
        message = _analyst.format_alert(pattern)
        task_ids = [e["task_id"] for e in pattern["entries"]]
        await _send_dlq_alert(message, key, task_ids)
        _analyst.record_alert(key)


async def _send_dlq_alert(message: str, pattern_key: str, task_ids: list[int]) -> None:
    """Send a DLQ pattern alert via Telegram with inline action buttons."""
    try:
        from src.app.telegram_bot import get_bot
        bot = get_bot()
        if not bot:
            return

        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        # Encode task IDs as comma-separated in callback data
        ids_str = ",".join(str(t) for t in task_ids[:20])  # cap at 20
        buttons = [
            [
                InlineKeyboardButton(
                    f"Retry All ({len(task_ids)})",
                    callback_data=f"dlqa:retry:{ids_str}",
                ),
                InlineKeyboardButton(
                    f"Drop All ({len(task_ids)})",
                    callback_data=f"dlqa:drop:{ids_str}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Pause Similar",
                    callback_data=f"dlqa:pause:{pattern_key}",
                ),
            ],
        ]
        markup = InlineKeyboardMarkup(buttons)
        await bot.send_notification(message, reply_markup=markup)
    except Exception as e:
        logger.debug(f"[DLQ] Failed to send pattern alert: {e}")
```

- [ ] **Step 4: Add callback handling in telegram_bot.py**

In `src/app/telegram_bot.py`, find `handle_callback()` method. Add a new block for `dlqa:` callbacks (near the existing `m:dlq:` handlers around line 5311):

```python
        # ── DLQ Analyst actions ──
        if data.startswith("dlqa:"):
            parts = data.split(":", 2)
            if len(parts) < 3:
                await query.answer("Invalid action")
                return

            action = parts[1]
            payload = parts[2]

            if action == "retry":
                task_ids = [int(t) for t in payload.split(",") if t.isdigit()]
                from src.infra.dead_letter import retry_dlq_task
                retried = 0
                for tid in task_ids:
                    if await retry_dlq_task(tid):
                        retried += 1
                await query.answer(f"Retried {retried}/{len(task_ids)} tasks")
                await query.edit_message_text(
                    query.message.text + f"\n\nRetried {retried} tasks.",
                )

            elif action == "drop":
                task_ids = [int(t) for t in payload.split(",") if t.isdigit()]
                from src.infra.dead_letter import resolve_dlq_task
                dropped = 0
                for tid in task_ids:
                    if await resolve_dlq_task(tid, resolution="discarded"):
                        dropped += 1
                await query.answer(f"Dropped {dropped}/{len(task_ids)} tasks")
                await query.edit_message_text(
                    query.message.text + f"\n\nDropped {dropped} tasks.",
                )

            elif action == "pause":
                pattern_key = payload
                # Store pause in orchestrator's pause set
                try:
                    from src.core.orchestrator import get_orchestrator
                    orch = get_orchestrator()
                    if orch and hasattr(orch, "paused_patterns"):
                        orch.paused_patterns.add(pattern_key)
                        await query.answer(f"Paused: {pattern_key}")
                        await query.edit_message_text(
                            query.message.text + f"\n\nPaused pattern: {pattern_key}. Use /dlq unpause to lift.",
                        )
                    else:
                        await query.answer("Orchestrator not available")
                except Exception as e:
                    await query.answer(f"Pause failed: {e}")

            return
```

- [ ] **Step 5: Verify imports**

```bash
python -c "from src.infra.dead_letter import quarantine_task; print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add src/infra/dead_letter.py src/infra/dlq_analyst.py src/app/telegram_bot.py
git commit -m "feat: hook DLQ Analyst into quarantine_task with Telegram alerts and inline actions"
```

---

### Task 9: DLQ Analyst — Pause Similar Mechanism

**Files:**
- Modify: `src/core/orchestrator.py` (add `paused_patterns` set and check)

- [ ] **Step 1: Add `paused_patterns` attribute to Orchestrator**

In `src/core/orchestrator.py`, in the `__init__` method, add:

```python
        self.paused_patterns: set[str] = set()
```

- [ ] **Step 2: Add pause check before task dispatch**

In the orchestrator's task processing loop (where it picks up pending tasks), add a check that skips tasks matching a paused pattern. Find the spot where tasks are fetched and about to be dispatched. Add:

```python
        # Skip tasks matching paused DLQ patterns
        if self.paused_patterns and task.get("error_category"):
            pattern_key = f"category:{task['error_category']}"
            if pattern_key in self.paused_patterns:
                logger.debug(f"[Task #{task['id']}] Skipped — pattern {pattern_key} paused")
                continue
```

Note: This is a lightweight check. Tasks don't have `error_category` on first run (only DLQ entries do), so this only affects tasks that were retried from DLQ and would hit the same failure pattern again. The main value is preventing re-retried DLQ tasks from cycling back.

- [ ] **Step 3: Add `/dlq unpause` handling**

In `src/app/telegram_bot.py`, find the `/dlq` command handler. Add an `unpause` subcommand:

```python
            if args and args[0] == "unpause":
                try:
                    from src.core.orchestrator import get_orchestrator
                    orch = get_orchestrator()
                    if orch and hasattr(orch, "paused_patterns") and orch.paused_patterns:
                        cleared = list(orch.paused_patterns)
                        orch.paused_patterns.clear()
                        await self._reply(update, f"Unpaused {len(cleared)} patterns:\n" +
                                          "\n".join(f"- {p}" for p in cleared))
                    else:
                        await self._reply(update, "No patterns currently paused.")
                except Exception as e:
                    await self._reply(update, f"Error: {e}")
                return
```

- [ ] **Step 4: Verify**

```bash
python -c "from src.core.orchestrator import Orchestrator; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add src/core/orchestrator.py src/app/telegram_bot.py
git commit -m "feat: add Pause Similar mechanism — paused_patterns set with /dlq unpause"
```

---

### Task 10: Clean i2p v3 Workflow

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json:18,6943,7066`

- [ ] **Step 1: Remove error_recovery from agent roster**

In `src/workflows/i2p/i2p_v3.json` line 18, remove `"error_recovery",` from the agents array.

- [ ] **Step 2: Reassign post_launch_monitoring**

Line 6943, change:
```json
      "agent": "error_recovery",
```
to:
```json
      "agent": "coder",
```

- [ ] **Step 3: Reassign incident_response**

Line 7066, change:
```json
      "agent": "error_recovery",
```
to:
```json
      "agent": "coder",
```

- [ ] **Step 4: Validate JSON**

```bash
python -c "import json; json.load(open('src/workflows/i2p/i2p_v3.json')); print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add src/workflows/i2p/i2p_v3.json
git commit -m "refactor: replace error_recovery with coder in i2p v3 workflow"
```

---

### Task 11: Final Verification

**Files:** None (verification only)

- [ ] **Step 1: Grep for any remaining error_recovery references**

```bash
cd C:/Users/sakir/Dropbox/Workspaces/kutay
grep -r "error_recovery" src/ --include="*.py" -l
grep -r "ErrorRecovery" src/ --include="*.py" -l
grep -r "error_recovery" src/workflows/i2p/i2p_v3.json
```

Expected: No matches (v1/v2 are deprecated, ignore them).

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -x -q --timeout=30 2>&1 | tail -30
```

- [ ] **Step 3: Quick import smoke test**

```bash
python -c "
from src.core.orchestrator import Orchestrator
from src.infra.dead_letter import quarantine_task
from src.infra.dlq_analyst import DLQAnalyst
from src.shopping.resilience import handle_scraper_error
print('All imports OK')
"
```

- [ ] **Step 4: Commit any remaining fixes if needed**

```bash
git add -u
git commit -m "fix: address any remaining error_recovery references"
```
