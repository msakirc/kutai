# Error Recovery Evolution: Remove, Rename, Replace

**Date:** 2026-04-08
**Status:** Approved

## Problem

The `ErrorRecoveryAgent` was designed to diagnose and fix failing tasks using LLM reasoning. In practice, it never produced useful results — small local LLMs cannot reliably diagnose failures. Meanwhile, the retry pipeline (RetryContext, model rotation, difficulty bumps, exhaustion classification) now handles all in-flight failure recovery mechanically and reliably.

The system has two problems:
1. **Dead weight** — `ErrorRecoveryAgent` burns 300s of GPU time on every non-trivial task failure, producing nothing useful
2. **Terminology collision** — "error recovery" names both the useless LLM agent AND the useful shopping scraper error classifier, causing confusion

## Solution

Four work streams:

### 1. Remove ErrorRecoveryAgent (complete deletion)

Delete all code related to the LLM-based error recovery agent:

| File | What to remove |
|------|---------------|
| `src/agents/error_recovery.py` | Entire file |
| `src/core/orchestrator.py` | `_spawn_error_recovery()` method |
| `src/core/orchestrator.py` | `_process_recovery_result()` method |
| `src/core/orchestrator.py` | All call sites invoking these methods (exception handlers, timeout handlers) |
| `src/core/orchestrator.py` | `error_recovery` timeout entry (300s) |
| `src/models/capabilities.py` | `error_recovery` task profile |
| `src/core/task_classifier.py` | `error_recovery` keyword group and weight |
| `src/core/router.py` | `error_recovery` difficulty/routing entry |
| `src/memory/episodic.py` | `store_error_recovery()` method |
| `src/memory/decay.py` | `error_recovery` decay protection |
| `src/memory/rag.py` | `error_recovery` context retrieval special-casing |
| `src/memory/context_policy.py` | `error_recovery` context policy entry |
| `src/security/permissions.py` | `error_recovery` permissions entry |

**What happens on task failure after removal:** The retry pipeline handles it — `record_failure()` increments the appropriate counter (worker/infra), `compute_retry_timing()` decides immediate/delayed/terminal, and terminal failures go to DLQ via `quarantine_task()`. No behavior change except we stop wasting GPU.

### 2. Rename Shopping Error Classification

The shopping scraper error classifier is a completely different system — synchronous Python logic that classifies scraper errors (transient, rate_limit, blocked, parse_error, permanent) and returns recovery actions (retry/skip/fallback/abort). It works well and stays.

| Old | New |
|-----|-----|
| `src/shopping/resilience/error_recovery.py` | `src/shopping/resilience/scraper_failure_handler.py` |
| Class `ErrorRecovery` | Class `ScraperFailureHandler` |
| All imports referencing the old module/class | Updated to new names |

Functions `classify_error()`, `handle_scraper_error()`, `handle_llm_error()` keep their names — they're accurate.

### 3. Add DLQ Analyst

New module: `src/infra/dlq_analyst.py`

A pure Python class (NOT an agent, no LLM calls) that detects failure patterns across DLQ entries and alerts via Telegram.

#### Trigger

Event-driven — called from `quarantine_task()` in `dead_letter.py` every time a task enters DLQ.

#### Pattern Detection

Queries `dead_letter_tasks` table for entries in the last 3 hours. Groups by:
- **Tool name** — e.g. 3x `web_search` failures
- **Model name** — e.g. 3x failures on the same model
- **Error category** — e.g. 3x timeout
- **Mission ID** — e.g. a single mission bleeding tasks into DLQ

Threshold: 3 matches in a 3-hour sliding window triggers an alert.

#### Deduplication

Tracks last alert time per pattern key (in-memory dict). Same pattern won't re-alert within 1 hour.

#### Telegram Alert Format

```
DLQ Pattern Detected

3 tasks failed with timeout in the last 2h:
- Task #42: web_search -- "find coffee machines"
- Task #45: web_search -- "compare prices caffitaly"
- Task #48: web_search -- "best espresso under 5000tl"

Likely cause: web search API unavailable
```

#### Inline Action Buttons

- **Retry All (N)** — moves all matched tasks back to pending
- **Drop All (N)** — permanently removes from DLQ
- **Pause Similar** — sets a temporary block on new tasks that would hit the same failure pattern; lifted via `/dlq unpause` or an "Unpause" button sent with the pause confirmation

Callback handling: new entries in `telegram_bot.py` under `handle_callback()`.

#### Known Failure Signatures

Before alerting, the DLQ Analyst runs a quick diagnostic check based on the pattern's `error_category` and `failed_in_phase`. These are fast, deterministic, no-LLM checks that make the alert actionable:

| Pattern | Quick Check | Actionable Message |
|---------|------------|-------------------|
| 3x timeout | Ping llama-server `/health` | "llama-server not responding" or "responding but slow (Xs)" |
| 3x grading failure (`failed_in_phase=grading`) | Check if all used same grading model | "Model X can't grade these tasks -- try different model" |
| 3x same model failed | Check model's recent success rate | "Model X: 0/6 success in last 2h -- may be misconfigured" |
| 3x same tool (e.g. `web_search`) | Lightweight probe (test URL fetch) | "web_search API unreachable" or "API responding, likely prompt issue" |
| 3x `network_error` | Basic connectivity check | "Network connectivity issue detected" |

The diagnostic result is included in the Telegram alert message, below the task list.

#### Dependencies

None new. Uses existing `dead_letter_tasks` table, existing Telegram bot callback infrastructure.

### 4. Clean i2p v3 Workflow

Only v3 is active (v1 and v2 are deprecated).

**Agent roster (line 18):** Remove `error_recovery` from the available agents list.

**Steps (2 steps):**
- `post_launch_monitoring` → reassign to `coder`
- `incident_response` → reassign to `coder`

## What Stays Unchanged

- **Retry pipeline** — `RetryContext`, `compute_retry_timing()`, model rotation, difficulty bumps, exhaustion classification — all untouched
- **DLQ infrastructure** — `dead_letter_tasks` table, `quarantine_task()`, `/dlq` command — all stay, DLQ Analyst hooks into them
- **Grading pipeline** — binary grading, deferred grading, grade parse failure cascade — untouched
- **Shopping scraper logic** — same behavior, just renamed

## Terminology After This Change

| Term | Meaning |
|------|---------|
| Retry pipeline | In-flight failure handling: model rotation, difficulty bumps, exhaustion (RetryContext) |
| DLQ | Terminal failures quarantined for human review |
| DLQ Analyst | Pattern detection across DLQ entries, Telegram alerts with action buttons |
| Scraper failure handler | Shopping-specific error classification and recovery actions |
| ~~Error recovery~~ | Dead concept, no longer exists in codebase |
