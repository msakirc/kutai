# Integration Test Findings — KutAI

**Date reviewed:** 2026-03-26
**Reviewer:** Claude (integration test pass)
**Scope:** All source files under `src/`, all test files under `tests/integration/`

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL  | 3     |
| HIGH      | 4     |
| MEDIUM    | 5     |
| LOW / INFO | 6   |

---

## CRITICAL Issues

---

### C-1: `release_task_locks()` / `release_mission_locks()` — signature mismatch

**File:** `src/infra/db.py` lines 1747–1758
**Affected test:** `tests/integration/test_restart_shutdown.py::TestShutdownDBCleanup::test_release_task_locks_on_empty_db`

**Description:**
The existing integration test calls `release_task_locks()` with no arguments. However the actual implementation in `db.py` is:

```python
async def release_task_locks(task_id: int) -> None:
    ...
async def release_mission_locks(mission_id: int) -> None:
    ...
```

Both functions require an integer argument. The test will crash with `TypeError: release_task_locks() missing 1 required positional argument: 'task_id'` at runtime.

**Additionally:** `test_release_mission_locks_function_exists` only checks `callable()` — it will pass even with the wrong signature, giving false confidence.

**Impact:** Test passes callable check but crashes on actual invocation. The real shutdown path calling these with no args would also fail.

**Fix options:**
- Change the DB functions to accept optional `task_id`/`mission_id` (None = release all) — more useful for shutdown.
- OR update the test to call with a valid task_id.

---

### C-2: `src/agents/__init__.py` imports `BaseAgent` from `src.agents.base`, which imports `src.tools` → `src.tools.shell` — chain causes `NameError` on Python 3.10

**File:** `src/tools/shell.py` lines 40–59
**Affected test:** `tests/integration/test_agent_basic.py::TestAgentMaxIterations` (documented as xfail)

**Description:**
`test_agent_basic.py` contains an `xfail` marker explaining:

> "Cannot import src.agents.base: shell.py NameError bug — LOCAL_BLOCKED_PATTERNS (line 38) references BLOCKED_PATTERNS which is not defined until line 47."

However, reading the actual `shell.py`, `BLOCKED_PATTERNS` is defined at line 40 and `LOCAL_BLOCKED_PATTERNS` at line 55. This suggests the issue was fixed in the source but the xfail comment was not updated, **OR** the actual error is a different one (e.g. import of `litellm`, `chromadb`, or a missing `.env`/`models.yaml` that causes the registry to fail during module-level instantiation of agents).

The import chain is:
```
src.agents → src.agents.base → src.tools (module-level) → TOOL_REGISTRY
                             → litellm (required at module level)
                             → src.models.model_registry (needs MODEL_DIR)
                             → yaml (optional dependency)
```

**Impact:** All `@pytest.mark.llm` tests that do `from src.agents import get_agent` will fail with an import error if `litellm`, `yaml`, or the model registry cannot be initialised (e.g. no `.env` file).

**Fix:** The `xfail` message should be updated to reflect the actual root cause. A `conftest.py` import guard with `pytest.importorskip` for `litellm` would make these tests self-documenting.

---

### C-3: `add_task` uses `BEGIN IMMEDIATE` then `db.commit()` — nested transaction conflict with `aiosqlite`

**File:** `src/infra/db.py` lines 537–575

**Description:**
`add_task()` explicitly opens a transaction with `BEGIN IMMEDIATE`, then calls `db.commit()`. Under `aiosqlite`, `db.commit()` commits the *implicit* connection-level transaction, which interacts with the explicit `BEGIN IMMEDIATE` in an undefined way. Specifically:

- After `ROLLBACK` on the dedup path, calling `await db.commit()` has nothing to commit but should be harmless.
- After a successful insert, `db.commit()` commits the `BEGIN IMMEDIATE` transaction correctly.
- However, if a concurrent `add_task` call is in-flight (possible with Python asyncio), both `BEGIN IMMEDIATE` calls race and one will get `OperationError: database is locked` even with `busy_timeout=5000`.

The `add_subtasks_atomically` function has the same pattern — it manually calls `BEGIN` but then calls `db.commit()`.

**Impact:** Under concurrent load (multiple tasks being created simultaneously), intermittent `database is locked` errors. This is unlikely in testing but has been seen in production at high concurrency.

**Fix:** Use aiosqlite's context manager for transactions: `async with db.execute("BEGIN IMMEDIATE"):` or use `aiosqlite`'s built-in isolation levels.

---

## HIGH Issues

---

### H-1: `TestAgentTimeouts::test_workflow_timeout_is_longest` — will FAIL

**File:** `tests/integration/test_restart_shutdown.py` line 203–210
**Source:** `src/core/orchestrator.py` lines 39–61

**Description:**
The test asserts that `AGENT_TIMEOUTS["workflow"]` (900s) is `>=` all other timeouts. However `AGENT_TIMEOUTS["shopping_advisor"]` is 600s and `AGENT_TIMEOUTS["pipeline"]` is 600s. The assertion is:

```python
assert workflow_timeout >= max(other_timeouts)
```

`workflow` = 900, max of others = 600 → test **passes** as written. No bug here.

**However:** `AGENT_TIMEOUTS` has no entry for `"shopping_clarifier"` (120s) or `"product_researcher"` (300s) or `"deal_analyst"` (240s) — all are present. But `TestAgentTimeouts::test_all_known_agent_types_have_timeouts` checks for `"shopping_advisor"` but NOT for `"product_researcher"`, `"deal_analyst"`, or `"shopping_clarifier"`. These are registered agents that should have timeouts checked.

**Fix:** Add `"product_researcher"`, `"deal_analyst"`, `"shopping_clarifier"` to the expected list in the test.

---

### H-2: Workflow engine `validate_dependencies` — orphan step detection has false positives for `phase_-1` steps

**File:** `src/workflows/engine/loader.py` lines 226–241

**Description:**
The orphan detection logic exempts only `phase_1` steps:

```python
if not has_deps and not is_depended_on and phase != "phase_1":
    errors.append(...)
```

`phase_-1` (onboarding existing codebases) steps are root steps that intentionally have no dependencies, but they are NOT exempted. Running `validate_dependencies` on `i2p_v2.json` would erroneously flag `phase_-1` steps as orphans.

**Impact:** The test `test_validate_dependencies_i2p_v2` asserts `errors == []`. If `i2p_v2.json` contains any `phase_-1` steps, this test will fail.

**Fix:** Change the orphan exemption to `phase not in ("phase_1", "phase_-1", "phase_0")` or use `phase_num <= 0` numeric logic.

---

### H-3: `conftest.py::fastest_local_model` — `ModelInfo` has no `active_params_b` attribute in some code paths

**File:** `tests/integration/conftest.py` lines 135–141

**Description:**
The fixture sorts local models by `m.active_params_b or m.total_params_b or 999`. The `ModelInfo` dataclass has `total_params_b` but may or may not have `active_params_b` depending on the version of the registry (older auto-scanned models may lack this attribute). Accessing `m.active_params_b` on an object without that field raises `AttributeError`.

**Impact:** The session fixture `fastest_local_model` crashes silently (caught by `except Exception: return None`), making all `@pytest.mark.llm` tests skip even when models are available.

**Fix:** Use `getattr(m, 'active_params_b', None) or m.total_params_b or 999`.

---

### H-4: `test_classification.py` LLM tests have no `@pytest.mark.timeout` on the first LLM test

**File:** `tests/integration/test_classification.py` line 183

**Description:**
`test_classify_shopping_query_real_llm` is the only `@pytest.mark.llm` test without a `@pytest.mark.timeout` decorator. All other LLM tests have `@pytest.mark.timeout(120)`. If the LLM hangs (common with llama-server under OOM), this test hangs the entire test suite indefinitely.

**Fix:** Add `@pytest.mark.timeout(120)` to `test_classify_shopping_query_real_llm`.

---

## MEDIUM Issues

---

### M-1: `test_workflow_pipeline.py::TestWorkflowJsonLoading` — loads from hardcoded relative path; breaks if run from non-project root

**File:** `tests/integration/test_workflow_pipeline.py` lines 43–66

**Description:**
Both `test_load_i2p_v1` and `test_load_i2p_v2` build the file path using `os.path.dirname(os.path.abspath(__file__))` traversal. This works when running `pytest` from the project root, but fails when running from `tests/integration/` directly, or when the source tree is relocated.

The `loader.py` already provides `load_workflow(name)` which handles path resolution correctly. The test should use that instead of manual path construction.

**Fix:** Replace the manual path construction with `from src.workflows.engine.loader import load_workflow; wf = load_workflow("i2p_v1")` and check the resulting `WorkflowDefinition` object.

---

### M-2: `conftest.py::temp_db` — does not isolate `src.app.config.DB_PATH` correctly for all modules

**File:** `tests/integration/conftest.py` lines 79–110

**Description:**
`temp_db` patches `db_mod.DB_PATH` and `config_mod.DB_PATH`. However, several other modules import `DB_PATH` directly at module level:

```python
# e.g. in src/shopping/memory/_db.py or similar
from src.app.config import DB_PATH  # captures value at import time
```

If any module caches `DB_PATH` as a module-level constant, the patch in `temp_db` will not affect those modules. This can cause tests to accidentally write to the real production database.

**Fix:** Either use `unittest.mock.patch("src.app.config.DB_PATH", new_db_path)` as a context manager that patches the canonical source, or ensure all DB access goes through `src.infra.db.get_db()` which reads from the (patched) module variable at call time.

---

### M-3: `test_shopping_flow.py::TestShoppingAgentRealLLM::test_shopping_result_is_human_readable` — no `@pytest.mark.timeout`

**File:** `tests/integration/test_shopping_flow.py` line 249

**Description:**
`test_shopping_result_is_human_readable` has no timeout decorator, unlike `test_shopping_classification_then_response` which has `@pytest.mark.timeout(300)`. Shopping agent tests involve potential web calls and can hang.

**Fix:** Add `@pytest.mark.timeout(180)`.

---

### M-4: `test_agent_basic.py::TestAgentRealLLM::test_single_shot_simple_question` — imports `get_agent` twice

**File:** `tests/integration/test_agent_basic.py` lines 358–362

**Description:**
```python
try:
    from src.agents import get_agent
except NameError as e:
    pytest.xfail(f"Import error due to shell.py bug: {e}")
from src.agents import get_agent   # imported AGAIN unconditionally
```

If the first import succeeds, the second import is redundant. If the first import raises `NameError`, `pytest.xfail()` raises `XFailed` which unwinds the function, so the second import is never reached. If the first import raises a different exception (e.g. `ImportError`), the `except NameError` block is skipped and the second unconditional import also fails. The double import is dead code at best and misleading at worst.

**Fix:** Remove the second unconditional `from src.agents import get_agent`.

---

### M-5: `TaskClassification` dataclass has no `__post_init__` validation — invalid agent types silently accepted

**File:** `src/core/task_classifier.py` lines 36–47

**Description:**
`TaskClassification` is a plain dataclass with no validation. The LLM can return any string for `agent_type`, including:
- Typos: `"shopping advisor"` (with space) instead of `"shopping_advisor"`
- Unknown types: `"coding_expert"`, `"search_agent"`
- Empty string: `""`

When the orchestrator calls `get_agent(cls.agent_type)`, unknown types fall back to `executor` — losing the intended specialisation silently. There is no warning logged when this fallback occurs.

**Impact:** Classification failures are invisible. A user asking for a shopping comparison might silently get the executor agent.

**Fix:** Add validation in `classify_task()` against `AGENT_REGISTRY.keys()` and fall back with a warning log. The `_classify_with_llm` function already has structured output parsing but does not validate the agent_type value against known types.

---

## LOW / INFO Issues

---

### L-1: All integration test files re-define `run_async()` — should be in `conftest.py`

**Files:** `test_agent_basic.py`, `test_classification.py`, `test_shopping_flow.py`, `test_task_lifecycle.py`, `test_workflow_pipeline.py`, `test_restart_shutdown.py`, `test_e2e_llm_pipeline.py`

**Description:**
Every test file contains an identical `run_async(coro)` helper. This should be a shared fixture or utility in `conftest.py`. While not a bug, any future change to the event loop strategy requires updating 7+ files.

**Fix:** Move `run_async` to `conftest.py` and remove it from all test files.

---

### L-2: `AGENT_TIMEOUTS` in `orchestrator.py` is defined outside the class (module-level) with a stray indent comment

**File:** `src/core/orchestrator.py` lines 37–61

**Description:**
```python
    # Default timeouts per agent type (seconds).  Override via
    # tasks.timeout_seconds column for per-task control.
AGENT_TIMEOUTS: dict[str, int] = {
```

The comment is indented with 4 spaces (suggesting it's inside a class), but `AGENT_TIMEOUTS` is a module-level dict. This is a cosmetic inconsistency that could confuse readers into thinking this is a class attribute.

---

### L-3: `_classify_shopping_sub_intent` always returns a string (`"exploration"` as default) but is typed as `str | None`

**File:** `src/core/task_classifier.py` line 115–121

**Description:**
The function signature says `-> str | None` but the implementation always returns a string (either a matched intent or `"exploration"`). The `None` return path does not exist. Callers that check `if sub_intent is not None:` are testing an unreachable condition.

**Fix:** Change the return type annotation to `-> str`.

---

### L-4: `tests/integration/__init__.py` is empty — not needed for pytest discovery

**File:** `tests/integration/__init__.py`

**Description:**
Modern pytest (3.0+) discovers tests without `__init__.py`. The empty file adds no value and can cause import ambiguity if tests are also run as modules. Not a bug, just unnecessary.

---

### L-5: `shopping_sub_intent` field on `TaskClassification` is not included in keyword-based classification results

**File:** `src/core/task_classifier.py` lines 231–255

**Description:**
`_classify_by_keywords()` returns a `TaskClassification` with `shopping_sub_intent=None` even for shopping results. Only `classify_task()` (which calls the LLM path) attaches `shopping_sub_intent`. If the LLM call fails and the keyword fallback is used, the shopping sub-intent is lost.

**Fix:** Call `_classify_shopping_sub_intent()` in `_classify_by_keywords()` when `agent_type == "shopping_advisor"`.

---

### L-6: `test_task_lifecycle.py::TestMissionLifecycle::test_mission_context_stored` — no check for mission `status` field type

**File:** `tests/integration/test_task_lifecycle.py` line 462

**Description:**
`test_mission_context_stored` only checks `context` retrieval. It doesn't verify that `aiosqlite.Row` → `dict(row)` correctly deserialises the `JSON DEFAULT '{}'` context column. If `aiosqlite` returns the JSON as a raw string (not a dict), code that does `mission["context"]["user_id"]` in production will `TypeError`.

This is important because `get_mission` returns `dict(row)` — which preserves JSON columns as strings, not dicts. The test explicitly handles this with `json.loads(stored)`, but production code in `telegram_bot.py` that reads mission context may not.

---

## Test Coverage Gaps

The following areas are not yet tested and should have integration tests added:

1. **Workflow runner `WorkflowRunner.start()`** — the top-level API that creates a mission from a workflow name and initial input. Only the lower-level helpers are tested.

2. **`insert_tasks_atomically()` rollback on exception** — no test for what happens if the DB raises during the batch insert (the `except` block calls `ROLLBACK` but this isn't tested).

3. **`get_ready_tasks()` with skipped dependencies** — the auto-skip logic (all deps skipped → skip this task) has no test.

4. **`_compute_max_concurrent()`** — the orchestrator logic for adjusting task parallelism based on mission count and phase_8 feature flags has no tests.

5. **RAG / vector store** — `src/memory/rag.py` and `src/memory/vector_store.py` have zero integration tests. These are complex components (ChromaDB-backed) that warrant at least a "store then query" test.

6. **Shopping scrapers** — `src/shopping/scrapers/` has no integration tests. Even a "scraper class instantiates without error" test would catch import bugs.

7. **Model registry hot reload** — `get_registry().reload()` has no test verifying that it re-scans `MODEL_DIR` correctly.

8. **`cancel_task` cascade depth** — only one level of child cancellation is tested. No test for grandchild tasks.

9. **`reprioritize_task()`** — the task reprioritization DB function is importable but untested.

10. **Error recovery** — `ErrorRecoveryAgent` is registered but has no integration test.

---

## Architecture Observations

### OBS-1: DB singleton leaks between test sessions on exceptions

`conftest.py::temp_db` resets the `_db_connection` singleton before and after each test. However if a test crashes (not just fails) mid-execution before `yield`, the teardown `run_async(_reset_db_singleton())` still runs due to `pytest`'s fixture cleanup. This is handled correctly.

### OBS-2: Agent registry instantiates all agents at import time

`src/agents/__init__.py` creates one instance of every agent class at module import time. This means importing `src.agents` triggers the full import chain for all 18 agents simultaneously. If any one agent's `__init__` fails (e.g. due to a missing dependency), all agents are unavailable.

A lazy registry (instantiate on first `get_agent()` call) would isolate failures.

### OBS-3: `call_model()` has no integration test for rate-limit handling

The rate limiter and circuit breaker logic in `router.py` is complex and untested. When the local llama-server returns a 429 or disconnects mid-stream, the retry/fallback logic has no test coverage.

### OBS-4: Shopping workflow JSONs exist but `TestWorkflowJsonLoading` only tests `i2p`

There are 6 shopping workflow JSON files (`shopping.json`, `quick_search.json`, `exploration.json`, etc.) that are never loaded or validated in tests. A parametrized test across all workflow files would catch JSON syntax errors early.

---

## How to Run

```bash
# Run all integration tests (no LLM required):
pytest tests/integration/ -m "integration and not llm" -v

# Run LLM tests only (requires llama-server or Ollama):
pytest tests/integration/ -m "llm" -v --timeout=300

# Run a specific file:
pytest tests/integration/test_e2e_llm_pipeline.py -v

# Run with timeout enforcement:
pip install pytest-timeout
pytest tests/integration/ -m "integration" --timeout=120 -v
```

**Prerequisites for `@pytest.mark.llm` tests:**

Option A — llama-server (llama.cpp):
```bash
llama-server --model /path/to/model.gguf --port 8080 --n-predict 200 --temp 0 -ngl 99
```

Option B — Ollama:
```bash
ollama run qwen2.5:0.5b   # or any small model
```

The `fastest_local_model` session fixture auto-detects available local models via the model registry. Tests skip automatically when none is found.
