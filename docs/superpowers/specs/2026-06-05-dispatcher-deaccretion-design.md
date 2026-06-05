# Dispatcher De-accretion (Modularization Finish Plan — Phase 4)

*2026-06-05. Scope: `src/core/llm_dispatcher.py` only. Companion to `docs/2026-05-31-modularization-finish-plan.md` Phase 4. Goal: dispatcher = load→call pipe; evict accreted non-loop logic.*

## Problem
CLAUDE.md calls the dispatcher a "thin dumb pipe" but it is **721 LOC**. ~90 LOC is dead introspection/estimation and ~52 LOC is relocatable telemetry. Recon (2026-06-05) shows the plan's eviction list drifted: 4 of the targets now have **zero prod callers** — they should be **deleted**, not relocated.

## Recon — caller audit
| Method | Callers | Action |
|--------|---------|--------|
| `_estimate_prompt_tokens()` | none | **delete** |
| `get_loaded_model_speed()` | none | **delete** |
| `is_loaded_model_thinking()` | none | **delete** |
| `_get_loaded_model_name()` | none (test comment only) | **delete** |
| `_get_loaded_litellm_name()` | `src/agents/base.py:157` | relocate → `src/models/introspection.py`, repoint caller, delete method |
| `_record_pick()` | internal `execute()` ×3 + test monkeypatches | relocate body → `src/telemetry/pick_recorder.py`, keep thin delegator method |
| `get_stats()` + `_total_calls`/`_overhead_calls` | husam worker bumps counters; husam tests read | **keep** |
| `request()` shim + `_request_kwargs_to_spec` + `_task_result_to_request_response` | shopping (SP5-gated) | **keep untouched** |

## Changes

### 1. Delete 4 dead methods (~57 LOC)
Pure subtraction. `_estimate_prompt_tokens`, `get_loaded_model_speed`, `is_loaded_model_thinking`, `_get_loaded_model_name`. No prod callers; no test depends on their behavior (only a stale comment in `test_race_fixes.py:154`).

### 2. `_record_pick` body → `src/telemetry/pick_recorder.py`
- New module-level `async def record_pick(*, pick, task, category, success, error_category="", agent_type="", difficulty=None)` holding the existing 52-LOC body verbatim (contextvar task-id resolve, `DB_PATH` resolve, `pick_log.write_pick_log_row`, fire-and-forget swallow).
- Dispatcher keeps a 3-line `_record_pick` **delegator** → `await record_pick(...)`. Rationale: preserves the monkeypatch surface (`tests/migration/test_admission_gates_run_once.py`, `tests/test_llm_dispatcher.py:123 hasattr`, image-gen plan refs) with zero test churn while evicting the logic.
- `category.value` mapping stays in the delegator boundary so the module fn takes the same args as today.

### 3. `_get_loaded_litellm_name` → `src/models/introspection.py`
- New module `src/models/introspection.py` with `def get_loaded_litellm_name() -> str | None` (body verbatim: `local_model_manager` + `model_registry` lookup, swallow on error).
- Repoint sole caller `src/agents/base.py:157` to `from src.models.introspection import get_loaded_litellm_name`.
- Delete the dispatcher method (private, single caller).

### 4. Untouched
`execute`, `_ensure_local_model`, `_prepare_messages`, `get_stats` + counters, `request()` + its 2 spec/response helpers, singleton, `ModelCallFailed` re-export.

## Testing
- New `tests/telemetry/test_pick_recorder.py`: `record_pick` writes one `model_pick_log` row; swallows on bad db.
- New `tests/models/test_introspection.py`: `get_loaded_litellm_name` returns name when loaded, None on error.
- Repoint + smoke `src/agents/base.py` import.
- Regression: `tests/test_llm_dispatcher.py`, `packages/husam/tests/test_husam_run_migrated.py` (get_stats), `tests/migration/test_admission_gates_run_once.py` (monkeypatch surface).
- Import smoke: `python -c "from src.core.llm_dispatcher import get_dispatcher, LLMDispatcher, CallCategory"`.

## Outcome
- `llm_dispatcher.py` 721 → ~440 LOC.
- Dispatcher surface = `execute` + `_ensure_local_model` + `_prepare_messages` + `_record_pick` delegator + `get_stats` + `request` shim + singleton.
- Update CLAUDE.md L17: drop "NOT thin (721 LOC)" + "accreted ~231 LOC of telemetry/introspection"; restate as pipe with telemetry evicted and remaining SP5-gated `request()` shim.

## Out of scope (deferred)
- P1 router dead-code delete (`select_model`/`get_kdv` relocation) — still pending, separate session.
- Deleting `request()` + spec helpers — gated on SP5 shopping migration.
