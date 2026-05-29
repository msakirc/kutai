# SP3b — Task 0 baseline + SP3 reconcile notes

**Branch:** `feat/cps-sp3b` from `main` @ `170cc3e7` (SP3 merged below; cps-sp3 stack `5689cbfa`..`a9453628` present).

## Pre-existing reds (NOT SP3b — recorded so we don't blame this branch)

Run via worktree-absolute paths (the worktree root conftest is `__file__`-relative + evicts pre-imported packages, so absolute paths resolve worktree code from any cwd).

- `general_beckman/tests/test_admission_cache.py::test_cache_skips_redundant_scan_when_state_unchanged` — FAIL (pre-existing, DB/cache; documented).
- `general_beckman/tests/test_admission_cache.py::test_cache_invalidates_when_in_flight_changes` — FAIL (pre-existing, DB/cache).
- `tests/test_llm_dispatcher.py::TestRequestRetry::test_retries_on_call_error_then_succeeds` — FAIL (tests dispatcher-internal retry removed by SP3 Phase C.3; Task 2 migrates/rewrites this file → expected to be superseded).
- `tests/core/test_dispatcher_records_swap.py::test_dispatcher_records_swap_after_swap` — FAIL (pre-existing, flagged in SP2 handoff; Task 2 migrates to husam).
- `tests/core/test_dispatcher_in_flight.py::test_dispatcher_calls_begin_end_for_cloud` — FAIL (pre-existing; Task 2 migrates).
- `tests/core/test_dispatcher_in_flight.py::test_dispatcher_ends_call_even_on_exception` — FAIL (pre-existing; Task 2 migrates).

Green baseline: general_beckman 123 pass, coulson 53 pass, dispatcher/migration 54 pass.

## Path corrections to the plan

- `tests/core/test_llm_dispatcher.py` does **NOT** exist. The dispatcher tests are:
  `tests/test_llm_dispatcher.py` + `tests/core/test_dispatcher_{pick_log,budget,preselected_pick,records_swap,in_flight}.py` + `tests/migration/test_admission_gates_run_once.py` + `tests/migration/test_dispatcher_alias_compat.py`. Task 2 migrates the `_do_dispatch`/`dispatch()` callers among these to `husam.run`.

## Test invocation for implementer subagents

From any cwd, worktree-absolute paths, ALWAYS timeout-prefixed, and run colliding roots (`tests/` vs `packages/*/tests/`) in SEPARATE invocations (single-invocation mixing → pluggy "Plugin already registered" collision):

```
timeout 150 C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest \
  C:/Users/sakir/Dropbox/Workspaces/kutay/.claude/worktrees/cps-sp3b/<test-path> -q -p no:cacheprovider
```

## SP3 shapes verified present on the merged tree (build on these)

- `general_beckman/posthooks.py`: `POST_HOOK_REGISTRY` (~40 kinds), `PostHookSpec(kind, verb, default_severity, cost_band, auto_wire_triggers, description)`, `determine_posthooks(task, task_ctx, result)` → `["grade", …extras]`, `_NO_POSTHOOKS_AGENT_TYPES` (recursion guard: includes `reviewer`/`summarizer`).
- `general_beckman/apply.py`: `_apply_posthook_verdict(task, a)` (~3779), `_enqueue_posthook_llm_child(kind, source, source_ctx, …)` (~1190, branches on `kind`, raises on unsupported), `_apply_request_posthook` (~1126). **Task 4/5 implementers: read the exact `PostHookVerdict` dataclass + the continuation-wiring (`on_complete`) shape before coding — adapt to whatever SP3 actually shipped.**
- `general_beckman/posthook_continuations.py` — SP3's CPS handlers (claim-then-fire idempotency).
- `src/core/llm_dispatcher.py`: `_task_result_to_request_response` consumers = grading (SP3), `posthook_handlers/brand_voice_lint.py`, `src/tools/vision.py` — keep stable.
- `src/core/orchestrator.py:317` — pump's raw_dispatch branch calls `get_dispatcher().dispatch(...)` → Task 2 repoints to `husam.run`.
