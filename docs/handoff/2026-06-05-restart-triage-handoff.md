# Handoff — Post-restart live-error triage + deferred-item closure

**Date:** 2026-06-05
**Author:** triage session (follows `2026-06-05-deterministic-materializer-handoff.md`)
**Status:** "Many live errors" triaged → NOT a mass cascade. Two real regressions fixed + pushed to `main`. Restart-gated.

---

## 0. TL;DR

The prior handoff warned of "MANY live errors" after a big restart-gated batch went live, and named **schema-gate 240-tightening** + **materializer** as top suspects. **Both cleared.** Bucketing 20k log lines: zero tracebacks/CRITICAL, only **4 DLQs total** (mission-81, category=quality). The real breakage was two concrete regressions, both now fixed:

| Commit | Fix | Was-live-broken |
|--------|-----|-----------------|
| `516e4ab4` | yalayut `payload` kwarg → nest under `context` | discovery + source-scout crashed every pump tick |
| `d07c8c64` | grade bonus check scans `produces` paths | mission-81 quality DLQs (real progress → DLQ'd without bonus) |

**Everything below is restart-gated — needs a KutAI restart to go live.**

---

## 1. How the triage went (repro for next time)

Live logs go to `logs/kutai.jsonl` (NOT `orchestrator.jsonl` — that's stale since Apr 6). Bucket recent levels:
```
tail -n 20000 logs/kutai.jsonl | python -c "<json parse, Counter on (level,msg[:40]) for WARNING/ERROR/CRITICAL>"
```
Result: no ERROR/CRITICAL, no tracebacks. Dominant signals:
- `add_task() got an unexpected keyword argument 'payload'` ×2 (yalayut discovery + source-scout enqueue failed)
- Mission-81 quality DLQ cluster ×4 → auto-paused (hallucination sub-corrections, `Permission denied: analyst→shell`, step 3.1 schema-fail)

**Lesson:** the handoff's blast-radius ranking was wrong. No mass DLQ from schema-gate. Always bucket the live log FIRST.

---

## 2. Fix #1 — yalayut `payload` kwarg (`516e4ab4`)

**Bug:** `src/core/orchestrator.py` `_check_yalayut_discovery` + `_check_source_scout` built enqueue specs with a **top-level `payload`** key. `general_beckman.enqueue()` forwards the spec via `add_task(**spec)`, and `add_task` has no `payload` param → both periodic checks raised every pump tick, killing daily discovery + source-scout since the restart.

**Fix:** nest `payload` under `context` (canonical mechanical shape — matches `mr_roboto/mention_monitor_sweep.py`). Dispatch routing already lifts `ctx["payload"] → t["payload"]` for `mr_roboto.run` (`orchestrator.py:343-351`); executors `yalayut_discovery`/`source_scout` are registered (`mr_roboto/__init__.py:5077,5082`).

**Test trap:** `tests/yalayut/test_phase4_orchestrator_checks.py` mocked `enqueue`, so `add_task(**spec)` never executed — the test asserted the broken top-level `payload` shape and stayed **green** while live crashed. Rewrote to assert `"payload" not in spec` + nested form. 3 passed.

---

## 3. Fix #2 — grade bonus check produces-blind (`d07c8c64`)

**This is the mission-81 quality-DLQ root cause.**

Since the materializer (`4a15fbec`) became the **sole writer** of declared `produces` paths, produces-having steps **no longer** get a mission-root `<name>.md` write (only produces-LESS steps do — `hooks.py:1524`). But `src/core/grading.py::apply_grade_result`'s bonus-attempt progress check (grants extra attempts before DLQ when the task made real progress) scanned **only** `mission_<id>/<name>.{md,json,txt}` at the root. A task that correctly wrote its declared artifact (e.g. `.research/foo.json`) registered `has_progress=False` → DLQ'd at the quality cap with no bonus attempts.

**Fix:** the progress check now also scans every declared `ctx["produces"]` path (abs, or WORKSPACE_DIR-relative), mirroring `hooks.py` produces resolution.

---

## 4. Materializer deferred-item audits (prior handoff §2)

Ran read-only audits before touching code:

- **Risk #3 (non-workflow producers lost in-loop auto-persist) = NOT a regression.** `produces`/`is_workflow_step` are set **exclusively** by `expander.py` (lines 375 / 257). Every dispatch path converges at `beckman.on_task_finished`, gated `if is_workflow_step(ctx) → post_execute_workflow_step → materialize_produces`. Non-workflow tasks (ad-hoc `/task`, single_shot, mechanical, husam raw_dispatch) **cannot declare produces**, so nothing to lose. The removed `react.py` AUTO-PERSIST/CANONICALIZE blocks were workflow-step recovery forks, now superseded.
- **Risk #5 (root-`.md` readers) = REAL, but only ONE broken reader** — the grading bonus (fixed in §3). The other candidate, `hooks.py:1300-1367` internal recovery, **reads produces paths too** (1301-1323) so it's fine. Artifacts also land in `store.store` (the canonical artifact channel, `hooks.py:1480`), independent of disk path.

---

## 5. Emit test (prior handoff §4) — STALE, not regressed

`test_emit_child_spec_is_raw_dispatch` failed (`enqueue` awaited 0×). Root cause is the post-2026-06-05 `should_skip_emit` contract: emit fires **only when the draft FAILS `validate_artifact_schema`** (lock-step with the schema gate). The test's draft was prose (`"the connection is verified, all good"`) which the **deliberately loose** validator ACCEPTS against an object schema → emit correctly skipped. The test mocked too deep and asserted the old "non-JSON always fires" contract.

**Fixed in `d07c8c64`:** firing test now uses an object-missing-required-field draft (`{"connection": {"present_but_incomplete": true}}`) that genuinely fails validation; added `test_emit_skipped_when_draft_validates` to lock the skip path. 11 passed.

**Latent (NOT touched):** the validator accepting bare prose for an object schema is handoff §2 #7 looseness. Tightening it has a **240-schema blast radius** → high mass-DLQ risk right after a restart. Left alone deliberately.

---

## 6. Still open

1. **`tests/test_grading.py::TestApplyGradeResultPass` ×2** (`test_pass_with_rich_verdict`, `test_pass_empty_verdict_uses_mechanical_fallback`) FAIL — assert `add_skill` called on grade-PASS. **STALE, not a regression:** the PASS path replaced `add_skill` with `capture_exemplar` (`workflow_exemplars`); non-workflow passing tasks now mint nothing (`grading.py:381-385` comment). Needs a test rewrite to the exemplar-capture contract. Confirmed pre-existing via `git stash` of grading.py (still red without my edit). Separate from the materializer line of work.
2. **Validator looseness** (§5) — object schema accepts prose. Pre-existing handoff §2 #7. Make `validate_artifact_schema` markdown/object branches stricter only with a full mission-replay first; mass-DLQ risk.
3. **Materializer §2 cuts** still deferred: `#1` `_schema_version` stamping (needs expander threading), `#2` multifile produces (result→N-artifact splitter).
4. **#6 dead helpers** `autopersist_candidate` + `recanonicalize_candidate` in `grounding.py` — KEPT (pure + tested; risk #3 being non-regression means unneeded but harmless). Remove-or-keep still an open call.

---

## 7. Validate once restarted

- Confirm `enqueued yalayut daily discovery task` / `enqueued yalayut source-scout task` INFO lines appear (no more "enqueue failed: add_task() ... 'payload'").
- Retry mission-81 DLQ'd quality tasks; watch for `grade bonus attempt | task_id=...` logs on produces-having steps that made progress (should no longer DLQ blind).
- `SELECT lane,status,COUNT(*) FROM tasks GROUP BY 1,2` — watch for a DLQ drop on quality category.
