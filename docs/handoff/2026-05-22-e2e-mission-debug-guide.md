# Debug field guide — first end-to-end i2p mission run

**Date:** 2026-05-22
**For:** the agent driving §3c of `2026-05-21-residual-reanalysis.md` — the
first real prototype-tier i2p mission run, end to end.
**Purpose:** accelerate that debugging session and prevent mis-fixes. Every
wiring sweep (Z0–Z10 + yalayut) closed *wiring* by static grep; **no mission has
ever actually run**. This run is where the bodies that grep can't see surface.
Read this before you launch. It is a map of where things break, how to tell a
real break from a decoy, and the traps that have wasted prior sessions.

---

## §0 — The one rule

**A green unit suite proved nothing about this pipeline.** Every bug closed in
the 05-18/05-19/05-21 sweeps passed its unit tests *with the bug live*. The
recurring shape: a registry-shape test asserts "X is wired" while X's verdict
never reaches the apply layer, or a field-name mismatch silently zeroes a
payload. Trust the **running mission's DB rows + logs**, not the test result.
When you fix something, the fix is not done until a **host-path test** (drive
the real call site, assert a row/state change) goes red→green.

---

## §1 — How to launch (and what mode)

**Launch path** (verified):
- Telegram: `/mission <description> --workflow` → forces `i2p_v3`. Without
  `--workflow`, the classifier decides (`telegram_bot.py:2204`); `--workflow`
  is the deterministic way in.
- Code: `WorkflowRunner().start(workflow_name="i2p_v3",
  initial_input={"raw_idea":..., "product_name":...}, title=...)`
  (`telegram_bot.py:2213`, `src/workflows/engine/runner.py`). This creates the
  mission + expands the **first wave** of steps into `tasks`. The orchestrator
  pump does the rest.

**Mock vs real:**
- `KUTAI_ENV != prod` → **vendor calls are mocked** (`src/integrations/registry.py:36`):
  PostHog, Stripe, Apple/Google, email, etc. return deterministic fakes. Keep
  it non-prod for the first run — you do NOT want live vendor side-effects.
- **There is no LLM mock for agent execution.** Mechanical steps with
  deterministic builders (intake_todo, charter consolidation) run offline, but
  real `coder`/`implementer`/`reviewer` steps go through
  `LLMDispatcher → fatih_hoca → DaLLaMa/HaLLederiz Kadir` and need a **loaded
  local model (llama-server up) or cloud creds**. Plan for llama-server running.
  If you only want to exercise wiring, a stripped mission that reaches the
  mechanical-heavy phases (0,3,13,14) surfaces most wiring bugs without burning
  many LLM calls.

**Watch progress:** `/wfstatus <mission_id>`, `/mission <id>` (pacing block),
`/dlq` (dead-letter — failed tasks land here and **block dependent phases** by
design; `/dlq retry` is the human intervention).

---

## §2 — The pipeline trace (where to put your eyes)

```
WorkflowRunner.start (runner.py)            # mission + first-wave tasks
        │
        ▼
orchestrator pump (orchestrator.py:494)     # while running:
   general_beckman.next_task()  :506        #   admission gate lives HERE
   intersect.flash(task)        :519        #   yalayut skill match (graceful)
   self._dispatch(task)         :524        #   fire-and-forget
        │
        ├── agent_type == "mechanical" ──► mr_roboto.run(task)
        │       (packages/mr_roboto/src/mr_roboto/__init__.py)
        │       _safety_guard_check(task) :411  ► pre_action gate (irreversible+locked)
        │       big `if action == "..."` dispatch chain
        │
        └── else ──► coulson (packages/coulson/src/coulson/)
                ReAct loop → LLM via LLMDispatcher
                _apply_tools_hint / _apply_hint_from_targets  (dispatch-time now)
        │
        ▼
post-hooks (general_beckman): determine_posthooks → apply.py
   _apply_posthook_verdict          ◄── verdict MUST reach here or it's dead
   _record_and_resolve_confidence   :48     (Z10 calibration)
        │
        ▼
next-wave expansion (expander.py) ── conditional_groups, multifile, fallback_steps
```

**Admission gate** (`general_beckman.next_task`): Z6 parks tasks needing absent
credentials (`founder_action(vendor_enroll)`), and Z0 lifecycle blocks
paused/budget-exceeded missions. If a task never dispatches, check here first —
it is probably parked, not lost.

---

## §3 — The 4 failure modes (from the 2026-05-16 results audit)

A step that "passes" can be lying. The four ways:

1. **Dead wiring** — verb/posthook exists, nothing triggers it.
   *Tell:* the step never appears in logs; `grep '"action": "X"'` on
   `i2p_v3.json` returns nothing; posthook `auto_wire_triggers=[]` + no step.
2. **Stub-returns-success** — executor returns `{status:"completed"}` without
   doing the work. *Tell:* completed task, but the artifact file is missing /
   empty / scaffold-shaped (e.g. the empty-string `hero` marketing_copy we
   found). **Always open the produced artifact, don't trust the status.**
3. **Scaffolding ≠ integration** — the module is real and unit-tested, but its
   output is never consumed downstream. *Tell:* a collection/table that stays
   empty across the whole run (support_docs, confidence_outcomes pre-fix,
   growth_events metric_emit).
4. **Trust-gate bypassed** — a review/approval gate runs but its verdict is not
   enforced (B3 incident-review historically published unreviewed). *Tell:* the
   gate logs a verdict but the next step proceeds regardless.

For each suspicious step: **(a)** is it in the logs? **(b)** did it write the
row/file it claims? **(c)** did the next step read it?

---

## §4 — Recurring root-cause catalog (recognize on sight, fix at root)

These exact shapes recurred across every zone. When a step misbehaves, check
these before inventing a theory:

| Shape | Where it bit | Root fix (not the symptom) |
|---|---|---|
| **Field-name mismatch zeroes a payload** | `a.raw` vs `a.result` (Z4); `platforms` vs `target_platform` (Z5); `lifecycle_state` default `'terminal'` (Z0) | grep the actual attr/field on the object, don't assume |
| **Posthook verdict never reaches apply** | many; `feedback_verify_verdict_roundtrip` | trace verdict to `_apply_posthook_verdict`; registry-shape test ≠ wired |
| **`auto_wire_triggers=[]` + no i2p step** | A6, inject_lessons, integration_review | add a real trigger (glob) OR an explicit step; verify the expander attaches |
| **Verb registered, no caller** | derive_token_tag_signature, changelog/publish, synthetic_check | grep call sites, not docstrings (`feedback_audit_call_sites`) |
| **Column never written by prod code** | confidence_outcomes, metric_emit, missions.product_id | find the writer; if none, that's the bug |
| **Conditional group reads wrong field** | mobile_app_submission group (Z5) | check `condition_check` against the artifact's real fields |
| **Lane mismatch orphans tasks** | yalayut tasks enqueued `lane="mechanical"` (no such lane) | lanes are `oneshot`/`ongoing` only — pump selects oneshot |
| **Recipe dir layout** | Z8 ops recipes one-level deep → undiscoverable | `list_recipes()` needs `<name>/<version>/recipe.yaml` |
| **Singleton DB ignores db_path** | `write_pick_log_row` (just bit in `1607474b`) | writes through `get_db()` singleton = live kutai.db, not the passed path |

---

## §5 — Environment traps that waste hours

- **DB / WAL lock (BIG for a live mission):** the running orchestrator holds the
  `kutai.db` WAL lock. Several writers (`write_pick_log_row`) **ignore their
  `db_path` arg and write through the `get_db()` singleton**. If you run a
  script/test against a temp DB while the orchestrator is up, you may (a) assert
  against the wrong DB and (b) hang on the lock. Either stop the orchestrator
  first, or isolate via `init_db` + singleton patch (see `test_pick_log_task_id.py`).
- **conftest collision:** never mix `tests/` and `packages/*/tests/` in one
  pytest invocation — dual `conftest.py` → pluggy "Plugin already registered".
  Run dirs separately.
- **Missing dev deps (NOT regressions):** `fastapi` + `sentence_transformers`
  aren't installed → all webhook-route + embedding tests fail/skip everywhere.
- **`safety_guard` editable install:** needs `pip install -e packages/safety_guard`
  on a fresh clone (now in conftest path too). Without it, `mr_roboto.run`'s
  `_safety_guard_check` import dies on every mechanical dispatch.
- **`i2p_v3.json` is UTF-8** — read with `io.open(path, encoding='utf-8')`;
  cp1252 default chokes.
- **scheduled_tasks datetime:** store `strftime("%Y-%m-%d %H:%M:%S")`, NEVER
  `isoformat()` (the `T` breaks SQLite string comparison vs `datetime('now')`).
- **never `pytest` without a timeout** — zombie pytest holds the SQLite write
  lock and crash-loops KutAI. `timeout 120 ...`.
- **never `taskkill` llama-server** — corrupts VRAM/model state. Kill the
  orchestrator (Yaşar Usta restarts it) only if hung; never the wrapper.

---

## §6 — Verify-before-fix discipline (anti-mis-fix)

The memory feedback entries exist because each was learned the hard way:

- **Audit call sites, not docstrings/TODOs** — they go stale (`feedback_audit_call_sites`).
  A docstring saying "fires when X" is not evidence X fires.
- **Verify the verdict round-trip** — a posthook is only wired when its verdict
  reaches `_apply_posthook_verdict` (`feedback_verify_verdict_roundtrip`).
- **Zero traffic ≠ dead** — `code_reviewer`/`visual_reviewer` show 0 picks only
  because i2p hasn't reached late phases; confirm against workflow refs before
  declaring something unused (`feedback_zero_traffic_not_dead`).
- **A stale handoff premise is common** — two of the five P3s this week had
  premises that were already false (the verb already had a caller). Re-verify
  the premise against current code before building the "fix."
- **Host-path test or it didn't happen** — drive the real call site and assert
  the row/state. Unit-level mocks hid every sweep bug.

---

## §7 — Phase-by-phase fragility map (expect breakage here)

From the 2026-05-16 results audit + sweeps:

- **Z0 / Phase 0 (preflight)** weakest historically (~35-40% built); the
  founder-action gate was dead until `27c11d18` (this week). Watch admission +
  lifecycle state transitions closely on the first run.
- **Phase 7→8 boundary** had the most non-terminal dead-ends in the original
  i2p wiring audit (21). Multifile expansion (`expand_steps_with_multifile`)
  was dead until the Z3 cascade fix — confirm `integration_review` actually
  fires when a step touches multiple files.
- **Phase 13/14 (real-world bridge + launch)** — the `irreversible+locked`
  safety steps live here. The founder-gate now sits on `14.8.submit` /
  `14.8.submit_play` (moved this week, `9efea092`). In mock mode the uploads
  fake out; verify the gate *requests* founder presence rather than proceeding.
- **B3 incident-review gate** historically published unreviewed — verify the
  verdict is enforced, not just logged.
- **Cross-mission learning** (mission_lessons inject, metric_emit) — confirm
  rows actually populate during the run; these were the classic empty-table
  scaffolds.

---

## §8 — Tooling for the session

- **yazbunu** — JSONL structured logs, web viewer on **port 9880**. Per-package
  logs in `logs/` (orchestrator.jsonl, dallama.jsonl, etc.). This is your
  primary trace surface — grep the mission_id across them.
- **`model_pick_log`** — every Fatih Hoca pick + the reinforce nudges
  (`call_category`). `SELECT picked_model, AVG(picked_score), COUNT(*) FROM
  model_pick_log GROUP BY 1` to see selection behavior.
- **`mission_lessons`** — cross-mission verdicts; should gain rows as the
  mission hits DLQ/posthook-fail/hypothesis paths.
- **`/dlq`** — failed tasks; they block dependent phases. `/dlq retry` to unblock.
- **Regression guard after any fix:**
  ```
  .venv/Scripts/python -m pytest packages/mr_roboto/tests/ -q   # 749 passed (post-P3s)
  # + the §5 wiring-sweep smoke gate in 2026-05-21-residual-reanalysis.md
  ```

---

## §9 — What NOT to do

- Don't "fix" a step by making its test pass — confirm the *artifact/row* first
  (failure mode 2 + 3).
- Don't add a posthook trigger without confirming the verdict reaches apply
  (failure mode + §6).
- Don't retag reversibility to silence a safety test — the `14.8` story
  (`9efea092`) shows the lock belongs on the act, not the parent. Move it, don't
  delete it.
- Don't run tests against a temp DB while the orchestrator is live (§5 WAL lock).
- Don't kill llama-server or the Yaşar Usta wrapper.
- Don't trust a handoff premise (including this one) without checking current
  code — cite file:line you actually read.
```
