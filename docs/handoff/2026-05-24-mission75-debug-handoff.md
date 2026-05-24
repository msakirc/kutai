# Handoff — mission 75 debugging + open items (2026-05-24)

**For:** the next session continuing i2p mission debugging. Two fixes shipped + pushed to `main` this
session; this doc is so you don't re-investigate from scratch.

---

## §0 — State at handoff

KutAI orchestrator was **LIVE** the whole session (wrapper PID 31628 + orchestrator + nerd_herd +
yazbunu, started 2026-05-23 22:28). Consequence: any pytest suite that opens the live DB
(`C:/Users/sakir/ai/kutai/kutai.db`) **deadlocks on the WAL lock** — one full `coulson` run hung
with zero output and had to be killed. Only DB-free / isolated-DB tests are safe to run while it's up.
**Never kill llama-server / wrapper / orchestrator; killing a zombie pytest is fine.**

Mission **75** is active. Reverse pitch confirmed. Task **165064** (`0.0a.draft` intake_todo_draft)
DLQ'd with `max_iterations_reached` → **fixed this session, pending restart + retry** (see §2).

---

## §1 — SHIPPED this session (2 commits on `main`, pushed)

### Commit 1 — RC-A: admission↔worker selection divergence (the `no_candidates` DLQ flood)
Spec + full record: `docs/handoff/2026-05-24-admission-worker-selection-divergence.md` §5.
- **A.** `fatih_hoca.is_servable(model, reqs)` (new) + `coulson/dispatch_helpers.py::pick_for_iter`
  now reuses the held pick across **every** no-failure iter while servable (was iter-0 only);
  re-selects only on failures or held-gone. Stamps `task["_held_pick"]`.
- **B.** `fatih_hoca/requirements_builder.py::requirements_for` input estimate switched from
  char-based `(desc+ctx)//4` to `estimate_for` (btable) — matches admission. **Data inverted the
  original premise**: char-based floored to 1000 vs actual median 8877 (9998 calls). Probe:
  `scripts/_probe_estimate_compare.py`.
- Also fixed a **stale** `tests/sim/run_swap_storm_check.py` (bare `SimpleNamespace` lacked
  `pressure_for`; pre-existing, unrelated).
- Validation: sim `run_scenarios.py` + `run_swap_storm_check.py` PASS; `fatih_hoca/tests/` 341 passed.

### Commit 2 — `.json` auto-persist (this mission-75 DLQ)
- **Root cause:** `react.py` auto-persist recovery only fired for produces ending in `.md`. Step
  `0.0a.draft` produces `intake_todo_draft.json` under the "return JSON as final_answer, engine
  persists, write_file disabled" contract. So the `final_answer` JSON was never written → the
  produces-grounding guard (clears only on a written path) looped the writer agent
  `final_answer`↔forbidden-`write_file` → `guard_burns` → `max_iterations_reached` → DLQ.
- **Fix:** extracted pure `coulson/grounding.py::autopersist_candidate(produces, written, result)`,
  extended to `.json` (serialize dict/list, validate it parses, then write + append the synthetic
  `write_file` entry so the guard clears). `.md` path unchanged. Wired into `react.py` ~line 760.
  Tests: `packages/coulson/tests/test_autopersist_candidate.py` (9) + grounding regression (32).

---

## §2 — IMMEDIATE next action (verify the fix end-to-end)

The orchestrator was running OLD code when 165064 DLQ'd; both fixes are on disk + committed but not
yet loaded.
1. Founder runs `/restart` (loads both fixes).
2. Founder runs `/dlq retry 165064` (re-runs `0.0a.draft`).
3. **Expected:** step completes in ~1 iteration; `workspace/mission_75/.intake/intake_todo_draft.json`
   appears; mission proceeds to mechanical `0.0a` (generate_intake_todo).
4. **If it still loops:** grep `logs/kutai.jsonl` for `[Task #…] auto-persisted` and `guard_burns`.
   No "auto-persisted" line ⇒ the helper isn't firing (check `_task_ctx["produces"]` shape / the
   result isn't valid JSON). A different DLQ reason ⇒ new bug, triage fresh.

Forensics tool: `python scripts/_probe_task.py <task_id>` (read-only; prints agent/ctx/picks/
per-iter `model_call_tokens`).

---

## §3 — OPEN / deferred (not done this session)

### RC-A follow-ups (from the admission-divergence handoff §5)
1. **Flag-forwarding** — admission still omits `needs_thinking`/`prefer_local`/`prefer_speed`/
   `local_only`/`exclude_models` from its `select()` (ranking-only divergence; not the eligibility
   gate). To fully unify, route admission through `requirements_for` (heavier: per-candidate async
   DB reads in the look-ahead loop).
2. **Poisoned learned-p90** — `step_token_stats.in_p90` fires on only ~1% of calls and is inflated by
   single huge-call outliers (+48k bias on that 1%). Clamp/winsorize, or prefer p50.
3. **Huge-context under-estimate** — dropping char means a genuinely large stored context now
   under-estimates (both sides equally → no admission/worker divergence; degrades to a call-time 413
   that A's failure-re-select handles). `max(estimate_for, char)` was the rejected alternative.
4. **Mission 74 re-run** still pending (RC-A validation). Watch
   `SELECT reason, COUNT(*) FROM admission_violations GROUP BY 1` → `no_candidates` → ~0.

### Other live i2p bugs (reported in the prior handoff §4 — NOT investigated this session)
- **RC-B1** — skipped legacy steps (0.2 / 0.4 / 0.5) get schema-validated → false DLQ.
- **RC-D** — `non_goals` `<id>` template not substituted (general template-substitution gap; note
  the `0.0a.draft` produces path WAS correctly substituted, so this is step/field-specific).
- **artifact_summarizer** — empty-result.

### Validation gaps (blocked by the live orchestrator)
- Full `packages/coulson` + `packages/general_beckman` suites were **not** run (WAL deadlock).
  `general_beckman` is unchanged by this session; `coulson` changes are covered by targeted tests.
  Run both after a `/stop`. Hazards: dual-conftest (run dirs separately), never `pytest` without a
  timeout, no DB isolation in those suites (they open the configured `DB_PATH`).

---

## §4 — Evidence pointers & known-good probes
- Live logs: `logs/kutai.jsonl` (orchestrator.jsonl is stale, April). Filter `grep -F "<task_id>"`.
- DB (read-only): `file:C:/Users/sakir/ai/kutai/kutai.db?mode=ro`. Key tables: `tasks`,
  `model_call_tokens` (per-call ground-truth prompt/completion tokens), `step_token_stats` (btable),
  `model_pick_log`, `admission_violations`.
- Workspace artifacts: `workspace/mission_<id>/...`.
- Probes (untracked scratch unless noted): `scripts/_probe_task.py <id>`,
  `scripts/_probe_estimate_compare.py` (committed), `scripts/_probe_missions.py <id>`.
- Guard machinery: `coulson/grounding.py` (pure produces↔written matching + `autopersist_candidate`),
  `coulson/react.py` (auto-persist ~760, sub-iter guard `check_sub_iter_guards` ~816, produces-done
  ~1581, exhaustion classify ~1745), `coulson/guards.py`.
- Selection: `fatih_hoca/selector.py` (`_check_eligibility` ~390, `is_servable` ~592),
  `general_beckman/__init__.py` admission (`select()` ~551).

### Venv / test invocation
`.venv/Scripts/python.exe`. Safe targeted runs (no live DB):
`python -m pytest packages/coulson/tests/test_autopersist_candidate.py -q -p no:cacheprovider`.
`fatih_hoca/tests/` is isolated (tmp registry_store) — safe even while live.
Sim harness (no DB): `python packages/fatih_hoca/tests/sim/run_scenarios.py` and
`MODEL_DIR=/c/Users/sakir/ai/models python .../run_swap_storm_check.py`.
