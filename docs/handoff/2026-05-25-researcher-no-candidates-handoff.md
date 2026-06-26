# Handoff — researcher `No model candidates available` (FC-eligibility + GPU-pressure starvation)

**Date:** 2026-05-25
**For:** the next session fixing the `1.0 prior_art_search` DLQ on mission 77 (and the
researcher agent more generally). Root cause is CONFIRMED below — don't re-investigate;
act on §3.

---

## §0 — Symptom

Mission 77, step `1.0 prior_art_search` (agent `researcher`), task **166110**:
```
❌ 1.0 priorartsearch  —  All models failed for 'researcher': No model candidates available
```
Fired 5× (15:21 → 16:04), then DLQ. `worker_attempts=5`.

---

## §1 — Confirmed root cause (do NOT re-derive)

`researcher` requires a **function-calling** model and none is currently *servable*, so the
selector returns an **empty pool**:

1. `researcher` declares a 10-tool `allowed_tools` list (`src/agents/researcher.py:25-36`:
   web_search, find_prior_art, read_file, write_file, file_tree, api_lookup, api_call,
   play_store, github, pharmacy).
2. `packages/fatih_hoca/src/fatih_hoca/requirements_builder.py:233-235` — any agent with a
   non-empty `allowed_tools` ⇒ `reqs.needs_function_calling = True` (also set at :72-73 when
   classification has `needs_tools`). So researcher ALWAYS needs FC.
3. Eligibility keeps only FC-capable models. FC-capable **cloud** (gemini/cerebras/…) is
   rate/rpd/tpm-filtered; FC-capable **local** is scarce + **pressure-blocked** by the other
   mission-77 tasks contending the single GPU.
4. ⇒ `select()` / `pick_for_iter` returns `None` ⇒ empty pool.

### Decisive evidence (live DB, read-only)
- All 5 violations for task 166110: `site=coulson_pool_empty`, `reason=no_candidates`,
  `error_message="No model candidates after 0 failure(s)"`, `failures_count=0`,
  `failure_models=[]`. **Zero models were even tried** — pure eligibility/pressure, NOT call
  failures and NOT the DNS outage.
- Raise site: `packages/coulson/src/coulson/react.py:587-591` (the `pick is None` branch,
  after `pick_for_iter` at :573). (NOT the dispatcher `llm_dispatcher.py:478` path.)
- `admission_violations` in the **last 3 hours** = `no_candidates: 5` (this task) + `rpm: 1`.
  The big historical counts (`daily_exhausted 978`, `rate_limit 607`, `no_candidates 573`)
  are CUMULATIVE all-time, NOT this window — this is NOT broad cloud exhaustion right now.
- `researcher` allows `cheap`→`medium` tier (`researcher.py:15-16`), so tier is NOT the wall;
  the FC requirement + pressure is.
- Other mission-77 tasks got picks fine (model_pick_log: 166122→gemini/gemma,
  166340→Qwen3.5-9B-thinking) — only the FC-requiring researcher starves.

---

## §2 — Why this session's committed fixes did NOT cover it

The 2026-05-25 work (`086d90df` legacy removal, `1a80144a`/`9fc991cd` non_goals, and the
checkpointed RC-A work in `370d240f`) did not and was not meant to touch this path:
- The **RC-A** fix targeted admission↔worker **divergence** no_candidates — admission admits
  a task, the worker re-selects with a *different input estimate* and can't serve. That is a
  `dispatcher_pool_empty` mismatch.
- **This** is `coulson_pool_empty, failures=0` — the selector genuinely has no FC-capable
  servable model. It is the **researcher cloud-eligibility + GPU-pressure starvation** that
  was explicitly listed as OPEN in:
  - `docs/handoff/2026-05-24-mission75-debug-handoff.md` §3 RC-A follow-up #1 ("Flag-
    forwarding … researcher cloud eligibility-filtered → local-only → starves") — NOT done.
  - memory `project_pressure_concurrency_20260524` ("researcher cloud eligibility-filtered
    (FC+tpm+rpd20) → local-only → starves; availability burns worker_attempts→DLQ").

---

## §3 — Two workstreams

### WS-1 — Instrument the pool-empty forensics FIRST (cheap, high-value)
`record_pool_empty_forensics` (called at `react.py:581`) wrote an **empty `snapshot_summary`**
for all 5 violations — so the DB cannot tell us WHICH filter emptied the pool (FC? pressure?
rate? per-provider rpd?). Fix it to capture, per candidate model, the eligibility-rejection
reason at the moment of the empty pool.
- Look at: `packages/fatih_hoca/src/fatih_hoca/selector.py` `_check_eligibility` (~390) —
  it presumably already computes a per-model reject reason; surface those into the forensic
  `snapshot_summary`/`extra_json` instead of dropping them.
- The selector also enforces pool pressure as the "single source of truth"
  (`llm_dispatcher.py:484-487`) and returns `None` when nothing clears the urgency threshold
  — capture the threshold + each candidate's score vs threshold too.
- **Acceptance:** re-run (or `/dlq retry 166110`) under contention and read
  `admission_violations.snapshot_summary` for task 166110 — it must now name the rejected
  FC-capable models + the exact reason (e.g. `gemini/*: rpd_exhausted`, `Qwen-FC-local:
  pressure_blocked`). Then WS-2's direction is data-driven, not inferred.

### WS-2 — The starvation itself (design — get founder direction; avoid invented knobs)
An FC-requiring agent has no fallback when cloud-FC is rate-capped AND local-FC is GPU-
contended. Candidate directions (pick ONE with founder, per `feedback_no_invented_surfaces`):
- **a.** Reserve / prioritize one FC-capable local model for tool-using agents under
  contention (selection-weight or a small reserved slot).
- **b.** Relax cloud rpd/tpm eligibility filtering for FC when local-FC is contended (let a
  rate-limited-but-reachable cloud FC model through with backoff rather than excluding it).
- **c.** Lower `ONESHOT_CONCURRENCY` (currently 4 static) so 4 multi-call tasks don't thrash
  the 1 GPU slot and starve the local pool — see `project_pressure_concurrency_20260524`
  (pressure limits CALLS not TASKS; the static task cap is the thrash source).
- **d.** Let researcher degrade to a non-FC model with prompt-driven (non-FC) tool use when
  no FC model is servable — a real fallback instead of DLQ.
- Validate any change with `packages/fatih_hoca/tests/sim/run_scenarios.py` +
  `run_swap_storm_check.py` (the documented tuning harness).

---

## §4 — Unblock mission 77 now (founder)
- `/dlq retry 166110` once the GPU frees (other mission-77 tasks finish) AND cloud rate
  windows reset — researcher should then get an FC cloud model. The retry won't help while
  the contention persists.
- If it keeps DLQ'ing, WS-2(c) (drop `ONESHOT_CONCURRENCY` to 2) is the lowest-risk
  immediate lever to test.

---

## §5 — Pointers
- Raise site: `packages/coulson/src/coulson/react.py:573-591` (`pick_for_iter` → None).
- FC requirement: `packages/fatih_hoca/src/fatih_hoca/requirements_builder.py:72-73, 233-235`.
- Agent def: `src/agents/researcher.py` (10 tools, tier cheap→medium).
- Eligibility / pressure: `packages/fatih_hoca/src/fatih_hoca/selector.py` (`_check_eligibility`
  ~390, `is_servable` ~592), pressure via `nerd_herd` signals + `combine.py` (NOT scarcity.py
  — that name is stale).
- Forensics: `record_pool_empty_forensics` (react.py:581) →
  `src/infra/admission_forensics.py::record_admission_violation` → `admission_violations`
  table (cols incl. `snapshot_summary`, `extra_json`, `timestamp`, `agent_type`).
- DB (read-only): `file:C:/Users/sakir/ai/kutai/kutai.db?mode=ro`. Probe used:
  `SELECT ... FROM admission_violations WHERE task_id=166110`.
- Constraints: orchestrator may be LIVE — DB-touching pytest deadlocks on WAL; run only
  isolated/DB-free tests with a `timeout`. Never kill llama-server/wrapper/orchestrator.
- NOT in scope / ruled out: today's DNS outage (`getaddrinfo failed`) is unrelated to this
  no_candidates (failures=0 = no calls attempted); the aiohttp leak has its own handoff
  (`2026-05-25-network-aiohttp-error-hunt.md`).
