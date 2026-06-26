# Handoff — mission-77 crash loop + the capacity root (2026-05-27, ~01:10 local)

**For:** the next session. Read §1 FIRST — there is a live crash loop. Then §4 (the
real root the founder keeps re-raising) and §5 (the two genuine open fixes).
Tonight shipped 7 gate fixes (all on `main`, verified) but the system is now in a
crash loop that those fixes do NOT cause and do NOT cure.

---

## §1 — LIVE INCIDENT: orchestrator crash loop (do this first)

**Symptom:** "Kutay online. Buttons ready." every ~3 min; DLQ spam; founder: "its a
crash loop". Restarting does NOT help.

**Confirmed mechanism (guard.jsonl, UTC):** orchestrator restarted ~10× between
21:36–22:04 UTC (00:36–01:04 local), every **~2.5–3 min**. Each instance: starts →
selects a model for a **mission-77 grader/reviewer/posthook** task → **goes silent
~2 min** (last log line is `fatih_hoca.selector: model filtered … task=grader|reviewer`)
→ Yaşar Usta's **heartbeat watchdog kills the hung orchestrator** → restart → re-picks
the SAME ready task → hangs again. Regular cadence = watchdog timeout, not a Python
crash (no traceback in guard.jsonl).

**Why restart can't fix it:** the poison tasks stay `status=ready`; every fresh
instance re-picks them and re-hangs. It is a **poison-queue + hung-call** loop.

**Root of the hang:** model calls are not returning (capacity-starved + the 05-25
network/DNS instability). A single hung call **outlives the Yaşar Usta heartbeat**, so
the watchdog restarts the whole orchestrator instead of just that one call failing.

**It is NOT tonight's gate fixes:** `reviewer` tasks hang too, and reviewers were
never touched. None of the gate fixes go near call execution or the heartbeat.

### Immediate remediation (founder, via Telegram — lifecycle is founder-owned)
**Pause or kill mission 77.** `get_ready_tasks` filters `m.lifecycle_state='active'`
(`src/infra/db.py:4489`), so once mission 77 leaves `active`, its tasks stop being
`ready` → orchestrator stops picking poison → loop ends. Mission 77 is
capacity-starved and unsalvageable in its current state (see §4). Every DLQ/hang
tonight was mission 77 (#166110 researcher, #166124 analyst, their grade/review/
posthook children, new spawns 177xxx/178xxx).
**Do NOT taskkill llama-server / wrapper.** Killing the orchestrator process is OK
(Yaşar Usta auto-restarts) but won't help while m77 tasks are ready.

---

## §2 — What shipped tonight (all on `main`, HEAD; verified RED→GREEN)

| commit | fix | tests |
|--------|-----|-------|
| `8fa5e5d1` | forensics: `select(diag_out=…)` names which filter empties the pool → `admission_violations.snapshot_summary` | 6 + 51 reg |
| `ea0d5b2d` | beckman: mechanical tasks EXEMPT from `ONESHOT_CONCURRENCY` lane cap (`count_in_flight` excludes mechanical + `has_ready_mechanical` gate + per-task LLM guard) | 6 + 13 reg |
| `14c7141e` | beckman: grade/summarize post-hooks spawn `kind=overhead` not `main_work` (`_posthook_kind`) | 5 + 30 reg |
| `68cea171` | orchestrator: bare `asyncio.TimeoutError` → `error_category=timeout` + real msg (`_dispatch_exc_to_result`) | 4 |
| `581c3b86` | **(NOT this session — founder/other)** `litellm.disable_aiohttp_transport=True` | 1 |
| `17a91386` | beckman: mechanical post-hook `needs_review` → mark task completed, no reviewer spawn (`_apply_review`) | 2 |
| `6b0a88cb` | mr_roboto: `compliance_template_present` resolves relative `overlay_path` vs `WORKSPACE_DIR` | 2 + 9 reg |
| `8283c573` | orchestrator: propagate mechanical `needs_review` (was collapsing to `failed`→worker DLQ) (`_mech_action_to_result`) | 4 |
| `1cc2d2b0` | mr_roboto: harden `compliance_template_present` vs non-list `required_documents` (prose placeholder) | 2 + reg |

All green at commit time. Test-run quirk: mixing repo `tests/` and `packages/*/tests/`
in one pytest invocation collides conftests ("Plugin already registered") — run them
in **separate** invocations.

**Caveat — verify post-stabilization:** because the system has been crash-looping,
none of these were observed working e2e on a live mission. The needs_review chain was
verified by source-reading (`on_task_finished:898-904`), not a live run.

---

## §3 — DLQ symptoms this session, classified

- `find_similar_missions` (#166396) — **fixed** (`8283c573`+`17a91386`): needs_review
  was collapsing to worker DLQ; now propagates → RequestReview → completes.
- `compliance_template_present` (#166560/#166124) — **fixed** (`6b0a88cb`+`1cc2d2b0`):
  relative path + prose `required_documents` crash.
- `product_charter_shape_check` (#166100), `prior_art_min_coverage` (#166605),
  `compliance_overlay` schema (#166124) — **CASCADE** from §4 (thin/empty analyst +
  researcher output). Not gate bugs. Will persist until §4 is fixed.
- `Grade task` `TimeoutError:` (#166463/#166466) — cloud call rode the 600s wall-clock
  cap (hung); see §5.1. `68cea171` makes it visible/correctly-categorized but does not
  stop the hang.

---

## §4 — THE RECURRING ROOT (founder re-asks every session) → memory `project_capacity_dlq_root`

Founder, verbatim: *"If there is no quota why the hell these tasks got dispatched at
the first place."* They are right; this is the root behind every DLQ cascade.

**Evidence (kutai.db read-only, 2026-05-26):**
- `166110` researcher: `coulson_pool_empty / no_candidates` — no FC model servable.
- `166124` analyst: `kdv_pre_call_refusal / tpm` + `daily_exhausted` — cloud quota capped.
- All DLQ'd with `error_category=worker`, `worker_attempts=5-6`, **`infra_resets=0`**.

**Mechanism:** KutAI HAS a "no capacity → wait" path — the availability/infra ladder in
`general_beckman/sweep.py` (60s→…→7200s via `tasks.infra_resets`). It never engaged
(`infra_resets=0`) because the failure is categorized **`worker`** (task-fault, capped
at `max_worker_attempts` → DLQ), not **`availability`** (infra-wait). **Mechanical
failures get NO category** (orchestrator mechanical branch returns
`{"status":"failed"}` with none) → `apply.py` defaults to `"worker"`. So "no quota" is
recorded as "bad task" and DLQ'd instead of held.

Plus admit↔worker race: admission admits on a momentary pick; capacity is gone by
worker execution (RC-A); nothing reserves capacity through to the call.

**Fix shape (per `feedback_no_invented_surfaces` — stay in admit/hold + the existing
ladder; invent nothing):**
1. Capacity-class signals (`no_candidates`, KDV `tpm`/`daily_exhausted`/`rate_limit`/
   `circuit_breaker`, uncategorized capacity-shaped mechanical failures) → categorize
   **`availability`**, ride `infra_resets` (terminal at 2h), NOT `worker`→DLQ.
2. **Verify `decide_retry` (general_beckman/retry.py:53-112) does not hard-DLQ
   `availability` at the worker cap** — it currently appears to (attempts≥max → DLQ
   for ALL non-quality categories), which contradicts the wait-ladder design. This is
   likely the real lever: availability must wait, not count toward the 6-cap.
3. Optionally: admission HOLDS (doesn't admit) when nothing can serve the task.

---

## §5 — Two open fixes (do after the loop is stopped; TDD + sim harness, not live)

### 5.1 — A hung call must not outlive the heartbeat (THE crash-loop fix)
A single model call that hangs (no response) blocks progress longer than Yaşar Usta's
heartbeat → the watchdog restarts the WHOLE orchestrator. One bad call should fail fast
(timeout < heartbeat) or bump the heartbeat during streaming, so it degrades to one
DLQ'd task, not a system-wide crash loop.
- Look at: Yaşar Usta heartbeat/watchdog timeout vs `packages/hallederiz_kadir`
  streaming inactivity watchdog (180s first chunk / 20s between) vs the 600s cloud
  wall-clock cap (`llm_dispatcher.py:717`). The inactivity watchdog is NOT catching
  connection-phase hangs (only the 600s outer cap fires — and 600s ≫ heartbeat).
- `src/core/orchestrator.py` per-task `_watchdog` (`:326`) cancels on
  `PROGRESS_TIMEOUT_SECONDS`; confirm that vs the wrapper-level heartbeat that triggers
  restart. The wrapper restart is firing first.

### 5.2 — The capacity root — see §4.

---

## §6 — Pointers / constraints
- HEAD: `1cc2d2b0` on `main`. Pre-session baseline: `9fc991cd`.
- New helpers this session: `orchestrator._mech_action_to_result`,
  `orchestrator._dispatch_exc_to_result`, `apply._posthook_kind`,
  `apply._apply_review` (mechanical branch), `lanes.has_ready_mechanical`,
  `compliance_template_present._resolve_overlay_path`,
  `fatih_hoca.selector.select(diag_out=…)`.
- Side note (not blocking): `add_task() got an unexpected keyword argument 'payload'`
  WARNINGs at startup from yalayut discovery + source-scout enqueues — those callers
  pass `payload=` which `add_task` doesn't accept; their enqueues silently fail. Worth
  a separate fix.
- DB (read-only): `file:C:/Users/sakir/ai/kutai/kutai.db?mode=ro`. WORKSPACE_DIR =
  `C:/Users/sakir/Dropbox/Workspaces/kutay/workspace`.
- Constraints: orchestrator may be LIVE — DB-touching pytest deadlocks on the WAL lock;
  run only isolated/DB-free tests, always with a `timeout` prefix. Never kill
  llama-server / wrapper. guard.jsonl is FLOODED with LiteLLM DEBUG — grep/scan with
  filters, don't tail blindly.
- PARALLEL FLEET hazard: ~40 worktrees share HEAD; a non-session commit (`581c3b86`)
  landed mid-session. Check `git log` before assuming what's deployed.
- Sim harness for any selection/retry change (CLAUDE.md):
  `packages/fatih_hoca/tests/sim/run_scenarios.py` + `run_swap_storm_check.py`.
