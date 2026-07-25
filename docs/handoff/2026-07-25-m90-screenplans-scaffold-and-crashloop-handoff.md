# Handoff — m90 screen-plan root fixes (scaffold-then-fill) + Kutay crash-loop (2026-07-25)

## ⚠️ TL;DR / DO FIRST
1. **Kutay is in / recovering from a crash-loop** (Yaşar restarting orch every ~2-3 min). Root =
   **infra, NOT the code work below**: intermittent slow orch startup (4.4GB ChromaDB cold-load +
   Posthog component init) exceeding Yaşar's `heartbeat_stale_seconds: 120` → killed before first
   heartbeat → self-perpetuates (rapid restarts never warm the disk cache). **Fix already applied:**
   bumped `heartbeat_stale_seconds: 120 → 300` in `../yasar_usta/registry.yaml:46`. **This needs a
   Yaşar HUB restart to load** (`load_registry` runs once at hub boot; no hot-reload). After the hub
   restart, one cold boot completes → warms cache → stable.
2. **All screen-plan/ADR code is committed locally but NOT pushed and NOT live-validated** (blocked by
   the crash-loop). origin/main = `c1f84631`. Unpushed mine: `ecda4096`, `6b968a65`, `72e6fce2`
   (+ `0ee67bcd` which is a parallel-session orchestrator commit, not mine — leave it).
3. **3 tasks parked** (`failed`) during crash-loop diagnosis: **567455** (5.20b), **567436** (4.9 ADR),
   **567433** (4.6 ADR). Re-pend them (SQL below) once the orch is stable to live-validate the fixes.

---

## 1. CRASH-LOOP (the active fire) — infra, not the code
**Symptom:** `🔴 Kutay dondu — Yaşar 5sn içinde yeniden başlatıyor` every ~2-3 min.
**Diagnosis (grounded):**
- Freeze point in `logs/kutai.jsonl` = last line `Starting component Posthog` (startup phase), before
  the pump loop / first heartbeat. A *successful* boot reached `Vector store initialized: 10
  collections, 1773072 total` at ~75s and the pump at ~90s — so it's **slow, not a hard hang**.
- `data/chroma/chroma.sqlite3` = **4.4 GB** (1.77M vectors; daily backups `chroma.bak.*` also ~4.3GB).
  Cold-loading it + Posthog init occasionally exceeds the **120s** `heartbeat_stale_seconds` → Yaşar
  declares "hung" → kill → restart → cold again (never warms) → loop.
- Trigger was a fresh orch boot ~17:51 on 07-24 (user restart for the scaffold fix) that happened to
  cross 120s; then it self-perpetuated.
- **My committed code does NOT run at startup** (import 0.57s, verified) — ruled out. Task at the freeze
  (#892377) was routine `mission_event_drain`.
- **Contributing noise (my fault):** I had leaked a background e2e script (PID 14164, a stuck
  `asyncio.run(mr_roboto.run())` from ~11:15) that ran 7h holding a torch/embedding resource. **Killed
  it.** Process-hygiene lesson: kill background scripts you spawn.

**Fix applied:** `../yasar_usta/registry.yaml:46` → `heartbeat_stale_seconds: 300`. **Restart the Yaşar
hub to load it.**

**Better permanent fixes (next session, pick one):**
- Orch writes an **early heartbeat** before the 4.4GB vector-store + Posthog load (so slow init never
  looks hung). Cleanest. (orch startup code, `src/app/run.py`.)
- **Prune/compact the ChromaDB** (1.77M vectors, 4.4GB is bloated) → faster every boot.
- Investigate whether `Starting component Posthog` makes a **hanging network call** (no timeout) — if so,
  disable/timeout it. (Not yet confirmed Posthog-vs-vectorstore; grace bump covers both.)

**Current status at handoff:** orchestrator heartbeat ~9h stale (orch down / loop escalated to long
backoff, or hub stopped). Confirm hub state on resume.

---

## 2. m90 SCREEN-PLAN PHASE — the real root fixes (the main work)
**Problem chain (all fixed, TDD'd, committed, restart-gated, live-validated where noted):**

The user pushed past 3 layers of "it's the model / re-pend it" with "no band-aids, dive deeper, no
guesses." Each layer was a real root:

| Layer | Root | Fix | Commit | Live status |
|---|---|---|---|---|
| A. Dir-authoring pipeline | strip/grounding/shape-gate all blind to a **directory** produces → agent couldn't author `.screens/` | `_produces_has_directory`; grounding prefix-match; verifiers self-expand recursive glob; i2p checks +paths | `b6db5d1f` | ✅ pushed + validated (567454 = 9 files) |
| B. 5.20a/b invent screens | plans **disconnected from `screen_inventory.md`** (567454 called Dashboard/Habit-Tracker the "first chunk" when chunk[0] = Landing/SignUp/Login/ForgotPassword); 14/19 screens unplanned; NO correspondence gate | `verify_screen_plans_match_inventory` — route-keyed, cumulative, mechanical, re-pends producer | `45ee6df8` | ✅ pushed + validated (5.20a → faithful chunk-0) |
| C. 5.20b won't converge | weak model degenerate-repeats routeless frontmatter; gate correctly DLQs but can't make a model comply | **scaffold-then-fill**: engine writes authoritative frontmatter (screen_id/route/mission_id from inventory) + grafts model BODY, drops invented → drift structurally impossible | `ecda4096` | ⏳ committed, NOT validated |
| C-bug | scaffold wired as standalone `materialize_*` kind → **runtime `ValueError: unknown posthook kind`** (`_CHECK_KINDS` only routes `verify_*`). My earlier dismissal of `test_every_z1_mechanical_kind_dispatches` masked it (critic_gate failed first) | **folded** materialize INTO `verify_screen_plans_match_inventory` (mutate-then-verify, like `verify_user_flow_shape`→`normalize_user_flow`) | `6b968a65` | ⏳ committed, NOT validated |

**Key file map (screen phase):**
- `packages/mr_roboto/src/mr_roboto/verify_screen_plans_match_inventory.py` — correspondence gate (now
  materializes-then-verifies). Route-keyed (models rename screen_ids; route is the contract). Cumulative
  (chunk N → chunks 0..N). Vacuous-safe + `wiring_suspect`.
- `packages/mr_roboto/src/mr_roboto/scaffold_screen_plans.py` — pure `build_screen_plan_files` (the heart;
  8 TDD cases). Also `verify_screen_plan_shape.py` (dir self-expand + `normalize_screen_plan` = mission_id
  stamp only; the inherits_shell modal-guess was DROPPED as a band-aid per user).
- i2p `5.20a`/`5.20b`: checks = `[verify_screen_plans_match_inventory, verify_screen_plan_shape,
  (verify_screen_consistency)]`; instructions retargeted "author the BODY, frontmatter is mechanical."

**Validated OFFLINE on real m90 disk** (before crash-loop): materialize chunk-1 from the garbage disk →
exactly chunks 0∪1 (`/`,`/signup`,`/login`,`/forgot-password`,`/onboarding`,`/dashboard`,`/habits`,
`/habits/:id`), correspondence ok=True, all shape-valid. **NOT yet validated live** (crash-loop).

**Design boundary (the through-line the user enforced):** *mechanical = deterministic (routes/ids/
mission_id from inventory), semantic = LLM (the body)*. Scaffold makes drift/routeless impossible; the
grade gate still governs body *quality* → a thin body re-pends for enrichment (intended).

---

## 3. m90 ADR PHASE (unblocked by the nullable fix, then two more roots)
- `26e6aea7` **ADR `supersedes_adr_id` nullable** — schema dialect `is_empty_required_value(None)=True`
  (mission-46 guard) false-rejected the honest `null` for a first/non-superseding ADR (8 ADR steps).
  Added `nullable` field modifier. **✅ pushed + live-validated** (567427 completed).
- `72e6fce2` **`verify_adr_shape` unwraps artifact-name envelope** — 4.6/4.9 returned a VALID ADR nested
  under `{...domain..., <artifact_name>: {ADR}}`; gate checked flat top → `adr_id=None` → reject.
  `_unwrap_adr_envelope` picks the nested dict with adr_id + most required fields. Same instruction drives
  4.4/4.6/4.8/4.9/4.10 → recurs. **⏳ committed, NOT validated** (re-pend 567436/567433).

---

## 4. GIT STATE
- **origin/main = `c1f84631`.** Pushed + live-validated: `b6db5d1f`, `26e6aea7`, `45ee6df8` (+ ancestors
  `ade1c348`).
- **Local unpushed (mine, restart-gated, NOT live-validated):** `ecda4096` (scaffold), `6b968a65`
  (materialize-fold), `72e6fce2` (ADR unwrap). Also `0ee67bcd` (parallel-session orchestrator commit —
  not mine; it'll ride along on push).
- **Also uncommitted:** `../yasar_usta/registry.yaml` grace bump (separate repo; commit there if keeping).
- **Push AFTER live-validation** of 567455 + 567436 + 567433.

---

## 5. RE-PEND (after the orch is stable) — validate the 3 fixes live
```sql
UPDATE tasks SET status='pending', worker_attempts=0, grade_attempts=0, result=NULL, error=NULL,
  task_state=NULL, sleep_state=NULL,
  context=json_remove(context,'$._rejection_ledger','$._schema_error','$._schema_error_for_attempt','$._prev_output')
WHERE id IN (567455,567436,567433);
```
- **567455 (5.20b):** the folded correspondence gate should MATERIALIZE chunks 0∪1 (authoritative
  frontmatter + grafted bodies, drop invented) → pass correspondence + shape → complete. Verify `.screens/`
  = exactly the 8 inventory routes with valid frontmatter. (5.20a/567454 already completed.)
- **567436 (4.9) + 567433 (4.6):** `verify_adr_shape` unwraps the nested ADR → pass. Watch the phase-4
  design-ADR chain (4.4/4.8/4.10) — same envelope pattern may surface; the unwrap covers them.
- m90 open DLQs currently **0** (resolved during earlier re-pends; the 3 above are parked `failed`).

**⚠️ Process hygiene:** run monitors with a bounded loop + timeout; **kill any background python you
spawn** (a leaked `mr_roboto.run()` caused the torch-resource contention this session). `pytest` always
with `timeout` (memory rule).

---

## 6. OPEN / NEXT STEPS (deferred, user-steer)
1. **Crash-loop permanent fix** (see §1) — early-heartbeat OR chroma prune OR posthog timeout. Grace bump
   is the band-aid; do a real one.
2. **Coverage 2-vs-5 chunks (product call):** inventory declares 19 screens / 5 chunks; only 5.20a/b +
   5.30a/b exist (≤8 screens). Gate is per-chunk-cumulative so it's *internally consistent* (not a
   failure) — but the full app needs either **dynamic per-chunk fanout** (engine has
   `multifile/expand_template`, wired for code-feature templates not screen chunks — real build) or an
   **intentionally-bounded inventory** (5.0d emits a prioritized MVP set). User: "capping strictly risky."
3. **5.30a/b (HTML prototypes)** are the same directory-authoring pattern (`.web/`) — they'll hit the SAME
   drift once phase-5.30 runs. Consider a `verify_html_prototypes_match_inventory` + scaffold analog.
4. **Stale-dir-clear on re-pend** for dir-produces (invented files persist across retries; the materialize
   now drops them, so lower priority).
5. **Pre-existing red test** `test_every_z1_mechanical_kind_dispatches` fails on `critic_gate` (parallel
   session, not mine — stash-confirmed). Worth a look by whoever owns critic_gate.
6. **ChromaDB bloat** (4.4GB) — vector-store retention/pruning policy.

---

## 7. MEMORY
`project_directory_authoring_pipeline_root_20260723.md` has the FULL saga (Parts 1–3c: dir-authoring,
correspondence gate, scaffold-then-fill, materialize-fold bug, ADR unwrap, + the crash-loop). This handoff
is the operational companion.
