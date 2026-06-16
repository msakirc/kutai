# Handoff: DB Phase B — deferred follow-ups

**Date:** 2026-06-16. **Predecessor:** Phase B registry slice DONE+MERGED+PUSHED `origin/main e523f889` (model-registry domain → `fatih_hoca`; code-ownership only, file-split deferred). See `project_db_dabidabi_phaseB_registry_20260615` memory + `docs/superpowers/plans/2026-06-15-db-dabidabi-phaseB-registry-slice.md`.

This lists everything intentionally LEFT for later, in rough priority order. Each is independent unless noted.

---

## 0. RESTART-GATE verification (do this FIRST, after `/restart`)

Not a code task — the verification that couldn't run during dev (live bot held the prod DB lock; one subagent then accidentally killed the wrapper). Env is already re-pip'd to main + import-clean.

- [ ] Boot smoke: logs show NO `no such table: model_pick_log / registry_events / model_stats`. `/benchpicks` renders the 7-day table (exercises `get_pick_summary`).
- [ ] `record_action_event` writes a `scope='action'` row post-restart (via `record_vendor_cost` / mr_roboto `_audit_action` → shim → fatih).
- [ ] Full beckman suite (WIN flags): `python -m pytest packages/general_beckman/tests -o addopts="" -p no:aiohttp -q`. Beckman persists `model_pick_log` via `pick_log.write_pick_log_row` → `fatih_hoca.db.insert_pick_log_row`; this delegation was NOT exercised in a beckman flow during dev.
- [ ] Integration (real-DB, bot stopped): `python -m pytest tests/integration/ -m "not llm" -o addopts="" -p no:aiohttp -q`.

---

## 1. Physical `registry.db` file-split (the actual moonshot landing — own slice)

Phase B did code-ownership only; tables still live in `core.db`. Splitting the FILE is the deferred payoff but has real prereqs (this is why it was deferred — the first plan draft tried it and an adversarial review found it unsafe):

**Blockers to clear FIRST:**
- **Multiple connection openers** must all see the split file. Today registry is read/written through 3 paths that a singleton-only ATTACH would miss:
  - `fatih_hoca/registry_store.py` — its OWN sync `sqlite3` singleton (`_get_conn`), opens `DB_PATH`.
  - `connect_aux` / `connect_aux_sync` (dabidabi) — open a NEW connection per call at the passed path.
  - the async `get_db()` singleton.
  Each must ATTACH `registry.db` + qualify table names, OR be unified onto one access path. ATTACH on one connection is invisible to the others.
- **Cross-file atomicity dies under WAL** — a data migration `INSERT INTO registry.<t> SELECT * FROM main.<t>; DROP main.<t>` is NOT atomic across attached files in WAL mode (documented SQLite limitation). Needs per-table existence guards + idempotent verify + crash-safe recovery, NOT a single cross-file transaction.

**To build:** the `attach_db()` / `attached()` engine primitive (designed + reviewed-sound in plan draft 1, omitted from the slice since unused without the split). Then qualify registry CREATEs with `registry.` schema prefix; the registered schema creates into the attached file.

**Reversible final move** once code-ownership + conn-unification are clean: flip the path, run the one-shot guarded data migration.

---

## 2. Read-side ownership leaks (cheap; not guarded today — write-guard only covers INSERT/UPDATE/DELETE)

Registry-table READS still outside `fatih_hoca` (code-ownership gaps, not safety issues):
- [ ] `packages/db/src/dabidabi/__init__.py` `record_confidence_claim` (~lines 8579, 8591) — raw `SELECT ... FROM model_pick_log` (two-tier join by task_id then task_name) inside the ENGINE. Add a `fatih_hoca.db` read helper (e.g. `get_latest_pick_for_task`) + repoint.
- [ ] `src/app/telegram_bot.py` `cmd_ops_log` (~line 11545) — raw `SELECT ... FROM registry_events WHERE scope='action'`. Route through a fatih read-API.
- [ ] `packages/mr_roboto/.../executors/record_verdict.py` (~223/238/251) — reads `model_pick_log` (the 2 task JOINs + fallback). Currently SANCTIONED (works as plain SQL while all in core.db). If you want strict read-ownership, give it a fatih read-API too. NOTE: these are the JOINs that will need ATTACH when §1 happens.

---

## 3. Cold-init entry points (low severity, fresh-DB only)

Registry tables are created ONLY when `fatih_hoca` is imported before `init_db()` (registration side-effect). `run.py` + `tests/conftest.py` import it. These CLIs call `init_db` WITHOUT importing fatih → on a brand-new DB they'd create it missing the 5 registry tables (self-heals next time a fatih-importing process runs init_db, since CREATE IF NOT EXISTS):
- [ ] `src/infra/dlq_feedback.py` (~line 198, `python -m src.infra.dlq_feedback`)
- [ ] `src/infra/mission_lessons.py` (~line 297, `python -m src.infra.mission_lessons emit-dlq`)
Fix = mirror run.py's `import fatih_hoca  # noqa: F401` in each `__main__`. (Trade-off: pulls fatih's heavier import into lightweight DLQ CLIs. Alternative long-term: make registration not import-dependent — but that couples engine→fatih, which the design rejected.)
- [ ] Consider a NEGATIVE guard test: `init_db()` without importing fatih → assert registry tables missing, so a future package conftest that adds `init_db` + a registry write can't silently regress (a regression of exactly this kind was found+fixed in `tests/conftest.py` during the slice).

---

## 4. Remaining Phase B domain splits (each its own slice, repeats the §0-pattern)

- [ ] `kdv_state` → **`kuleden_donen_var`** (NOT fatih — Phase B lumped it under "registry" but the table is KDV's; owner code = `src/infra/kdv_persistence.py`).
- [ ] `ledger.db`: `cost_budgets` + `model_call_tokens` (zero atomicity loss — already excluded from `restore_mission_db_rows`'s MISSION_SCOPED_TABLES). Owner: general_beckman or a small ledger module — decide.
- [ ] `yalayut.db` → leaf, yalayut already owns its tables.
- [ ] `core.db` spine → **`general_beckman`** schema relocates in but stays ONE file (never split — `restore_mission_db_rows` green-tag rollback is atomic across missions/tasks/artifacts/growth; breaks the instant any crosses a file boundary).
- [ ] `product.db` (comms) → **BLOCKED/deferred by user** (no single owner; unvalidated fresh subsystem). Revisit after first live mission exercises comms.

---

## 5. Phase A tail (makes `dabidabi` truly standalone)

- [ ] Invert the 5 lazy/optional engine UP-reaches inside `dabidabi/__init__.py` (`src.tools.shell`, `src.memory.vector_store` ×3, `src.infra.dead_letter`) → injected callbacks/hooks so the engine depends on nothing in `src/`.
- [ ] Relocate the ~70 in-app `src/*` callers off the `src.infra.db` / `src.infra.times` alias shims → `dabidabi`, then delete the alias shims. (`src/infra/registry_store.py` is now ALSO an alias shim → its callers can move to `fatih_hoca.registry_store` and the shim deleted.)

---

## 6. Hygiene / debt (cheap, anytime)

- [ ] **Stale doc:** `CLAUDE.md` still says "Database: SQLite via `src/infra/db.py`" and implies registry tables are engine-owned. Now `src/infra/db.py` + `src/infra/registry_store.py` are alias shims and `fatih_hoca` owns the registry domain — add a line.
- [ ] **Pre-existing test debt (NOT from this slice):** `tests/fatih_hoca/test_pick_telemetry.py` — 5 fails reproduce on pristine main (`bb86b00e`). Test-isolation defect: it monkeypatches `src.infra.db.DB_PATH`/`_db_connection` but the dabidabi singleton binds to the REAL prod `kutai.db` → these tests touch the LIVE DB during runs. Fix the isolation (use `dabidabi.configure(tmp)`); also a latent prod-DB-corruption risk.
- [ ] **Write-guard is line-based:** `packages/fatih_hoca/tests/test_registry_write_guard.py` regex won't catch a multi-line `INSERT\n... INTO model_pick_log`. Already hardened for `INSERT OR IGNORE/REPLACE`. Optional: make it AST/multiline-aware.

---

## 7. Cross-cutting GOTCHAS (carry forward — bit us this session)

- **WIN pytest hang:** ALWAYS run package tests with `-o addopts="" -p no:aiohttp` (the aiohttp plugin orphans a PATH-`python` child that deadlocks + holds SQLite locks).
- **NEVER bulk-kill `python.exe`.** The live KutAI stack runs on the SYSTEM **Python310** interpreter (same as pytest), NOT only `.venv`. A subagent's `Stop-Process` on "stray Python310 procs" killed the live wrapper this session. Kill only by exact PID after matching CommandLine; exclude `kutai_wrapper`/`run.py`/`nerd_herd`/`yazbunu`/`yasar` and the PID in `logs/guard.lock`. See `feedback_subagent_killed_live_wrapper` memory.
- **Editable-install `.pth` cross-worktree rot:** multiple packages' `.pth` pointed at DEAD worktrees from prior sessions (incl. `yasar_usta` → `agent-ac37b05d`). Repointed all to main this session. Before any `/restart`, audit: `pip list | grep worktrees` should be empty; re-`pip install -e ./packages/<X> --no-deps` from MAIN path for any stray.
- **Concurrent agent sessions cross `main`** — use a git worktree for implementation; integrate via real 3-way merge (NEVER force/reset/`-X theirs` — main advanced 4 commits during this slice and a naive overwrite would have reverted prior_art + selector work).
