# Handoff: DB Phase B deferred — remaining (§5b-db, §1, §4)

**Date:** 2026-06-17. **Predecessor:** `docs/handoff/2026-06-16-db-phaseB-deferred-handoff.md`.

## What shipped (all MERGED to `main`, NOT pushed — restart-gated)

| Slice | Commit | Notes |
|---|---|---|
| §2 read-ownership | `4b743458` (merge `1d769200`) | 3 fatih_hoca.db read helpers, 3 raw-SQL callers repointed |
| §3 cold-init | `20c9d7eb` | `import fatih_hoca` before `init_db()` in dlq_feedback + mission_lessons CLIs |
| §6 doc | `4af8fc2e` | CLAUDE.md DB-ownership line |
| §5a engine→src inversion | merge `324fc277` | `dabidabi.hooks` registry + `src/infra/db_hooks.py::wire()` (called in run.py after DB writable); engine drops 5 `src.*` up-reaches; unset hooks = best-effort no-op |
| §5b times | merge `caab8e0f` | `src.infra.times` → `dabidabi.times`, **shim DELETED**; 14 importers (8 abs + 6 relative) + 6 test files |
| §5b registry_store | merge `6f9088bb` | `src.infra.registry_store` → `fatih_hoca.registry_store`, **shim DELETED**; ~18 sites |

Verified green: §2 11/11, §5a 9+2+5, registry_store 100. Also: uninstalled orphan `salako` editable install (pointed at a deleted worktree). Concurrent sessions landed `d70b6bfa` / `bf776ef3` / etc. on main during the work — all merges were clean 3-way; my commits verified as ancestors.

**Pre-existing failure (NOT introduced here, do not chase in these slices):** `tests/unit/test_bench_picks_command.py` ×2 — `cmd_bench_picks` reads `fatih_hoca.db.get_pick_summary` → the dabidabi singleton (env `DB_PATH`), so the test's `monkeypatch.setattr(tg_mod, "DB_PATH", ...)` never reaches the command → it reads PROD, not the seeded tmp DB. Proven failing on MAIN. This is §6 isolation debt (same family as `tests/fatih_hoca/test_pick_telemetry.py`). Real fix = `dabidabi.configure(tmp)` per-test with teardown.

---

## §5b-db — the LAST shim: `src/infra/db.py` (own slice, biggest)

`src/infra/db.py` is a pure `sys.modules` alias to `dabidabi`. Deleting it = the Phase-A-tail payoff, but it has **100 importers** + relative-form landmines.

**Decision recorded:** this is **pure hygiene** (the alias is harmless and correct) vs **HIGH live-restart breakage risk** (100 files; one missed import → bot fails to boot on next `/restart`). It was deliberately deferred, not forgotten. Do it as its own focused slice only when worth the churn.

**The landmine (learned the hard way on `times`):** an absolute-path grep MISSES relative imports. `times` had 8 absolute + **6 relative** (`from ..infra.times`, `from .times`) — the relative ones only surfaced as a `ModuleNotFoundError` at `telegram_bot` import AFTER the shim was deleted. For `db`, grep ALL forms BEFORE deleting:
```
grep -rn "from src\.infra\.db import\|from src\.infra import db\b\|import src\.infra\.db\b\|from \.\.infra\.db\|from \.db import\|from \.\.infra import db" src packages tests --include=*.py
```
Also check string references (`"src.infra.db"` in monkeypatch/sys.modules stubs).

**Procedure (mirrors times/registry_store):**
1. Grep all forms → repoint each to `from dabidabi import X` (verify dabidabi exports every X used — it should; the shim was a full rebind).
2. Watch for files that import BOTH (e.g. `from src.infra.db import get_db` + `from src.infra import db`).
3. `git rm src/infra/db.py`; re-grep all forms repo-wide → must be empty (incl. docstrings/strings).
4. py_compile every touched file; run affected test packages.
5. Note: many callers already use `dabidabi` directly; the 100 is the long tail.

---

## §1 — physical `registry.db` file-split (moonshot; migration + restart gated)

Code-ownership is done (fatih owns the registry domain); tables still live in `core.db`. Splitting the FILE has real prereqs — see predecessor handoff §1. Summary:
- **3 connection openers** must all ATTACH the split file or be unified: `fatih_hoca/registry_store.py` sync singleton, `connect_aux`/`connect_aux_sync` (dabidabi, per-call), async `get_db()` singleton.
- **Cross-file atomicity dies under WAL** — the data migration needs per-table guards + idempotent verify + crash-safe recovery, not one cross-file transaction.
- Build the `attach_db()`/`attached()` engine primitive (designed in plan draft 1), qualify registry CREATEs with `registry.` prefix, then flip path + run the one-shot guarded migration.

---

## §4 — remaining domain splits (each own §0-pattern slice)

- `kdv_state` → **`kuleden_donen_var`** (NOT fatih; owner code = `src/infra/kdv_persistence.py`).
- `ledger.db`: `cost_budgets` + `model_call_tokens` (zero atomicity loss — already excluded from MISSION_SCOPED_TABLES). Owner: general_beckman or small ledger module.
- `yalayut.db` → leaf, yalayut owns its tables.
- `core.db` spine → **`general_beckman`** schema relocations IN, stays ONE file (never split — `restore_mission_db_rows` green-tag rollback is atomic across missions/tasks/artifacts/growth).
- `product.db` (comms) → **BLOCKED/deferred by user** (no single owner; unvalidated). Revisit after first live comms mission.

---

## §5 Phase A tail — leftover

`§5b-db` above is the remaining alias relocation. Still open from Phase A tail:
- Invert the lazy/optional engine UP-reaches NOT covered by §5a hooks (only the 5 service reaches + the `src.infra.times` self-reach were done). Audit `dabidabi/__init__.py` for any `from src.` that re-appears.

---

## CARRY-FORWARD GOTCHAS (bit us this session)

1. **NEVER run pytest via `run_in_background` on Windows.** It orphans a `python -m pytest` child that holds the prod `kutai.db` SQLite lock for HOURS (user saw "hung for hours"). Run FOREGROUND with a hard shell `timeout`, and reap by exact PID in the same command. pytest.ini's `timeout=` is INERT (pytest-timeout not installed). See memory `feedback_no_background_pytest_orphans`.
2. **Worktree pytest MUST keep `--import-mode=importlib`** (it's in `pytest.ini` addopts). The old `-o addopts=""` memory flag DROPS it → true collection HANG. Use `-p no:aiohttp` ONLY; keep ini addopts.
3. **Raw `python -c` loads MAIN, not the worktree** — the editable-install import finder shadows PYTHONPATH. Only pytest (via the root + tests/ conftests that evict + prepend `packages/*/src`) resolves worktree code. The conftests' `_PACKAGE_SRCS` now include `packages/db/src` (was missing → masked worktree dabidabi).
4. **`dabidabi.configure()` requires an ABSOLUTE path.**
5. **Relative-import landmine** (see §5b-db above) — grep relative forms before any shim delete.
6. **Concurrent agent sessions cross `main`** — use a worktree; integrate via real 3-way merge (never force/reset). Main advanced ~3 commits during this session; merges stayed clean.
7. **NEVER bulk-kill `python.exe`** — live wrapper runs on the SAME system Python310 as pytest. Kill only by exact PID after matching `pytest` in CommandLine; exclude `kutai_wrapper`/`run.py`/`guard.lock` PID.

## USER restart-gate (predecessor §0 still applies)
`/restart` + verify: boot smoke (no `no such table` for registry tables), `/benchpicks`, `/ops_log <mid>`, full beckman suite (`-p no:aiohttp`, keep ini addopts), then push.
