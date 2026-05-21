# Handoff — Residual reanalysis + next-session prep

**Date:** 2026-05-21
**Session theme:** re-analyze the pile of partial-progress handoffs accumulated
2026-05-16 → 2026-05-20, separate done-from-open, close the small/clear items,
and leave an accurate map for next session.

---

## §1 — What this session shipped (8 commits on `main`)

| Commit | What | Kind |
|---|---|---|
| `b88ed845` | marketing_copy writes to mission `workspace_path`, not repo root | prod fix |
| `bb71bb2c` | doc: Z9 reinforce vs Z10 calibration are deliberately separate | coherence |
| `27c11d18` | new missions start `lifecycle_state='active'` (was `'terminal'`) | **prod bug** |
| `404fc60b` | founder_actions stable ordering on same-second `created_at` ties | **prod bug** |
| `c8052a8a` | intake_todo test mocks telegram so `keyboard_sent` path runs | test |
| `9efea092` | move app-store founder-gate lock 14.8 → real upload sub-steps | design fix |
| `ae004547` | close 7 mr_roboto suite reds (safety_guard conftest + 2 more) | test/infra |
| (this) | next-session prep doc + RESOLVED banners + `data/artifacts/` gitignore | docs |

Two were **real production bugs** masked by test drift, not test-only issues:
- **Founder-action gate was dead for every post-Z0 mission.** `add_mission`
  never set `lifecycle_state`, so fresh missions inherited the Z8 column
  default `'terminal'`; the gate (`_missions_lifecycle_column` → `lifecycle_state`
  post-Z0) saw `'terminal' != 'active'` and never blocked. Also broke the
  ongoing-mission resumption query.
- **`founder_actions` "most recent first" was non-deterministic** — `ORDER BY
  created_at` only; same-second inserts tie and SQLite fell back to rowid.

---

## §2 — Prior-handoff status map (re-verified)

| Handoff | Status | Remaining |
|---|---|---|
| 2026-05-16 yalayut-demand-signals-gap | RESOLVED (05-18, `project_yalayut_demand_wiring_closed`) | — |
| 2026-05-17 z6-phase14-mobile-test-failures | RESOLVED (`7e5ecc6d` + `9efea092`) | — |
| 2026-05-17 z7-unwired-features | §1.A wired; bucket-2 mostly addressed | verify-on-mission |
| 2026-05-18 wiring-sweep z1/z6/z8/z9 | CLOSED by 05-19 | — |
| 2026-05-18 wiring-sweep z2/z3/z4/z5/z10 | CLOSED by 05-19 | — |
| 2026-05-18 z0-and-backlog | Z0 merged; Z9/Z10-loop call made (doc-split) | scoped-unbuilt (below) |
| 2026-05-19 wiring-sweep-residuals | §1+§3 done (§3 this session); §2 OPEN | 5 deferred P3s |
| 2026-05-20 mr-roboto-suite-reds | RESOLVED (`ae004547`) | — |

---

## §3 — OPEN work, carried to next session

### 3a. Deferred P3 queue (from 2026-05-19 §2) — 5 items, ~12-18h

Suggested order (from that handoff §4, cheapest blast radius first):

1. **§2.D Z3** — `run_semgrep_layer_filtered` has no trigger. Add a
   `domain_layer_check` posthook kind (`auto_wire_triggers=["**/domain/*.py"]`)
   dispatching it with `forbidden_in_domain.yml`. ~2-3h.
2. **§2.E Z2** — `_apply_hint_from_targets` no-ops on fresh missions (workspace
   dir doesn't exist at expansion time). Move call to per-step dispatch, OR
   document re-expansion-only. 30min–2h.
3. **§2.B Z9** — reinforce model-resolver joins `tasks.title =
   model_pick_log.task_name` (fragile). Add `task_id` column + join by id;
   keep title fallback for old rows. ~2-3h (incl. backfill migration).
4. **§2.A Z1** — `propose_spec_patch_from_html_diff` has no caller. Add a
   `[Propose spec patch]` Telegram inline button on `annotate_html_oids` /
   `regen_artifact` results → enqueue the verb → founder review. ~2-4h.
5. **§2.C Z10** — confirmation gate is a 60s busy-poll holding the worker slot.
   Replace with `asyncio.Event` keyed by `confirmation_id` + a
   `confirmation_resolved` continuation. ~4-6h; do last — changes worker-slot
   semantics, land after a release window.

All five are "connect existing correct code" except §2.B (one migration) and
§2.C (event machinery). Each needs a host-path test (the unit suites passed
*with* the original bugs — that is the recurring lesson).

### 3b. Scoped-but-unbuilt (from 2026-05-18 z0-backlog §2b)

- **`gorsel_ustasi`** — image-gen provider-abstraction package. Scoped Z1→Z2,
  built by neither. Z1 emits placeholder images only. (`project_z1_strategic_locks_20260509`
  named the image-provider abstraction a strategic lock.)
- **Web preview hosting** (C10/F1) — `emit_preview_url` verb exists; the
  cloudflared / local-port / GitHub-Pages host + viewer was never built.
- **Z8 on-call cloud impls** — `restart_service` / `scale_up` etc. fail loud
  (honest stubs); need vendor cloud-API wiring when accounts exist.

### 3c. The real reliability proof — still not done

No i2p mission has ever run end-to-end. Per every wiring sweep: static grep
closed the *wiring*, but a real prototype-tier mission run is the only thing
that surfaces what grep missed. Highest-signal, heaviest. Do after 3a.

---

## §4 — Environment / hygiene notes for next session

- **`safety_guard` editable install:** the package was shipped by Z0 but never
  `pip install -e`'d into the venv (its 31 unit tests had never run). Fixed
  in-session via `pip install -e packages/safety_guard` + the conftest
  `_PACKAGE_SRCS` add (`ae004547`). A fresh clone still needs the editable
  install — worth adding to the venv setup script.
- **`data/artifacts/` + `data/mission_*/`** now gitignored — production mission
  writes were leaking into the repo (marketing_copy default path, fixed
  `b88ed845`). If you see them in `git status`, the gitignore covers them.
- **Pre-existing env skips (NOT regressions):** `fastapi` +
  `sentence_transformers` not installed → all webhook-route + embedding tests
  fail/skip everywhere. Install or accept.
- **conftest collision:** never mix `tests/` and `packages/*/tests/` in one
  pytest invocation — the dual `conftest.py` triggers a pluggy
  "Plugin already registered" error. Run dirs separately.
- **Worktree fleet:** the ~45 stale `worktree-agent-*` worktrees were pruned
  this session (commit before `b88ed845`). Branch list is clean (`main` only).

---

## §5 — Verification gate (run before declaring next-session work done)

```
# originally-failing tests this session closed (expect all green)
.venv/Scripts/python -m pytest \
  tests/founder_actions/test_lifecycle.py \
  tests/founder_actions/test_repo.py \
  tests/i2p/test_intake_todo.py \
  tests/workflows/test_z5_t5_distribution.py \
  tests/workflows/test_z6_t6a_reversibility.py \
  tests/workflows/test_z6_polish_phase14_mobile.py \
  tests/workflows/test_i2p_v3_reversibility_tags.py -q
# -> 45 passed

# full mr_roboto suite
.venv/Scripts/python -m pytest packages/mr_roboto/tests/ -q
# -> 732 passed, 2 skipped, 0 failed

# wiring-sweep smoke gate (regression guard for the 05-18/05-19 closures)
.venv/Scripts/python -m pytest \
  tests/test_wiring_sweep_20260518.py tests/test_wiring_sweep_p2_20260518.py \
  tests/test_z3_p2_cascade_20260518.py tests/test_z8_sweep_20260518.py \
  tests/test_z6_sweep_20260518.py tests/test_z1_sweep_20260518.py -q
# -> 45 passed + 25 subtests
```
