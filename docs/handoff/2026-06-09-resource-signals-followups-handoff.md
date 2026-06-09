# Handoff — Desktop Resource Signals: follow-ups & loose ends

**Date:** 2026-06-09
**Status of the feature:** SHIPPED to local `main` (HEAD `e045d8d6`, fast-forward merge, conflict-free). **NOT pushed. Restart-gated** (KutAI not yet restarted on this code).
**Spec:** `docs/superpowers/specs/2026-05-31-resource-signals-design.md` (Status: Implemented).
**Plan:** `docs/superpowers/plans/2026-06-09-desktop-resource-signals.md` (all 19 tasks + 4 review fixes done).
**Memory:** `[[project_resource_signals_20260609]]`, `[[feedback_test_serialization_boundary]]`.

## What shipped (one paragraph)

No "governor" package. Two new **local-only, negative-only** pressure signals feed the *existing* Nerd-Herd `pressure_for` fold: **S13 user-presence** (`packages/nerd_herd/src/nerd_herd/signals/s13_presence.py`) and **S14 machine-contention** (`s14_contention.py`). Load mode is reborn as modifier **M4** (`modifiers.py::M4_load_mode_weights`) weighting S13/S14 only (full=0 silence / heavy=1.5 / shared=2.0 / minimal=eligibility veto). Fullscreen + external-GPU≥30% use the `−10` sentinel → scalar pegs −1.0 → admission blocked. Placement (cloud↔local) and deferral fall out of the existing pressure→ranking→admission path — **no new policy code**. `need_ctx` moved out of the dispatcher into Fatih's `Pick` (`fatih_hoca/need_ctx.py::compute_need_ctx`, `_resolve_load_ctx` in `llm_dispatcher.py`). VRAM-cap deleted (`get_vram_budget_mb` returns raw free; load_manager sync stubs + dead P1 context symbols gone). **S12 is `s12_pool_balance`** (free-cloud, untouched) — that's why the new signals are S13/S14.

Tests: **707 pass** across nerd_herd + fatih_hoca (2 pre-existing failures, see below).

## ⚠️ Two hard gates before this is "real"

1. **Live verification (USER must restart KutAI).** All tests are in-process or unit; the round-trip test proves the HTTP-snapshot path now carries the fields. But *"presence actually shifts a live mission local→cloud"* has NOT been observed. After restart: watch a real **present-user** session (idle < 30s) and a **fullscreen/game** session — confirm picks move to cloud and low-urgency tasks WAIT. Check `model_pick_log` and the pressure scalar in selector logs.
2. **Push** — local `main` only.

## Cheap wins (knock out first — ~20 min total)

1. **Pre-existing 2 red tests (NOT caused by this work, but worth fixing).**
   `packages/fatih_hoca/tests/test_counterfactual.py::test_cli_runs_on_empty_db` and `::test_cli_reports_agreement_rate`. They `subprocess`-spawn the counterfactual CLI with `PYTHONPATH` set to **only** `fatih_hoca/src`, so the child can't `import nerd_herd` (pulled in by `requirements.py`). Fail identically on the base commit. **Fix:** in the test's subprocess env, add `packages/nerd_herd/src` (and any other package dirs the CLI imports) to `PYTHONPATH`. ~1-line.
2. **README drift** — `packages/nerd_herd/README.md` lines ~62/95/302/336 still describe `get_vram_budget_mb()` as "free VRAM × mode fraction". It now returns **raw free** (cap removed). Reword.
3. **Prometheus metric name** — `nerd_herd/exposition.py` exports a `vram_budget_mb` gauge that now reports raw free (no mode scaling). Name is mildly misleading; rename or update its help text.
4. **M4 comment nit** — `modifiers.py` `_M4_BY_MODE` comment says "otomatik"; there is no "otomatik" mode string (modes are full/heavy/shared/minimal; auto is the `_auto_managed` flag). Cosmetic.

## Deferred by design (do NOT do unless triggered)

- **VRAM safety margin (spec §3.5)** — ~0.5–1 GB brake baked into `--fit`. Skipped on purpose: P1 need-ctx + small load windows already removed the OOM; this is hardening. **Reopen only if a spike-OOM is observed post-launch.**
- **BUG-3 cold-local permissiveness** — raw-free VRAM makes S9's cold-local-VRAM-OK branch (`s9_perishability.py`) fire more readily in shared/heavy. This is intended ("placement not capping"); M4's cloud-bias now covers contention. The §3.5 margin is its only belt-and-suspenders. No action unless over-admission is seen.
- **Telegram relabel** — buttons still say full/heavy/shared/minimal; spec's preset language (Full/Otomatik/Heavy-Shared/Minimal) not applied. Pure UI string change, out of scope.

## Tuning (after a few live missions — needs real data)

All S13/S14 constants are **starting guesses**, tune against `kutai.jsonl` idle/RAM distributions the way S1/S9 thresholds were:
- `s13_presence.py`: `ACTIVE_IDLE_S=30`, `PRESENT_IDLE_S=300`, penalty band −0.6→0.
- `s14_contention.py`: `RAM_USED_FLOOR=0.80`, `RAM_USED_CAP=0.95`, `EXTERNAL_GPU_VETO_FRACTION=0.30`.
- M4 weights `heavy=1.5 / shared=2.0` — dial cloud-bias strength.
After any change, re-run `packages/fatih_hoca/tests/sim/run_scenarios.py` + `run_swap_storm_check.py` (set `PYTHONPATH` to all `packages/*/src` — the swap-storm script only injects fatih_hoca/src itself).

## Gotchas the next session MUST know

- **Serialization boundary (the bug that 19 green commits hid):** `SystemSnapshot` is built in TWO places — the sidecar (`nerd_herd.py::snapshot()`, emitted via `asdict` in `exposition.py`) AND the client deserializer (`client.py::_parse_snapshot`, what prod reads via `run.py:518`). Any NEW snapshot field must be added to BOTH + a `test_parse_snapshot_preserves_*` round-trip test + bump `exposition.py::API_VERSION`. In-process tests will pass while the feature is dead in prod otherwise. See `[[feedback_test_serialization_boundary]]`.
- **Concurrent agent sessions write to `main`'s shared working tree.** During this merge, another session's uncommitted `prompt-foundry.md` edits (96 lines) appeared in the tree. Use a **worktree** (`EnterWorktree`) for any non-trivial work to isolate from them. Founder's uncommitted files were preserved (stash → ff → pop → resolve CLAUDE.md → unstage).
- **`is_servable` carve-outs:** the continuation path now allows held-loaded-local under both `no_vram_available` AND `load_mode_minimal` (selector.py). If you add more eligibility rejections, decide whether continuation should bypass them too.

## Fast-start checklist for next session

1. `git -C <repo> log --oneline -1` → confirm `e045d8d6` (or later if pushed/advanced).
2. Cheap wins #1–#4 above (counterfactual PYTHONPATH first — turns the suite fully green).
3. If KutAI restarted: do the live verification (present-user + fullscreen). Capture pressure scalars from logs.
4. Tune S13/S14 constants from observed data; re-run the two sim scripts.
5. Push when founder says.
