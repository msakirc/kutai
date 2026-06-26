# Handoff — "No model candidates for analyst" saga: resolution + what's left

Date: 2026-06-21
Scope: the multi-day recurring `All models failed for 'analyst': No model candidates available → DLQ`
(mission 86, steps 1.4a competitive_positioning / 5.0a design_tokens; tasks 459160 / 459220),
plus the fixes layered during the hunt.

---

## TL;DR

The recurring DLQ had **two stacked causes**, both now fixed in code (merged to `main`, NOT
pushed, restart-gated):

1. **THE operative root — frozen requirements on checkpoint resume** (`8f7baede`).
   `coulson/react.py` restored a stale `reqs` object from the task_state checkpoint *verbatim*
   on every resume, skipping `requirements_for`. So a once-poisoned constraint
   (`local_only=True` on 459220, `est=164750` on 459160) became **immortal** — it survived
   restarts and DLQ re-pends and bypassed every requirements-layer fix we shipped. Admission
   built fresh/correct reqs (admitted cloud), the worker restored the frozen poison and refused
   → the empty pool. Fix: `reqs_for_run()` always re-derives selection requirements fresh on
   resume (react.py:204, 291-298); only non-selection checkpoint state is restored.

2. **The estimate that did the original poisoning — uncapped context-layer budget** (`d19f0051`).
   `src/memory/context_policy.py::compute_layer_budgets` sized the pool as `model_ctx * 0.40`
   with no ceiling → a gemini-class 1M window gave a 400k pool → the `deps` layer (legacy
   `get_completed_dependency_results` full-dump, highest weight) filled ~190k tokens for a
   ~2.6k-input task → B-table learned in_p90 173k → ctx_needed ~226k → every model filtered.
   Fix: `CONTEXT_ABS_CAP=32768` (env `KUTAI_CONTEXT_ABS_CAP`), rollup sanity filter
   (`prompt_tokens <= 64k`, preserves the cost ledger), and a one-shot prune of the poisoned
   `step_token_stats` row. NOTE: the earlier "board layer = 102k" claim was a **misattribution**
   — the blackboard is content-capped (≤3000 chars) before injection; the 190k was deps. See
   `38d2ad47`.

3. **Byte-slice truncation bugs surfaced along the way** (`6ccfdb11`, `16b4bb7e`).
   `format_blackboard_for_prompt` and `fetch_deps` raw-sliced mid-content (broke ```json fences,
   cut sections mid-line) and the caller double-sliced. Replaced with item-/dep-granular
   truncation that keeps or drops whole units and emits an honest omission note.

Supporting: `d0d30fe7` (transient-empty backoff in react), `d2038c67` (degenerate
control-token detection), `d3bbc408` (learned B-table wired into demand projection).

Tasks 459160 / 459220 are **no longer in the tasks table** (resolved/swept) — the stuck loop is
broken.

---

## DO THIS FIRST (deploy — everything below is restart-gated)

1. **Push the backlog.** `git rev-list --count origin/main..main` = **14**. HEAD `62608022`.
   Nothing is on `origin` yet. `git push origin main`.
2. **Full restart** (NOT just `/restart` if it has been flaky — a full wrapper relaunch from the
   main dir guarantees the orchestrator + sidecars reload). All fixes are restart-gated.
3. **Run the B-table prune** if any step still shows a poisoned estimate (>64k in_p90):
   `python scripts/prune_btable_context_poison.py C:/Users/sakir/ai/kutai/kutai.db --apply`
   (dry-run without `--apply`). Run it AFTER the restart so cleared rows can't be re-poisoned.
   It only touches `step_token_stats`, never the cost ledger.
4. **Verify**: mission 86 advances past 1.4a / 5.0a; analyst selections pick models (cloud under
   minimal, or local off-minimal) instead of empty-pool DLQ.

---

## OPEN ITEMS (not yet done)

### P1 — Boot warmup fires a spurious "No candidates" (causes long-backoff wedging)
The FIRST analyst selection after each restart runs against a **cold B-table cache** (stale/empty
in-memory `btable_cache` before the first in-process rollup) and a cold-start `local_only`
default → one spurious empty-pool failure per restart. Harmless on its own (the next selection is
fine), BUT it counts as a real attempt and pushed 1.4a/5.0a onto a ~10-hour backoff ladder
(`worker_attempts=14`, next retry +10h), so they never got a *warm* retry to prove the fix —
which is exactly why it looked unfixed across many restarts. The manual unblock used during the
hunt was resetting `next_retry_at=now, worker_attempts=0`.
- Fix options: warm/load the `btable_cache` from `step_token_stats` *before* the first selection
  at boot; OR don't count a cold-cache empty-pool as a retry-consuming attempt; OR seed
  `local_only` only after classification is ready.

### P2 — New downstream failures (different problem, surfaced once selection was unblocked)
Both tasks now select + execute models, then fail downstream:
- **1.4a → `max_iterations_reached`** (picks gemini/gemma succeed, ReAct can't produce a valid
  final answer in N iterations). Quality/complexity, not selection. Needs its own look:
  iteration budget, prompt, or artifact-shape gate.
- **5.0a → `context_overflow`** on local `Qwen3.5-9B` (+ transient `server_error`). Even at the
  32k cap the built prompt (~40k) overflows small local windows. Either tighten the cap a notch
  for small-window models, or ensure local models load with `need_ctx` matching the actual
  prompt (need_ctx sizing). Had intermittent `success=1` runs, so it's borderline.

### P3 — Deferred architectural / review items
- **`load_mode_minimal` = hard veto on local with no cloud fallback.** Under minimal, when the
  cloud pool is genuinely empty there is no local escape → DLQ. Same class as the phantom-veto
  ruling: pressure/mode should **rank-demote, not veto**, keeping local as last-resort. Owner
  decision — see `docs/superpowers/specs/2026-06-17-phantom-veto-architecture-spec.md` and the
  phantom-veto residuals handoff (`62608022`).
- **Empty-backoff retry placement** (`d0d30fe7`): the transient-empty backoff lives in
  `coulson/react.py`, but CLAUDE.md:97 says retry belongs in `hallederiz_kadir`. Non-ReAct
  callers (overhead grader/classifier/single-shot) don't get the protection. Relocate for
  universal coverage. (In-place fix is correct for the reported ReAct-agent failures; this is
  scope, not a bug.)

### P4 — Hygiene
- **Stale worktrees**: ~9 leftover dirs under `.claude/worktrees/` (prompt-foundry,
  image-plan2-3, fix-thinking-token-budget, db-phaseB-*, etc.) from prior sessions. Each holds a
  full stale code copy. Not on the orchestrator's import path (editable installs point at main),
  but they pollute repo-wide greps/finds and are confusing. Prune the ones whose branches are
  merged.

---

## KEY FILES / COMMITS

| Concern | Commit | File(s) |
|---|---|---|
| Frozen reqs on resume (THE root) | `8f7baede` | `coulson/react.py` (`reqs_for_run`, ~204/291-298) |
| Context-layer budget cap | `5d9c11ea`, `d19f0051` | `src/memory/context_policy.py`, `packages/general_beckman/.../btable_rollup.py`, `scripts/prune_btable_context_poison.py` |
| Blackboard item-granular truncation | `6ccfdb11`, `38d2ad47` | `coulson/context.py` (`format_blackboard_for_prompt`) |
| Deps structural truncation | `16b4bb7e` | `coulson/context.py` (`fetch_deps`, `_fit_dep_blocks`, `_line_safe_truncate`) |
| Transient-empty backoff + finish_reason telemetry | `d0d30fe7` | `coulson/react.py`, `hallederiz_kadir/{response,caller,types}.py`, `coulson/dispatch_helpers.py` |

Memory entries (recall context): `project_checkpoint_freezes_reqs_20260620`,
`project_context_budget_cap_20260618`, `project_blackboard_truncation_fix_20260620`,
`project_empty_response_backoff_20260618`.

---

## LESSONS (so the next session doesn't repeat the loop)

- **Checkpoint-restored state can immortalize a bug.** If a fix is in the requirements/selection
  layer but the symptom survives restarts AND DLQ, suspect a frozen/cached copy on the resume
  path before re-diagnosing the layer itself. (Cost us several rounds.)
- **A "fresh restart" can still run stale behavior** if state is restored from checkpoint or an
  in-memory cache is cold — the process start time being recent does NOT prove the fix is live.
- **Long backoff ladders hide fixes.** A task at `worker_attempts=14` won't retry for ~10h;
  reset `next_retry_at`/`worker_attempts` to force a warm validation instead of inferring.
- **Measure before theorizing.** Five wrong root-cause theories were proposed and disproven by
  direct DB/log measurement; the user's "this doesn't make sense" pushback was correct each time.
