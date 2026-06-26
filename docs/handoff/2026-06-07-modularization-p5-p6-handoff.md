# Modularization P5/P6 Handoff
*2026-06-07. Continues `docs/2026-05-31-modularization-finish-plan.md`. P1–P4 done this session; P5 (thin orchestrator), P6 (split local_model_manager), P7 (guardrail) remain.*

## ⚠️⚠️ READ FIRST — live-bot commit-storm hazard
The running KutAI system (i2p `git_commit` steps / mr_roboto) **auto-commits to `main` every few seconds with `git add -A`**. This session it repeatedly **swept my in-flight git-mv'd files and staged shims into unrelated bot commits** (`fix(image)…`, `fix(materializer)…`), and once left HEAD transiently broken (a `git mv` DELETE committed without its shim → `from src.core.grading import …` would ImportError on the live system).

**RULE for P5/P6 (hot-path — a swept broken intermediate can crash the running orchestrator/loader):**
- Do the work in a **git worktree** (separate dir → own index; the main-dir `git add -A` cannot see it). `git worktree add ../kutay-p5 -b modularization-p5`. Commit on the branch, merge to main when quiet. OR
- Have the user **stop the live runtime auto-commit** (not just parallel Claude sessions) before touching main.
- Do NOT rely on "fast atomic commits" — the bot window is seconds.
- Do NOT use `git stash` here — it un-staged a `git rm` this session and nearly dropped a deletion from a commit.
- After any move, immediately verify HEAD coherence: `git cat-file -e HEAD:<shim path>` and an import smoke.

## What's already DONE (do not redo)
- **P1** delete dead `router.select_model/select_for_task/check_cost_budget` + `grading.apply_grade_result`; relocate `ModelCallFailed`→`src/core/exceptions.py`, `get_kdv`→`src/infra/rate_limiter.py`. `router.py` now a 22-LOC re-export shim.
- **P2** moves: `models.py`→`coulson/actions.py`; `auto_tuner.py`→`fatih_hoca/`; `fast_resolver.py`→`src/tools/`. **`gpu_scheduler.py` KILLED** (was dead — zero slot-API callers; decision doc `docs/2026-06-07-gpu-scheduler-kill-decision.md`). Removed `LocalModelManager.acquire_inference_slot/release_inference_slot` + `inference_busy` field. Deleted obsolete `tests/unit/test_idle_watchdog_race.py`.
- **P3** `grading.py`/`code_review.py`/`reflection_posthook.py` → `packages/coulson/src/coulson/posthooks/` (+`__init__`); `src/core/*` are thin `from coulson.posthooks.X import *` shims; `coulson/reflection.py` imports impl directly (needs private `_GENERIC_REFLECTION_BLOCK`).
- **P4** already done 2026-06-05: `_record_pick` delegates to `src/telemetry/pick_recorder.py`; `src/models/introspection.py` exists; `_estimate_prompt_tokens`/`_get_loaded_*` already evicted. Dispatcher's remaining `get_stats` is a trivial call-counter (keep). Only leftover = delete the `request()` shim, **SP5-gated** (shopping migration) — NOT a P4 task now. See memory `project_shopping_sp5_coupling_20260605`.

Memories: `project_modularization_p1_shipped_20260607`, `_p2_`, `_p3_`, base `project_modularization_debt_20260531`.

---

## P5 — Thin the orchestrator (`src/core/orchestrator.py`, currently **820 LOC**, target ~400)
NOTE: debt-map said 793 / CLAUDE.md says ~746 — it's **820 now and growing** (bot re-adds feature logic). **Re-audit line numbers before editing — they drift fast.** Current pointers (2026-06-07):
1. **Periodic checks → new `src/core/periodic_checks.py`.** `_check_yalayut_discovery` (L139), `_check_source_scout` (L165), `_check_mcp_idle_sweep` (module fn L88). All are timestamp-gated enqueue-a-mechanical-task jobs. Extract into a `run_due(orch)` the pump calls. The `_last_yalayut_discovery`/`_last_source_scout` state (L116-117) moves with them.
2. **Workflow agent-type refresh → `src/workflows/engine/task_refresh.py`.** Inside `_dispatch` (L192). This is the logic that re-resolves a workflow step's agent_type/difficulty at dispatch.
3. **Self-reflection bridge (L259-294 in `_dispatch`) → coulson or `general_beckman` posthook wiring.** Bridges `profile.enable_self_reflection` (agent CLASS attr) into task context so the completion path spawns `self_reflect`. SP3b Task 7 artifact. Belongs with the reflection posthook wiring, not the pump.
4. **`_mech_action_to_result` needs_review handling (L51-68)** — keep (it's result-router glue) but verify it's not duplicated.
5. **Founder sweep / mission resumption `_rebind_ongoing` / heartbeat loop** — grep current lines (past L420). Founder sweep → call existing `founder_actions.sweep_*` inline. Heartbeat → reuse `yasar_usta` HeartbeatWriter where possible (heartbeat impl is `src/core/heartbeat.py`).
6. **skip-when gate (L305-321) + raw_dispatch sentinel (L356-390) + watchdog (L407+)** inside `_dispatch._run` — these are core dispatch flow; leave unless they pull weight better placed in workflow engine (skip-when arguably → engine/conditions).

**Goal:** `_dispatch` back to ~200 LOC, `run_loop` a pure pump that calls `beckman.next_task()` + `periodic_checks.run_due()`.

**Verify:** full boot + 1 multi-step mission + 1 workflow with a live JSON edit (the agent-type-refresh guard) + `timeout 120 pytest tests/ -q`. Boot smoke: `python -c "import src.core.orchestrator"`.

---

## P6 — Split `local_model_manager.py` (currently **583 LOC**, target ~300)
NOTE: debt-map said 684 — already shrank (P2 gpu_scheduler removal). The plan's "keep the GPU-scheduler bridge" is **OBSOLETE** (that bridge was deleted in P2). Current structure (2026-06-07):
- **EASY WIN — delete dead code first:** `_floored_baseline_ctx` (L60) is marked `DEPRECATED 2026-05-31 — no longer called`. Confirm zero callers (`grep _floored_baseline_ctx`) and delete (~50 LOC free).
- **`_need_ctx` (L40) context-sizing → DaLLaMa** (it owns load config). This is the active ctx-window calculator used in `ensure_model` (L306). Push the math into DaLLaMa; LMM calls it.
- **`ensure_model` (L252) — extract the DB scheduling + Beckman notifications → task layer** (orchestrator/beckman callback). Keep the load/swap orchestration.
- **KEEP** (LMM as the DaLLaMa-facing wrapper): `set_nerd_herd`/`_push_to_nerd_herd` (L158/165), `_on_ready` (L199), lifecycle props (`current_model`/`is_loaded`/`idle_seconds`/`loaded_context_length` L477-507), `begin/end_inference`, `get_status`/`get_metrics`, `run_idle_unloader`/`run_health_watchdog`/`_health_check` (verify these delegate to DaLLaMa's watchdog — `packages/dallama/src/dallama/watchdog.py` exists; idle-unload race coverage is `packages/dallama/tests/test_watchdog.py`).
- `ModelRuntimeState` dataclass (L~95) + `get_runtime_state` (L581) stay — tested by `tests/test_runtime_state.py`.

**Verify:** model swap + load + idle-unload cycle; VRAM smoke; `tests/test_runtime_state.py`, `tests/test_dallama_shim.py`, `packages/dallama/tests/`. **DO NOT** run a mix of `tests/` and `packages/` in one pytest invocation (conftest plugin-name collision — run separately).

---

## P7 — Guardrail (after P5/P6)
New `tests/test_root_stays_thin.py`:
- assert each `src/models/*.py` shim < ~80 LOC;
- assert `src/core/` modules do NOT `import hallederiz_kadir` / `litellm` directly (LLM execution stays in packages);
- **tree-wide extension** (debt-map §D): assert package `__init__` stubs aren't masking fat `src/` twins (catches the `workflow_engine` half-extraction: `packages/workflow_engine` is a 250-LOC stub while the real engine is `src/workflows/engine/` 8.8k).
- optional LOC ceilings on orchestrator/dispatcher with comment "raise only with a refactor PR".
Also rewrite `docs/architecture-modularization.md` to match reality (it still calls router/auto_tuner/etc. shims wrongly; list the moved posthooks + telemetry + introspection modules).

---

## Test / verify playbook (learned this session)
- Python: **`.venv/Scripts/python`** (Windows; venv at `.venv/`).
- ALWAYS `timeout 120 pytest …` (zombie pytest holds SQLite WAL lock → crash-loops live KutAI). Long runs → background (`run_in_background`) and wait for notification.
- **conftest collision:** never pass both `tests/…` and `packages/…` paths to one `pytest` call (`ValueError: Plugin already registered`). Run them in separate invocations.
- **Shim/monkeypatch rule:** when a test does `patch("src.core.X.fn")` and you move `X`, keep callers importing the OLD path (shim) OR retarget the patch — a re-export shim preserves `patch("old.path.fn")` only if the live caller also imports from the old path.
- **git rename ambiguity:** `git mv old→new` + writing a new shim at `old` confuses rename detection; verify with `git diff --cached --stat -M` and `git ls-files --stage` that BOTH the moved impl and the shim are present before commit.
- **Audit call sites, not docstrings:** grep imports + invocations; docs/CLAUDE.md drift (this session: CLAUDE.md claimed dispatcher 544 LOC — actually 623; claimed pick telemetry moved — it's a thin delegate, body did move).
- **Known pre-existing red (NOT yours):** `tests/test_lifecycle_fixes.py` 5 fails on `BaseAgent._execute_react_loop` missing — stale since the runtime extraction gutted BaseAgent (base.py 4092→179). Orthogonal; leave or delete separately.

## Verification commands
```
.venv/Scripts/python -c "import src.core.orchestrator, src.models.local_model_manager; print('boot ok')"
timeout 120 .venv/Scripts/python -m pytest tests/test_runtime_state.py tests/test_dallama_shim.py -q
timeout 120 .venv/Scripts/python -m pytest packages/dallama/tests -q
timeout 150 .venv/Scripts/python -m pytest tests/ --co -q   # collection sanity after moves
```
