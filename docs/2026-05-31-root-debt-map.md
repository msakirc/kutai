# Root Debt Map — `src/core/` + `src/models/` (+ tree-wide survey)
*2026-05-31→06-01. What actually sits in KutAI root vs what the docs claim. Built from 4 parallel call-site audits (prod call sites only; tests/docs/.md excluded). Tree-wide section added 06-01.*

## TL;DR
- Docs claim: thin orchestrator + thin dispatcher + thin shims. Reality: **~3000 LOC of fat logic squatting in root.**
- 3 categories of debt: **(A) dead fossils**, **(B) doc-lies (size + nature)**, **(C) misplaced packages** (real logic in the wrong directory).
- **But core is the SMALLEST fish.** Tree-wide (section D below), the real monoliths dwarf it: `telegram_bot.py` 13.8k, `db.py` 8.4k, `workflow_engine` half-extraction, `shopping` 25.9k sprawl.
- Pattern across the whole tree: **"extraction declared, bulk never moved."** Nothing on fire — but the architecture story is fiction in several places. This is the cleanup backlog; founder tackles items one-by-one.

---

## A. DEAD CODE (delete)

### `router.py` — scoring fossil
- `select_model()` (L131–687), `select_for_task()` (L692–694), `check_cost_budget()` (L700–705) = **ZERO prod callers.** Only 5 test files + docs.
- Live model selection path is `dispatcher → fatih_hoca.select() → fatih_hoca/ranking.py::rank_candidates()`. Router's copy never executes.
- **DRIFT (router copy is stale + wrong):**
  | Layer | router.py (dead) | fatih_hoca/ranking.py (live) |
  |-------|------------------|------------------------------|
  | Performance | hardcoded `perf_score=50` | dynamic grading+tps blend |
  | Availability | live KDV read | snapshot read |
  | Stickiness | fixed 1.40×/0.75× | 1.10–1.80× + anti-flap (call_category aware) |
  | Specialty | 1.15× applied | removed (Phase 2a, double-score fix) |
  | Failure adapt | **none** | full per-model penalties |
  | Utilization/pressure | none | Phase 2d equation + S7 rebalance |
  | quality_mode dial | absent | present |
- **Action:** delete the 3 dead funcs (L131–706). Port/delete 4 tests (`test_scoring_layers`, `test_sibling_rebalancing`, `test_real_pipeline`, `test_gap_fixes`) → point at `rank_candidates`.
- **Keep & relocate the live bits:** `ModelCallFailed` (L16–28) → `src/core/exceptions.py`; `get_kdv()` (L48–128) → `src/infra/rate_limiter.py`. Leave back-compat re-exports for `llm_dispatcher`, `orchestrator`, `task_classifier`, `run.py`, `telegram_bot`.

### `llm_dispatcher.py` — deprecated shim
- `request()` (L202–287, 86 LOC) = legacy shopping bridge, marked "do NOT add new callers." Delete after SP5 shopping migration.

### `grading.py` — suspected dead helper
- `apply_grade_result()` (L326) has no found prod callers. Verify, then deprecate.

---

## B. DOC-LIES (CLAUDE.md says X, reality is Y)

| File | CLAUDE.md claim | Reality |
|------|-----------------|---------|
| `orchestrator.py` | "~366 lines, thin ~30-line pump" | **793 LOC** (2.16×). +363 since Task 13 |
| `llm_dispatcher.py` | "413 lines, thin ask→load→call→**retry** loop" | **721 LOC**. **No retry loop** — deleted 2026-03-26 (d3e3c4fa); retries now in hallederiz_kadir + selector feedback |
| `local_model_manager.py` | "thin re-export shim" | **684 LOC**, ~350 real logic (incl. 158-LOC `ensure_model()`) |
| `gpu_scheduler.py` | (listed under thin shims) | **233 LOC**, 100% real priority-queue impl, 0 delegation |
| `models.py` | (listed under thin shims) | **325 LOC**, 100% real action-validation/schema lib |
| `auto_tuner.py` | (listed under thin shims) | **403 LOC**, 100% real ML tuning pipeline |
| grading/code_review/reflection_posthook | unlisted in architecture-modularization.md | live LLM-child prompt builders, undocumented |

**Genuinely thin (docs honest):** `model_registry.py` (54), `gpu_monitor.py` (20), `rate_limiter.py` (59), `result_router.py` (10), `task_context.py` (4), core plumbing `decisions/heartbeat/startup_recovery/state_machine/in_flight`.

---

## C. MISPLACED LOGIC (real code, wrong home)

### LLM-child prompt builders — loose in `src/core/`, all LIVE via `general_beckman/apply.py`
| File | LOC | Callers | Should live |
|------|-----|---------|-------------|
| `grading.py` | 561 | apply.py:1645, posthook_continuations.py:146 | `coulson/` posthooks or documented core resident |
| `code_review.py` | 155 | apply.py:1660, posthook_continuations.py:218 | same |
| `reflection_posthook.py` | 393 | apply.py:1696/1748, coulson re-export | colocate w/ above (lowest risk) |

These 3 build `raw_dispatch=True` specs (no direct LLM calls) → architecturally clean, just homeless. By the "LLM logic lives in packages" rule they belong in `coulson/`. Low–med risk (≤2 callers, no cycles).

### `fast_resolver.py` (392) — tool, not core
- API fast-path resolver, zero LLM. Only caller: `src/tools/smart_search.py:47`. → move to `src/tools/`. Med risk (private `_`-API, DB logging).

### `src/models/` packaging mistakes → belong in `src/core/`
- `gpu_scheduler.py` (233) — single-GPU arbitration. Caller: local_model_manager.
- `models.py` (325) — agent action validation. Caller: `coulson/react.py`.
- `auto_tuner.py` (403) — capability-score tuning. Caller: telegram_bot, fatih_hoca metrics.
- All 3 are Kutay orchestration, not model shims; buried under `src/models/` only because facade dir.

### `local_model_manager.py` (684) — split concern
- Keep as DaLLaMa wrapper: nerd_herd push, GPU-scheduler bridge, lifecycle.
- Move out: `_need_ctx()` context sizing → DaLLaMa; DB scheduling + Beckman notifications inside `ensure_model()` → task layer.

---

## Bloat re-added to orchestrator since Task 13 (366→793)
Workflow agent-type refresh (+60, _dispatch L179–238) · self-reflection bridge (+35, L240–277) · yalayut discovery+source-scout (+70, L126–169) · mission resumption `_rebind_ongoing` (+39) · founder sweep (+24) · heartbeat loop (+30) · error classification (+25) · skip-when (+35) · intersect.flash (+20) · needs_review propagation (+15). Each belongs in workflows/engine, coulson, founder_actions, or a `periodic_checks` module — not the pump.

## Dispatcher accreted non-loop logic (231 LOC)
`_record_pick()` telemetry (58) → `src/telemetry/` · `_get_loaded_*`/`is_loaded_*`/`get_stats` introspection (55+9) → `src/models/introspection.py` · `_estimate_prompt_tokens()` (23) → dallama/hallederiz · `request()` shim (86) → delete post-SP5.

---

## Suggested cleanup order (low risk → high)
1. **Delete router dead funcs** + relocate ModelCallFailed/get_kdv (mechanical, prod-safe).
2. **Move `fast_resolver.py` → src/tools/** (1 caller).
3. **Relocate src/models fat** (DECISIONS LOCKED 2026-05-31): `auto_tuner → packages/fatih_hoca`, `models.py → packages/coulson`, `gpu_scheduler → DISSOLVE into dallama or delete` (founder: "ideally none" — spike first). Keep re-export shims during migration.
4. **Relocate 3 LLM-child builders → coulson/** (or formally document as core residents).
5. **Carve dispatcher telemetry/introspection out** to src/telemetry + src/models/introspection.
6. **Thin the orchestrator pump** — extract periodic checks + workflow refresh + reflection bridge.
7. **Split local_model_manager** concerns; delete dispatcher `request()` after SP5.
8. **Rewrite CLAUDE.md + architecture-modularization.md** to match reality (update LOC, drop false "thin/shim" labels, list the undocumented LLM-child builders). *(CLAUDE.md size/shim/retry/call_model lies already fixed 2026-05-31; telegram_bot 5800→13.8k fixed 06-01.)*

> Authoritative sequenced plan for core/models = `docs/2026-05-31-modularization-finish-plan.md` (7 phases + guardrail). Tree-wide targets below (section D) are separate future tracks.

---

## D. TREE-WIDE SURVEY (all of `src/`, 2026-06-01)
*Recursive LOC. Core/models above are the SMALLEST debt. These dwarf them.*

### The two true monoliths
- **`src/app/telegram_bot.py` = 13,787 LOC** — largest god-file in repo. CLAUDE.md said "~5800" (lie, fixed). Split → `src/app/handlers/{commands,callbacks,menus,...}` or a package. Highest single-file leverage.
- **`src/infra/db.py` = 8,453 LOC** — split by domain (tasks / missions / memory / shopping / todo schema+queries) or extract a data-layer package.

### Half-done extraction (same pattern as core)
- **`workflow_engine`**: `packages/workflow_engine/` = **250 LOC stub** (`advance.py` only). Real engine still in `src/workflows/engine/` (8.8k: `hooks.py` 2308, `expander.py` 1085, `schema_dialect.py` 601, conditions/policies/artifacts). Finish the move; leave shims. This is exactly the trap the P7 guardrail must catch.

### Folder map (recursive LOC)
| Folder | LOC / files | Health | Note |
|--------|-------------|--------|------|
| shopping | 25.9k / 84 | 🔴 sprawl | chronically reverted/broken; 15 scrapers + unwired intelligence; own redesign track |
| app | 23.1k / 29 | 🔴 monolith | telegram_bot 13.8k + jobs/meetings/interview/run |
| infra | 13.7k / 28 | 🔴 monolith | db.py 8.4k + recipes/registry_store |
| tools | 10.9k / 38 | 🟡 fat | `__init__.py` 1664 (registration bloat → move registry out), `shell.py` 1202 |
| workflows | 8.8k / 23 | 🟡 mis-homed | real engine here, not in the workflow_engine package stub |
| memory | 5.0k / 16 | 🟢 ok | vector_store 983 / rag 682 / skills 617 — reasonable |
| core | 5.5k | 🟡 | sections A–C above |
| agents | 2.5k / 26 | 🟡 | pure config classes (see kill-agents memory); candidate for data/yaml |
| models | 1.8k | 🟡 | sections A–C above |
| integrations/security/ops/parsing/context | 0.7–2k | 🟢 | scoped, fine |
| founder_actions / growth / collaboration / languages / util / telemetry | <0.6k | 🟢 | thin, fine |
| **runtime** | 83 / 12 | ✅ | pure shims → coulson. **Extraction done RIGHT — the template.** |
| **workspace** | 0 | ⚰️ dead | empty scaffold → delete |

### Tree-wide future tracks (ranked, tackle one-by-one)
1. `telegram_bot.py` 13.8k split — biggest leverage.
2. `db.py` 8.4k split by domain.
3. Finish `workflow_engine` extraction (src/workflows/engine → package).
4. `shopping` 25.9k redesign (separate, known-hard track).
5. `tools/__init__.py` 1664 registration de-bloat.
6. `src/agents/` config → data/yaml (ties to kill-agents work).
7. Delete dead `src/workspace/`.

### Guardrail must be tree-wide (extends P7)
`test_root_stays_thin` should assert package `__init__` stubs aren't masking fat `src/` twins (catches the workflow_engine pattern), not just police core LOC.
