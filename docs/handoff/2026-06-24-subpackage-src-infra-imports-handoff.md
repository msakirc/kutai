# Handoff — Remaining `src.infra` imports in sub-packages

**Date:** 2026-06-24
**Predecessor work:** logging migration — commit `03526dcf` (pushed to `main`).
**Goal:** zero `src.*` imports from `packages/*` so each package is independently installable/releasable.

---

## What just shipped (context)

`src.infra.logging_config` was a 12-line re-export shim over the **yazbunu** package
(editable-installed leaf, `packages/db` + `packages/nerd_herd` already used it).
Migrated all sub-packages off the shim:

- 215 `.py` files across 9 packages: `from src.infra.logging_config import get_logger` → `from yazbunu import get_logger` (alias forms `as _gl` / `as _get_logger` preserved).
- `yazbunu>=0.2.0` added to 9 `pyproject.toml` (clair_obscur, coulson, fatih_hoca, general_beckman, hallederiz_kadir, husam, intersect, mr_roboto, yalayut).
- Tests: 2281 passed, 2 skipped, **0 failures** (all 9 touched packages).
- `src/infra/logging_config.py` shim **kept** — `src/` core (orchestrator, telegram_bot, run.py) still imports it. That is a *separate* cleanup, NOT in scope here.

This removed **98%** of the prior package→`src.infra` coupling (227 of ~244 import sites).

---

## What remains

**17 files** in `packages/*/src` still import non-logging `src.infra.*` — **12 distinct modules.**
(One more match, `packages/db/src/dabidabi/__init__.py:21`, is a **comment** mentioning `src.infra.db`, not an import — ignore it.)

DB ops themselves are already abstracted: sub-packages get connections from `dabidabi`
(e.g. `fatih_hoca/db.py` → `from dabidabi import get_db`). There are **zero** real
`from src.infra.db import` statements in package code. The 12 modules below are misc
cross-cutting utilities still living under `src/infra/`.

### The 12 modules, by difficulty tier

Difficulty is driven by what the **module itself** imports (moving a module that imports
`src.app.*` just relocates the coupling). LOC + own-deps audited 2026-06-24.

#### Tier 1 — clean leaves: move now, low risk (4 modules)
No `src.*` coupling (or only `dabidabi`). Just relocate into an owner package + repoint callers.

| module | LOC | own deps | callers (file:line) | suggested owner |
|---|---|---|---|---|
| `audit` | 136 | none/stdlib | hallederiz_kadir/caller.py:282; coulson/react.py:1390,1653 | leaf telemetry pkg **or** dabidabi (audit_log table) |
| `tracing` | 121 | `dabidabi` | coulson/react.py:1407 | dabidabi (trace table) — already on dabidabi |
| `artifacts_register` | 53 | none/stdlib | mr_roboto/executors/legal_document_render.py:228 | mr_roboto (single caller) or dabidabi |
| `monitoring` | 74 | none/stdlib | mr_roboto/executors/monitoring_check.py:24 | mr_roboto (single caller, pure HTTP checks) |

#### Tier 2 — db-shim swap, then leaf (4 modules)
Only `src.*` coupling is `src.infra.db`. Swap `from src.infra.db import …` →
`from dabidabi import …` (the proven `fatih_hoca/db.py` pattern), which makes them leaves,
then relocate to the owner package.

| module | LOC | own deps | callers (file:line) | suggested owner |
|---|---|---|---|---|
| `admission_forensics` | 90 | `src.infra.db` | hallederiz_kadir/caller.py:792,826; husam/worker.py:463; coulson/dispatch_helpers.py:201 | **leaf telemetry pkg** (multi-package caller — see circular note) |
| `cost_wiring` | 195 | `src.infra.db` | coulson/react.py:423 (`quality_mode_profile`); mr_roboto/mission_deliverable_bundle.py:118 (`format_mission_cost`) | kuleden_donen_var (cloud cost domain) |
| `recipes` | 645 | `src.infra.db` | general_beckman/apply.py:257; mr_roboto/{classify_signals.py:48, instantiate_picked_recipes.py:54, instantiate_recipe.py:62, pick_recipe.py:56} | yalayut or mr_roboto (most callers in mr_roboto) |
| `mission_lessons` | 306 | `src.infra.db` | general_beckman/apply.py:693; mr_roboto/{record_verdict.py:134, launch_lessons_writeback.py:41, launch_drafts.py:110, __init__.py:354,2371} | general_beckman (mission lifecycle) |

#### Tier 3 — blocked on the `telegram_bot` god-file (3 modules)
These import `src.app.telegram_bot` (the ~12.7k-line god-file) and/or other `src.app.*`.
**Cannot move cleanly until the notification/event path is inverted** (callers should emit
through an interface — e.g. `mr_roboto.notify_user` / an event bus — not import the bot).
This is real architectural work, tied to the known telegram_bot split debt
(`docs/2026-05-31-root-debt-map.md`).

| module | LOC | own deps (the blocker in **bold**) | callers | note |
|---|---|---|---|---|
| `dead_letter` | 482 | `dabidabi`, **`src.app.telegram_bot`**, `src.infra.db`, `src.infra.dlq_analyst`, logging | general_beckman/apply.py:740 (`quarantine_task`, `is_unresolved_dlq`) | DLQ domain = Beckman, but bot-coupled |
| `mission_pacing_cron` | 198 | **`src.app.mission_events`**, **`src.app.telegram_bot`**, `src.infra.db`, `src.infra.pacing`, logging | general_beckman/cron.py:170 (`check_and_post_tradeoff_prompts`) | pulls bot + mission_events + pacing |
| `dlq_feedback` | 206 | `src.infra.db`, `src.infra.dead_letter`, logging | mr_roboto/__init__.py:2385 (`mine_dlq_patterns`) | transitively blocked via `dead_letter` |

Note: `metrics` (217 LOC) imports `src.shopping.resilience.detection_monitor` — sits between
Tier 2 and Tier 3. Callers: hallederiz_kadir/caller.py:274 (`track_model_call_metrics`),
coulson/react.py:1419,1668 (`record_tool_call`). Decoupling needs the shopping import resolved
first (or that helper moved). Treat as its own item.

---

## ⚠️ Circular-dependency trap (read before picking owners)

`audit`, `metrics`, `admission_forensics` are **telemetry writers called from multiple
packages** (hallederiz_kadir, coulson, husam). If you move one *into a domain package that
those callers already depend on inversely*, you create a cycle — e.g. putting
`admission_forensics` into `general_beckman` while `hallederiz_kadir` imports it, when Beckman
owns the dispatcher that drives hallederiz.

**Recommendation:** route multi-package telemetry writers through a **leaf package** (mirror
the yazbunu/dabidabi pattern) — either fold the pure-DB writers into `dabidabi`, or create a
small `telemetry`/`forensics` leaf pkg. Leaves are import-safe from everywhere. Domain-owned
moves (`mission_lessons`→beckman, `monitoring`→mr_roboto) are only safe for **single-caller**
or same-domain modules.

---

## Suggested execution order

1. **Tier 1 (4 modules)** — mechanical, no shim work. Do first; immediate win, builds the
   relocation muscle. `tracing`/`audit` → likely dabidabi; `artifacts_register`/`monitoring`
   → mr_roboto. Add owner pkg to each caller's `pyproject` deps (as done for yazbunu).
2. **Tier 2 (4 modules)** — per module: swap `src.infra.db`→`dabidabi`, run module's tests,
   relocate, repoint callers, add deps. `recipes` (645 LOC) is the biggest; isolate it.
3. **`metrics`** — resolve the `src.shopping` import, then treat as Tier 2.
4. **Tier 3 (3 modules)** — gated on inverting the `telegram_bot` notification path. Do NOT
   attempt as a string-move; it needs a notify/event interface. Tie to the telegram_bot split.

Per-step verification (proven this session): after each move,
`python -c "import <pkg>"` smoke + `timeout 180 pytest packages/<pkg>/tests -q`.
Collect package test dirs **one at a time** — collecting multiple together throws a
pytest `conftest.py` name-collision (both register as `tests.conftest`); not a real failure.

## Reproduce the inventory
```
# remaining non-logging src.infra imports in package prod code:
rg "from src\.infra(\.\w+)? import|import src\.infra" packages -g "**/src/**/*.py" -n \
  | rg -v "logging_config"
# per-module own-deps audit:
for m in metrics audit admission_forensics tracing cost_wiring recipes mission_lessons \
         dead_letter mission_pacing_cron artifacts_register monitoring dlq_feedback; do
  echo "== $m =="; rg "^from (src\.|dabidabi|yazbunu|nerd_herd)" "src/infra/$m.py"
done
```

## Definition of done
Zero hits for `rg "src\.(infra|app|core|shopping|memory|models|tools|agents)" packages -g "**/src/**/*.py"`
(currently: 17 files / 12 modules under `src.infra`, plus any `src.app`/`src.shopping`
transitively pulled by Tier 2/3 modules once they relocate).

---

## ✅ Update 2026-06-25 — Tier 1 DONE (4 modules)

All four Tier-1 leaves relocated. Zero `src.infra.{audit,tracing,monitoring,artifacts_register}`
imports remain in `packages/*`. Verified green: kara_kutu 6 / hallederiz_kadir 81 /
coulson 136 / mr_roboto 1021 / monitoring integration 3. (mr_roboto's lone failure
`test_resend_review_halt_path` is **pre-existing** — that test file was already `M` in the
working tree before this work, unrelated review-halt change; zero overlap with the moves.)

**Decision — observability is three pillars, not one bucket** (owner-confirmed after pushback):

| pillar | package | nature |
|---|---|---|
| metrics | `nerd_herd` | live system state → Prometheus, **DB-free standalone** exporter |
| events/history | **`kara_kutu`** (NEW leaf) | durable append-only SQLite via dabidabi, human-facing |
| logs | `yazbunu` | log lines |

- `audit` + `tracing` → **new `kara_kutu` leaf** (`packages/kara_kutu/`, deps dabidabi+yazbunu,
  editable-installed). Rejected dabidabi (would push domain vocab ACTOR_*/ACTION_* + formatters
  into the DB *engine* — contradicts the model-registry→fatih_hoca ownership precedent) and
  rejected nerd_herd (it is deliberately DB-free/standalone; audit/tracing are SQLite writers
  and would drag in dabidabi + mix the metrics pillar with the event pillar).
- `src/infra/audit.py` + `src/infra/tracing.py` **kept as shims** re-exporting from `kara_kutu`
  (src core still imports them: `telegram_bot.py`, `workflows/engine/hooks.py`) — same pattern as
  the `logging_config` shim. The src→src cleanup is separate/out of scope.
- `monitoring` + `artifacts_register` → **`mr_roboto`** (single caller, same domain). Originals
  in `src/infra/` **deleted** (no src prod caller). Integration test patch targets updated
  `src.infra.monitoring` → `mr_roboto.monitoring`.

**Status: local, NOT committed — restart-gated.** `kara_kutu` is editable-installed in `.venv`,
so a `/restart` picks it up. Verify a live audit/trace write post-restart, then commit + push.

**Bearing on later tiers:** when `metrics` (217 LOC) is freed from its `src.shopping` import it
fits the **nerd_herd** pillar. `mission_lessons` + the DLQ history (`dlq_feedback`/`dead_letter`
records) are durable-history → they belong in **`kara_kutu`**, growing it from ~257 to a
~600–1000 LOC real package.

---

## ✅ Update 2026-06-25 — Tier 2 DONE (4 modules) — committed `90e44e5c`, pushed `main`

All four Tier-2 modules relocated. `src.infra.db`→`dabidabi` + logging→`yazbunu` swap per
module, then moved to owner package; callers repointed; deps wired.

| module | LOC | → owner | original |
|---|---|---|---|
| admission_forensics | 90 | `kara_kutu` | **deleted** (no src caller) |
| mission_lessons | 306 | `kara_kutu` | shim kept (telegram_bot:9364) |
| cost_wiring | 195 | `kuleden_donen_var` | shim kept (telegram_bot:3525) |
| recipes | 645 | `yalayut` | shim kept (playbooks `_load_yaml`) |

Owner notes:
- **recipes → yalayut, NOT mr_roboto** (handoff suggested either). yalayut already owns recipes
  as a first-class concept: `yalayut/executor.py::run_recipe`, `__all__` exports it, and
  `discovery/sources/cookiecutter_template.py` ingests templates as `shell_recipe` artifacts.
  The 4 mr_roboto callers are the *mechanical-executor shim layer* that calls INTO yalayut
  (yalayut's own design), not the owner. Imported via `yalayut.recipes` submodule (NOT top-level
  `__init__`) to keep the disk-YAML recipe vocab separate from yalayut's `run_recipe`/shell_recipe.
- ⚠️ **Two parallel recipe systems now co-located in yalayut, NOT converged**: (1) the moved
  disk-YAML library (`recipes/<name>/v1/recipe.yaml`, tech_stack match, token-substitute,
  `pin_recipe`/`recipe_picks` table); (2) yalayut's `yalayut_index` `shell_recipe` rows +
  cookiecutter discovery + `run_recipe` shell-steps. Convergence = separate design task (follow-up).
- `cost_wiring` → kuleden_donen_var verified cycle-safe (kuleden imports no coulson/mr_roboto).
- `admission_forensics`/`mission_lessons` → kara_kutu (multi-pkg telemetry writers / durable
  history; leaf-safe per the circular-trap rule).

Deps added: `kara_kutu`→husam/general_beckman/mr_roboto; `kuleden_donen_var`→coulson/mr_roboto;
`yalayut`→general_beckman/mr_roboto.

Patch targets repointed (caller now does lazy `from <owner> import …`):
`coulson/tests/test_pool_empty_diag.py` → `kara_kutu.record_admission_violation`;
`mr_roboto/tests/test_mission_deliverable_bundle.py` → `kuleden_donen_var.format_mission_cost`.
(`test_visual_review_notify` self-patches the shim attr → unchanged; root `test_z2_t4` patches
`src.infra.db.get_db` which is a sys.modules alias of dabidabi → unchanged.)

Verified green: beckman **371**, kara_kutu+kuleden **190**, mr_roboto targets **41**, root **65**.
2 root failures in `test_z2_t4_mission_lessons::TestDLQEmitter` are **pre-existing** (proven by
stash-revert to original module → identical `sqlite3 no such column: t.retry_reason`; the test
hand-rolls a `tasks` table without `retry_reason`, prod schema has it — dabidabi:833). Unrelated
to this work; left as-is.

**Status: committed `90e44e5c` + pushed to `origin/main`** (FF `692abe5a..90e44e5c`, also carried
the prior restart-gated local commits). kara_kutu/yalayut/kuleden are editable-installed → a
`/restart` picks up the new modules; verify a live mission-lessons / recipe-pin / mission-cost
path post-restart.

**Remaining (out of Tier 2 scope):**
- `metrics` (217 LOC) — its own item; blocked by `src.shopping.resilience.detection_monitor`.
  Once freed → nerd_herd pillar.
- Tier 3 (3 modules) — `dead_letter` / `mission_pacing_cron` / `dlq_feedback`, gated on inverting
  the `telegram_bot` notification path. Do NOT string-move.
- DoD grep now shows ONLY `metrics` (×3) + the 3 Tier-3 modules in `packages/*/src`.
