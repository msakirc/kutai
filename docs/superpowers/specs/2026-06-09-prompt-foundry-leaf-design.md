# Prompt Foundry — Leaf Package for Prompt/Profile Content & Unanimous Building

*2026-06-09. Supersedes the mechanical framing of `docs/handoff/2026-06-09-kill-agents-yaml-handoff.md`. Continues `project_kill_agents_20260504` + the 2026-05-16 prompt-ownership refinement (session `c67274dc`).*

## Problem

Two coupled architectural defects, surfaced by a founder design review:

1. **Agents are residue.** The original design had agents *owning* work; everything else served them. The system inverted — each module now owns its work (beckman=lifecycle, fatih_hoca=selection, coulson=react execution, husam=single-shot execution, hallederiz_kadir=call). The 29 `src/agents/*.py` classes are now **pure config**: class attributes + a static `get_system_prompt`. Verified 2026-06-09: zero custom methods outside `get_system_prompt`. The class machinery buys nothing.

2. **Prompt content has no coherent owner and no uniform build path.** LLM prompts are built ad-hoc by whichever module spawns the call:
   - **Agent execution prompts** (29) — seeded from `get_system_prompt`, live source is the `prompt_versions` DB table (DB-versioned, A/B-tuned, auto-promoted ≥10 tasks).
   - **Coulson posthook prompts** — grading, code_review, reflection, self_critique, constrained-emit — **frozen code strings** in `packages/coulson/`.
   - **Husam-caller prompts** — husam is prompt-agnostic (caller passes `messages`); its callers each build their own: `packages/general_beckman/posthook_handlers/brand_voice_lint.py`, `…/copy_compliance_review.py`, `packages/yalayut/.../synthesize.py`, `src/tools/vision.py`.
   - **Message-routing classifier** — `CLASSIFIER_PROMPT` in `src/core/task_classifier.py:299` (agent_type `"classifier"`): a frozen code string with NO DB override. (NOTE: `signal_classifier` is a *different* thing — a normal registry agent that DOES get a DB override and migrates like the other 26.)

   Result: same artifact (an LLM prompt) lives under **two governance regimes** — agent prompts are tunable data, everything else is frozen code requiring an edit + restart. And there is **no single way to build a prompt** — every spawner does it bespoke.

## Founder requirements (hard constraints)

- **No package → `src` dependency.** Packages must not import `src/*` for prompt content. (`src/memory/prompt_versions.py` imports `src/infra/db.py`; packages reading it = the dependency inversion being eliminated.) This is the first instance of a broader goal: kill the `src` DB dependency of all packages.
- **Unanimity in prompt building.** One uniform mechanism for building prompts, not N bespoke assembly sites.
- **KEEP-or-DROP, never MODE-merge** (`feedback_no_agent_modes`). The change is class→data, NOT 29→1. The registry still has 29 distinct keys after. No `mode`/`phase`/`artifact_type` flag consolidation.
- **Each module keeps owning its work.** Do not re-bloat one module into a prompt god-store (the smell P1–P7 just undid). The content owner must be *passive data*, not an active module.

## Ownership model (the core decision)

Ownership is **three axes**, not one "prompt owner":

| Axis | Owner | Status |
|------|-------|--------|
| **Content** — prompt strings, rubrics, profile rows | **Prompt Foundry** — a new foundational **leaf** package; passive data + the one build API | NEW |
| **Assembly** — turning content + task into final `messages` | **Foundry's single build API**, called by every spawner (unanimity) | NEW path; spawners stop hand-rolling |
| **Selection** — model | `fatih_hoca` | unchanged |
| **Execution shape** — react / single-shot / mechanical | worker (`coulson` / `husam` / `mr_roboto`) | unchanged |
| **Lifecycle / admission** | `general_beckman` | unchanged |

**Agents die** → become Profile rows in the Foundry registry: `{prompt-seed, tier, min_tier, allowed_tools, max_iterations, can_create_subtasks, execution_pattern, enable_self_reflection, min_confidence, confidence_gate, routing-key}`. Read by coulson, husam-callers, classifier, fatih_hoca.

### Why a leaf package (not "coulson owns it")

Husam proves coulson cannot be the universal prompt owner: husam-caller prompts are built in `general_beckman`, `yalayut`, and `src/tools` — outside coulson. A shared **leaf** that depends on **nothing** upstream and is depended-upon by all gives unanimity *without* the sideways feature-package coupling the founder forbids (same dependency status as stdlib; cf. the enforced `test_husam_does_not_import_coulson` purity rule).

## Design

### The Foundry package (leaf — depends on nothing in `src` or feature packages)

Proposed location: `packages/<foundry>/` (name = open founder decision; repo uses Turkish-character names — placeholder `prompt_foundry` below). Contents:

1. **`Profile`** — a frozen dataclass holding the field surface above (harvested from `base.py`). `_suppress_clarification` stays a **runtime** attr, NOT a profile field (ruled in the 2026-05-04 session). `get_system_prompt(task)` → returns the seed string for the 27 static profiles.
2. **Profile registry (static seed data)** — the 29 ex-agents shipped **as data inside the package** (per-profile YAML under `profiles/`, lean per-agent for clean diffs, matching the i2p per-step style). Loaded once at import into per-type singletons. `get_profile(type)` returns a **stable cached instance per type** (see Invariant 1).
3. **Rubric/template content** — grading, code_review, reflection (+ REFLECTION_BLOCKS / STACK_BLOCKS / LAYER_BLOCKS), self_critique, constrained-emit, brand_voice, copy_compliance, classifier — all current frozen strings, relocated here as data keyed by purpose.
4. **One build API** — `build_messages(content_key | profile, task, *dynamic_blocks) → list[dict]`. Every spawner calls this. Uniform skeleton; dynamic specifics (mission lessons, calibration, tools block) passed **in as params** — the leaf cannot see mission/model state, so it composes blocks it is handed rather than fetching them.
5. **`PromptStore` Protocol (port)** — tiny, mirrors the current surface:
   - `async get_active(key) -> str | None`
   - `async save_version(key, text, notes="", activate=False) -> int`
   - `async record_quality(key, score) -> None`
   - `async list_versions(key) -> list[dict]`
   The Foundry uses an **injected** store for tuned overrides. With no store wired, `build_messages` returns the in-package seed → the Foundry works standalone; the store is a *plug-in*, not a *requirement*.

### Forward-compat: a future DB package will own all DB ops

The founder plans a dedicated DB-layer package that lifts DB ops out of every package. The Foundry is built to plug straight into it: the `PromptStore` **port IS the seam** that future package sits behind, and its methods are deliberately **storage-generic** (`get_active(key)`, `save_version`, `record_quality`, `list_versions`) — no `prompt_versions`-table vocabulary leaks into the port. The concrete `DbPromptStore` adapter (below) is **disposable scaffolding**: when the DB package lands, re-point the adapter at it and the leaf never changes. Therefore: **no foundry-owned sqlite file now** (it would be thrown away); stay on the shared-table adapter (zero migration).

### Storage stays the DB — behind the port

Concrete adapter lives in the **app layer (`src`)**, wrapping the existing `prompt_versions` table (reuse `get_active_prompt`/`save_prompt_version`/`record_prompt_quality`/`list_prompt_versions` verbatim). Wired into the Foundry at startup via dependency injection. **No new storage tech, no data migration, no schema churn.** Packages stop *importing* the DB; the app *injects* it. This adapter is the **reference domino** for the broader "kill `src` DB dep in all packages" migration — the port pattern proven once here is replicated package-by-package later.

### The 2 carve-outs (dynamic prompts) — NOT symmetric (corrected per review B2)

| Profile | Dynamic bit | Handling |
|---------|-------------|----------|
| `writer` | branches on `_detect_markdown_schema(task)` — **pure** dict inspection, no external deps | leaf `Profile` subclass (`WriterProfile`) overriding `get_system_prompt`; `markdown_prompt` field on base `Profile` |
| `oncall_agent` | embeds an **action whitelist** fetched via `from coulson.agent_handlers.registry import get_whitelist` (`oncall_agent.py:40`); `domain` selects the verb set but never appears as prompt text | **STAYS a thin class in `src/agents/`** — it needs `coulson`, which the leaf may NOT import. A genuine carve-out OUTSIDE the Foundry, merged into `AGENT_REGISTRY` |

Recommendation: **hybrid** — 27 static → data; `writer` → leaf subclass; `oncall_agent` → stays a `src/agents` class. (The original spec wrongly modeled oncall as a `{{domain}}` string substitution — it is a `get_whitelist` call. A leaf reproduction would violate purity.)

## Coupled surfaces that must stay in sync

1. **3 prompt-quality invariants** (`tests/agents/test_prompt_quality.py`): each prompt — first line `You are …`, body has must/always + don't/never, body has `final_answer` + fenced ` ```json ` schema. **Retarget the test to load from the Foundry registry.**
2. **Per-agent self-reflection blocks** (`packages/coulson/.../reflection.py::REFLECTION_BLOCKS`, keyed by agent name) + the `enable_self_reflection` flag (now a Profile field) gated via the P5 bridge (`src/core/dispatch_prep.py`). Move the *content* (blocks) into the Foundry; preserve the flag and the bridge wiring.
3. **Classifier / workflow keys ⊆ registry keys** — `signal_classifier.py` + `src/workflows/i2p/i2p_v3.json` reference `agent_type` strings that must equal registry keys exactly; a typo silently routes to the `executor` fallback. **Add a test asserting classifier/workflow agent_types ⊆ registry keys.**
4. **Prompt seed path** (`/prompt seed`, `seed_from_agents()`) currently reads the code string via `get_agent().get_system_prompt()`; re-point at the Foundry seed.
5. **Singleton identity** — see Invariant 1.

## Invariants (DO NOT violate)

1. **Stable per-type singletons.** `get_agent(x)`/`get_profile(x)` must return the SAME object across calls. The runtime mutates per-execution instance attrs (`_original_allowed_tools`, `progress_callback`, `_suppress_clarification`, `allowed_tools`) and restores them in `execute()`'s `finally`; tests patch a resolved instance and rely on identity (e.g. `tests/core/test_orchestrator_self_reflect_bridge.py`). The data-driven resolver MUST hand back cached singletons, not fresh objects per call.
2. **DB override still wins at runtime.** `prompt_versions.is_active=1` (now via the injected store) beats the in-package seed. Unchanged behavior; only the seed *source* moves (code → package data).
3. **Back-compat surface.** Keep `AGENT_REGISTRY` name + `get_agent(type)` signature (re-exported / aliased) so no external caller breaks. No external module imports a concrete `*Agent` class (verified); only tests do — those get updated.
4. **`execute()` + `_build_context()` are NOT on the data Profile** (review B1). They live on `BaseAgent` today and are called at `src/core/orchestrator.py:227` (`get_agent(x).execute(task)`) and in coulson `react.py`/`single_shot.py` (`profile._build_context(task)`). A data Profile has neither → AttributeError on dispatch. The fix (plan Task 5.5) moves both onto the worker: `coulson.execute(profile, task)` + `coulson.build_context(profile, task)`. This MUST land before any agent is served from data. Verify with a real `coulson.execute(get_agent("summarizer"), mock_task)` end-to-end test — NOT a `build_system_prompt`-only smoke (that path misses `.execute`/`._build_context`; the green-test-dead-prod trap, `feedback_test_serialization_boundary`).

## Phased plan (canonical-first, low → high risk)

1. **Decouple execute/_build_context (Task 5.5) + scaffold the leaf + 1 profile.** First move `execute`/`_build_context` off the profile onto coulson (Invariant 4). Then create the Foundry package: `Profile` dataclass, `profiles/` dir, `PromptStore` Protocol. Migrate ONE static agent (`summarizer`). Wire the app-side DB adapter + injection. `get_agent` serves the migrated one from data; the rest stay classes. Prove: singleton identity, **real `coulson.execute(get_agent("summarizer"), mock_task)` e2e**, DB-override, 3-invariant test. **Land to main.**
2. **Bulk-migrate the 26 static profiles** to package data; delete their `.py`. `writer` → leaf subclass; `oncall_agent` → stays a `src/agents` class (coulson whitelist dep). Rebuild `AGENT_REGISTRY` = Foundry profiles + the oncall carve-out.
3. **Migrate overhead/husam-caller prompts** into the Foundry: coulson posthooks (grading/code_review/reflection/self_critique/emit + blocks), `general_beckman` posthook_handlers (brand_voice/copy_compliance), yalayut synth, vision tool, classifier. Each spawner switches to `build_messages`. This is where unanimity lands.
4. **Retarget tests + seed path:** `test_prompt_quality.py` reads Foundry; replace `from src.agents.X import XAgent` with `get_agent("x")`; add classifier/workflow-keys-⊆-registry test; re-point `seed_from_agents()` at Foundry seed.
5. **Guardrails:** (a) dep-purity test — Foundry imports nothing from `src` or feature packages (mirror `test_husam_does_not_import_coulson`); (b) extend `tests/test_root_stays_thin.py` — `src/agents/*.py` count stays small (Profile shim + 2 carve-outs + `__init__`), so new agents land as data.

## Verification

- `… python -c "from src.agents import get_agent; print(get_agent('coder').name, get_agent('coder') is get_agent('coder'))"` → `coder True`.
- `timeout 60 .venv/Scripts/python -m pytest tests/agents/ -q`.
- Dep-purity: subprocess import test asserting Foundry has no `src.`/feature-package imports.
- Live: 1 multi-step mission touching coder/reviewer/researcher; confirm prompts resolve + self-reflection fires for coder + a posthook (grading) runs via the new build path.
- **DO NOT** mix `tests/` and `packages/` in one pytest call (conftest collision). Use `.venv/Scripts/python`; always `timeout`.

## Decisions (resolved 2026-06-09)

1. **Foundry package name** — DEFERRED; founder will name at the end. Placeholder `prompt_foundry` throughout; do not hardcode the final name into the plan until set.
2. **Profile data format** — ✅ **per-profile YAML** (`profiles/<name>.yaml`). Block scalars for multi-line prompts. The 2 carve-outs stay Python subclasses alongside.
3. **Broader src-DB-dep kill** — ✅ **separate track.** This spec ships ONLY the `PromptStore` port as the reference domino; replicate per-package later. This spec stays prompt-scoped.
4. **base.py final deletion** — ✅ **sequential / separate.** CORRECTION (review S1): `_build_model_requirements` / `_maybe_constrained_emit` **do not exist** (zero grep matches; model requirements come from `fatih_hoca.requirements_for`). The actual base.py methods are `execute` (delegates to coulson) and `_build_context` (delegates to `build_user_context` + mutates `allowed_tools`). The Foundry plan's **Task 5.5** moves BOTH off the profile contract onto coulson (`coulson.execute(profile, task)`, `coulson.build_context(profile, task)`) — that is on-thesis and is IN scope (it's the B1 blocker fix). The only out-of-scope remainder is the final `git rm src/agents/base.py` once it has no callers.

## Risk notes

- `get_agent`/`build_messages` are hot-path (every dispatch). Parse YAML once at import → per-type singletons; **never parse per-call**.
- Live-bot `git add -A` storm on `main` — do this in a **git worktree**, merge in a quiet window (`project_modularization_p5_p6_p7_20260608`; `feedback_avoid_git_stash_foreign`).
- Medium-high risk: touches every dispatch's profile resolution, the prompt seed path, all overhead/husam-caller build sites, and 3 coupled test/classifier surfaces. **Canonical-first (Phase 1 lands before bulk) is mandatory.**
- Serialization boundary: the injected store crosses an async/DB boundary — if any DTO/field is added, round-trip test it (`feedback_test_serialization_boundary`).
