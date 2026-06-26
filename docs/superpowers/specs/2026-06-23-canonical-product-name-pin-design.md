# Canonical product-name pin (i2p_v3) — design

**Date:** 2026-06-23
**Status:** design (two sub-Opus review passes + code-seam extraction folded in; simplified to artifact-store-as-source-of-truth)
**Author:** session work on mission-89 review-halt (1.13 check 10 naming drift)

## Problem

i2p_v3 generates a product's artifacts via independent LLM steps with **no canonical
product name**. Step `0.0z` (reverse_pitch) invented "HabitTrack"; step `0.1`
(product_charter) invented "FlowState". Reviewer step `1.13` (research_quality_review,
check 10) caught the contradiction and **halted mission 89 to the founder**.

Re-running the producers cannot converge: each writer re-invents a name from
`raw_idea`/`strategic_context` with nothing to converge *to*. This is a **structural
gap**, not a retry-able fault — so the reviewer exhausts the producers' attempts and
escalates. Fix the structure: decide ONE name once, hand it to every step, enforce it
mechanically at the two chokepoints, keep the LLM reviewer as backstop.

## Goal / non-goals

**Goal:** product-name drift between artifacts is structurally prevented at two producer
chokepoints, the name is injected into every step's prompt, and the LLM reviewer
(check 10) remains a final backstop for the remaining naming sites.

**Non-goals:**
- Name *quality* is **unguarded** — one LLM call picks the name with no human review. We
  trade "drift detected late" for "one name chosen early, possibly awkward." Acceptable
  because the target is *drift*, not aesthetics.
- We do **not** mechanically check every artifact that mentions the product (intake_todo,
  competitive_positioning, downstream branding). The mechanical check covers
  `reverse_pitch` + `product_charter` only; the reviewer backstop covers the rest. This
  backstop is **load-bearing and must stay**.

## Key design decision: the artifact store is the single source of truth

`0.0y` emits a `product_name` artifact via `output_artifacts` → the engine materializes it
into the **artifact store** (the same path `success_metrics`/`north_star` use, read by
`inject_north_star.py:42` via `get_artifact_store().retrieve(mission_id, "<name>")`).

Both consumers read it **fresh from the store at execution time**:
- **Prompt injection** happens in the **async** `build_user_context`
  (`coulson/context.py:757`), which can legitimately `await store.retrieve(...)`. This
  sidesteps the dead sync path: `build_system_prompt` (`:251`) calls the loop-blocked
  `_get_mission_lessons_cached` (`:480`, returns `[]` whenever `loop.is_running()`, which
  is always true at the `react.py:373` call site, and is never primed in prod) — we do NOT
  use that path.
- **The mechanical check** (`verify_contains_product_name`) does its own
  `await store.retrieve(...)` in its `mr_roboto._run_dispatch` branch.

Because the name lives only in the artifact store (durable) and is read fresh, there is
**no `missions.context` write, no task-context stamping, no expander seed, and therefore
no lost-update race** with `inject_lessons` (which RMWs `missions.context`). This is the
central simplification over the first draft.

## Architecture

### 1. ORIGIN — new phase_0 naming step `0.0y product_name_pick`

LLM step that decides the name once and emits it to the store.

- `agent: "analyst"` (NOT `writer`: `writer` + object schema is flagged by
  `_INCOMPATIBLE_AGENT_SCHEMA`, `loader.py:233` — logged-only but a real threat to a clean
  structured emit). Instruction carries the explicit JSON-skeleton note ("produce your
  answer as a JSON object; your response text IS the artifact") like existing object steps
  (e.g. `-1.1` at `i2p_v3.json:454`). Instruction must require a **non-empty, real** name
  and avoid any "product (CODE)" adjacency (`test_i2p_v3_phase_code_leak.py`).
- `depends_on: []`. **Physically the first step in the steps array** (array order matters —
  Risks #3).
- `input_artifacts: ["raw_idea", "strategic_context"]` — mission-seed artifacts are
  delivered to a step ONLY when declared here (deps fetcher, `context.py:647`); they are
  NOT auto-available via `workflow_context`.
- `output_artifacts: ["product_name"]` — this is what lands the result in the artifact
  store.
- `difficulty: "easy"`, `tools_hint: []` (required by `test_i2p_v3.py:377,390`).
- **Structured object schema, NO `produces` file path:**
  ```json
  "artifact_schema": {
    "product_name": { "type": "object", "required_fields": ["product_name"] }
  }
  ```
  Object type → `_schema_is_structured_only` (`coulson/__init__.py:62`) True →
  `_apply_auto_strip` (`:299`) removes write tools → engine materializes the result into
  the store. (A `"string"` type would NOT be structured-only —
  `_STRUCTURED_SCHEMA_TYPES = {object,array}` only — and would keep write tools.) With **no
  `produces`**, the grounding guard bails (`guards.py:371`), so no auto-strip/grounding DLQ
  trap. (Doubly important: the grounding-skip mitigation `guards.py:374` is **uncommitted**
  on this machine — we must not depend on it.)
- **Validation:** the object emit is validated against `required_fields: ["product_name"]`
  by the grade gate (`apply.py:1842`). Do NOT set `requires_grading: false`. Worst case (an
  empty-string name slips the gate): injection finds nothing → no line; checks degrade to
  pass; the reviewer backstop still catches drift. Graceful, not silent.

**No separate pin/mechanical step.** The store write is intrinsic to `output_artifacts`;
nothing else needs to persist the name.

`0.0z.depends_on` flips `[]` → `["0.0y"]` so the name exists in the store before
reverse_pitch runs. The phase-0 spine (`0.0z → 0.0z.confirm → 0.0a.draft → 0.0a → 0.1`)
follows transitively; nothing that names the product runs before `0.0y`.

### 2. PROPAGATION — async store read in `build_user_context`

Add `_load_product_name(mission_id)` to `coulson/context.py` (mirrors
`inject_north_star._load_success_metrics`: `get_artifact_store().retrieve`, dict-or-JSON
handling, best-effort `None`). In `build_user_context` (async), right after the
`task_context` parse block (`context.py:~790`), inject when present:

```python
_pn = await _load_product_name(task.get("mission_id") or task_context.get("mission_id"))
if _pn:
    parts.append(
        "## Product Name (canonical)\n"
        f"The product is named **{_pn}**. Use this name EXACTLY in every artifact. "
        "Do NOT invent or vary the name."
    )
```

Reads fresh every dispatch; correct after `0.0y` completes; silent before it (returns
`None`). No expansion-time `{product_name}` template (`_substitute_payload` resolves only
`{mission_id}`; unmatched placeholders survive verbatim — confirmed `expander.py:756`).

### 3. ENFORCEMENT — `verify_contains_product_name` check on 0.0z + 0.1

New mechanical check declared in each producer's `checks[]`:

```json
{ "kind": "verify_contains_product_name",
  "payload": { "action": "verify_contains_product_name",
               "artifact_paths": ["mission_{mission_id}/.charter/reverse_pitch.md"] } }
```
(0.1 uses `product_charter.md`.)

- Registered via `_shape_check_spec("verify_contains_product_name", ...)` in
  `posthooks.py` `POST_HOOK_REGISTRY` (mirrors `verify_charter_shape` at `:505`) → picked
  up by `_CHECK_KINDS` (`apply.py:2091`) → verbatim payload (`apply.py:2137`).
- New branch in `mr_roboto/_run_dispatch` (`__init__.py:~1100`, mirroring the
  `verify_charter_shape` branch shape). It:
  1. `mid = task.get("mission_id")`;
  2. `name = await _load_name_from_store(mid)` (`get_artifact_store().retrieve(mid,
     "product_name")` → `dict["product_name"]`, stripped);
  3. if `not name`: return `Action(status="completed", result={"ok": True,
     "skipped": "no product_name pinned"})` — defensive, never hard-block on our own
     missing precondition (reviewer backstop covers it);
  4. `paths = _resolve_path_list(payload.get("artifact_paths"))`; read each file's text;
  5. **whole-word, case-insensitive** presence test of `name` in the artifact text
     (`re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE)` — guard against a
     name that is regex-empty after strip);
  6. present → `Action(status="completed", result={"ok": True})`;
     absent → `Action(status="failed", error="artifact does not contain canonical product
     name '<name>'", result={...})`.
- A `checks[]` `verify_*` returning `Action(status="failed")` → `PostHookVerdict(passed=
  False)` → `_apply_simple_blocker_verdict` re-pends the **source producer** (`apply.py:
  4779`, `worker_attempts` bumped, error stamped as feedback), DLQ on exhaustion. The
  producer re-pends **with the canonical name already injected in its prompt** (§2) →
  converges. (This is the exact loop `verify_charter_shape`/`verify_reverse_pitch_shape`
  already drive on these same steps — no new apply-layer code.)

The reviewer (1.13 check 10) is unchanged and remains the backstop for all other naming
sites.

## Data flow

```
intake (raw_idea, strategic_context)
        │
   0.0y product_name_pick (analyst, object schema, output_artifacts:[product_name])
        │            └── engine materializes → artifact store["product_name"] = {product_name:"X"}
        │
   0.0z reverse_pitch ── build_user_context awaits store → "Product Name: X" in prompt
        │              └── check verify_contains_product_name → reverse_pitch.md contains "X"? blocker if not
        │
   ... 0.0a ...
        │
   0.1 product_charter ── "Product Name: X" in prompt ── check → product_charter.md contains "X"? blocker if not
        │
   ... downstream steps all get "Product Name: X" injected ...
        │
   1.13 reviewer check 10 (backstop)
```

## Error handling

- **Empty/garbage name from 0.0y:** grade gate rejects a missing `product_name` field;
  re-pend/DLQ via the normal path. If an empty *string* slips through, injection/checks
  degrade to no-op and the reviewer backstop catches drift (graceful).
- **Check finds name absent in artifact:** blocker → producer re-pend (bounded by
  `worker_attempts`); converges because the name is in the prompt.
- **Name not yet in store when check runs** (shouldn't happen — `depends_on` ordering):
  check returns `completed/skipped` (defensive).
- **Restart mid-phase-0:** the artifact store is durable; re-pended 0.0z reads the same
  pinned name fresh; no in-flight state to lose.

## Risks / open items (validate in plan via TDD)

1. **Injection point reachability:** confirm `build_user_context` runs for every i2p
   workflow step and its output reaches the model prompt (the `react.py:373-374` pair).
   TDD: a step dispatch with `product_name` in the store renders the line; empty store →
   no line.
2. **Store materialization for object schema:** confirm a step with `output_artifacts` +
   object schema + no `produces` actually writes `{product_name:"X"}` into the store
   retrievable by name (the `success_metrics`/`north_star` precedent says yes). TDD against
   the real `get_artifact_store()`.
3. **Array-ordering / forward-reference (REAL):**
   `tests/workflows/test_i2p_v3_dep_integrity.py::test_no_forward_reference_dependencies`
   requires every `depends_on` to point at a step defined **earlier in the array**. `0.0y`
   must be inserted **physically first** in the steps array, ahead of `0.0z`.
4. **Step-count/structure tests:** `test_i2p_v3.py` asserts `150 ≤ steps ≤ 350` (range — 1
   new step OK) and that every step has `difficulty` + `tools_hint`.
5. **Scope honesty:** mechanical check covers 2 of N naming sites; reviewer backstop must
   stay; name quality unguarded (stated in non-goals).
6. **Per-dispatch store read cost:** `build_user_context` gains one async store read per
   step. Negligible (one small DB read), best-effort, `None` on any error.

## Test plan (high level)

- `_load_product_name` helper: store dict → name; JSON string → name; empty/missing →
  None; never raises.
- `build_user_context` injection: store has name → "Product Name (canonical)" line
  present; store empty → no line. (Async test against a fake/real store.)
- `verify_contains_product_name` executor branch: name present in artifact → pass; absent
  → fail blocker; whole-word (substring inside a larger word does not falsely match);
  unpinned store → completed/skipped (defensive).
- Check-fail → producer re-pend: a `checks[]`-driven failed verdict re-pends the source
  step (reuse existing review-routing/blocker test patterns).
- i2p_v3 structural tests green: `test_i2p_v3.py` (counts, difficulty, tools_hint),
  `test_i2p_v3_dep_integrity.py::test_no_forward_reference_dependencies` (array order),
  `test_i2p_v3_phase_code_leak.py`.
- No expansion-time `{product_name}` leakage.

## Files touched

- **Create:** `packages/mr_roboto/src/mr_roboto/verify_contains_product_name.py` (the
  whole-word presence helper, pure) — or inline in the dispatch branch if trivial.
- **Modify:** `packages/mr_roboto/src/mr_roboto/__init__.py` (`_run_dispatch` branch).
- **Modify:** `packages/general_beckman/src/general_beckman/posthooks.py` (registry entry).
- **Modify:** `packages/coulson/src/coulson/context.py` (`_load_product_name` +
  `build_user_context` injection).
- **Modify:** `src/workflows/i2p/i2p_v3.json` (new `0.0y` step physically first; `0.0z`
  `depends_on`; `checks[]` on `0.0z` and `0.1`).
- **Tests:** `packages/mr_roboto/tests/test_verify_contains_product_name.py`,
  `packages/coulson/tests/test_product_name_injection.py` (or nearest existing coulson test
  module).

## Deployment

Restart-gated (i2p_v3.json + coulson + mr_roboto + general_beckman). TDD green locally →
`/restart` → live-verify on a fresh mission (name pinned, both artifacts agree) → commit →
push. (Per repo norm: do not push restart-gated work until the user has restarted and
verified.)
