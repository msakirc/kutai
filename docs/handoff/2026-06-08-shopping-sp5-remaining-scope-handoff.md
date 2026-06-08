# Handoff — Shopping SP5 remaining scope (what actually unblocks `await_inline` deletion)

**Date:** 2026-06-08
**For:** a parallel session picking up the `src/shopping/` migration (and whoever finishes quick_search/product_research).
**From:** the session that launched shopping_v3 live + fixed 6 launch bugs.
**Read first:** memory `project_shopping_sp5_coupling_20260605`, spec `docs/superpowers/specs/2026-06-05-shopping-workflow-step-migration-design.md`, prior handoff `docs/handoff/2026-06-06-shopping-sp5-coupling-continuation.md`.

---

## TL;DR — is SP5 unblocked? NO. Three live `request()` caller groups remain.

The original plan migrated **only `shopping_v3`** (the category/deep_research path). It is now LIVE and a clean variant run is confirmed (mission #84). But `LLMDispatcher.request()` (the shim SP5 needs gone) still has **three** live caller groups the plan never touched:

```
request() live callers (2026-06-08, post-v3-launch):
  GROUP 1 — src/workflows/shopping v2-style handlers (LIVE via quick_search + product_research)
    src/workflows/shopping/pipeline_v2.py:363  _grouping_llm_call      (← step_group)
    src/workflows/shopping/pipeline_v2.py:499  _synthesis_llm_call     (← step_synthesize_reviews)
    src/workflows/shopping/labels.py:22        _label_llm_call         (← step_label)
      used by handlers: group_label_filter_gate, synth_one, format_compare, group_and_synthesize
      used by workflows: quick_search_v2.json, product_research_v2.json  (BOTH LIVE)
  GROUP 2 — src/shopping/intelligence/_llm.py:42   (SEPARATE subsystem, own spec — this handoff's focus)
  GROUP 3 — dead/uncertain: coulson/reflection.py:88, src/workflows/engine/constrained_emit.py:147
            (handoff history calls these "dead-legacy" — VERIFY liveness before deleting; constrained_emit
             is a documented live feature, so :147 may be a live path, not dead)
  single_shot.py:46 — dead (no agent sets execution_pattern="single_shot" after T9) but TEST-ENTANGLED
    (5 tests patch coulson._single_shot_run: coulson/tests/test_z6_polish_detect_bail_e2e.py,
     test_z6_t7c_detect_and_bail.py ×4; tests/integration/test_agent_basic.py:340)
```

**`request()` cannot be deleted until GROUP 1 and GROUP 2 are migrated.** `await_inline` (SP5) deletes only after `request()` is gone.

---

## What shipped this session (all on `main`)

- **T8** compare-all = Approach B native-join (founder pick over CPS), terminal delivery.
- **T9** `shopping_clarifier` single_shot→react + canonical `clarify` action (fixed a LATENT no-op: single_shot dropped the clarify question, so in-workflow clarify never paused).
- **T10** launch flipped to `shopping_v3` (wf_map + quick_search escalation_target); compare-all re-routed into the pump.
- **shopping_v2.json retired** (orphaned).
- **6 live launch bugs** (all pre-existing dormant→live wiring debt, each guarded by a regression test):
  1. `0.1 understand_query_check_clarity` had no `_STEP_HANDLERS_V2` entry → added `handler_understand_query`.
  2. `2.1 clarify_variant` wrong mechanical shape (nested `context.executor:"clarify"` instead of top-level `executor:"mechanical"` + `payload:{action:"clarify",...}`) — fixed in ALL 4 shopping workflows.
  3. Branch discriminator: `skip_when` on a MISSING artifact defaults to RUN, and a skip_when-skipped step completes status=`completed` (NOT `skipped`) so transitive skip never propagates → spurious branch runs. FIX: gate (1.1e) SEEDS `clarify_choice` {chosen|pending|escalation}; tap overwrites {variant|compare_all}; every branch keys on `clarify_choice.kind` (always present).
  4. Delivery picker (`advance.py::_maybe_complete_mission`) shipped a skipped step's `{"skipped":true}` JSON as the final answer → now ignores skip-sentinels.
  5. `dogru_mu_samet` degeneracy gate false-rejected `groups_state` (64915 chars; repeated candidate-JSON structure reads as repetition) → `post_execute` skips the gate for deterministic dispatchers (`shopping_pipeline_v2`).
  6. **Cache coherence:** `ArtifactStore._cache` is PER-INSTANCE and `retrieve` returns cached values without re-reading DB (artifacts.py:33,76). The pump reads via the `get_artifact_store()` singleton; `_resume_mission_at_step` wrote the tap's `clarify_choice` via a FRESH `ArtifactStore()` → singleton cache stayed stale → branches skipped on stale `pending`. FIX: `_resume` writes through the singleton. **⚠️ BROADER LATENT DEBT:** many call sites do fresh `ArtifactStore()` (clarify.py, `pipeline_v2._read_artifacts`) — per-instance caches can diverge from the singleton. Only the clarify_choice write-path is fixed. A shared/class-level cache is the real fix (test-isolation risk — fresh missions share cache across tests).

**Live status:** variant path CLEAN (mission #84: 2.0*/2.3* skip, 2.2*+3.0 run, card delivered). Chosen + compare-all paths not yet confirmed clean live (founder paused testing). Producer agents (grouper/labeler/synthesizer) all run live and produce valid shapes.

---

## The migration pattern (already built — REUSE it)

Every inline `request()` call becomes a `prep (deterministic) → agent:<producer> (react one-pass) → apply (deterministic)` triad. ALL the machinery exists from T1-6:

- **Producer agents** (registered in `src/agents/__init__.py`): `shopping_grouper`, `shopping_labeler`, `shopping_synthesizer`.
- **Triad handlers** in `pipeline_v2.py` (`_STEP_HANDLERS_V2`): `group_prep`, `group_apply_label_prep`, `label_apply_filter_gate`, `synth_prep`, `synth_apply`, `compare_prep`, `compare_line_apply`, `compare_assemble`, `understand_query_check_clarity`.
- **Reference workflow:** `src/workflows/shopping/shopping_v3.json` — copy its step structure.
- **Mechanical clarify shape (CRITICAL):** top-level `"executor":"mechanical"` + `"payload":{"action":"clarify","kind":"variant_choice","payload_from":"gate_result"}` (NOT nested in context — bug #2).
- **Branch discriminator (CRITICAL):** the gate handler `handler_label_apply_filter_gate` seeds `clarify_choice` via `_gate_with_seed_choice`; branch steps key `skip_when` on `clarify_choice.kind` (NEVER on an artifact that may be absent — bug #3).
- **Delivery:** the last/highest-id step must return `{formatted_text,...}`; the picker ignores skip-sentinels (bug #4).

---

## Remaining work

### TASK A — migrate `quick_search_v2.json` + `product_research_v2.json` to producer triads
Both currently use the fused v2 handlers (`group_label_filter_gate`, `synth_one`, `format_compare`) → `step_group`/`step_synthesize_reviews`/`step_label` → `request()`. Rewrite each JSON to the v3 triad structure (reusing the existing handlers + producers above). Their flows differ from v3 (quick_search = fast top-3, product_research = deep single product), so preserve each one's intent — don't just clone v3.
- After both migrate, GROUP 1 `request()` callers (`_grouping_llm_call`, `_synthesis_llm_call`, `_label_llm_call`) + the fused handlers (`_handler_group_label_filter_gate`, `_handler_synth_one`, `_handler_format_compare`, `_handler_group_and_synthesize`) become dead → delete them. KEEP the deterministic reused fns (`step_filter`, `step_variant_gate`, `format_group_card`, `step_compare_all`, `gather_review_snippets`, `_parse_*`, `*_view`).
- Live-validate each via Telegram (simple query → quick_search; specific product → product_research).

### TASK B — `src/shopping/` spec (the parallel-session focus the founder requested)
`src/shopping/intelligence/_llm.py:42` is GROUP 2 — the `ShoppingPipeline` / `src/shopping/` two-tier subsystem (25.9k LOC, the simple-query/two-tier path, NOT `src/workflows/shopping/`). Per the dispatch handoff it needs its **own brainstorm + spec** — its carrier may be **interactive CPS resume / husam-inline** (a user waiting synchronously), NOT workflow steps. Steps:
1. `superpowers:brainstorming` on the carrier (workflow-steps vs husam-inline vs CPS) given a synchronous waiting user.
2. Map every `request()` / inline-LLM call in `src/shopping/`.
3. Write spec + plan; implement; validate.
This is independent of TASK A and is the clean parallel unit.

### TASK C — T11 final (after A + B both land)
1. Re-grep: `rg -n "\.request\(|_request_kwargs_to_spec|_task_result_to_request_response|single_shot|execution_pattern\s*=\s*.single_shot" src packages --glob '!*/tests/*'` — STOP if any unexpected live caller.
2. Verify GROUP 3 liveness (reflection.py:88, constrained_emit.py:147) — if live, they are separate migrations, NOT free deletes.
3. Delete `single_shot.py` + the `single_shot` branch in `coulson/__init__.py:90-91` + rewrite the 5 entangled tests (they patch `coulson._single_shot_run`; switch them to react or remove).
4. Delete `LLMDispatcher.request()` + `_request_kwargs_to_spec` (+ `_task_result_to_request_response` after re-grep confirms 0 callers).
5. **THEN** SP5 deletes `await_inline`.

---

## Build/test discipline (carry over)
- `timeout 120 pytest` always; NEVER concurrent pytest (SQLite lock crash-loops live KutAI); `tests/` and `packages/*/tests/` in SEPARATE invocations (colliding conftest → "Plugin already registered" error).
- Bare `python -c` misses editable packages (`fatih_hoca`, `dogru_mu_samet`) — validate via pytest, not `python -c`.
- Live KutAI runs from the `main` checkout; new code loads on the founder's `/restart` (Telegram). NEVER `taskkill`. Telegram "✅ [step]" ticks show for SKIPPED steps too — inspect `tasks.result` (`{"skipped":true}`) in the DB to tell run-vs-skip, not the ticks.
- DB inspection: `DB_PATH` from `.env` (`C:\Users\sakir\ai\kutai\kutai.db`); artifacts live in the `blackboards` table `data.artifacts` JSON, NOT an `artifacts` table. Use `PYTHONIOENCODING=utf-8` (Turkish text / emoji crash cp1252).

## Live validation recipe (per migrated workflow)
Via Telegram: vague query (clarifier should ask), specific query (chosen → one card), multi-variant → tap a variant (its card), → tap "compare all" (stacked cards). Then `SELECT lane,status,COUNT(*) FROM tasks WHERE mission_id=<m> GROUP BY 1,2` — producer children complete, none orphaned; inspect `blackboards.data.artifacts.clarify_choice` + per-step `result` for run/skip correctness.
