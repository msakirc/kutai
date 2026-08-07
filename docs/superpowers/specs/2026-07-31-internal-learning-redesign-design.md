# KutAI Internal Learning — memory & context redesign (design)

**Date:** 2026-07-31 (final rev after 2 adversarial reviews + founder decisions)
**Track:** Started as "smart-RAG Phase 5 (task_result TTL)"; the vector store was a symptom. **Supersedes** `docs/handoff/2026-07-31-smart-rag-phase3-done-p5-taskresult-handoff.md`.
**Status:** design FINAL — reviewed twice (SHIP-WITH-CHANGES both), founder-decided. Ready for `writing-plans` (P1).

---

## 0. Decision log (final)

- **Home = extend `kara_kutu`** (not a new package). kara_kutu is the existing "black box / flight recorder" package and **already owns `mission_lessons`** (relocated from `src/infra`, now a shim). Adding the per-step exemplar store + `capture`/`recall` there avoids fragmenting "durable execution history for recall" a third time (§3's root disease). No dependency cycle: `intersect→kara_kutu→dabidabi` acyclic; `general_beckman→kara_kutu` already live.
- **Injection = `intersect` multi-source gate, refactored in P2** (founder call, made with eyes open). Both reviews flagged this as a **medium refactor that risks the live skill path** and buys nothing until a 3rd source exists — the founder accepts that cost for the unified end-state. Honest accounting (§5): experience candidates **bypass** `scoring`/`exposure.classify`/`binding`/`render_variant` (else default-tier-3 → quarantine, and `_slot_key` collision); genuine reuse is only the merge loop + `budget.apply_caps` pass-through. P2 carries a **regression guard**: the existing yalayut skill envelope behavior must be byte-unchanged.
- **`mission_lessons` — absorb WRITES only; keep its readers.** `inject_lessons`/`top_mission_lessons` have 4+ live consumers (launch drafts, lessons_writeback, visual-review calibration, DLQ populator; auto-wired to every mission's phase_0). Route new writes through kara_kutu's `capture`; **do not retire those readers.**
- **Interface = two verbs** (`capture`+`recall`) + pure key-helpers + `Experience`. `invalidate`→capture upsert; `sweep`→internal prune (no public verb).
- **Corrections from review** (were wrong in earlier revs): capture hook = **`apply.py:6073`** (not 5999); the exemplar reader is **LIVE** (`context.py:1349`) but keyed on `profile.name` — the key-mismatch trap is *already shipped*, not hypothetical; i2p "0% lessons" is a **writer + read-timing bug**, not a missing reader.

## 1. Problem — KutAI does not learn

Multiple internal learning loops, each a **disconnected or mis-keyed half** — no single component owns a whole loop, so halves rot unnoticed.

| loop | reader | writer | status |
|---|---|---|---|
| `workflow_exemplars` (per-step success) | ✅ **LIVE** `context.py:1349` — but keyed on `profile.name` | ❌ `capture_exemplar` zero prod callers | reader fires but **empty + key-inconsistent**; writer unwired |
| `mission_lessons` (per-error-domain) | ✅ LIVE (`inject_lessons` phase_0, launch track) | ✅ LIVE (`apply.py` posthook-fail, DLQ, calibration) | both alive; **0 on i2p** = writer emits no i2p-domain lessons + stack detected mid-mission (`i2p_v3.json:482`) after phase_0 read → **writer+timing bug** |
| `errors` chroma collection | ✅ RAG (coder/fixer) | ❌ none | dead, 0 rows |
| past-experience (`recall_similar_tasks`) | ❌ never called | — | dead |
| skills library (vector) | fossil | fossil | dead (8% fossils) |
| episodic `task_result` (vector) | generic-RAG noise | ✅ 18k/day | wrong data — success event-logs, 0.03% read |

## 2. Evidence (live, 2026-07-31; ✅confirmed across both reviews)

episodic 68,107 rows / 100% task_result / **0.032% ever read** / zero failed rows (`metrics_push.py:119,133`). web_knowledge **648/660 (98%) past `ttl_days`** (unenforced). prefs double-injected (`preferences.py:169` == RAG semantic query). memory double-stored (`dabidabi:5531`+`:5541`). Context telemetry (325 builds): deps 52.7%, RAG 3.6% (low-value), `Watch out for`/`Past Experience`/`Known Issues` = 0%. `capture_exemplar` zero prod callers. *(Probes: `scratch_episodic_quality_probe.py`, `scratch_context_layers.py`.)*

## 3. Root cause

**Scattered ownership** (capture in `metrics_push`, store in `src/memory/*`, read in `coulson/context`, lessons in `apply.py`+`kara_kutu`, dead curation in `decay.py`) + **wrong substrate** (cosine where exact-key structured recall is needed; cosine AUROC ~0.59 for identity ≈ chance).

## 4. Vision — operation-keyed experience

Stable **operation key**; at the grade/completion event capture **success** (output + working tool-recipe) and **failure→fix** (pattern + resolution); inject only when the same keyed operation recurs. i2p key `(workflow, step_id, agent_type)` = instance #1 (densest, highest-pain = the founder's manual m90 grind). Generalizes to shopping `(query_class)`, LLM `(task_kind)`, later multimodal. KutAI is general orchestration; i2p is first, not the whole vision.

## 5. Architecture — capture → kara_kutu → intersect multi-source gate

```
grade/completion event (apply.py:6073  `a.kind=="grade" and a.passed`, + fail path)
        │  capture(key, …)  [success | failure mode]     ← own loop, structured, auto, unvetted
        ▼
 kara_kutu (extended)  owns TWO key namespaces:
     per-step (workflow,step_id,agent_type) → exemplar store  |  per-error-domain (stack,domain) → mission_lessons
     store+curate: upsert-supersede on write, prune top-N per key, filter-stale on read
        │  recall(key)  [dispatches by key-kind: exact top-N-by-quality  |  domain top-N-by-recency+suppress]
        ▼
 task dispatch → intersect.flash = MULTI-SOURCE gate (P2 refactor)
     ├─ capability adapter : yalayut.query + scoring + exposure.classify + binding   (existing path, UNCHANGED)
     └─ experience adapter : kara_kutu.recall(key) → hand-set exposure_class="inject", experience render
                             (BYPASSES scoring/classify/binding/render_variant — else tier-3 quarantine + slot collision)
     flash: merge loop + budget.apply_caps pass-through + kind-aware _slot_key/telemetry/envelope → ONE envelope
        ▼
 coulson.context renders the one envelope. Regression guard: skill envelope byte-unchanged.
```

- **Capture ≠ yalayut** (discovers/vets external capabilities; experience is internal auto-captured feedback — the `auto:` merge already minted noise).
- **Honest reuse:** only merge loop + `budget.apply_caps` are shared; experience skips 4 of 6 flash modules. This is a medium refactor (kind-aware `_slot_key`/`telemetry.record_usage`/envelope-build + candidate protocol), accepted by the founder.

## 6. Interface (into `kara_kutu`)

Two verbs; everything else is a pure key-helper or the return shape. Call sites trivial — recipe reconstructed from `task_id` internally (`_tool_recipe_for_task`).

```python
# key helpers (pure; SINGLE source of truth — capture AND the live reader context.py:1349 must both use these)
def i2p_step_key(workflow, step_id, agent_type) -> str      # agent_type := task['agent_type'], NOT profile.name (B1/B4)
def error_domain_key(stack, domain) -> str                  # the mission_lessons namespace

async def capture(key, *, task_id, mission_id=None,
                  result=None, quality=None,                # success mode
                  pattern=None, fix=None, severity="info") -> bool
    # GUARD exactly one mode: (result) XOR (pattern and fix); else raise.
    # upsert-supersede (keep-newest = invalidate-not-delete); prune top-N per key on write
    #   (per-step: top-N by quality; per-domain: top-N per (stack,domain) — bounds the lessons table).
    # success: recipe reconstructed from task_id. failure: reuses kara_kutu.upsert_mission_lesson.

async def recall(key, *, limit=3) -> list[Experience]
    # dispatch by key-kind: per-step = exact top-N-by-quality; per-domain = top-N-by-recency, honor `suppressed`.

@dataclass(frozen=True)
class Experience:
    kind: str; key: str; text: str; fix: str | None
    tool_recipe: list[dict]; quality: float
    occurrences: int; severity: str; suppressed: bool     # carried by mission_lessons; consumers render these
    task_id: int; created_at: str
```

- **`quality`**: derive from `GradeResult` booleans (`relevant/complete/coherent/well_formed`), no LLM. **Default gracefully** when booleans absent (short-circuit grade-passes build `raw` without them → treat missing as pass/mid, never 0).
- **Internal:** schemas, dedup, upsert-supersede, top-N prune (both namespaces — bounds the lessons table the sub flagged as unbounded), stale-filter, recipe reconstruction.

## 7. Delete / Keep / Hygiene

**DELETE (~213 MB):** episodic firehose (gate write; rebuild-swap purge), `recall_similar_tasks`, skills-library vector path, `semantic.memory` mirror, `user_preference` RAG double-read, code-agent RAG noise layers, stale web rows. *(Git + this doc = preservation.)*

**KEEP + wire:** `workflow_exemplars` → **move into kara_kutu**, repoint its `src.infra` imports to `dabidabi`/`yazbunu`, wire `capture` at grade-pass, **fix the live reader `context.py:1349` to key on `task['agent_type']`**; `mission_lessons` → route new writes via `capture`, **keep `inject_lessons`/`top_mission_lessons` readers** (launch track); `intersect` → P2 multi-source refactor; `yalayut` + Fatih Hoca calibration untouched. Separately fix the **i2p lessons writer+timing** bug.

**HYGIENE (secondary):** durable facts as plain structured memory (no vector mirror); small relevance-gated prefs; enforce `web_knowledge.ttl_days`.

## 8. Phasing

- **P1 — capture into kara_kutu + fix the live key, round-trip proof.** Hook `apply.py:6073` (`source`, `a.source_task_id`, `a.raw` in scope). Wire `capture` (success); quality from booleans (default-safe); key via the single builder on `task['agent_type']`; **fix `context.py:1349` to use it too**. Add `capture` (failure) migrating `_maybe_emit_lesson_from_posthook_fail` onto it (reuse `upsert_mission_lesson`, no double-write; keep launch readers). **Exit: a row captured at `[X.Y]` grade-pass is returned by the REAL reader (`context.py:1349` path), not a synthetic recall.**
- **P2 — intersect multi-source gate + i2p lessons fix.** Refactor `flash`: candidate protocol; kind-aware `_slot_key`/`telemetry`/envelope-build; capability adapter = current path (byte-unchanged, regression-guarded); experience adapter over `recall(key)` querying both namespaces, hand-setting exposure/render (bypassing classify/scoring/binding). Fix i2p lessons: emit i2p-domain lessons + move the stack read after `tech_stack_detected`. **Exit: one envelope carries the experience item at a repeat `[X.Y]`; skill envelope unchanged.**
- **P3 — delete wrong-substrate + reclaim.** After P1/P2 migrate readers. Gate episodic write; rebuild-swap purge + separate-process reclaim + vacuum. **Exit: chroma < ~150 MB, no regression.**
- **P4 — hygiene.** prefs/facts/freshness + `ttl_days`.
- **Later:** external-source adapters (web/docs/APIs) into the P2 gate — the payoff that justifies the multi-source refactor; cross-mission solution reuse; multimodal keys.

## 9. Success metrics

Real-reader round-trip works (today: 0), unified-envelope experience fire-rate at repeat operations, **retry/DLQ-rate at previously-failing `(workflow, step)`**, founder-intervention frequency, chroma size.

## 10. Hard constraints (research)

Exact-key/structured identity — never cosine for identity (holds inside the experience adapter). Invalidate-not-delete; keep-newest. Chroma deletion = rebuild-swap + separate-process reclaim after vacuum. No new LLM call; failure `fix` reuses grader `a.raw`. MNAR: never-retrieved ≠ junk for curated types; age-gate only the firehose.

## 11. Open items

**None blocking.** First P1 verification (cheap, read-only, settles the design's hinge): confirm whether `tasks.agent_type` == the `profile.name` coulson resolves for completed i2p `[X.Y]` steps — if they diverge, B1 is live and the single-key-builder must cover both capture and `context.py:1349`.

*(Resolved: home=extend kara_kutu; injection=intersect multi-source refactor (P2, eyes-open); interface=2 verbs; mission_lessons=absorb-writes-keep-readers; quality=grade-boolean default-safe; fix=grader `a.raw`; hook=`apply.py:6073`; key=`task['agent_type']` on both sides.)*

## 12. Provenance

Predecessors: smart-RAG handoffs; research `docs/research/2026-07-27-agent-memory-forgetting-research.md`. Memories: `project_skills_loop_dead_chroma_retrieval_20260726`, `project_smart_rag_phase3_feedback_purge_20260730`, `project_smart_rag_phase2_vacuum_orphan_hnsw_20260727`. Reviews: two general-purpose subs, 2026-07-31 (both SHIP-WITH-CHANGES); founder decisions: extend kara_kutu, intersect refactor now.
