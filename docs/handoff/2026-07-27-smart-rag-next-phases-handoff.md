# Handoff — Smart-RAG next phases (Phase 2+); Phase 1 done+pushed

**Date:** 2026-07-27
**Track:** Smart-RAG memory redesign (chroma bloat + retrieval quality). **Separate from the m90 workflow-engine work** (see `docs/handoff/2026-07-27-m90-multiartifact-adr-fix-handoff.md`).
**Predecessors:** design + root cause `docs/handoff/2026-07-26-smart-rag-memory-design-handoff.md`; Phase 1 detail `docs/handoff/2026-07-27-smart-rag-phase1-done-handoff.md`; research `docs/research/2026-07-27-agent-memory-forgetting-research.md` (21/21 claims source-confirmed).

---

## 0. Status
- **Phase 1 DONE + LIVE-VERIFIED + PUSHED** — `origin/main` @ `c790fc27`, commits `01d93c6f`..`dbf33ec0`. Killed the implicit `user_feedback` firehose (the ~886K-row whale) at source (`_push_preference_feedback` off the `metrics_push` dispatch tuple) + write-time `encode_policy.should_store` gate at both `embed_and_store` seams. Killswitch `KUTAI_ENCODE_POLICY=off`. Verified at source in the running modules + 15/15 tests; end-to-end count-divergence unobserved only because the agent queue was idle (not a defect).
- **P3 forgetting signal LOCKED** = founder chose "Option A refined + documented path". Option B (per-doc retrieval-outcome attribution) REJECTED on 4 proofs (0.1% retrieval → no denominator; co-retrieval confound unlearnable on e5; MNAR bias; cost). Do not re-open without new evidence.

## 1. Key research corrections that constrain the design (all source-confirmed)
- **ChromaDB DOES have vacuum** — `chroma utils vacuum --path <dir>` + auto-pruning since **v0.5.6** (3-0 confirmed). The predecessor's "no vacuum API" conflated *no Python compaction API* (true — `collection.delete` doesn't shrink files) with *no vacuum at all* (false).
- **Cosine is UNUSABLE for dedup/merge** — contradicting facts are *more* embedding-similar than true duplicates (AUROC 0.59 ≈ chance). Merge on **structured identity**, never a cosine threshold.
- **Recency is the WORST retention signal** in a blind-regime benchmark (0.368 vs 0.770 for a learned multi-factor value fn). So recency = within-type tie-breaker ONLY; a learned value fn is the documented future path (needs training signal the 0.1% store doesn't currently produce).
- **MNAR** — "absence of retrieval is not evidence of low utility": never-retrieved ≠ junk for CURATED types; only firehose types (0.03–0.10% hit) are genuinely disposable → per-type policy resolves this.

## 2. Next phases (execute in order)
- **Phase 2 — P4 loadability.** **FIRST STEP:** check the prod chroma version and whether `chroma utils vacuum --path <chroma_dir>` reclaims HNSW segments on the namespaced `mission_{id}__*` collections (those are outside every static row cap). If vacuum works, the predecessor's rebuild-swap complexity collapses to a vacuum call. Then: bytes-budget invariant (`os.walk`) + boot-time size gate + delete-in-batches with WAL-checkpoint + enumerate via `list_collections()`.

  **FIRST-STEP RESULTS (2026-07-27, read-only / on a copy — memory `project_smart_rag_phase2_vacuum_orphan_hnsw_20260727`):**
  - chromadb **1.5.5**. Vacuum CLI is top-level **`chroma vacuum --path <dir> --force [--timeout N]`** — NOT `chroma utils vacuum` (that `utils` subcommand does not exist in 1.x; the v0.5.6 doc syntax is wrong for prod).
  - Live `data/chroma` = **254MB, 44,755 rows, 10 collections** (episodic 22,409 + semantic 21,594 dominate; `errors`=0 confirms §3 starved reader). **NO `mission_{id}__*` / `global__*` namespaced collections exist** in the restored-from-backup store — the "outside every row cap" concern is moot for the current store; it bites only when missions complete and their collections get deleted (see orphan leak).
  - **DECISIVE — vacuum is only HALF a reclaim.** Copy experiment: `delete_collection('episodic')` (22.4k rows) freed ~nothing on disk (confirms "delete doesn't shrink files"). Then `chroma vacuum` reclaims **SQLite only** (chroma.sqlite3 120.5→84.0MB, −37.6MB of deleted vector records) but **does NOT remove the orphaned HNSW segment dir** — episodic's 71.8MB index folder stayed byte-identical, is provably orphaned (on-disk dir with no row in the `segments` table), and a fresh `PersistentClient` open does **not** GC it either → **orphaned HNSW segment dirs leak permanently.**
  - **Revised P4 reclaim = `chroma vacuum` (sqlite) + deterministic orphan-segment-dir GC (HNSW):** `set(on-disk VECTOR seg dirs) − set(segment_ids in the segments table) = orphans → rmtree`, run at boot with the store closed. This orphan-GC is the missing half that rebuild-swap used to cover; for the 9.6GB whale (`project_kutai_coldboot_heartbeat_chroma_prune_20260725`: 4.4GB sqlite + 2×2.8GB HNSW) vacuum alone would leave ~5.6GB of orphaned HNSW.
  - **Live store is HEALTHY today (254MB, 0 orphans, no whale — Phase 1 killed it)** → Phase 2 is a **preventive guardrail**, not an urgent fix. Phase 3 (P5a read filters) remains the handoff's self-described "biggest precision win". Prioritization is a founder call.

  **PHASE 2 IMPLEMENTED 2026-07-27 (TDD, restart-gated, NOT committed — memory `project_smart_rag_phase2_vacuum_orphan_hnsw_20260727`):**
  - NEW `src/memory/chroma_maintenance.py` — `store_size_bytes` (os.walk bytes budget), `find_orphan_segment_dirs` / `reclaim_orphan_segment_dirs` (the orphan-HNSW GC: on-disk UUID dirs minus `segments`-table ids; **fail-safe → reclaim nothing if the table can't be read**; `min_age_seconds` guard skips in-flight segments; read-only sqlite open), `vacuum_store` (injectable `chroma vacuum` CLI wrapper), `enforce_size_budget` (boot decision; over budget → **cross-process `_destructive_lock`** then reclaim + vacuum, else skip).
  - `src/memory/vector_store.py::_init_store_locked` — boot gate runs `enforce_size_budget` in a thread **before the client opens** (store closed → GC/vacuum safe), **fail-OPEN** (a maintenance error never blocks the store from opening), killswitch `KUTAI_CHROMA_SIZE_GATE=off`, budget `KUTAI_CHROMA_MAX_BYTES` (default 2 GB), `wal_checkpoint` after over-budget reclaim.
  - Tests: `tests/memory/test_chroma_maintenance.py` 19 green (incl a chroma-gated **"healthy real store has ZERO orphans"** safety invariant + a real cross-process lock-exclusivity test); 27-test regression across the chroma/vector_store suite green. **E2E twice on a copy of the real prod store: reclaimed episodic's orphaned 68.5 MB HNSW dir + vacuum → 256.8→147 MB, all 9 live collections intact after reopen.** Opus adversarial review: SHIP-WITH-NITS → addressed cross-process race (added the lock), read-only open, missing-table fail-safe test.
  - **Deferred (honest scope):** "delete-in-batches with WAL-checkpoint" belongs with Phase 5 row-level TTL pruning (this gate reclaims via orphan-GC + vacuum, not row deletes); `list_collections()` per-collection enumeration is unnecessary for the orphan GC (it works at the segment-dir/table level). **Residual:** commit/push (a concurrent m90 session has an unpushed commit on local `main`) + a restart to confirm the boot-log line `Chroma size gate: N bytes ok`.

  **CLI-syntax note for the vacuum tool:** `chroma vacuum --path <dir> --force [--timeout N]` (chromadb 1.5.5).
- **Phase 3 — P5a read filters.** Add `where`/`type` filters to the raw semantic read at `rag.py:423` (biggest precision win). Do NOT force all readers through one global 0.72 threshold (breaks web-cache/support-RAG readers with different precision needs).
- **Phase 4 — P1b/P2 consolidation/novelty-merge.** Key merges on **structured identity** (not cosine). Occurrence-bump + keep-newest, never discard-new.
- **Phase 5 — P3 forgetting (decided).** Per-type hard TTL (firehose short; curated long/none) + deterministic **(subject, relation, object) bi-temporal supersession, invalidate-not-delete, no LLM** (MemStrata/Graphiti pattern). Recency as within-type tie-breaker only. Documented future: learned multi-factor value fn IF retrieval instrumentation is revived.

## 3. Adjacent findings (separate tickets)
- **Classic skills loop is DEAD in prod** — `find_relevant_skills`/`add_skill`/`record_injection`/`record_injection_success` have zero live callers; DB counters frozen since 2026-04-12; superseded by the yalayut exemplar/envelope system (June prompt-foundry refactor). `/skillstats` reports frozen April fossil data. Decide: revive, remove, or leave.
- **`errors` collection has a reader (`recall_error_patterns`) but no writer** — the "Known Issues" RAG section is structurally starved.
- **`analysis_key_finding`** polluter named by the original handoff was never located — grep for a writer; gate if it exists.
- **Store state note:** the 2026-07-25 prune (→2,255 rows) is NOT reflected in the live `data/chroma` (42K rows back to 2026-04-05) — it was restored from backup during the crash-loop recovery. Small (~sub-100MB), not the 9.6 GB whale, but relevant to P4 sizing.

## 4. Do NOT repeat
- Do not un-scope the `encode_policy` quality filter to all types — it would drop curated data (the plan-review blocker). Curated types are kill-only/pass-through by design.
- Do not re-introduce the implicit per-task feedback write. The whale is the implicit `accepted` firehose, not explicit corrections (`modified`/`rejected` still stored).
- Do not trust "chroma has no vacuum" — it does (v0.5.6+). Verify the version, then use it.
- Do not build Option B. Three internal + one external proof say it cannot work on this store.
