# Handoff — Smart-RAG Phase 3 (P5a read filters); Phases 1 & 2 done

**Date:** 2026-07-27
**Track:** Smart-RAG memory redesign (chroma bloat + retrieval quality). **Separate from the m90 workflow-engine work.**
**Predecessors:**
- Design + root cause: `docs/handoff/2026-07-26-smart-rag-memory-design-handoff.md`
- Phase 1 (encode_policy firehose gate): `docs/handoff/2026-07-27-smart-rag-phase1-done-handoff.md`
- **Phase 2 + all first-step diagnostics: `docs/handoff/2026-07-27-smart-rag-next-phases-handoff.md`** (read its "PHASE 2 IMPLEMENTED" block — has the vacuum/orphan-HNSW findings that constrain P4)
- Research (21/21 claims source-confirmed): `docs/research/2026-07-27-agent-memory-forgetting-research.md`
- Memory: `project_smart_rag_phase2_vacuum_orphan_hnsw_20260727`, `project_skills_loop_dead_chroma_retrieval_20260726`

---

## 0. Status
- **Phase 1 DONE + PUSHED + LIVE-VERIFIED** — `origin/main c790fc27` (`01d93c6f..dbf33ec0`). Killed the implicit `user_feedback` firehose at source + write-time `encode_policy.should_store` gate. Killswitch `KUTAI_ENCODE_POLICY=off`.
- **Phase 2 DONE + PUSHED + LIVE-VERIFIED** — `origin/main 761d465e`. Chroma P4 loadability guardrail: boot-time bytes gate + orphaned-HNSW-segment GC + `chroma vacuum` wrapper in new `src/memory/chroma_maintenance.py`, wired fail-open into `vector_store._init_store_locked` before the client opens. Killswitch `KUTAI_CHROMA_SIZE_GATE=off`; budget `KUTAI_CHROMA_MAX_BYTES` (default 2 GB). Independent 2nd validator = SHIP; live boot log confirmed `Chroma size gate: 273723760 bytes (budget 2000000000) ok` → clean start, +0.25s, no regression.
- **P3 forgetting signal LOCKED** — founder chose "Option A refined + documented path"; Option B (per-doc retrieval-outcome attribution) REJECTED on 4 proofs. Do not re-open.

## 1. Phase 3 — P5a read filters (THIS SESSION'S WORK)

**Goal:** raise retrieval *precision* by filtering at read time in the main RAG pipeline. Per the design, this is the **biggest single precision win** left.

**The exact site — grounded:**
- `src/memory/rag.py:423`:
  ```python
  results = await _vs_query(text=q, collection=col_name, top_k=top_k)
  ```
  This is the primary RAG read. It passes **no metadata filter** — it pulls `top_k` from each collection unfiltered, then relies on post-hoc `_rank_results` (`rag.py:200`, relevance filtering) + `_deduplicate` to trim. Low precision by construction: junk enters, then you rank it out.
- **The plumbing already exists.** `vector_store.query` accepts a `where` ChromaDB metadata filter (`vector_store.py:859` def, applied at `:907`). Other readers already use it: `preferences.py:173` (`where={"type":"user_preference"}`), `skills.py:244/271` (`where={"type":"skill"}`), `conversations.py:110/169` (`where={"chat_id":...}`), `episodic.py:140`. The RAG read is the one hot path that skips it.
- `_vs_query` is the injected alias bound at `rag.py:38` (`_vs_query = query`), so it already forwards `**` to `vector_store.query` — you can pass `where=` through.

**What to build (confirm scope via brainstorming skill first):**
1. A per-collection (and/or per-agent-type) metadata filter map — e.g. which `type` values are admissible for each collection given the querying `agent_type`. Thread it into the `rag.py:423` call as `where=`.
2. Keep it **per-reader**, not global. See constraint below.

**HARD CONSTRAINTS (from research — do not violate):**
- **Do NOT force all readers through one global 0.72 similarity threshold.** Different readers need different precision (web-cache and support-RAG readers legitimately want looser recall). A single global cosine cutoff breaks them. Filter on **metadata** (`where`/`type`), not a blanket similarity floor.
- **Cosine is unusable for identity/dedup decisions** (contradicting facts embed *more* similar than true dupes, AUROC 0.59 ≈ chance). This matters for Phase 4, but keep it in mind: don't add similarity-threshold "precision" logic here either.
- **MNAR** — never-retrieved ≠ junk for curated types. A metadata filter that hard-excludes a curated type would silently starve it. Scope filters to *admit* the right types, not to *exclude* by usage.

**Validation approach (mirror Phase 1/2):** TDD. There is a RAG hit log (`_rag_log.info("rag_hit", ...)` around `rag.py:454`) recording `raw_counts`, `ranked`, per-collection counts — use it to measure precision before/after on a real query without needing the 0.1%-retrieval instrumentation. Live-verify after a restart; the pipeline is on the agent hot path so a bad filter starves context — guard with a killswitch env and default-open.

## 2. Remaining phases (after Phase 3)
- **Phase 4 — P1b/P2 consolidation/novelty-merge.** Key merges on **structured identity** (subject/relation/object or explicit keys), never a cosine threshold. Occurrence-bump + keep-newest, never discard-new.
- **Phase 5 — P3 forgetting (decided).** Per-type hard TTL (firehose short; curated long/none) + deterministic **(subject, relation, object) bi-temporal supersession, invalidate-not-delete, no LLM**. Recency = within-type tie-breaker only. **This is where "delete-in-batches with WAL-checkpoint" belongs** (deferred from Phase 2 — the Phase 2 gate reclaims via orphan-GC + vacuum, not row deletes; row-level TTL pruning needs the batched delete + `wal_checkpoint`, which already exists at `vector_store.py:530`).

## 3. Adjacent findings (separate tickets — decide, don't silently carry)
- **Classic skills loop is DEAD in prod** — `find_relevant_skills`/`add_skill`/`record_injection`/`record_injection_success` have zero live callers; DB counters frozen since 2026-04-12; superseded by the yalayut envelope (June prompt-foundry refactor). `/skillstats` reports April fossil data. Decide: revive, remove, or leave. (memory `project_skills_loop_dead_chroma_retrieval_20260726`)
- **`errors` collection has a reader (`recall_error_patterns`) but no writer** — 0 rows live; the "Known Issues" RAG section is structurally starved.
- **`analysis_key_finding`** polluter named by the original design handoff was never located — grep for a writer; gate at write time if it exists.
- **Store state:** live `data/chroma` ≈ 262 MB, 10 collections, ~46K rows, **0 orphans** (healthy). Not the 9.6 GB whale (Phase 1 killed it). The 2026-07-25 prune-to-2,255 was restored from backup during crash-loop recovery, so row history goes back to 2026-04-05.

## 4. Do NOT repeat
- Do not un-scope the `encode_policy` filter (Phase 1) to curated types — kill-only/pass-through by design.
- Do not re-introduce the implicit per-task `accepted` feedback write.
- Do not force a global similarity threshold in the RAG read (breaks per-reader precision). Metadata filter only.
- Do not build P3 Option B (per-doc retrieval attribution) — four proofs say it can't work on this store.
- Vacuum syntax for chromadb 1.5.5 is `chroma vacuum --path <dir> --force` (NOT `chroma utils vacuum`).
