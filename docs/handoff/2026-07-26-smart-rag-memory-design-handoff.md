# Handoff — Smart RAG memory: root cause PROVEN, design HARDENED, one fork open

**Date:** 2026-07-26
**Status:** INVESTIGATION + DESIGN COMPLETE. **No code written. Nothing committed.**
**Predecessor:** `docs/handoff/2026-07-25-chroma-memory-bloat-investigation.md` (this answers its 6 questions)
**Blocked on:** one founder decision (P3 signal) that needs research finished first — see §6.

---

## 0. TL;DR for whoever picks this up

The 9.6 GB bloat was **not** a storage bug. Two writers fire **once per completed task**, forever,
with no gate, and the retention mechanism that was supposed to stop them **was never wired to
anything**. The fix is a 5-pillar deterministic design (below). Two adversarial reviews **killed
two of the five pillars as originally specced** and proved empirically that the obvious fix
(row-count caps + `col.delete`) **would not have worked** — it would pass its own assertions and
still cold-load 9.6 GB.

**Read §5 before writing any code.** It is the part that is counter-intuitive and will be
re-invented wrongly by anyone who skips it.

---

## 1. Root cause (PROVEN, not suspected)

Answers to the predecessor handoff's 6 questions:

**Q1 — why decay never held caps.** `src/memory/decay.py::run_decay_cycle` has **ZERO callers.**
Verified by full-tree grep: only its own def (`decay.py:103`), its docstring (`decay.py:12`), and
doc references exist. `cron_seed.py` seeds `vector_maint_wal` + `vector_maint_snapshot`, but
`packages/mr_roboto/src/mr_roboto/executors/vector_maint.py` only does WAL-checkpoint +
directory-snapshot — **neither prunes a single row.** The mechanism was dead on arrival.
(Handoff's suspects #2 `is_ready()` chicken-egg and #3 weak `PRUNE_FRACTION=0.2` are real
**secondary** flaws — see §5 — but the primary fact is it simply never ran.)

**Q2 — episodic firehose.** `src/core/metrics_push.py:118-134 _push_episodic_memory` →
`src/memory/episodic.py:94 store_task_result` embeds **every completed task**
(`type=task_result`, doc_id `task-{id}-{ts}` = timestamped ⇒ blind append, never upserts).
→ 885 K rows. It IS read (`recall_similar_tasks`), so it has real value; it has no dedup and no TTL.
Only writer in the system with any quality gate (degenerate-output check, `episodic.py:60`).

**Q3 — semantic firehose.** `src/core/metrics_push.py:155-163 _push_preference_feedback` fires
`record_feedback(task, "accepted")` per completed task → `src/memory/preferences.py:98`
embeds `type=user_feedback` into **semantic**. → 886 K rows. **Never read** (see §4). Real
feedback learning lives in the SQLite `task_feedback` table (`src/memory/feedback.py:50-67`),
a completely separate mechanism. So ~half of all 1.77 M embeddings were never retrievable.

**Q4 — 22 M metadata rows.** ChromaDB `embedding_metadata` is EAV: one row per
(embedding × metadata key). 1.77 M × ~12.5 keys ≈ 22 M. Proportional, not a separate bug.
**Design consequence:** every metadata key you add multiplies sqlite rows by the embedding count.
Do NOT persist an N-key score breakdown (see §5, Flaw 2b).

**Q5/Q6 — governance/loadability.** None existed. No size metric, no alert, no write-time cap.

**Two whales, both from `metrics_push.py`, both once-per-task.** Everything else is a rounding error.

---

## 2. Founder's standing constraints (from `docs/issues/memory-redesign-context.md` + this session)

- Precision over coverage ("one relevant result beats five vaguely related ones").
- Binary, not numeric (small LLMs give noisy 1-5 scores).
- Static/heuristic routing, not LLM routing.
- 40 % context budget cap.
- **Loadability invariant** (top requirement): store must never grow past what cold-loads inside
  the boot heartbeat window.
- Utility = real retrieval-success rate, not grader opinion. ← **contested, see §6**
- Founder directives this session: *"no guesses, no band aids, no quality loss"*;
  *"running llm for each llm job might inflate things"*; *"aim for the stars"*.

### ⚠️ STALE PREMISE CORRECTED BY FOUNDER (2026-07-26)
The prior design docs repeatedly say "cloud not connected yet" and rule out LLM summarization /
HyDE "until cloud connects". **Founder: cloud models were injected months ago.** Verified —
`packages/kuleden_donen_var/` (KDV) actively tracks cloud provider rate limits across 25 modules.
**Therefore the categorical ban on memory-LLM calls is void.** It is now a *cost/quota* decision
(batched + idle-gated + OVERHEAD lane), not a hard no. Any doc asserting "no cloud" is stale.
This may reopen LLM-assisted semantic consolidation and contradiction detection, which were
deferred on the old premise.

---

## 3. The design — 5 deterministic pillars (post-adversarial-review)

Core reframe: **`src/memory/skills.py` is already a complete self-curating store** — selective
encode (PASS + iters≥2 + tools), merge-on-duplicate (cosine ≥0.93 + keyword-overlap guard,
strategies capped at 5, prune-worst-first), utility-forget (injection-success-rate, cold-start
neutral, A/B lift), restrained inject. **The design is: generalize skills' proven curation to the
whole store, deterministically.** "Consolidation" = the merge idiom, NOT an LLM distiller.

**P1 · Selective encoding** at the write choke points `vector_store.py:710 embed_and_store`
**and** `vector_store.py:179 embed_and_store_for_mission` (⚠️ there are TWO; the second bypasses
everything and currently has no prod callers — route it or document it as dead):
- (a) Kill **only** the two verified whales: the `user_feedback` **vector** write and the episodic
  `task_result` firehose (via gating, not blind kill). **Do NOT kill** `fact`,
  `ingested_document`, `cross_agent_insight`, `conversation_summary` — see §4.
- (b) Novelty-merge for firehose types: threshold **≥0.93** (skills' proven value) **plus the
  keyword-overlap guard** (`skills.py:119-133`, `STRATEGY_RELEVANCE_MIN_OVERLAP=0.25`).
  Semantics = **occurrence-count bump + keep-newest, NEVER discard-new** (discarding collapses
  two distinct failures with different fixes into one — violates "no quality loss").
- (c) Quality filter: reuse `skills.py:42-53 _DESC_POLLUTION_RE` + length cap.
- (d) Per-type policy: episodic stores **failures always**, successes **only if novel**.
- (e) For facts/prefs use **temporal supersession** (new fact expires conflicting old —
  findings #9), not merge. Better fit than cosine-novelty for user-stated facts.

**P2 · Consolidation** = deterministic merge-on-write (P1b) + a paginated, row-capped,
off-heartbeat merge-sweep executor. LLM prose-distillation **deferred** — but see §2, the
premise that forced that deferral is now stale; revisit as batched/idle-gated OVERHEAD.

**P3 · Forgetting** — **OPEN FORK, see §6.**

**P4 · Loadability** — **completely re-specced by adversarial review, see §5.**

**P5 · Retrieval precision** (split in two):
- (a) **SOUND** — add `type`/`where` filters to raw semantic reads. `rag.py:423` queries with
  **no `where` filter**, so every type competes on raw cosine distance. Biggest precision win.
- (b) **REJECTED** — do NOT force all readers through rag.py's 0.72 + dedup. Each raw reader has
  deliberately different semantics and would break:
  | Reader | file:line | Why 0.72 breaks it |
  |---|---|---|
  | web-search cache | `web_search.py:454,483` | uses ~0.5 + keyword guard + <12 h recency; comment `:459-465` documents that tight cosine gave false hits |
  | support RAG | `support_rag.py:63-74` | **intentionally no threshold** — weak matches drive the escalation signal |
  | `recall_similar_tasks` | `episodic.py:132-134` | needs its `where={"agent_type":...}` preserved |
  | shopping | `vector_bridge.py:444-450` | `where={data_type,user_id}`, sparse data |
  | `/recall` | `telegram_bot.py:7022` | user command, must stay loose |
  Make rank/rerank **opt-in helpers**, not a mandatory funnel. Note `rag.py:78 RERANKER_ENABLED=True`
  — the reranker is ON (docs claiming it's disabled are stale); funneling adds ~200 ms CPU
  cross-encoder cost to latency-sensitive paths.

---

## 4. ⚠️ The "dead write" kill list was HALF WRONG — do not repeat my error

My first draft killed 5 types as "never read". Adversarial review **refuted 3 of 5**. The
load-bearing fact: **`rag.py:423 retrieve_context` queries with NO `where` filter**, so a type with
no dedicated reader **still surfaces** via raw semantic distance. "No type-filtered reader" ≠ "never read".

| Type | Verdict | Evidence |
|---|---|---|
| `user_feedback` | ✅ **KILL** (vector write only) | no filtered reader; real feedback is SQLite `task_feedback` (`feedback.py:61-89`) |
| `task_result` | ✅ **GATE** (not kill) | read by `recall_similar_tasks`; the volume whale |
| `fact` | ❌ **KEEP** | `/remember` writes (`telegram_bot.py:6995`), `/recall` reads (`:7022`). Killing breaks a user feature |
| `ingested_document` | ❌ **KEEP** | `/ingest` (`telegram_bot.py:4541`) — killing silently discards user documents |
| `cross_agent_insight` | ❌ **KEEP** | grader-extracted real text, importance 7, negligible volume; the 2026-07-25 prune deliberately KEPT it |
| `conversation_summary` | ⚠️ moot | lives in `conversations` (which is 0 rows); killing saves nothing |

Empirical check: post-prune semantic = **1,508 rows total** across memory/user_preference/
cross_agent_insight/skill. **The ~50 % volume win comes from `user_feedback` + episodic gating
ALONE.** The other four buy nothing and cost features.

Also: the predecessor handoff names **`analysis_key_finding`** as a semantic polluter — I never
located its writer. **Open TODO:** grep for it and gate it.

Inverse finding: the **`errors` collection has a reader** (`recall_error_patterns`,
`episodic.py:159`) **but no writer anywhere.** The "Known Issues" RAG section is structurally
starved. Worth fixing separately.

---

## 5. 🔴 THE CRITICAL PART — row caps do NOT bound disk size (empirically proven)

My P4 (per-collection row caps + `col.delete` + write-time eviction) **was REJECTED**. A reviewer
ran real experiments against chroma 1.5.5 in the project venv:

1. **sqlite DELETE does not shrink the file.** 3000 rows in, 2500 deleted by `where` → file size
   unchanged; pages go to the freelist. Only `VACUUM` reclaimed (2.2 MB → 397 KB, −82 %).
   **Corroborated by the prod forensics:** 4.39 GB `chroma.sqlite3` with a freelist of **0.004 GB**
   ⇒ it had never been VACUUMed.
2. **HNSW segments never shrink.** 4000 vectors in, 3800 deleted → `data_level0.bin` stayed
   6.5 MB through delete AND reopen. `data_level0.bin` is pre-allocated to max capacity; deletes
   are tombstones. In prod that was **2 × 2.8 GB = 5.6 GB of the 9.6 GB — unreclaimable by any delete.**
3. **chroma 1.5.5 exposes no compaction/vacuum API** (only `fork`, which copies).

⇒ **`vector_maint_prune` as I specced it would report "deleted 800 K rows" and leave the directory
at 9.6 GB. Next cold boot: same crash loop.** The one-time prune only worked because it was a
**rebuild-swap** (new store → copy survivors → atomic dir swap).

### Corrected P4
- **Invariant = total on-disk BYTES**, not row count. Measure via `os.walk(_DB_DIR)`. Promote the
  size-check from "Telegram alert" to the **enforcement gate**.
- Reclaim sequence: prune → `wal_checkpoint(TRUNCATE)` (exists, `vector_store.py:524`) → **VACUUM**
  (no helper exists — must add) → **drop-and-recreate over-budget collections from survivors**
  (generalize the existing rescue path `vector_store.py:407 _rescue_and_rebuild_sync` into a
  `compact_collection(name, keep_ids)`) — the only way to shrink HNSW.
- **Enumerate via `_client.list_collections()`**, NOT the static `COLLECTIONS` list. Mission
  collections `mission_{id}__*` live in a separate dict `_namespaced_collections`
  (`vector_store.py:122-176`) — **outside every cap**, one set per mission, unbounded. A
  "sum of per-collection caps" invariant is meaningless when the *number* of collections is unbounded.
- **Boot-time bytes gate**: `os.walk` needs no `is_ready()`, so check size and compact **before**
  declaring the store ready. This is what fixes the chicken-and-egg (`decay.py:116` returns `{}`
  when not ready ⇒ during the crash loop it never pruned because it never loaded).
- **Delete in bounded batches with `wal_checkpoint(TRUNCATE)` between batches**, all in
  `asyncio.to_thread`, wall-time-capped per tick. A single 880 K-row delete bloats the WAL and
  risks the same >120 s heartbeat wedge (cf. `vector_store.py:904`).
- **DROP write-time eviction entirely.** `col.count()` is not O(1) (reads INDEX+WAL), it taxes the
  hot write path near the pump, and it reclaims no bytes. Replace with an amortized every-Nth-write
  *enqueue* of the prune task. The write path stays a pure upsert.
- `delete(where={"stored_at": {"$lt": cutoff}})` **is** valid in 1.5.5 (verified, even accepts
  `limit`). Keep `compute_relevance` (`decay.py:54-98`) — the math is sound. Discard the
  load-all-into-RAM scan (`decay.py:143`) and the `PRUNE_FRACTION` bleed.
- **Concurrency gap:** drop-and-recreate races with live queries holding cached handles
  (`_collections`, `_namespaced_collections`). Take `_namespace_lock`, swap atomically, evict stale
  handles (pattern: `purge_mission_chroma_collections`, `vector_store.py:322-325`). Do not run
  `snapshot_chroma` (daily `copytree`) concurrently with a compaction — serialize the vector_maint executors.

Mechanical registration (3 files): executor in `mr_roboto/executors/`, `if action ==` branch in
`mr_roboto/__init__.py:803+ _run_dispatch`, verb in `reversibility.py VERB_REVERSIBILITY`
(classify **"full"** — local, snapshot-recoverable ⇒ no critic gate), plus a `cron_seed.py` row.
Contract: `async def run(task: dict) -> dict`.

---

## 6. 🔶 OPEN FORK — P3 forgetting signal (needs the unfinished research)

**Founder was asked to choose and correctly refused:** *"I can't reply this out of blue. This has
to be a deeply researched and analyzed approach."* **Do not ask him again without evidence.**

**The tension.** Founder's standing principle: *utility = real retrieval-success rate, not grader
opinion.* Adversarial reviewer argues that is **unobservable here**:
- `rag.py:560-606` injects ~5-15 docs/task. On PASS, crediting all of them **rewards co-injected
  noise** — the junk-protected-by-`access_count` bug reborn under a nicer name.
- Task PASS/FAIL is dominated by model capability, not memory quality ⇒ per-doc correlation ≈ 0.
- `skills.py` escapes this by injecting **few** and tracking a real **denominator**
  (`record_injection` `:534` numerator `record_injection_success` `:560`, rate `:138`).
- **The plumbing does not exist:** `top_hits` is only *logged* (`rag.py:594-633`), never threaded
  into task context. Grep for `injected_doc_ids` = zero hits.
- Reviewer further suspects the **skills loop itself may be broken in prod**: `metrics_push.py:174`
  reads `task_ctx["injected_skills"]`, but `coulson/context.py:1526` returns injected **tools**,
  not skill names. **If true, the "proven precedent" I based the design on is illusory.**
  ← THIS IS THE SINGLE HIGHEST-VALUE THING TO VERIFY.

**Options:** (A) recency + hard TTL only (deterministic, honest; a 30-day TTL on the firehose types
alone would have prevented the whole incident); (B) build real per-doc attribution (thread
`injected_doc_ids`, add a denominator, credit only top-1/2 not all); (C) A now, B deferred.

---

## 7. ⛔ RESEARCH THAT DID NOT FINISH — re-run these first

Two background agents were launched and **both died on the monthly spend limit with ZERO output**
(0 tokens, no report). Their findings are **not** in this doc. Re-run them:

**Agent 1 — empirical, in-repo.** Measure, don't speculate. Read-only.
Forensics snapshot: `C:\Users\sakir\kutai_chroma_forensics\chroma_snapshot_20260724-163413\chroma.sqlite3`
(`sqlite3.connect('file:<path>?mode=ro&immutable=1', uri=True)`); also `data/chroma.preprune.20260725-1844/`.
1. Real distribution of docs injected per task (parse `logs/*.jsonl` for the `memory.rag` logger
   `top_hits`) — decides whether credit assignment is diluted or narrow.
2. **Does the skills injection loop actually work in prod?** Read the full path and query the DB:
   how many skills have `injection_count>0`, distribution of success rates, is the denominator
   incrementing? Decisive for §6.
3. `access_count` distribution in the snapshot: what fraction of 1.77 M embeddings were EVER
   retrieved? If ≈0, pure recency/TTL is provably sufficient and attribution is moot.
4. What tables could support a durable doc-id→outcome join, and its row cost (remember EAV 12.5×).
5. Current cloud provider/model/quota reality (KDV + fatih_hoca catalog) — what a batched LLM job
   would really cost.

**Agent 2 — external prior art** (WebSearch + fetch, cite URLs).
RAG credit attribution (leave-one-out ablation, citation-based, LLM-judge — what's production-real
vs research-only, and cost); agent-memory forgetting in MemGPT/Letta, Mem0, Zep/Graphiti
(bi-temporal fact expiry), LangMem, A-MEM, Generative Agents (recency×importance×relevance),
MemoryBank (Ebbinghaus), HippoRAG — retention signal, deterministic vs LLM, consolidate vs delete;
critiques of access-frequency signals protecting stale docs; consolidation dedup thresholds actually
used and evidence on information loss; temporal validity/contradiction expiry; published
ChromaDB/sqlite+HNSW disk-reclamation guidance. End with a recommendation on Option A vs B + the
strongest counter-argument.

(Prior agent ids `a2af3f60a869dcc4a` / `a1a59b7face0dd49e` are dead — spawn fresh, don't resume.)

---

## 8. Recommended order of work

1. Re-run the two research agents (§7). Verify the skills-loop question first.
2. Present P3 to the founder **with evidence**; get the call.
3. Implement in this order (each independently valuable, TDD):
   **Phase 1** P1a+P1c+P1d — stop the two whales at the choke point. Kills the bloat class immediately.
   **Phase 2** P4 corrected — bytes-budget + VACUUM + rebuild-compaction + boot gate + `list_collections()`.
   **Phase 3** P5a — `where`/type filters on semantic reads.
   **Phase 4** P1b/P2 merge (≥0.93 + keyword guard, occurrence-bump).
   **Phase 5** P3 per §6 outcome.
4. Verify against a real cold boot before pushing. Commit at each milestone.

## 9. Do NOT repeat these mistakes
- Do not assume row caps bound disk size. They do not. (§5)
- Do not kill a type because it has no `where`-filtered reader. Unfiltered semantic reads surface
  everything. (§4)
- Do not force all readers through one 0.72 threshold. (§3 P5b)
- Do not add N metadata keys for a score breakdown — EAV multiplies by embedding count. (§1 Q4)
- Do not put `count()`+evict in the write path. (§5)
- Do not trust docs saying "cloud not connected" or "reranker disabled" — both stale. (§2, §3)
