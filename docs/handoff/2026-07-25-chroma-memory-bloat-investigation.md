# Handoff — Vector memory grew unbounded to 9.6 GB / 1.77 M embeddings (WHY + prevent)

**Date:** 2026-07-25
**For:** a **separate KutAI session** dedicated to memory-write discipline + retention governance.
**Status:** INVESTIGATION ONLY. The boot crash-loop this bloat caused, and a one-time
prune, are being handled in the originating session (see "Sibling work" below). Do **not**
re-run a destructive prune here — start read-only on the preserved snapshot.

---

## Why this exists
The 2026-07-24 R3/R4 restart forced KutAI's first **cold boot in months**. Cold-loading the
ChromaDB store took longer than the heartbeat-stale window → Yaşar Usta kill-restart loop
(full analysis: `../yasar_usta/docs/handoff/2026-07-25-kutai-boot-heartbeat-starvation.md`).
Root of *that* boot failure is the heartbeat gap + a 9.6 GB store on the boot critical path.
Root of *this* handoff is the deeper question: **why did the vector store reach 9.6 GB at all,
and how do we make KutAI write smarter memories with hard size governance so it never happens
again.**

## What was found (evidence)
Live store `data/chroma` = **9.6 GB**:
- `chroma.sqlite3` (metadata) = **4.39 GB** — freelist ~0.004 GB, so it is **live data, not
  fragmentation/queue-bloat** (`embeddings_queue` = only 875 rows).
- `embeddings` = **1,773,084** rows. `embedding_metadata` = **22,155,661** rows
  (~12.5 metadata rows per embedding — investigate why so many).
- Two HNSW vector segments at **2.8 GB each**.

Per-collection embedding counts:

| collection | embeddings | note |
|---|---:|---|
| **semantic** | 886,876 | facts/prefs — but polluted with auto-captured `analysis_key_finding` + task-`feedback` |
| **episodic** | 885,451 | one row per task result (`store_task_result`) — execution history |
| web_knowledge | 648 | negligible |
| shopping / support_docs_* / mission_ideas | ~110 total | negligible |
| codebase / errors / conversations | 0 | unused |

**The two whales are `semantic` + `episodic` — together 1.772 M of the 1.773 M total.**

## Smoking gun — retention exists but was never enforced
`src/memory/decay.py` already implements caps + relevance pruning:
```
COLLECTION_CAPS = {episodic: 10_000, semantic: 10_000, codebase: 15_000, errors: 5_000, conversations: 5_000}
PRUNE_FRACTION  = 0.2   # removes only 20% of the *excess* per cycle
RELEVANCE_THRESHOLD = 0.05
PROTECTED_TYPES = {"user_preference"}
```
Yet episodic/semantic sit at **~886 K = 88× the 10 K cap.** So the mechanism is present but
effectively dead. Prime suspects (verify each):
1. **`run_decay_cycle()` is never scheduled** (no cron / mechanical maintenance task wiring it),
   or is scheduled but silently failing.
2. **Chicken-and-egg:** decay gates on `vector_store.is_ready()`. During the crash loop the store
   never loaded → `is_ready()` never true → decay never ran → store stayed huge → couldn't load.
   Even in healthy operation, if decay only runs after a full warm load, a store this size may
   never get a window.
3. **`PRUNE_FRACTION=0.2` is far too weak** — from 886 K toward 10 K it would take dozens of
   cycles even if it ran daily. It can never catch a fast writer.
4. **Write rate outruns prune rate** — every task result is written to episodic
   (`store_task_result`), and semantic is fed auto-captured analysis/feedback with no
   write-time dedup or relevance gate.

## Preserved snapshot for forensics (read-only)
A full pre-fix copy was moved **out of Dropbox** (same-volume rename, zero copy cost) to:
```
C:\Users\sakir\kutai_chroma_forensics\chroma_snapshot_20260724-163413\   (9.52 GB, incl chroma.sqlite3 4.18 GB)
```
This is the 2026-07-24 16:34 daily snapshot — representative of the accumulated data (same
1.77 M scale as live). Inspect it **read-only** so you never touch the live store:
```python
import sqlite3
uri = r'file:C:\Users\sakir\kutai_chroma_forensics\chroma_snapshot_20260724-163413\chroma.sqlite3?mode=ro&immutable=1'
con = sqlite3.connect(uri, uri=True)
# e.g. distribution of metadata "type" keys in semantic, age of oldest rows, dupes, etc.
```
(The originating session's one-time prune will also move the exact pre-prune **live** store aside
— check `data/` for a `chroma.preprune.*` dir; add it to forensics if you want the 18:44 state.)

## Investigation questions to answer
1. **Why is `run_decay_cycle` not holding the caps?** Trace where/if it is scheduled (orchestrator
   cron, `mr_roboto` mechanical maintenance, `beckman_cron`). Make it actually run + prove it prunes.
2. **Episodic write policy** (`src/memory/episodic.py::store_task_result`): should we embed **every**
   task result? Candidates: only store failures + novel successes; dedup near-identical outcomes;
   TTL episodic hard (e.g. 30 days) regardless of relevance.
3. **Semantic pollution** (`src/memory/preferences.py`, `ingest.py`, `db_hooks.py`, and whatever
   writes `analysis_key_finding`/`feedback-*`): add a **write-time** relevance/dedup gate so
   low-value auto-captures never enter, instead of relying on after-the-fact decay.
4. **22 M metadata rows** — why ~12.5 per embedding? Check `embedding_metadata` /
   `embedding_metadata_array` fan-out; trim metadata we never query on.
5. **Size governance / observability:** emit a per-collection count + on-disk size metric; alert
   (Telegram) when any collection exceeds e.g. 2× cap or the store exceeds N GB — **long before**
   it reaches the point where a cold load can't finish inside the heartbeat window.
6. **Loadability invariant:** the store must never grow past what cold-loads within the boot budget.
   Decide the design guarantee (hard row caps enforced at write time? periodic compaction/VACUUM?
   sharding cold-rarely-queried collections to lazy-load-on-demand only?).

## Pointers (write paths + retention)
- `src/memory/decay.py` — caps, relevance, prune (exists, not enforced). Start here.
- `src/memory/episodic.py` — `store_task_result` (episodic writer; per-task).
- `src/memory/preferences.py`, `src/memory/ingest.py`, `src/memory/conversations.py`,
  `src/infra/db_hooks.py` — semantic/other writers.
- `src/memory/vector_store.py` — `embed_and_store`, `is_ready`, `get_all_counts`,
  `COLLECTIONS`; `run_decay_cycle` consumers.
- `src/memory/rag.py`, `src/memory/skills.py` — read + auto-capture paths.

## Sibling work (already in flight, do not redo)
- **Boot fix** (originating session): lazy/background chroma init + gapless heartbeat so a large
  cold load never blocks boot or starves the heartbeat again.
- **One-time prune** (originating session): shrink episodic/semantic now so the store cold-loads
  quickly. This handoff is the **durable prevention** that must follow, so we never rebuild the
  same 9.6 GB.
