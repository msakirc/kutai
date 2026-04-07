# Memory Subsystem Findings — Parallel Fix Targets

Date: 2026-04-07
Context: Audit during memory system redesign. Each section is an independent fix that can be tackled in a separate session.

---

## 1. Fake HyDE Implementation

**Location**: `src/memory/rag.py:247-274`

**Problem**: HyDE (Hypothetical Document Embeddings) is supposed to generate a hypothetical answer via LLM, then embed that for better semantic matching. Current implementation is a hardcoded template:
```python
f"The task '{title.strip()}' was completed successfully. {desc_part.strip()} 
The approach involved analyzing the requirements and implementing a solution..."
```
This is not HyDE — it's a generic string that pollutes embedding queries with boilerplate language. Worse than no HyDE because it biases retrieval toward generic "task completed successfully" language.

**Impact**: Degrades retrieval precision. Every query gets mixed with the same generic template embedding.

**Fix options**:
- A) Remove it entirely. Raw query embedding is better than fake HyDE.
- B) Implement real HyDE using the loaded model via OVERHEAD call. Only viable when cloud models are connected (too expensive for local 8K context).
- C) Replace with lightweight query reformulation heuristics (synonym expansion, Turkish/English bilingual query) that don't need LLM.

**Recommendation**: A now, C later if retrieval precision still insufficient.

---

## 2. Dead Conversation Summarization

**Location**: `src/memory/conversations.py:224-294`

**Problem**: Every 10 exchanges, a ~300-char summary is generated and embedded into the "semantic" collection. The summarization code works correctly. However, **nothing ever queries or retrieves these summaries**. No code path in RAG, context building, or any other module reads them back.

**Impact**: Wastes embedding compute and storage. Pollutes the semantic collection with summaries that compete with actual facts during retrieval.

**Fix options**:
- A) Wire summaries into RAG retrieval — query the semantic collection filtered by type="conversation_summary" when conversation context is needed.
- B) Remove the summarization entirely if conversation context isn't valuable.
- C) Move summaries to their own collection so they don't pollute semantic search, then wire into context building as an optional layer.

**Recommendation**: B for now (remove dead code), revisit A/C when context budget allows.

---

## 3. Keyword-Based Preference Detection (~50% precision)

**Location**: `src/memory/preferences.py:212-350`

**Problem**: Preference detection uses simple keyword matching:
```python
if "python" in combined:
    patterns.append({"preference_text": "Prefers Python", ...})
```
"Python" appearing in an error message = "user prefers Python". Confidence is calculated as `min(0.5 + count * 0.1, 0.9)` — two mentions = 0.7 confidence regardless of context.

**Impact**: ~50% of detected preferences are false positives. These get injected into every task's context via `_build_context()`, actively misleading small LLMs with wrong information.

**Fix options**:
- A) Remove preference detection entirely. Stop injecting noise.
- B) Replace keyword matching with classification piggybacked on the grading LLM call (add an optional "preference_signal" field to grading output).
- C) Require explicit user confirmation before storing preferences (e.g., "I noticed you prefer Python — is that right?").

**Recommendation**: A immediately (stop the noise), B when ready. The unified grading prompt already has progressive parsing infrastructure for optional fields. Adding `PREFERENCE: one-line user preference signal or "none"` to the grading prompt would replace keyword matching with LLM-based detection at zero additional LLM cost. See `docs/issues/memory-redesign-context.md` decision #3 for the piggybacking approach.

---

## 4. Reformatted "Insights" (Not Real Extraction)

**Location**: `src/memory/episodic.py:296-349`

**Problem**: Cross-agent insight extraction claims to summarize learnings from successful tasks. Actual implementation:
```python
f"Insight from {agent_type}: Task '{title}' — {result_preview}"
```
This is just string formatting, not extraction. Every completed task generates an "insight" — including trivial tasks like timezone lookups. These pile up in the semantic collection and degrade retrieval.

**Impact**: Noise in semantic collection. More "insights" = worse retrieval precision for actual knowledge.

**Fix options**:
- A) Remove entirely. Episodic memory already stores task results — the "insight" adds nothing.
- B) Gate by task complexity (only extract for iterations >= 3 AND tools_used >= 2). Still won't be real extraction but at least reduces noise.
- C) Replace with real extraction piggybacked on grading (add "reusable_insight" field to grading output for score >= 4).

**Recommendation**: A now (remove noise source), C when ready. Same piggybacking approach as #3: add `INSIGHT: one-line reusable learning or "none"` to the grading prompt. The progressive parsing already handles optional fields — empty = silently skipped. See `docs/issues/memory-redesign-context.md` decision #3.

---

## 5. Disabled Reranker

**Location**: `src/memory/rag.py:49` — `RERANKER_ENABLED = False`

**Problem**: Cross-encoder reranker is fully implemented but permanently disabled. Uses `ms-marco-MiniLM-L-6-v2`. Would significantly improve retrieval precision by reranking the initial vector search results.

**Impact**: None currently (disabled). But enabling it would improve the quality of what gets injected.

**Fix options**:
- A) Enable it. Test latency impact. The cross-encoder runs on CPU, so no VRAM cost.
- B) Enable selectively — only for tasks where RAG actually fires and returns 3+ results worth reranking.
- C) Leave disabled until context diet is in place (no point reranking if we're still injecting everything).

**Recommendation**: B after context diet ships. Reranking 5 results on CPU adds ~200ms, worth it for precision on 8K context.

---

## 6. Two Disconnected Grading Systems

**Location**: 
- `src/core/router.py:1498-1579` — old rich grader (1-5 score + JSON with situation/strategy/tools)
- `src/core/grading.py:19-139` — new binary grader (YES/NO only)

**Problem**: The binary grader was introduced because small LLMs couldn't reliably produce JSON. But it lost skill extraction as collateral — deferred grades (majority of tasks) never produce `grader_data`, so skill extraction falls back to mechanical `f"Used {tools} in {iterations} iterations"`.

**Impact**: ~60%+ of tasks produce low-quality skill descriptions. The skill library fills with mechanical entries that don't help future tasks.

**Fix**: This is the core of the memory redesign — unified grading with progressive extraction. Documented in the main design spec, not here.

---

## 7. Loose Retrieval Thresholds

**Locations**:
- `src/memory/rag.py` — RAG_MIN_RELEVANCE = 0.5
- `src/memory/skills.py:30` — skill match threshold = 0.6

**Problem**: On 8K context with small LLMs, injecting "vaguely related" content is worse than injecting nothing. 0.5 cosine similarity means "shares some semantic space" — a Python web framework query matches JavaScript frontend libraries.

**Impact**: Context filled with marginally relevant content that small LLMs can't distinguish from noise.

**Fix**: Raise thresholds as part of context diet redesign. Suggested: RAG 0.5→0.72, skills 0.6→0.75. Needs empirical tuning.

---

## 8. Uncapped Context Layers in _build_context()

**Location**: `src/agents/base.py:347-577`

**Problem**: 13 context layers, almost all unconditionally injected. No total budget enforcement — each layer independently decides its own size. Dependency results have no per-dependency cap. Skills don't respect remaining budget after RAG.

Layers that always inject (no gating):
- Ambient context (~400 tokens)
- Project profile (unbounded)
- Blackboard (unbounded if mission_id exists)  
- RAG (2K-12K tokens)
- User preferences (unbounded)
- Project memory (up to 3K tokens)

**Impact**: 15K+ tokens of context on tasks that need 200 tokens. On 8K context models, this literally doesn't fit.

**Fix**: Core of the memory redesign — task-type gating + hard total budget. Documented in the main design spec.

---

## Priority Matrix

| Issue | Severity | Independence | Suggested Order |
|-------|----------|-------------|-----------------|
| #8 Uncapped context layers | Critical | Part of main design | Main design |
| #6 Two grading systems | Critical | Part of main design | Main design |
| #7 Loose thresholds | High | Part of main design | Main design |
| #3 Keyword preferences | High | Independent | Parallel session |
| #4 Fake insights | Medium | Independent | Parallel session |
| #1 Fake HyDE | Medium | Independent | Parallel session |
| #2 Dead summaries | Low | Independent | Parallel session |
| #5 Disabled reranker | Low | Depends on #8 | After main design |

Issues #6, #7, #8 are the main design. Issues #1-4 are independent cleanup that can happen in parallel. Issue #5 is best enabled after the context diet is in place.

---

## 9. No Temporal Validity on Facts

**Location**: `src/memory/vector_store.py`, `src/infra/db.py` (memory table)

**Problem**: Facts stored in memory have no expiry or validity window. "User lives in Istanbul" stored in January stays forever, even if wrong. Importance-protected facts (user_preference, error_recovery with importance 8-9) never decay at all. Decay helps for low-importance facts (30-day half-life) but can't distinguish "still true" from "was true once".

**Impact**: Stale facts retrieved with high confidence because they were important when stored. On tight context budgets, stale facts displace current relevant facts.

**Fix**: Add `valid_from`/`valid_to` fields to memory entries. New facts about the same entity/topic should invalidate (set `valid_to`) on older conflicting facts. Query-time filtering: only retrieve facts where `valid_to IS NULL OR valid_to > now()`.

**Complexity**: Medium. Requires contradiction detection heuristic at store time (when storing "user prefers dark mode", find and expire "user prefers light mode"). Vector similarity on the same entity + opposing content is one approach.

---

## 10. No Bilingual Query Expansion

**Location**: `src/memory/rag.py` (query pipeline)

**Problem**: Turkish users type Turkish queries, but much stored knowledge is in English (web results, code patterns, error messages). multilingual-e5-base handles some cross-language similarity but isn't perfect — especially for domain-specific terms. No query expansion bridges this gap.

**Impact**: Retrieval misses cross-language matches. A Turkish query about "fiyat karşılaştırma" may not find English-stored results about "price comparison".

**Fix**: Lightweight bilingual query expansion heuristic — a small dictionary of common term pairs (Turkish ↔ English) for shopping, tech, and common domains. No LLM needed. Expand query with translated terms, embed both, take best results. Could also use the existing query decomposition infrastructure.

**Complexity**: Low. Dictionary maintenance is the ongoing cost.

---

## 11. No Precomputed Essentials Cache

**Location**: `src/agents/base.py:_build_context()`

**Problem**: Every task runs the full retrieval pipeline even for trivial queries. A timezone lookup triggers RAG, skill search, preference lookup, memory recall — all returning either nothing useful or noise. MemPalace's L0/L1 concept: precompute a tiny (~200 token) user/project brief, cache it, inject it as a baseline. Simple tasks get this and nothing else.

**Impact**: Unnecessary compute on every task. More importantly, simple tasks that should be fast get delayed by retrieval overhead.

**Fix**: Daily-refreshed identity cache with: user language, timezone, communication style, active project context, key preferences (only confirmed ones). Stored as a plain text block. Injected as the first context layer for all tasks. For task types with empty context policies (reviewer, router), this would be the only context beyond the task itself.

**Complexity**: Low. Trigger: startup + daily cron. Storage: single file or DB row.

---

## 12. Arbitrary RAG/Decay Weights (Unvalidated)

**Location**: 
- `src/memory/rag.py:174-206` — ranking: `relevance * 0.4 + recency * 0.25 + importance * 0.2 + access_freq * 0.15`
- `src/memory/decay.py:54-98` — decay: `access * 0.3 + recency * 0.4 + importance * 0.3`

**Problem**: These weights are hardcoded without empirical justification. They "feel right" but have never been validated against actual retrieval quality. Different task types likely need different weight profiles (code tasks should weight recency differently than factual queries).

**Impact**: Unknown — could be fine, could be significantly suboptimal. Without measurement, we're guessing.

**Fix**: After the context gating system ships and thresholds are tuned, run retrieval quality analysis: for completed tasks with grade=PASS, check whether the injected context was actually referenced in the output. Use this signal to tune weights. Could also make weights per-task-type using the same policy map infrastructure.

**Complexity**: Medium. Requires instrumentation + analysis tooling. Not urgent — current weights aren't broken, just unvalidated.

---

## 13. Single-Pass Retrieval (No Reranking Active)

**Location**: `src/memory/rag.py:49` — `RERANKER_ENABLED = False`

**Problem**: Retrieval is embed → cosine search → take top-k. The cross-encoder reranker exists but is disabled. Cross-encoders are significantly more accurate than bi-encoder cosine similarity for ranking — they see both query and document together rather than comparing independent embeddings.

**Impact**: Top-k results are ordered by embedding similarity, not by actual relevance to the query. On tight context budgets (8K), the difference between rank 1 and rank 3 matters.

**Fix**: Enable reranker after context gating ships. The reranker uses `ms-marco-MiniLM-L-6-v2` on CPU — no VRAM cost. Only rerank when RAG returns 3+ results (not worth it for 1-2). Expected latency: ~200ms for 5 results.

**Depends on**: Issue #8 (context gating) — no point reranking if we're still injecting everything.

**Complexity**: Low. Code exists, just needs enabling + selective triggering.

---

## 14. Stale References in skill-system.md

**Location**: `docs/skill-system.md` lines 125, 127, 186

**Problem**: After the grading unification, three references still describe the old system:
- Line 125: "Reads `grader_data`, calls `add_skill`" — skill extraction now lives in `grading.py:apply_grade_result()`, not the orchestrator
- Line 127: "`grade_response` returns `(bool, dict)`" — this function no longer exists; `grade_task()` returns `GradeResult` dataclass
- Line 186: Same stale reference to `grade_response` returning a tuple

**Impact**: Misleading documentation for anyone modifying the grading or skill system.

**Fix**: Update these lines to describe the current flow: `grade_task()` returns `GradeResult`, `apply_grade_result()` handles skill extraction using `verdict.situation`/`verdict.strategy`/`verdict.tools`.

**Complexity**: Trivial. Text-only change.

---

## 15. Missing Test Coverage for apply_grade_result

**Location**: `tests/test_grading.py`

**Problem**: The test file covers parsing well (18 tests) but does not test:
- `apply_grade_result()` PASS path (skill extraction, state transition, injection success tracking)
- `apply_grade_result()` FAIL path (retry logic, DLQ quarantine)
- `grade_task()` with empty/trivial result (auto-fail at <10 chars)
- `drain_ungraded_tasks()` flow

These are the most critical business logic paths in the grading module. Currently only tested indirectly via integration.

**Impact**: Regressions in grading behavior (e.g., broken skill extraction, incorrect retry logic) would not be caught by unit tests.

**Fix**: Add tests with mocked DB, dispatcher, and skills module. The PASS path needs mocks for `transition_task`, `record_model_call`, `add_skill`, `record_injection_success`. The FAIL path needs mocks for `compute_retry_timing`, `update_exclusions_on_failure`, `quarantine_task`.

**Complexity**: Medium. Heavy mocking required but the test structure is straightforward.

---

## 16. _parse_text_field Regex is Single-Line Only

**Location**: `src/core/grading.py:_parse_text_field()`

**Problem**: The regex `KEY\s*:\s*(.+)` only captures content on the same line as the key. If a small LLM wraps the TOOLS line across multiple lines (e.g., `TOOLS: smart_search,\n  web_search, api_call`), only `smart_search,` would be captured.

**Impact**: Low currently — the grading prompt asks for single-line answers and most small LLMs comply. But could cause silent data loss for models that wrap long tool lists.

**Fix**: Either make the regex multiline-aware (capture until next KEY or end of string) or document that the prompt design relies on single-line responses. A multiline-aware version: `rf"{key}\s*:\s*(.+?)(?=\n[A-Z]+\s*:|$)"` with `re.DOTALL`.

**Complexity**: Low, but needs careful testing to avoid matching across unrelated fields.

---

## Updated Priority Matrix

**Resolved by main design (2026-04-07):** #6 (two grading systems), #7 (loose thresholds), #8 (uncapped context layers)


| Issue | Severity | Independence | Status |
|-------|----------|-------------|--------|
| #8 Uncapped context layers | Critical | Main design | ✅ Resolved 2026-04-07 |
| #6 Two grading systems | Critical | Main design | ✅ Resolved 2026-04-07 |
| #7 Loose thresholds | High | Main design | ✅ Resolved 2026-04-07 |
| #3 Keyword preferences | High | Independent | Parallel session |
| #4 Fake insights | Medium | Independent | Parallel session |
| #1 Fake HyDE | Medium | Independent | Parallel session |
| #10 Bilingual query expansion | Medium | Independent | Parallel session |
| #9 Temporal validity | Medium | Independent | Parallel session |
| #15 Missing grading test coverage | Medium | Independent | Parallel session |
| #11 Precomputed essentials | Medium | Independent | Parallel session |
| #14 Stale skill-system.md refs | Low | Independent | Parallel session (trivial) |
| #2 Dead summaries | Low | Independent | Parallel session |
| #16 Single-line text field parsing | Low | Independent | Parallel session |
| #13 Reranker enablement | Low | Depends on #8 | Ready now (#8 resolved) |
| #12 Weight validation | Low | Depends on #8 | Ready now (#8 resolved) |
| #5 Disabled reranker | Low | Same as #13 | Ready now (#8 resolved) |
