# KutAI Intelligence & RAG Architecture Plan

**Date:** 2026-03-26
**Scope:** Comprehensive analysis of all memory, context, retrieval, and knowledge systems, plus a plan to maximize intelligence through vector search, RAG, embeddings, and better context injection.

---

## Part 1: What Exists Today (Full Inventory)

### 1.1 Memory System (`src/memory/`)

| Module | Purpose | Status |
|--------|---------|--------|
| `vector_store.py` | ChromaDB-backed store with 5 collections (episodic, semantic, codebase, errors, conversations) | **Implemented, core plumbing works** |
| `embeddings.py` | Shared embedding utility: Ollama (nomic-embed-text/all-minilm/mxbai-embed-large) -> sentence-transformers fallback | **Implemented**, in-memory cache (5000 entries), truncates at 2000 chars |
| `rag.py` | RAG pipeline: queries episodic + semantic + errors, ranks by relevance*recency*importance, deduplicates, formats within 2000-token budget | **Implemented**, injected into every agent via `BaseAgent._build_context()` |
| `episodic.py` | Stores task outcomes (success/failure) and error recovery patterns in vector store | **Implemented** |
| `conversations.py` | Embeds user-AI exchanges, follow-up detection via similarity | **Implemented** |
| `decay.py` | Memory lifecycle: relevance scoring, pruning at 80% cap, protected types (user_preference, error_recovery) | **Implemented** |
| `ingest.py` | Document ingestion (URL, PDF, DOCX, text) -> chunk (500-token windows, 50-token overlap) -> embed -> store in semantic collection | **Implemented** |
| `preferences.py` | User preference learning from feedback (accepted/modified/rejected), keyword-based pattern detection | **Implemented but primitive** |
| `prompt_versions.py` | Versioned agent prompts in DB, auto-promotion after 10+ tasks if quality improves | **Implemented** |
| `skills.py` | Reusable task approach library in DB, matched via regex trigger_pattern | **Implemented but regex-only matching** |
| `feedback.py` | Feedback loop: accept/partial/reject -> score -> feeds model stats + prompt quality | **Implemented** |
| `self_improvement.py` | Weekly analysis: failure rates, negative feedback, model degradation, cost anomalies | **Implemented** |

### 1.2 Context Assembly (`src/context/`)

| Module | Purpose | Status |
|--------|---------|--------|
| `assembler.py` | Queries codebase vector collection for relevant symbols, reads file sections, resolves imports, finds related tests, includes recent git changes. Token-budgeted (4000 tokens). Also builds ambient context (time, load mode, active missions, blackboard decisions) | **Implemented** |
| `repo_map.py` | Generates structural map: dependency graph, entry points, test mapping, config files, directory purposes | **Implemented** |
| `onboarding.py` | Project onboarding: detect language/framework, run indexing, build code embeddings, generate repo map, detect conventions | **Implemented** |

### 1.3 Code Intelligence (`src/parsing/`, `src/tools/`)

| Module | Purpose | Status |
|--------|---------|--------|
| `tree_sitter_parser.py` | Multi-language parsing (Python, JS, TS, Go, Rust, Java, C, C++) via tree-sitter -> Python AST -> regex fallback. Extracts functions, classes, imports, exports with signatures, docstrings, body previews | **Implemented, solid** |
| `code_embeddings.py` | Embeds function/class signatures + docstrings + body previews into codebase vector collection. Incremental re-indexing via file hash. search_code() for semantic code search | **Implemented** |
| `codebase_index.py` | Structural in-memory index: function/class/import lookup by name. Supplements vector search with exact-name queries | **Implemented** |

### 1.4 Knowledge Stores (`src/infra/db.py`)

**Main SQLite DB tables:**
- `missions` — high-level goals with context, workflow, repo_path, language, framework
- `tasks` — individual task execution with status, result, error, context JSON, retry_count
- `conversations` — LLM conversation logs per task (role, content, model_used, cost)
- `memory` — key-value store with category, mission-scoped. **Simple exact-key lookup, no search**
- `model_stats` — per-model performance tracking (grade, cost, latency, success rate)
- `cost_budgets` — budget tracking per scope
- `blackboards` — per-mission structured state (architecture, files, decisions, open_issues, constraints)
- `prompt_versions` — versioned agent prompts with quality scores
- `skills` — reusable task approaches
- `task_feedback` — explicit feedback with scoring
- `todo_items` — user todo tracking

### 1.5 Shopping Intelligence (`src/shopping/`)

**Separate DB systems:**
- `shopping_cache.db` — product cache (TTL-based), review cache, price_history, search_cache
- `shopping_memory.db` — user_profiles (dietary restrictions, location), owned_items, preferences (key-value), behaviors, price_watches with history

**Intelligence modules (all rule-based / LLM-prompt-based, no embeddings):**
- `product_matcher.py` — EAN/UPC > MPN > fuzzy name > spec fingerprint matching
- `query_analyzer.py`, `search_planner.py` — query understanding
- `review_synthesizer.py` — review analysis
- `alternatives.py`, `combo_builder.py`, `value_scorer.py` — recommendation logic
- 14+ special intelligence modules (fraud, fake discount, warranty, TCO, seasonal, etc.)

### 1.6 Collaboration (`src/collaboration/`)

- `blackboard.py` — per-mission shared state in DB with in-memory cache. Agents read/write structured data (architecture, files, decisions, open_issues, constraints). Formatted for prompt injection.

### 1.7 Context Injection in Agent Prompts (`BaseAgent._build_context()`)

Current injection order:
1. Task description (title + description)
2. Workspace snapshot (if in context)
3. Prior tool results
4. Additional context (JSON dump of remaining context keys)
5. Dependency results from DB
6. Inline prior_steps
7. Recent conversation (for follow-up)
8. Ambient context (time, load mode, active missions, blackboard decisions)
9. Project profile (language, framework, conventions)
10. Blackboard data (architecture, decisions, files)
11. Relevant skills from library (regex-matched)
12. **RAG context** (episodic + semantic + errors, ranked, 2000 tokens)
13. User preferences (from vector store)
14. Key-value memories from `memory` table

---

## Part 2: Gaps and Weaknesses

### 2.1 Embedding System Gaps

| Gap | Impact | Severity |
|-----|--------|----------|
| **No Turkish-aware embedding model** | Ollama models (nomic-embed-text, all-minilm) are English-centric. Turkish product names, conversations, and queries get poor vector representations | **Critical** |
| **Embedding fallback is broken** | `_get_st_embedding()` does `import SentenceTransformer` instead of `from sentence_transformers import SentenceTransformer` — the fallback never works | **High** |
| **No batch embedding efficiency** | `get_embeddings()` calls `get_embedding()` serially in a loop — no batching at the Ollama or sentence-transformers level | **Medium** |
| **Text truncated at 2000 chars** | Long documents, code files, and shopping reviews are brutally truncated before embedding, losing important tail content | **Medium** |
| **No embedding dimension awareness** | Different Ollama models produce different dimensions (nomic=768, all-minilm=384, mxbai=1024). Switching models mid-run corrupts similarity scores | **High** |
| **Cache evicts randomly** | LRU-ish eviction removes first 10% of dict keys — not actually LRU, just insertion-order based | **Low** |

### 2.2 Vector Store Gaps

| Gap | Impact | Severity |
|-----|--------|----------|
| **ChromaDB optional dependency** | If chromadb isn't installed, the entire vector store silently returns empty — all RAG, episodic memory, code search, conversation continuity degrades to nothing | **Critical** |
| **No shopping data in vector store** | Products, reviews, price patterns, user shopping preferences are all in separate SQLite DBs with no vector/semantic search | **High** |
| **No conversation summarization** | Full user+response text stored as-is. Old conversations waste space without distillation | **Medium** |
| **No cross-collection queries** | RAG queries each collection separately and merges — no unified embedding space, no cross-domain retrieval | **Medium** |
| **Collection cap limits are arbitrary** | 10K-15K caps were set without measuring actual token budgets or retrieval quality | **Low** |

### 2.3 RAG Pipeline Gaps

| Gap | Impact | Severity |
|-----|--------|----------|
| **2000-token RAG budget is too small** | With 128K+ context windows available, only 2000 tokens (~8K chars) of retrieved knowledge is injected. Massive waste of context window | **Critical** |
| **No query expansion/rewriting** | Single query text used for all collections — no HyDE, no query decomposition, no keyword extraction | **High** |
| **No relevance threshold filtering** | All top-K results are included regardless of distance — irrelevant results pollute context | **Medium** |
| **Deduplication is naive** | Jaccard word-set overlap misses semantic duplicates and over-filters similar-but-different content | **Medium** |
| **No reranking** | Results are scored by a simple linear combination (0.5*relevance + 0.3*recency + 0.2*importance) — no learned reranker, no cross-encoder | **Medium** |
| **Code context assembly doesn't use RAG scores** | `assembler.py` queries codebase collection but doesn't benefit from the ranking in `rag.py` | **Low** |

### 2.4 Knowledge & Memory Gaps

| Gap | Impact | Severity |
|-----|--------|----------|
| **`memory` table is keyword-only** | `recall_memory()` filters by exact category and mission_id only — no semantic search over stored memories | **High** |
| **Skills matched by regex only** | `find_relevant_skills()` does regex pattern matching on task text — misses semantically similar tasks with different wording | **High** |
| **No web search result caching in vector store** | Search results are used once and discarded — no knowledge accumulation from web research | **High** |
| **Preferences detected by keyword patterns** | `_extract_patterns()` uses hardcoded keyword matching ("python", "snake_case", etc.) — can't detect nuanced preferences | **Medium** |
| **No entity extraction from conversations** | Conversations are embedded as raw text — no extraction of entities, facts, or structured knowledge | **Medium** |

### 2.5 Shopping Intelligence Gaps

| Gap | Impact | Severity |
|-----|--------|----------|
| **No product embeddings** | Product matching uses EAN/MPN/fuzzy-name — no semantic similarity for "similar products" queries | **High** |
| **No review embeddings** | Reviews are synthesized per-request via LLM — no persistent review knowledge base | **High** |
| **No price pattern vectors** | Price history is tracked but not embedded — can't answer "products that tend to drop in March" | **Medium** |
| **Shopping memory is isolated** | `shopping_memory.db` has no connection to the main vector store — shopping insights can't inform coding tasks and vice versa | **Medium** |

---

## Part 3: Proposed Architecture for Maximum Intelligence

### 3.1 Embedding Model Selection

**Recommendation: `intfloat/multilingual-e5-base`** (278M params, 768 dimensions)

| Model | Dims | Size | Turkish | Speed (CPU) | Quality |
|-------|------|------|---------|-------------|---------|
| all-MiniLM-L6-v2 | 384 | 80MB | Poor | Fast (15ms) | Good English |
| multilingual-e5-base | 768 | 1.1GB | **Good** | Medium (40ms) | Excellent multilingual |
| multilingual-e5-small | 384 | 470MB | **Good** | Fast (20ms) | Good multilingual |
| nomic-embed-text (Ollama) | 768 | 274MB | Fair | Fast (GPU) | Good English |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 470MB | **Good** | Fast (20ms) | Good multilingual |

**Strategy:**
1. **Primary:** Run `multilingual-e5-small` via sentence-transformers on CPU (384 dims, fast, good Turkish)
2. **Upgrade path:** If GPU time is available (llama-server idle), use `multilingual-e5-base` via Ollama for higher quality
3. **Fix the import bug** in `embeddings.py` line 93: `import SentenceTransformer` -> `from sentence_transformers import SentenceTransformer`
4. **Lock embedding dimensions**: Store model name + dimension in a config. Refuse to mix models within a collection. Add a migration path for re-embedding if model changes.

**Why not Ollama-only:** Ollama competes with llama-server for GPU. sentence-transformers on CPU keeps embedding independent of LLM inference.

### 3.2 Vector Store Selection

**Recommendation: Keep ChromaDB** but make it a hard dependency and improve configuration.

| Option | Pros | Cons |
|--------|------|------|
| **ChromaDB (current)** | Already integrated, persistent, HNSW index, metadata filtering, Python-native | Extra process/memory, no sqlite integration |
| sqlite-vss | Single DB file, integrates with existing aiosqlite | Experimental, limited features, manual index management |
| FAISS | Fastest search, GPU support, battle-tested | No persistence API, no metadata filtering, manual save/load |
| LanceDB | Serverless, fast, native Python | New, less mature ecosystem |

**Actions:**
1. Make `chromadb` a required dependency (not optional)
2. Add collection for `shopping` data (products, reviews)
3. Add collection for `web_knowledge` (search results, extracted facts)
4. Configure HNSW parameters per collection (ef_construction, M) based on expected size
5. Add model-dimension tracking to prevent mixed-dimension corruption

### 3.3 RAG Pipeline Overhaul

#### 3.3.1 Query Processing

```
User query / Task description
    |
    v
[Query Analyzer]
    |-- Extract keywords (Turkish + English)
    |-- Identify entities (product names, function names, error signatures)
    |-- Generate query variants (HyDE: hypothesize ideal answer, embed that)
    |-- Language detection (TR/EN/mixed)
    |
    v
[Multi-Collection Query]
    |-- episodic (past tasks)        top_k=5
    |-- semantic (facts, prefs)      top_k=5
    |-- errors (failure patterns)    top_k=3
    |-- codebase (if code task)      top_k=8
    |-- conversations (recent)       top_k=3
    |-- shopping (if shopping task)  top_k=5
    |-- web_knowledge                top_k=3
    |
    v
[Reranker]
    |-- Filter by minimum distance threshold (e.g., cosine > 0.3)
    |-- Cross-encoder reranking (optional, CPU-friendly model)
    |-- Score: 0.4*relevance + 0.25*recency + 0.2*importance + 0.15*access_frequency
    |-- Deduplicate with semantic similarity (embedding distance, not word overlap)
    |
    v
[Token Budget Manager]
    |-- Dynamic budget based on task complexity:
    |      Simple task: 2000 tokens
    |      Code task:   6000 tokens (more code context)
    |      Research:    4000 tokens
    |      Shopping:    4000 tokens
    |-- Priority allocation: errors > episodic > code > semantic > conversations
    |-- Truncate individual items, not entire sections
    |
    v
[Formatted Context Block]
```

#### 3.3.2 Dynamic Token Budget

Current: Fixed 2000 tokens. Proposed: Dynamic based on model context window and task type.

```python
def _compute_rag_budget(task: dict, model_context_window: int) -> int:
    """Use up to 15% of available context window for RAG."""
    # Reserve space for: system prompt (~2000), tools (~3000),
    # task description (~1000), conversation history (~4000)
    reserved = 10000
    available = model_context_window - reserved
    # Cap at 15% of available, min 2000, max 12000
    budget = int(available * 0.15)
    return max(2000, min(12000, budget))
```

### 3.4 Semantic Memory (Replacing Keyword Search)

**Goal:** The `memory` table (`store_memory`/`recall_memory`) currently uses exact key lookup. This should be augmented with vector search.

**Plan:**
1. On every `store_memory(key, value, category, mission_id)` call, also embed the value into the `semantic` vector collection with metadata `{source: "memory_table", key, category, mission_id}`
2. Add `semantic_recall(query_text, category=None, mission_id=None, top_k=5)` that does vector search on the semantic collection filtered by metadata
3. Modify `recall_memory()` to return union of exact-key matches AND semantic matches
4. Skills library: embed skill descriptions into semantic collection with `{type: "skill"}` metadata for vector-based skill matching alongside regex

**Effort:** ~2 hours, low risk

### 3.5 Code RAG

**Goal:** Instead of dumping whole file sections, embed and retrieve at function/method granularity with smart context expansion.

**Current state:** `code_embeddings.py` already embeds functions/classes. `assembler.py` already queries the codebase collection and reads file sections. This is largely working.

**Improvements:**
1. **Embed docstrings separately** from code bodies — docstrings are natural language and embed well; code needs different chunking
2. **Include import context** in code embeddings — `from X import Y` tells the embedding model about relationships
3. **Embed test functions** linking them to the code they test via metadata `{tests_module: "X"}`
4. **Add call-graph edges** — when function A calls function B, store this relationship so retrieving A also surfaces B
5. **Triggered re-indexing** — currently only on project onboarding. Add file-watcher or post-tool-execution hook to `reindex_file()` after every `write_file`/`edit_file`/`patch_file` tool call

**Effort:** ~4 hours, medium risk

### 3.6 Conversation RAG

**Goal:** Make past conversations searchable and useful for long-term context.

**Current state:** `conversations.py` embeds exchanges into the `conversations` collection. Follow-up detection works via similarity.

**Improvements:**
1. **Conversation summarization** — After every 10 exchanges, generate a summary embedding. Store both the summary and individual exchanges. This reduces noise while preserving detail.
2. **Entity extraction** — Extract key entities from conversations (project names, file paths, preferences stated, decisions made) and store as structured facts in the `semantic` collection
3. **Temporal clustering** — Group conversations by session (time gaps > 30 min = new session). Embed session summaries for coarse retrieval, individual messages for fine retrieval.
4. **Cross-chat learning** — If user discusses the same topic across multiple sessions, link them via embedding similarity

**Effort:** ~3 hours, medium risk

### 3.7 Shopping RAG

**Goal:** Make shopping data searchable via embeddings to answer complex queries like "what headphones did you recommend last time?" or "find something similar to what I bought."

**Proposed collections/data to embed:**

| Data Source | Embed Text | Collection | Metadata |
|-------------|-----------|------------|----------|
| Products (from scrapers) | `name + brand + key_specs + price_range` | `shopping` | source, price, category, url, fetched_at |
| Reviews | Individual review text + rating | `shopping` | product_url, source, rating, language |
| User purchase history | `product_name + category + satisfaction_notes` | `shopping` | user_id, purchase_date, price_paid |
| Price patterns | `product_name: price dropped from X to Y on date` | `shopping` | product_url, direction, magnitude |
| Shopping sessions | `User searched for X, compared Y products, chose Z because...` | `shopping` | user_id, session_id, outcome |

**Integration with existing shopping system:**
1. After every product scrape, embed the product into the `shopping` collection
2. After every review synthesis, embed the synthesized review summary
3. After every completed shopping session, embed a session summary
4. `product_matcher.py` can use vector similarity as an additional matching signal alongside EAN/MPN/fuzzy

**Turkish-specific considerations:**
- Product names mix Turkish and English ("Samsung Galaxy S24 Ultra Akilli Telefon")
- Reviews are primarily in Turkish
- The multilingual-e5 model handles this well

**Effort:** ~6 hours, medium risk

### 3.8 Cross-Agent Knowledge Sharing

**Goal:** Agents learning from each other's work through shared embeddings.

**Current state:** Blackboard provides structured inter-agent communication per mission. Episodic memory stores task outcomes. But there's no way for the coder agent to benefit from what the researcher agent discovered across missions.

**Proposed improvements:**
1. **Shared insight extraction** — After each agent completes a task, extract 1-3 key insights as semantic facts:
   - Researcher: "The best approach for X is Y because Z"
   - Coder: "Function X in module Y has bug Z"
   - Shopping advisor: "User prefers brand X for category Y"
2. **Tag insights with agent_type** — Allow cross-agent retrieval with optional agent_type filtering
3. **Blackboard -> vector store bridge** — When a blackboard decision is written, also embed it in the semantic collection for cross-mission retrieval

**Effort:** ~3 hours, low risk

### 3.9 Web Knowledge Accumulation

**Goal:** Web search results should build a persistent knowledge base instead of being discarded after one use.

**Plan:**
1. After every web search, embed the top results (title + snippet) into a `web_knowledge` collection (add this to COLLECTIONS)
2. Before running a new web search, check if there's already relevant knowledge in the vector store
3. For Perplexica results (which are more comprehensive), embed the full extracted content
4. Add TTL-based decay — web knowledge should have a shorter half-life (7 days) than task results (30 days)

**Effort:** ~2 hours, low risk

### 3.10 Incremental Indexing

**Current state:** `code_embeddings.py` has file-hash-based incremental indexing, but only runs during project onboarding.

**Proposed triggers:**
1. **Post-tool hook** — After `write_file`, `edit_file`, `patch_file`, `apply_diff` tool calls, call `reindex_file()` on the affected file
2. **Startup re-scan** — On KutAI startup, quick hash scan of project files to detect changes made externally
3. **Periodic background** — Every 30 minutes, scan for changed files in registered projects
4. **Git-aware** — After `git_commit`, `git_branch`, `git_rollback`, re-index changed files from `git diff`

**For shopping data:**
1. Products and reviews have natural TTLs already (from cache.py)
2. Re-embed when cache is refreshed with new data
3. Price history embeddings update on each price observation

**Effort:** ~3 hours, low risk

---

## Part 4: Integration Plan

### Phase A: Foundation Fixes (Day 1, ~4 hours)

**Priority:** Fix what's broken before adding new capabilities.

1. **Fix embedding import bug** in `embeddings.py` line 93
   - `import SentenceTransformer` -> `from sentence_transformers import SentenceTransformer`

2. **Install multilingual-e5-small** via sentence-transformers
   - `pip install sentence-transformers` (if not installed)
   - Change default model from `all-MiniLM-L6-v2` to `intfloat/multilingual-e5-small`
   - Test with Turkish text: `"en uygun fiyatlı bulaşık makinesi"` should be similar to `"ucuz bulaşık makinesi"` and `"dishwasher cheapest"`

3. **Make ChromaDB required**
   - Add to requirements.txt
   - Fail loudly on startup if not available (not silently)

4. **Add embedding dimension lock**
   - Store `{model_name, dimension}` in ChromaDB collection metadata
   - On init, verify all collections use same model. If mismatch, log warning and offer re-embed migration.

5. **Increase RAG token budget** from 2000 to dynamic (4000-8000 depending on task)

### Phase B: Semantic Search Upgrade (Day 2, ~4 hours)

6. **Semantic memory recall** — augment `recall_memory()` with vector search
7. **Semantic skill matching** — embed skill descriptions, vector search alongside regex
8. **Relevance threshold filtering** — skip results with cosine distance > 0.7
9. **Better deduplication** — use embedding distance instead of word overlap Jaccard

### Phase C: Shopping RAG (Day 3, ~6 hours)

10. **Add `shopping` collection** to vector store
11. **Product embedding pipeline** — embed after scraping
12. **Review embedding** — embed synthesized reviews
13. **Shopping session embedding** — embed session summaries
14. **Connect shopping memory to vector store** — bridge `shopping_memory.db` entities

### Phase D: Knowledge Accumulation (Day 4, ~4 hours)

15. **Add `web_knowledge` collection**
16. **Web search -> embed** pipeline
17. **Cross-agent insight extraction**
18. **Conversation summarization** (every 10 exchanges)

### Phase E: Code Intelligence (Day 5, ~4 hours)

19. **Post-tool reindexing hooks**
20. **Startup re-scan**
21. **Call-graph edge storage**
22. **Test-to-code linking in embeddings**

### Phase F: Advanced RAG (Day 6-7, ~6 hours)

23. **HyDE query expansion** — generate hypothetical answer, embed that for retrieval
24. **Dynamic budget manager** — scale RAG budget to model context window
25. **Optional cross-encoder reranker** — `cross-encoder/ms-marco-MiniLM-L-6-v2` on CPU for top-20 -> top-5 reranking
26. **Query decomposition** for complex multi-part questions

---

## Part 5: Technical Specifications

### 5.1 Embedding Pipeline Architecture

```
Text Input
    |
    v
[Preprocessor]
    |-- Detect language (TR/EN/mixed)
    |-- For multilingual-e5: prepend "query: " or "passage: " prefix
    |-- Truncate to model max (512 tokens for e5-small)
    |-- Handle code vs natural language differently
    |
    v
[Embedding Model]
    |-- Primary: multilingual-e5-small (CPU, sentence-transformers)
    |-- Optional: Ollama nomic-embed-text (GPU, when available)
    |
    v
[Cache Layer]
    |-- In-memory LRU cache (5000 entries, actual LRU not insertion-order)
    |-- Optional: SQLite cache for cross-restart persistence
    |
    v
[Storage]
    |-- ChromaDB collection with cosine similarity HNSW index
```

### 5.2 Collection Schema

| Collection | Document Text | Key Metadata | Expected Size |
|------------|--------------|--------------|---------------|
| `episodic` | Task title + description + result summary | task_id, agent_type, success, model_used, cost, duration, timestamp | 5K-10K |
| `semantic` | Facts, preferences, insights, memory values | type, category, source, importance, chat_id, confidence | 5K-10K |
| `codebase` | Function/class signature + docstring + preview | filepath, symbol_name, symbol_type, line_start, line_end, language | 10K-15K |
| `errors` | Error description + root cause + fix | task_id, error_signature, prevention_hint | 2K-5K |
| `conversations` | User message + AI response preview | chat_id, task_id, task_title, timestamp | 3K-5K |
| **`shopping`** (new) | Product name + specs / review text / session summary | source, price, category, user_id, product_url, data_type | 5K-10K |
| **`web_knowledge`** (new) | Search result title + snippet / extracted content | query, source_url, fetched_at, ttl | 3K-5K |

### 5.3 GPU/CPU Resource Management

```
[NVIDIA GPU]
    |-- Primary: llama-server (LLM inference)
    |-- Secondary: Ollama embedding (when llama-server idle)
    |
[CPU]
    |-- Primary: sentence-transformers (embedding, always available)
    |-- ChromaDB HNSW search (CPU-bound but fast)
    |-- Tree-sitter parsing
    |-- All other Python processing
```

**Scheduling strategy:**
- Embedding is CPU-primary (multilingual-e5-small via sentence-transformers)
- Ollama embedding is an optional GPU boost, not a requirement
- Batch embedding jobs (indexing, re-scan) run during idle periods
- Individual query embeddings are always fast (cached or CPU)

### 5.4 Turkish Language Handling

| Component | Turkish Support Strategy |
|-----------|------------------------|
| Embedding model | multilingual-e5-small handles Turkish natively (trained on 100+ languages) |
| Text preprocessing | `normalize_turkish()` from shopping/text_utils.py for consistent character handling |
| Query expansion | Detect Turkish queries, optionally generate English equivalent for bilingual search |
| Product names | Embed both original Turkish and normalized forms |
| Stop words | Use language-specific stop word lists for chunking (not for embedding) |

### 5.5 Key Configuration Constants

```python
# embeddings_config.py (new)
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSION = 384
EMBEDDING_MAX_TOKENS = 512    # model limit
EMBEDDING_CACHE_SIZE = 5000
EMBEDDING_BATCH_SIZE = 32

# rag_config.py (new)
RAG_MIN_BUDGET = 2000
RAG_MAX_BUDGET = 12000
RAG_BUDGET_FRACTION = 0.15    # of available context window
RAG_MIN_RELEVANCE = 0.3       # cosine distance threshold
RAG_DEDUP_THRESHOLD = 0.85    # embedding similarity
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # optional
RERANKER_ENABLED = False       # enable in Phase F
```

---

## Part 6: Dependency Matrix

```
Phase A (Foundation) ──────── no dependencies
    |
    v
Phase B (Semantic Search) ─── depends on A (working embeddings)
    |
    v
Phase C (Shopping RAG) ────── depends on A (working embeddings)
    |                          can run parallel with B
    v
Phase D (Knowledge) ────────── depends on A
    |                          can run parallel with B, C
    v
Phase E (Code Intelligence) ── depends on A
    |                          can run parallel with B, C, D
    v
Phase F (Advanced RAG) ─────── depends on B (semantic search working)
                               benefits from C, D, E (more data to rank)
```

**Critical path:** A -> B -> F
**Parallel work:** C, D, E can all proceed independently after A is done.

---

## Part 7: Expected Impact

| Metric | Current | After Phase A-B | After All Phases |
|--------|---------|----------------|-----------------|
| Memory recall accuracy | Keyword-only (~30% relevant) | Semantic search (~70%) | Semantic + reranked (~85%) |
| Code context relevance | Good for indexed projects | Same + incremental updates | + call-graph, test links |
| Shopping query understanding | Rule-based | + product embeddings | + review/session/history RAG |
| Turkish query quality | Poor (English models) | Good (multilingual-e5) | Same |
| Knowledge reuse | Per-mission only | Cross-mission semantic | + web knowledge, cross-agent |
| Context window utilization | ~5% (2K/128K) | ~10% (8K/128K) | ~15% dynamic |
| Agent learning from past | Episodic only | + semantic facts, skills | + conversation, web, shopping |

---

## Appendix: Files to Modify

| File | Changes |
|------|---------|
| `src/memory/embeddings.py` | Fix import bug, switch to multilingual-e5, add batch embedding, dimension lock |
| `src/memory/vector_store.py` | Add `shopping` and `web_knowledge` collections, make chromadb required |
| `src/memory/rag.py` | Dynamic token budget, relevance threshold, better dedup, HyDE (Phase F) |
| `src/memory/skills.py` | Add vector-based skill matching alongside regex |
| `src/memory/conversations.py` | Add summarization pipeline |
| `src/memory/preferences.py` | Replace keyword patterns with LLM-based extraction |
| `src/agents/base.py` | Dynamic RAG budget, post-tool reindexing hooks |
| `src/context/assembler.py` | Better code context ranking, larger budget |
| `src/parsing/code_embeddings.py` | Post-tool hooks, test linking, call-graph |
| `src/shopping/cache.py` | Bridge to vector store on cache refresh |
| `src/tools/web_search.py` | Embed search results into web_knowledge collection |
| `src/infra/db.py` | Augment `store_memory` with vector embedding, add `semantic_recall` |
| `requirements.txt` | Add `sentence-transformers`, `chromadb` as required |
