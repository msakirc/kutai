# KutAI Extraction Report v1

**Date:** 2026-03-31
**Status:** Analysis complete, extraction-ready refactoring done for 7 modules

## Strategy

**Don't extract anything yet.** Make modules extractable without actually extracting them. This gives 80% of the architectural benefit with 0% of the maintenance tax. Extract when a module stops changing for ~4-6 weeks.

**Why not now:**
- Dual maintenance (library + consumer) slows a WIP project
- Interface rigidity — published APIs can't freely reshape
- Premature abstraction freezes assumptions that are still evolving

## Extraction-Ready Refactoring (Completed)

7 modules decoupled from `src.infra.logging_config` — now use stdlib `logging.getLogger(__name__)` while still routing through root logger sinks. Injectable dependencies added where needed.

| Module | Changes |
|--------|---------|
| `src/memory/embeddings.py` | stdlib logging |
| `src/memory/vector_store.py` | stdlib logging + injectable `embed_fn`/`dimension_fn` via `init_store()` |
| `src/tools/shell.py` | stdlib logging |
| `src/tools/free_apis.py` | stdlib logging (DB imports already lazy) |
| `src/core/llm_dispatcher.py` | stdlib logging + 17 kv-arg log calls converted to f-strings |
| `src/tools/web_search.py` | stdlib logging + lazy `run_shell` import via `_get_shell_fn()` |
| `src/memory/rag.py` | stdlib logging + lazy vector_store/embeddings imports via `_load_deps()` |

## Extraction Candidates

Each candidate evaluated on two axes: what it does, and why the open-source/LLM community would care.

### Tier 1: Genuinely Novel

#### LLM Dispatcher (`src/core/llm_dispatcher.py`)
- **What it does:** Centralized LLM call coordinator with swap budgets (max N swaps per time window), call categorization (MAIN_WORK vs OVERHEAD), deferred grading queue, cold-start wait logic, and backpressure signaling.
- **Why it could be valuable:** Nothing like this exists in OSS. LiteLLM does routing but not swap budgets or deferred grading. vLLM and llama.cpp users managing local models on limited VRAM have no coordination layer — they manually restart servers. This solves the "I have one GPU and multiple model needs" problem that every local LLM user faces.

#### Model Router (`src/core/router.py`)
- **What it does:** Multi-dimensional task-aware model selection across 14+ scoring dimensions (context length, speed, capability, cost, GPU fit, thinking support, language proficiency). Circuit breakers per model, rate limiting, loaded-model affinity boosting, and runtime state awareness.
- **Why it could be valuable:** Most routing solutions are simple round-robin or cost-based. This is the only router that scores models on task characteristics (coding vs translation vs analysis), respects GPU constraints, and prefers the already-loaded model to avoid swaps. Useful for anyone running multiple local models or mixing local + cloud.

### Tier 2: Solid Utility

#### Web Search Pipeline (`src/tools/web_search.py`)
- **What it does:** Multi-tier search with intent inference (navigational vs informational vs time-sensitive), source quality tracking, DDG + Brave + GCSE fallback chain, search guard (blocks redundant/low-value searches), and depth control.
- **Why it could be valuable:** Every agent framework hand-rolls web search. Most stop at "call DuckDuckGo API." This adds the missing layers: intent-aware query rewriting, automatic fallback when a source fails, and quality scoring that improves over time. The search guard alone saves significant API costs by blocking searches that won't return useful results.

#### Tiered Web Scraper (`src/tools/scraper.py`)
- **What it does:** 4-tier scraper escalation: plain HTTP -> TLS fingerprint rotation -> stealth headers -> headless browser. Automatic tier selection based on target site's known requirements. Response quality validation.
- **Why it could be valuable:** Most scraping libraries are single-tier. This handles the real-world progression where sites block basic requests and you need to escalate. The automatic tier selection means consumers don't need to know which sites need which approach.

#### RAG Engine (`src/memory/rag.py`)
- **What it does:** Lightweight RAG with lazy-loaded dependencies, budget-based retrieval (limits token spend on context), and integration with ChromaDB vector store.
- **Why it could be valuable:** Lighter alternative to LlamaIndex/LangChain for projects that just need "embed, store, retrieve" without a 200-dependency framework. The lazy loading means it imports in <10ms even with heavy dependencies.

#### Free API Registry (`src/tools/free_apis.py`)
- **What it does:** Curated registry of 13 free APIs (weather, exchange rates, IP geolocation, etc.) with auto-discovery from public-apis/free-apis directories. Runtime health checking and automatic fallback.
- **Why it could be valuable:** Useful as a dependency-free utility for any agent that needs real-world data. The auto-discovery means the registry grows without manual updates.

### Tier 3: Niche / Low Standalone Value

#### Embedding Bridge (`src/memory/embeddings.py`)
- **What it does:** Unified embedding interface with LRU cache, CPU/GPU fallback, and multilingual-e5-base default.
- **Why it could be valuable:** Marginal — langchain-embeddings and sentence-transformers already cover this. The caching is the only differentiator.

#### Vector Store Wrapper (`src/memory/vector_store.py`)
- **What it does:** ChromaDB wrapper with semantic collections, injectable embedding function, and init-time dimension validation.
- **Why it could be valuable:** Low — ChromaDB's own API is already clean. This adds injectable deps and lazy loading, but not enough to justify a standalone package.

#### Shell Sandbox (`src/tools/shell.py`)
- **What it does:** Sandboxed shell execution with blocked command patterns, timeout management, and output capture.
- **Why it could be valuable:** Low — e2b, modal, and Docker SDK exist. This is simpler but not novel enough to publish.

#### Shopping Scrapers (`src/shopping/scrapers/`)
- **What it does:** 15 Turkish e-commerce scrapers with standardized output schema.
- **Why it could be valuable:** Niche — only useful for Turkish market. Not generalizable.

## Recommendations

1. **Next to decouple:** `router.py` and `llm_dispatcher.py` are the highest-value extraction candidates. They still import from `src.models` and `src.infra` — those need lazy loading.
2. **Watch for stability:** When `llm_dispatcher` and `router` stop changing for 4-6 weeks, they're ready to extract as a `llm-router` package.
3. **Web search + scraper** could become a `web-search-pipeline` package once the search guard and scraper tier logic stabilize.
4. **Don't extract** embeddings, vector_store, shell, or shopping scrapers — they don't have enough standalone value to justify the maintenance cost.
